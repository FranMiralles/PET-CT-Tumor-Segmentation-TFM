"""Build a YOLO detection dataset of axial CT slices with an optional PET channel.

Task
----
The Lung-PET-CT-Dx annotations are 2D bounding boxes drawn on **CT** slices
(PASCAL-VOC XML, one file per annotated slice, named after the slice SOP UID).
We therefore frame the problem as **2D lesion detection on axial CT slices** and
collapse the noisy histology labels into a single ``lesion`` class.

Input channels (per slice, 8-bit RGB so COCO-pretrained YOLO weights apply)
---------------------------------------------------------------------------
  - channel 0 : CT, lung window        (WL=-600, WW=1500)
  - channel 1 : CT, mediastinal window (WL=  40, WW= 400)
  - channel 2 : PET SUV-bw, resampled onto the CT slice grid; **all zeros when
                the patient has no PET**.

Because 222/355 patients are CT-only, the PET channel is genuinely empty for the
majority of training images, so the detector cannot become PET-dependent. The
optional ``pet_dropout`` augmentation can blank the PET channel even more often.

Geometry
--------
CT and PET are both acquired axially with identity orientation
(``ImageOrientationPatient == [1,0,0,0,1,0]``) in this dataset, so resampling PET
to a CT slice reduces to a per-axis coordinate mapping using the DICOM
``ImagePositionPatient`` / ``PixelSpacing`` tags (see :func:`resample_pet_slice`).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import map_coordinates

# CT display windows (window level, window width) in Hounsfield Units.
CT_WINDOWS = {"lung": (-600, 1500), "mediastinum": (40, 400)}
# SUV value mapped to the top of the PET channel (typical lesion SUVs are < 5-10).
SUV_CLIP = 5.0


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
def parse_annotation_xml(path: str | Path) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """
    Parses one PASCAL-VOC XML annotation file. The per-object class name is
    intentionally ignored: the dataset's class field is inconsistent (mixed case,
    leaked patient IDs) and histology is not reliably inferable from a single box,
    so every object is treated as one ``lesion``.
    Params:
        - path: str | Path, path to the XML file
    Returns:
        - width: int, annotated image width in pixels
        - height: int, annotated image height in pixels
        - boxes: list[tuple[int, int, int, int]], list of (xmin, ymin, xmax, ymax)
    """
    root = ET.parse(path).getroot()
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        boxes.append((
            int(round(float(bb.findtext("xmin")))),
            int(round(float(bb.findtext("ymin")))),
            int(round(float(bb.findtext("xmax")))),
            int(round(float(bb.findtext("ymax")))),
        ))
    return width, height, boxes


def boxes_to_yolo(
    boxes: list[tuple[int, int, int, int]], width: int, height: int
) -> list[str]:
    """
    Converts pixel boxes to normalized YOLO label lines. Boxes are clipped to the
    image and degenerate (zero-area) boxes are dropped.
    Params:
        - boxes: list[tuple[int, int, int, int]], (xmin, ymin, xmax, ymax) per box
        - width: int, image width in pixels
        - height: int, image height in pixels
    Returns:
        - lines: list[str], one "0 cx cy w h" line per valid box (class 0 = lesion,
          all coordinates normalized to [0, 1])
    """
    lines = []
    for xmin, ymin, xmax, ymax in boxes:
        xmin, xmax = sorted((max(0, xmin), min(width, xmax)))
        ymin, ymax = sorted((max(0, ymin), min(height, ymax)))
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


# ---------------------------------------------------------------------------
# CT windowing
# ---------------------------------------------------------------------------
def window_ct(hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """
    Maps a Hounsfield-Unit array to uint8 [0, 255] using a display window.
    Params:
        - hu: np.ndarray, CT slice in Hounsfield Units
        - wl: float, window level (center)
        - ww: float, window width
    Returns:
        - out: np.ndarray, uint8 array of the same shape, clipped to the window
    """
    lo, hi = wl - ww / 2, wl + ww / 2
    out = np.clip((hu - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# PET: SUV conversion + resampling onto a CT slice
# ---------------------------------------------------------------------------
def _parse_dicom_time(value: str | None) -> float | None:
    """
    Parses a DICOM TM string 'HHMMSS(.ffffff)' into seconds since midnight.
    Params:
        - value: str | None, DICOM time string
    Returns:
        - seconds: float | None, seconds since midnight, or None if empty/missing
    """
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    hh, mm = int(value[0:2]), int(value[2:4])
    ss = float(value[4:]) if len(value) > 4 else 0.0
    return hh * 3600 + mm * 60 + ss


def suv_bw_factor(meta_row: dict) -> float | None:
    """
    Computes the body-weight SUV scale factor for a PET series such that
    ``SUV = BQML * factor``. Uses the injected dose, decay-corrected from the
    radiopharmaceutical start time to the acquisition time, and the patient weight.
    Params:
        - meta_row: dict, mapping with keys patient_weight, radionuclide_total_dose,
          radionuclide_half_life, radiopharmaceutical_start_time, acquisition_time
    Returns:
        - factor: float | None, the BQML->SUV-bw factor, or None when any required
          field is missing (caller should fall back to robust per-slice normalisation)
    """
    weight = meta_row.get("patient_weight")
    dose = meta_row.get("radionuclide_total_dose")
    half_life = meta_row.get("radionuclide_half_life")
    t_start = _parse_dicom_time(meta_row.get("radiopharmaceutical_start_time"))
    t_scan = _parse_dicom_time(meta_row.get("acquisition_time"))

    if not all(v not in (None, 0) for v in (weight, dose, half_life)):
        return None
    if t_start is None or t_scan is None:
        return None

    dt = t_scan - t_start
    if dt < 0:                       # crossed midnight
        dt += 24 * 3600
    decayed_dose = dose * math.exp(-math.log(2) * dt / half_life)
    if decayed_dose <= 0:
        return None
    # BQML is Bq/ml; SUV-bw = activity / (dose / weight_in_grams)
    return (weight * 1000.0) / decayed_dose


def load_pet_volume(pet_files: list[str], suv_factor: float | None = None) -> dict | None:
    """
    Loads a patient's PET series into a single volume plus its geometry, applying
    the SUV factor when provided.
    Params:
        - pet_files: list[str], paths to the PET DICOM instances of one patient
        - suv_factor: float | None, output of suv_bw_factor; if None the volume is
          left in BQML and robustly normalised later
    Returns:
        - pet: dict | None, {'volume': (Z,H,W) float32, 'z': (Z,) sorted ascending,
          'origin_xy': (x0,y0), 'spacing_xy': (sx,sy), 'is_suv': bool}, or None if
          the series is empty
    """
    slices = []
    for f in pet_files:
        ds = pydicom.dcmread(f, force=True)
        ipp = ds.get("ImagePositionPatient")
        if ipp is None:
            continue
        slope = float(ds.get("RescaleSlope", 1.0))
        intercept = float(ds.get("RescaleIntercept", 0.0))
        bqml = ds.pixel_array.astype(np.float32) * slope + intercept
        slices.append((float(ipp[2]), float(ipp[0]), float(ipp[1]), bqml,
                       [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]))
    if not slices:
        return None

    slices.sort(key=lambda s: s[0])
    z = np.array([s[0] for s in slices], dtype=np.float32)
    volume = np.stack([s[3] for s in slices]).astype(np.float32)
    x0, y0 = slices[0][1], slices[0][2]
    sy, sx = slices[0][4]            # PixelSpacing is [row(y), col(x)]

    is_suv = suv_factor is not None
    if is_suv:
        volume = volume * suv_factor
    return {"volume": volume, "z": z, "origin_xy": (x0, y0),
            "spacing_xy": (sx, sy), "is_suv": is_suv}


def resample_pet_slice(
    pet: dict,
    ct_ipp: Sequence[float],
    ct_spacing: Sequence[float],
    ct_shape: tuple[int, int],
) -> np.ndarray:
    """
    Resamples the PET volume onto a single CT slice grid (linear interpolation).
    Assumes axial identity orientation for both modalities (verified for this
    dataset). Voxels whose physical z falls outside the PET extent are set to 0.
    Params:
        - pet: dict, output of load_pet_volume
        - ct_ipp: Sequence[float], CT slice ImagePositionPatient (x, y, z) in mm
        - ct_spacing: Sequence[float], CT PixelSpacing [row(y), col(x)] in mm
        - ct_shape: tuple[int, int], (rows, cols) of the CT slice
    Returns:
        - sampled: np.ndarray, PET values on the CT grid, shape ct_shape (float32)
    """
    rows, cols = ct_shape
    cx0, cy0, cz = float(ct_ipp[0]), float(ct_ipp[1]), float(ct_ipp[2])
    csy, csx = float(ct_spacing[0]), float(ct_spacing[1])

    if cz < pet["z"].min() or cz > pet["z"].max():
        return np.zeros(ct_shape, dtype=np.float32)

    x0, y0 = pet["origin_xy"]
    psx, psy = pet["spacing_xy"]

    # Physical coordinates of every CT pixel -> continuous PET indices.
    cols_phys = cx0 + np.arange(cols) * csx
    rows_phys = cy0 + np.arange(rows) * csy
    pj = (cols_phys - x0) / psx                       # PET column index per CT col
    pi = (rows_phys - y0) / psy                       # PET row index per CT row
    pk = float(np.interp(cz, pet["z"], np.arange(len(pet["z"]))))  # PET slice index

    grid_i, grid_j = np.meshgrid(pi, pj, indexing="ij")
    grid_k = np.full(grid_i.shape, pk, dtype=np.float32)
    coords = np.stack([grid_k, grid_i, grid_j])
    sampled = map_coordinates(pet["volume"], coords, order=1, mode="constant", cval=0.0)
    return sampled.astype(np.float32)


def pet_to_uint8(pet_slice: np.ndarray, is_suv: bool) -> np.ndarray:
    """
    Scales a PET slice to uint8: SUV clipped to [0, SUV_CLIP] when calibrated,
    otherwise a robust (99.5th-percentile) normalisation of the raw activity.
    Params:
        - pet_slice: np.ndarray, PET values on the target grid
        - is_suv: bool, True if the values are SUV-bw, False if raw BQML
    Returns:
        - out: np.ndarray, uint8 array of the same shape
    """
    if is_suv:
        out = np.clip(pet_slice / SUV_CLIP, 0, 1)
    else:
        hi = np.percentile(pet_slice, 99.5)
        out = np.clip(pet_slice / hi, 0, 1) if hi > 0 else np.zeros_like(pet_slice)
    return (out * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-slice image assembly
# ---------------------------------------------------------------------------
def build_slice_image(
    ct_path: str | Path, pet: dict | None = None
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Builds the 3-channel uint8 image for one annotated CT slice
    ([CT lung window, CT mediastinal window, PET]; PET channel is zeros if absent).
    Params:
        - ct_path: str | Path, path to the CT DICOM slice
        - pet: dict | None, output of load_pet_volume, or None for CT-only patients
    Returns:
        - image: np.ndarray, (H, W, 3) uint8 image
        - shape: tuple[int, int], (rows, cols) of the slice
    """
    ds = pydicom.dcmread(ct_path, force=True)
    slope = float(ds.get("RescaleSlope", 1.0))
    intercept = float(ds.get("RescaleIntercept", -1024.0))
    hu = ds.pixel_array.astype(np.float32) * slope + intercept
    rows, cols = hu.shape

    ch0 = window_ct(hu, *CT_WINDOWS["lung"])
    ch1 = window_ct(hu, *CT_WINDOWS["mediastinum"])

    if pet is not None:
        pet_slice = resample_pet_slice(
            pet, ds.ImagePositionPatient, ds.PixelSpacing, (rows, cols))
        ch2 = pet_to_uint8(pet_slice, pet["is_suv"])
    else:
        ch2 = np.zeros((rows, cols), dtype=np.uint8)

    return np.stack([ch0, ch1, ch2], axis=-1), (rows, cols)


def save_image(image_hwc: np.ndarray, path: str | Path) -> None:
    """
    Saves an (H, W, 3) uint8 array as a PNG, creating parent directories.
    Params:
        - image_hwc: np.ndarray, (H, W, 3) uint8 image
        - path: str | Path, destination PNG path
    Returns:
        - None
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_hwc, mode="RGB").save(path)
