"""Extraction of DICOM header metadata for the Lung-PET-CT-Dx dataset.

The dataset under ``data/raw/images`` is organised as::

    data/raw/images/<Patient>/<Study>/<Series>/<instance>.dcm

with ~250k DICOM instances spread over CT and PT (PET) series. This module
reads the header of every instance (the pixel data is skipped) and flattens a
curated set of tags into a tidy ``pandas`` DataFrame with one row per instance.

The relevant tags are grouped in three families:

  - GENERAL : identifiers, geometry and pixel representation shared by every
    modality.
  - CT      : acquisition parameters specific to computed tomography.
  - PET     : quantification parameters specific to positron emission
    tomography, including the radiopharmaceutical information sequence.

Because the full scan is expensive, :func:`build_metadata_dataframe` caches its
result to a parquet file and reuses it on subsequent calls.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pydicom
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Tag specification
# ---------------------------------------------------------------------------
# Each entry maps a pydicom keyword to (column_name, converter). The converter
# turns the raw DICOM value into a plain Python scalar so the resulting frame
# has clean, analysable dtypes.

def _to_float(value) -> float | None:
    """
    Converts a raw DICOM element value to float, returning None on failure.
    Params:
        - value: any, raw DICOM element value
    Returns:
        - result: float | None, the float value or None if not convertible
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    """
    Converts a raw DICOM element value to int, returning None on failure.
    Params:
        - value: any, raw DICOM element value
    Returns:
        - result: int | None, the int value or None if not convertible
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_str(value) -> str | None:
    """
    Converts a raw DICOM element value to str, mapping empty/None to None.
    Params:
        - value: any, raw DICOM element value
    Returns:
        - result: str | None, the string value or None if empty/missing
    """
    if value is None or value == "":
        return None
    return str(value)


def _join_multi(value) -> str | None:
    """
    Joins a multi-valued DICOM element (e.g. ImageType) into a back-slash string.
    Params:
        - value: any, scalar, list or pydicom MultiValue element
    Returns:
        - result: str | None, back-slash-joined string, the scalar as str, or
          None if missing
    """
    if value is None:
        return None
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return "\\".join(str(v) for v in value)
    return str(value)


# keyword -> (column, converter)
GENERAL_TAGS = {
    "PatientID": ("patient_id", _to_str),
    "PatientSex": ("patient_sex", _to_str),
    "PatientAge": ("patient_age", _to_str),
    "PatientSize": ("patient_size", _to_float),
    "PatientWeight": ("patient_weight", _to_float),
    "PatientPosition": ("patient_position", _to_str),
    "Modality": ("modality", _to_str),
    "Manufacturer": ("manufacturer", _to_str),
    "ManufacturerModelName": ("model_name", _to_str),
    "SoftwareVersions": ("software_versions", _join_multi),
    "StudyDate": ("study_date", _to_str),
    "SeriesDate": ("series_date", _to_str),
    "StudyDescription": ("study_description", _to_str),
    "SeriesDescription": ("series_description", _to_str),
    "ProtocolName": ("protocol_name", _to_str),
    "BodyPartExamined": ("body_part", _to_str),
    "StudyInstanceUID": ("study_uid", _to_str),
    "SeriesInstanceUID": ("series_uid", _to_str),
    "SOPInstanceUID": ("sop_uid", _to_str),
    "FrameOfReferenceUID": ("frame_of_reference_uid", _to_str),
    "ImageType": ("image_type", _join_multi),
    "Rows": ("rows", _to_int),
    "Columns": ("columns", _to_int),
    "SliceThickness": ("slice_thickness", _to_float),
    "SpacingBetweenSlices": ("spacing_between_slices", _to_float),
    "SliceLocation": ("slice_location", _to_float),
    "PhotometricInterpretation": ("photometric", _to_str),
    "SamplesPerPixel": ("samples_per_pixel", _to_int),
    "BitsAllocated": ("bits_allocated", _to_int),
    "BitsStored": ("bits_stored", _to_int),
    "PixelRepresentation": ("pixel_representation", _to_int),
    "RescaleSlope": ("rescale_slope", _to_float),
    "RescaleIntercept": ("rescale_intercept", _to_float),
    "RescaleType": ("rescale_type", _to_str),
    "WindowCenter": ("window_center", _to_str),
    "WindowWidth": ("window_width", _to_str),
    "InstanceNumber": ("instance_number", _to_int),
    "SeriesNumber": ("series_number", _to_int),
    "AcquisitionNumber": ("acquisition_number", _to_int),
}

CT_TAGS = {
    "KVP": ("kvp", _to_float),
    "XRayTubeCurrent": ("xray_tube_current", _to_float),
    "Exposure": ("exposure", _to_float),
    "ExposureTime": ("exposure_time", _to_float),
    "GeneratorPower": ("generator_power", _to_float),
    "FilterType": ("filter_type", _to_str),
    "FocalSpots": ("focal_spots", _join_multi),
    "ConvolutionKernel": ("convolution_kernel", _join_multi),
    "ScanOptions": ("scan_options", _join_multi),
    "SpiralPitchFactor": ("spiral_pitch_factor", _to_float),
    "SingleCollimationWidth": ("single_collimation_width", _to_float),
    "TotalCollimationWidth": ("total_collimation_width", _to_float),
    "GantryDetectorTilt": ("gantry_tilt", _to_float),
    "ReconstructionDiameter": ("reconstruction_diameter", _to_float),
    "DataCollectionDiameter": ("data_collection_diameter", _to_float),
    "TableHeight": ("table_height", _to_float),
}

PET_TAGS = {
    "Units": ("pet_units", _to_str),
    "DecayCorrection": ("decay_correction", _to_str),
    "CorrectedImage": ("corrected_image", _join_multi),
    "ActualFrameDuration": ("frame_duration", _to_float),
    "AcquisitionTime": ("acquisition_time", _to_str),
}


def _extract_pixel_spacing(ds: "pydicom.Dataset", record: dict) -> None:
    """
    Splits PixelSpacing into row/column floats (mm) and writes them into record.
    Params:
        - ds: pydicom.Dataset, the read DICOM header
        - record: dict, the per-instance record, mutated in place
    Returns:
        - None
    """
    ps = ds.get("PixelSpacing")
    if ps is not None and len(ps) == 2:
        record["pixel_spacing_row"] = _to_float(ps[0])
        record["pixel_spacing_col"] = _to_float(ps[1])
    else:
        record["pixel_spacing_row"] = None
        record["pixel_spacing_col"] = None


def _extract_image_position_z(ds: "pydicom.Dataset", record: dict) -> None:
    """
    Writes the axial (z) coordinate of ImagePositionPatient into record.
    Params:
        - ds: pydicom.Dataset, the read DICOM header
        - record: dict, the per-instance record, mutated in place
    Returns:
        - None
    """
    ipp = ds.get("ImagePositionPatient")
    record["image_position_z"] = _to_float(ipp[2]) if ipp is not None and len(ipp) == 3 else None


def _extract_radiopharmaceutical(ds: "pydicom.Dataset", record: dict) -> None:
    """
    Pulls the relevant fields out of RadiopharmaceuticalInformationSequence into
    record (radiopharmaceutical, total dose, half-life, start time).
    Params:
        - ds: pydicom.Dataset, the read DICOM header
        - record: dict, the per-instance record, mutated in place
    Returns:
        - None
    """
    record["radiopharmaceutical"] = None
    record["radionuclide_total_dose"] = None
    record["radionuclide_half_life"] = None
    record["radiopharmaceutical_start_time"] = None

    seq = ds.get("RadiopharmaceuticalInformationSequence")
    if seq is None or len(seq) == 0:
        return
    item = seq[0]
    record["radiopharmaceutical"] = _to_str(item.get("Radiopharmaceutical"))
    record["radionuclide_total_dose"] = _to_float(item.get("RadionuclideTotalDose"))
    record["radionuclide_half_life"] = _to_float(item.get("RadionuclideHalfLife"))
    record["radiopharmaceutical_start_time"] = _to_str(item.get("RadiopharmaceuticalStartTime"))


# All scalar tag families merged once, used by the extractor.
_SCALAR_TAGS = {**GENERAL_TAGS, **CT_TAGS, **PET_TAGS}


def extract_file_metadata(path: str | Path, root: str | Path | None = None) -> dict:
    """
    Reads one DICOM file header and returns a flat metadata record (pixel data is
    skipped).
    Params:
        - path: str | Path, path to the .dcm instance
        - root: str | Path | None, dataset root; when given, `patient_folder` is
          derived from the first path component relative to `root`
          (e.g. Lung_Dx-A0001), which is more reliable than the de-identified
          PatientID tag
    Returns:
        - record: dict, one record with the curated tags; on a read error it
          contains the file path and a `read_error` message, with all tags None
    """
    path = Path(path)
    record = {"file_path": str(path)}

    if root is not None:
        try:
            record["patient_folder"] = path.relative_to(root).parts[0]
        except ValueError:
            record["patient_folder"] = None

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:  # corrupt / non-DICOM file
        for _, (column, _conv) in _SCALAR_TAGS.items():
            record[column] = None
        record["read_error"] = str(exc)
        return record

    record["read_error"] = None
    for keyword, (column, converter) in _SCALAR_TAGS.items():
        record[column] = converter(ds.get(keyword))

    _extract_pixel_spacing(ds, record)
    _extract_image_position_z(ds, record)
    _extract_radiopharmaceutical(ds, record)
    return record


def iter_dicom_files(root: str | Path) -> Iterator[Path]:
    """
    Yields every .dcm file path under root (recursively).
    Params:
        - root: str | Path, directory to walk
    Returns:
        - paths: Iterator[Path], generator of .dcm file paths
    """
    root = Path(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".dcm"):
                yield Path(dirpath) / name


def build_metadata_dataframe(
    root: str | Path = "data/raw/images",
    cache_path: str | Path | None = "data/processed/image_metadata.parquet",
    force: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Builds (and caches) the per-instance DICOM metadata DataFrame. The header of
    every .dcm file under root is read once, written to cache_path (parquet) and
    reused on later calls unless force is True.
    Params:
        - root: str | Path, directory holding the DICOM tree
        - cache_path: str | Path | None, where to store/read the parquet cache;
          no cache is used when None
        - force: bool, recompute even if the cache file exists
        - show_progress: bool, display a tqdm progress bar during the scan
    Returns:
        - df: pd.DataFrame, one row per DICOM instance
    """
    root = Path(root)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return pd.read_parquet(cache_path)

    files = list(iter_dicom_files(root))
    iterator = tqdm(files, desc="Reading DICOM headers") if show_progress else files
    records = [extract_file_metadata(f, root=root) for f in iterator]

    df = pd.DataFrame.from_records(records)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df
