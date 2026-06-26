"""Logic to build the YOLO PET/CT lesion-detection dataset from raw DICOM + XML.

This module holds the building *methods*; the thin command-line entry point that
drives them lives in ``src.pipelines.build_location_dataset``.

:func:`build_dataset` produces, under its ``out_dir``::

    images/{train,val,test}/<patient>__<series>__<sop>.png   # 3-channel CT(+PET)
    labels/{train,val,test}/<...>.txt                        # YOLO boxes (class 0)
    data.yaml                                                # Ultralytics data config
    patient_splits.csv                                       # patient -> split
    manifest.csv                                             # one row per written image

The heavy DICOM metadata scan is read from the cached parquet produced by
``src.data.dicom_metadata.build_metadata_dataframe`` (the image-metadata EDA).
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.data.detection_dataset import (
    parse_annotation_xml, boxes_to_yolo, build_slice_image, save_image,
    load_pet_volume, suv_bw_factor,
)
from src.data.patient_split import make_patient_splits


def collect_annotations(annotation_root: str | Path) -> dict[str, str]:
    """
    Maps each annotated SOP UID to its XML path (the filename equals the SOP UID).
    Params:
        - annotation_root: str | Path, root folder of the PASCAL-VOC XML annotations
    Returns:
        - sop_to_xml: dict[str, str], {sop_uid: xml_path}
    """
    sop_to_xml = {}
    for x in glob.glob(os.path.join(annotation_root, "**", "*.xml"), recursive=True):
        sop_to_xml[os.path.basename(x)[:-4]] = x
    return sop_to_xml


def build_patient_table(annotated_rows: pd.DataFrame, pet_patients: set[str]) -> pd.DataFrame:
    """
    Builds a one-row-per-patient table with the cohort letter and PET availability,
    used as input to the stratified patient split.
    Params:
        - annotated_rows: pd.DataFrame, CT metadata rows of annotated slices
          (needs a `patient_folder` column)
        - pet_patients: set[str], patient_folder values that have a PET series
    Returns:
        - table: pd.DataFrame, columns patient, cohort (letter), has_pet (bool)
    """
    patients = sorted(annotated_rows["patient_folder"].unique())
    rows = [{
        "patient": p,
        "cohort": p.replace("Lung_Dx-", "")[0],
        "has_pet": p in pet_patients,
    } for p in patients]
    return pd.DataFrame(rows)


def sample_background_sops(
    series_rows: pd.DataFrame,
    positive_sops: set[str],
    n: int,
    rng: np.random.Generator,
    min_gap_mm: float = 20.0,
) -> list[str]:
    """
    Picks background (tumour-free) CT SOPs from a series, keeping only slices that
    are at least `min_gap_mm` away in z from every annotated slice.
    Params:
        - series_rows: pd.DataFrame, CT metadata rows of the candidate series
          (needs `sop_uid` and `image_position_z`)
        - positive_sops: set[str], SOP UIDs that are annotated (excluded as positives)
        - n: int, number of background SOPs to sample
        - rng: np.random.Generator, random generator for reproducible sampling
        - min_gap_mm: float, minimum z-distance (mm) to any annotated slice
    Returns:
        - sops: list[str], sampled background SOP UIDs (possibly fewer than n)
    """
    pos = series_rows[series_rows["sop_uid"].isin(positive_sops)]
    if pos.empty:
        return []
    pos_z = pos["image_position_z"].dropna().to_numpy()
    cand = series_rows[~series_rows["sop_uid"].isin(positive_sops)].copy()
    cand = cand.dropna(subset=["image_position_z"])
    if cand.empty or len(pos_z) == 0:
        return []
    far = cand[cand["image_position_z"].apply(
        lambda z: np.min(np.abs(pos_z - z)) >= min_gap_mm)]
    if far.empty:
        return []
    take = min(n, len(far))
    return far.sample(n=take, random_state=int(rng.integers(1e9)))["sop_uid"].tolist()


def build_dataset(
    out_dir: str | Path,
    meta_parquet: str | Path,
    annotation_root: str | Path,
    image_root: str | Path,
    background_ratio: float = 1.0,
    seed: int = 42,
) -> None:
    """
    Builds the full YOLO PET/CT lesion-detection dataset on disk: maps annotations
    to CT slices via the cached metadata, splits patients, and writes images,
    labels, data.yaml and the split/manifest CSVs.
    Params:
        - out_dir: str | Path, output directory for the YOLO dataset
        - meta_parquet: str | Path, cached DICOM metadata parquet
        - annotation_root: str | Path, root of the PASCAL-VOC XML annotations
        - image_root: str | Path, root of the raw DICOM images (kept for reference)
        - background_ratio: float, background slices per positive slice (0 disables)
        - seed: int, seed for the split and background sampling
    Returns:
        - None
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)

    df = pd.read_parquet(meta_parquet)
    ct = df[(df["modality"] == "CT") & (df["photometric"] == "MONOCHROME2")].copy()
    ct_by_sop = ct.set_index("sop_uid")
    pet_patients = set(df[df["modality"] == "PT"]["patient_folder"].unique())
    pet_files_by_patient = (
        df[df["modality"] == "PT"].groupby("patient_folder")["file_path"].apply(list).to_dict())
    pet_meta_by_patient = (
        df[df["modality"] == "PT"].groupby("patient_folder").first().to_dict("index"))

    # Annotated CT slices only.
    sop_to_xml = collect_annotations(annotation_root)
    annotated = ct_by_sop.index.intersection(sop_to_xml.keys())
    annotated_rows = ct.loc[ct["sop_uid"].isin(annotated)].copy()
    print(f"Annotated CT slices: {len(annotated_rows)} | "
          f"patients: {annotated_rows['patient_folder'].nunique()}")

    # Patient-level stratified split.
    patient_table = build_patient_table(annotated_rows, pet_patients)
    patient_table = make_patient_splits(patient_table, seed=seed)
    split_of = dict(zip(patient_table["patient"], patient_table["split"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_table.to_csv(out_dir / "patient_splits.csv", index=False)
    print(patient_table.groupby(["split", "has_pet"]).size())

    pos_by_patient = defaultdict(list)
    for _, r in annotated_rows.iterrows():
        pos_by_patient[r["patient_folder"]].append(r)

    manifest = []
    for patient, rows in tqdm(pos_by_patient.items(), desc="patients"):
        split = split_of[patient]

        # Load the PET volume once per patient (if available).
        pet = None
        if patient in pet_files_by_patient:
            factor = suv_bw_factor(pet_meta_by_patient.get(patient, {}))
            pet = load_pet_volume(pet_files_by_patient[patient], suv_factor=factor)

        positive_sops = {r["sop_uid"] for r in rows}

        # ---- positive slices ----
        for r in rows:
            width, height, boxes = parse_annotation_xml(sop_to_xml[r["sop_uid"]])
            image, (h, w) = build_slice_image(r["file_path"], pet=pet)
            stem = f"{patient}__{r['series_uid'][-8:]}__{r['sop_uid'][-8:]}"
            save_image(image, out_dir / "images" / split / f"{stem}.png")
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(boxes_to_yolo(boxes, w, h)))
            manifest.append({"patient": patient, "split": split, "stem": stem,
                             "kind": "positive", "n_boxes": len(boxes),
                             "has_pet": pet is not None})

        # ---- background slices (optional) ----
        if background_ratio > 0:
            n_bg = int(round(len(rows) * background_ratio))
            series_ids = annotated_rows.loc[
                annotated_rows["patient_folder"] == patient, "series_uid"].unique()
            bg_pool = ct[ct["series_uid"].isin(series_ids)]
            bg_sops = sample_background_sops(bg_pool, positive_sops, n_bg, rng)
            for sop in bg_sops:
                row = ct_by_sop.loc[sop]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                image, _ = build_slice_image(row["file_path"], pet=pet)
                stem = f"{patient}__{row['series_uid'][-8:]}__{sop[-8:]}__bg"
                save_image(image, out_dir / "images" / split / f"{stem}.png")
                (out_dir / "labels" / split / f"{stem}.txt").write_text("")
                manifest.append({"patient": patient, "split": split, "stem": stem,
                                 "kind": "background", "n_boxes": 0,
                                 "has_pet": pet is not None})

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "lesion"},
    }
    with open(out_dir / "data.yaml", "w") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)
    print(f"\nDone. Wrote {len(manifest)} images to {out_dir}")
