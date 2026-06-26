"""Command-line entry point that builds the YOLO PET/CT lesion-detection dataset.

All the building logic lives in ``src.data.location_dataset``; this pipeline only
parses the arguments and calls :func:`src.data.location_dataset.build_dataset`.

Run from the project root, e.g.::

    python -m src.pipelines.build_location_dataset --background-ratio 1.0

The heavy DICOM metadata scan is read from the cached parquet produced by
``src.data.dicom_metadata.build_metadata_dataframe`` (the image-metadata EDA).
"""

from __future__ import annotations

import argparse

from src.data.location_dataset import build_dataset


def main() -> None:
    """
    Command-line entry point: parses arguments and runs build_dataset.
    Params:
        - None (arguments are read from the command line)
    Returns:
        - None
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/processed/yolo_petct")
    ap.add_argument("--meta-parquet", default="data/processed/image_metadata.parquet")
    ap.add_argument("--annotation-root", default="data/raw/annotation")
    ap.add_argument("--image-root", default="data/raw/images")
    ap.add_argument("--background-ratio", type=float, default=1.0,
                    help="Background slices per positive slice (0 disables).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_dataset(args.out, args.meta_parquet, args.annotation_root,
                  args.image_root, args.background_ratio, args.seed)


if __name__ == "__main__":
    main()
