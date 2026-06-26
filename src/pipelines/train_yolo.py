"""Train a YOLOv12 lesion detector on the PET/CT slice dataset.

Expects the dataset produced by ``src.data.build_location_dataset`` (a
``data.yaml`` plus images/labels split by patient).

Channel layout of every image: [CT lung window, CT mediastinal window, PET SUV].
The PET channel is already empty for the 222 CT-only patients, so the detector
learns to work without PET. ``--pet-dropout`` additionally blanks the PET channel
of a random fraction of *training* batches, making the model explicitly robust to
missing PET (a form of modality dropout). Validation/test images are never
altered, so reported metrics reflect the real inputs.

Example::

    python -m src.pipelines.train_yolo --model yolo12m.pt --epochs 100 \
        --batch 32 --imgsz 512 --pet-dropout 0.3 --device 0
"""

from __future__ import annotations

import argparse
from collections.abc import Callable


def make_pet_dropout_callback(p: float) -> Callable:
    """
    Builds an Ultralytics callback that blanks the PET channel (channel 2) of a
    random subset of each training batch, implementing modality dropout so the
    detector stays robust to missing PET.
    Params:
        - p: float, per-image probability of zeroing the PET channel
    Returns:
        - on_train_batch_start: Callable, callback taking the Ultralytics trainer
          and mutating its current batch in place
    """
    import torch

    def on_train_batch_start(trainer) -> None:
        """
        Zeros the PET channel of a random subset of the current training batch.
        Params:
            - trainer: ultralytics trainer, exposes the current batch as
              trainer.batch["img"] (a (B, C, H, W) tensor)
        Returns:
            - None
        """
        img = trainer.batch["img"]
        if img.ndim == 4 and img.shape[1] >= 3:
            mask = torch.rand(img.shape[0], device=img.device) < p
            img[mask, 2] = 0  # channel 2 == PET
    return on_train_batch_start


def main() -> None:
    """
    Command-line entry point: builds the YOLOv12 model, optionally registers the
    PET-dropout callback, trains on the patient-split dataset and evaluates on the
    held-out test split.
    Params:
        - None (arguments are read from the command line)
    Returns:
        - None
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/yolo_petct/data.yaml")
    ap.add_argument("--model", default="yolo12m.pt",
                    help="YOLOv12 weights/config: yolo12{n,s,m,l,x}.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--pet-dropout", type=float, default=0.3,
                    help="Prob. of blanking the PET channel per training image.")
    ap.add_argument("--project", default="results/trained_models")
    ap.add_argument("--name", default="yolo12_petct")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.pet_dropout > 0:
        model.add_callback("on_train_batch_start",
                           make_pet_dropout_callback(args.pet_dropout))

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,
        # Lesion boxes are small and CT is not natural imagery: keep photometric
        # augmentation mild and disable flips that are anatomically misleading.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        fliplr=0.5, flipud=0.0, mosaic=1.0, degrees=5.0,
    )

    # Final evaluation on the held-out test split.
    metrics = model.val(data=args.data, split="test")
    print("Test metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
