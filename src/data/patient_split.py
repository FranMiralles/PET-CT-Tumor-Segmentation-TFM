"""Patient-level train/val/test splitting for the detection dataset.

All slices of a given patient must end up in the same split, otherwise nearly
identical axial slices of the same tumour would leak between train and
validation and inflate the metrics. The split is therefore done at the
*patient* level and stratified by (cohort, PET availability) so every split
keeps a comparable mix of sub-cohorts and of CT-only vs CT+PET patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_patient_splits(
    patients: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    stratify_cols: tuple[str, ...] = ("cohort", "has_pet"),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assigns each patient to a 'train' / 'val' / 'test' split at the patient level
    (so slices of one patient never leak across splits), stratified so every
    split keeps a comparable mix of the strata.
    Params:
        - patients: pd.DataFrame, one row per patient; must contain a `patient`
          column plus the columns listed in stratify_cols
        - ratios: tuple[float, float, float], (train, val, test) fractions; must
          sum to 1
        - stratify_cols: tuple[str, ...], columns whose combination defines the
          strata; splitting is done independently inside each stratum so the
          global ratios are preserved within every group
        - seed: int, seed for the shuffling
    Returns:
        - out: pd.DataFrame, copy of `patients` with an added `split` column
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1.")

    rng = np.random.default_rng(seed)
    train_r, val_r, _ = ratios

    out = patients.copy().reset_index(drop=True)
    out["split"] = ""

    for _, group in out.groupby(list(stratify_cols)):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        # Guarantee non-empty train when the stratum is tiny.
        n_train = min(n_train, n)
        out.loc[idx[:n_train], "split"] = "train"
        out.loc[idx[n_train:n_train + n_val], "split"] = "val"
        out.loc[idx[n_train + n_val:], "split"] = "test"

    return out
