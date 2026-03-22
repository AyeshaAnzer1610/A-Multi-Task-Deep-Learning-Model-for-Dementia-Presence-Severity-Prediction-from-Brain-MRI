"""utils.py — Reproducibility, subject-level data splitting, and shared helpers."""

import os
import re
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all relevant random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NON       = "Non Demented"
VERY_MILD = "Very mild Dementia"
MILD      = "Mild Dementia"
MOD       = "Moderate Dementia"
KEEP      = {NON, VERY_MILD, MILD, MOD}


def extract_zip(zip_path: str, extract_dir: str) -> Path:
    """Unzip the OASIS archive and return the base image directory."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    base = Path(extract_dir) / "input"
    if not base.exists():
        base = Path(extract_dir)

    print(f"Extracted to: {extract_dir}")
    print(f"Base folder:  {base}")
    return base


def build_dataframe(base: Path) -> pd.DataFrame:
    """Walk *base* and build a DataFrame with path, class, and derived labels."""
    rows = []
    for cls_dir in base.iterdir():
        if not cls_dir.is_dir() or cls_dir.name not in KEEP:
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                rows.append({"path": str(p), "class": cls_dir.name, "filename": p.name})

    if not rows:
        raise RuntimeError("No images found — check zip structure / base folder.")

    df = pd.DataFrame(rows).reset_index(drop=True)

    # Binary label: 0 = Non-Demented, 1 = Demented
    df["y_bin"] = df["class"].apply(lambda c: 0 if c == NON else 1).astype(int)

    # 3-class ordinal severity
    def _sev(c: str) -> int:
        if c == NON:       return 0
        if c == VERY_MILD: return 1
        return 2

    df["y_sev"] = df["class"].apply(_sev).astype(int)

    # Subject ID extracted from OASIS filename convention
    def _subj(fname: str) -> str:
        m = re.search(r"(OAS1_\d{4})", fname)
        return m.group(1) if m else fname.split("_")[0]

    df["subject_id"] = df["filename"].apply(_subj)
    return df


def subject_split(df: pd.DataFrame, random_state: int = 42):
    """
    Stratified group k-fold (k=3) split at the subject level.

    Returns
    -------
    train_df, val_df, test_df — non-overlapping by subject_id.
    """
    subj_sev = df.groupby("subject_id")["y_sev"].max().reset_index()
    subjects = subj_sev["subject_id"].values
    y_subj   = subj_sev["y_sev"].values

    sgkf  = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_state)
    folds = list(sgkf.split(subjects, y_subj, groups=subjects))

    test_subj  = set(subjects[folds[0][1]])
    val_subj   = set(subjects[folds[1][1]])
    train_subj = set(subjects) - test_subj - val_subj

    train_df = df[df["subject_id"].isin(train_subj)].reset_index(drop=True)
    val_df   = df[df["subject_id"].isin(val_subj)].reset_index(drop=True)
    test_df  = df[df["subject_id"].isin(test_subj)].reset_index(drop=True)

    print(f"\nSubject split: train={len(train_subj)}  val={len(val_subj)}  test={len(test_subj)}")
    print(f"Image  split:  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df
