#!/usr/bin/env python
"""
scripts/evaluate.py
Evaluate a saved checkpoint on validation and test sets.

Usage
-----
    python scripts/evaluate.py --checkpoint results/best_binary_auc.pth
    python scripts/evaluate.py --checkpoint results/best_binary_auc.pth --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import set_seed, extract_zip, build_dataframe, subject_split
from src.dataset import build_loaders
from src.model import ProposedEffNetOrdinal
from src.evaluate import (
    get_predictions,
    tune_binary_threshold,
    tune_ordinal_thresholds,
    report_binary,
    report_severity,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(checkpoint: str, cfg: dict) -> None:
    set_seed(cfg["data"]["random_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    base = extract_zip(cfg["data"]["zip_path"], cfg["data"]["extract_dir"])
    df   = build_dataframe(base)
    train_df, val_df, test_df = subject_split(df, cfg["data"]["random_state"])

    _, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df,
        batch_size=cfg["training"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = ProposedEffNetOrdinal(
        dropout=cfg["model"]["dropout"],
        se_reduction=cfg["model"]["se_reduction"],
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint}")

    # Tune thresholds on validation
    df_val  = get_predictions(model, val_loader, device)
    best_t,          _  = tune_binary_threshold(df_val)
    best_t1, best_t2, _ = tune_ordinal_thresholds(df_val)

    # Validation report
    print("\n" + "=" * 50 + "\nVALIDATION\n" + "=" * 50)
    report_binary(df_val, best_t, name="VAL", subject_level=False)
    report_binary(df_val, best_t, name="VAL", subject_level=True)
    report_severity(df_val, best_t1, best_t2, name="VAL")

    # Test report
    print("\n" + "=" * 50 + "\nTEST\n" + "=" * 50)
    df_test = get_predictions(model, test_loader, device)
    report_binary(df_test, best_t, name="TEST", subject_level=False)
    report_binary(df_test, best_t, name="TEST", subject_level=True)
    report_severity(df_test, best_t1, best_t2, name="TEST")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a dementia model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file.")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()
    main(args.checkpoint, load_config(args.config))
