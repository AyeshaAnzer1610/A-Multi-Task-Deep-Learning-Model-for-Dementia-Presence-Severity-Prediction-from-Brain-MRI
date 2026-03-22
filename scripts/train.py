#!/usr/bin/env python
"""
scripts/train.py
Entry point for a full training run.

Usage
-----
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import set_seed, extract_zip, build_dataframe, subject_split
from src.dataset import build_loaders
from src.model import ProposedEffNetOrdinal
from src.losses import build_loss_functions
from src.train import build_optimizer, train
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


def main(cfg: dict) -> None:
    set_seed(cfg["data"]["random_state"])

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem   = device.type == "cuda"
    use_amp   = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    base     = extract_zip(cfg["data"]["zip_path"], cfg["data"]["extract_dir"])
    df       = build_dataframe(base)
    train_df, val_df, test_df = subject_split(df, cfg["data"]["random_state"])

    print("\nClass distribution (train):\n", train_df["class"].value_counts().to_string())

    train_loader, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df,
        batch_size=cfg["training"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        pin_memory=pin_mem,
    )

    # ------------------------------------------------------------------
    # Model + Losses + Optimizer
    # ------------------------------------------------------------------
    model = ProposedEffNetOrdinal(
        dropout=cfg["model"]["dropout"],
        se_reduction=cfg["model"]["se_reduction"],
    ).to(device)

    combined_loss, _, _ = build_loss_functions(
        train_df, device,
        binary_loss_weight=cfg["training"]["binary_loss_weight"],
        ordinal_loss_weight=cfg["training"]["ordinal_loss_weight"],
    )

    optimizer = build_optimizer(
        model,
        lr_head=cfg["training"]["lr_head"],
        lr_backbone=cfg["training"]["lr_backbone"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    results_dir = cfg["output"]["results_dir"]
    train(
        model, train_loader, val_loader, combined_loss, optimizer,
        device=device,
        epochs=cfg["training"]["epochs"],
        freeze_epochs=cfg["training"]["freeze_epochs"],
        early_stop_patience=cfg["training"]["early_stop_patience"],
        grad_clip_norm=cfg["training"]["grad_clip_norm"],
        combined_auc_weight=cfg["training"]["combined_auc_weight"],
        combined_f1_weight=cfg["training"]["combined_f1_weight"],
        results_dir=results_dir,
        best_auc_filename=cfg["output"]["best_auc_checkpoint"],
        best_f1_filename=cfg["output"]["best_f1_checkpoint"],
        use_amp=use_amp,
    )

    # ------------------------------------------------------------------
    # Final evaluation (best binary-AUC checkpoint)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL EVALUATION  (best binary-AUC checkpoint)")
    print("=" * 60)

    auc_ckpt = Path(results_dir) / cfg["output"]["best_auc_checkpoint"]
    model.load_state_dict(torch.load(auc_ckpt, map_location=device))

    df_val = get_predictions(model, val_loader, device)
    best_t,      _  = tune_binary_threshold(df_val)
    best_t1, best_t2, _ = tune_ordinal_thresholds(df_val)

    print("\n-- Validation --")
    report_binary(df_val, best_t, name="VAL", subject_level=False)
    report_binary(df_val, best_t, name="VAL", subject_level=True)
    report_severity(df_val, best_t1, best_t2, name="VAL")

    df_test = get_predictions(model, test_loader, device)
    print("\n-- Test --")
    report_binary(df_test, best_t, name="TEST", subject_level=False)
    report_binary(df_test, best_t, name="TEST", subject_level=True)
    report_severity(df_test, best_t1, best_t2, name="TEST")

    # Save final model
    final_path = Path(results_dir) / cfg["output"]["final_checkpoint"]
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved -> {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the dementia multi-task model.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()
    main(load_config(args.config))
