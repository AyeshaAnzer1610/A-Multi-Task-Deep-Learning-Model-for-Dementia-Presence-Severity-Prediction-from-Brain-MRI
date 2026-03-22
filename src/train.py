"""train.py — Training loop with warm-up freeze, dual checkpointing, and early stopping."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from .evaluate import (
    aggregate_by_subject,
    get_predictions,
    report_severity,
)


def build_optimizer(model, lr_head: float, lr_backbone: float, weight_decay: float):
    """
    Two-parameter-group AdamW: backbone uses a 10x lower LR than the head
    to preserve pretrained features during fine-tuning.
    """
    return optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": lr_backbone},
            {
                "params": (
                    list(model.shared.parameters())
                    + list(model.head_bin.parameters())
                    + list(model.head_ge1.parameters())
                    + list(model.head_ge2.parameters())
                ),
                "lr": lr_head,
            },
        ],
        weight_decay=weight_decay,
    )


def train(
    model,
    train_loader,
    val_loader,
    combined_loss,
    optimizer,
    *,
    device: torch.device,
    epochs: int = 25,
    freeze_epochs: int = 5,
    early_stop_patience: int = 5,
    grad_clip_norm: float = 1.0,
    combined_auc_weight: float = 0.6,
    combined_f1_weight: float = 0.4,
    results_dir: str = "results",
    best_auc_filename: str = "best_binary_auc.pth",
    best_f1_filename: str = "best_severity_f1.pth",
    use_amp: bool = False,
) -> dict:
    """
    Full training loop.

    Phases
    ------
    1. Warm-up (epochs 1 .. freeze_epochs): backbone frozen, head only.
    2. Fine-tune (epochs freeze_epochs+1 .. epochs): full network, backbone at lower LR.

    Checkpointing
    -------------
    - best_binary_auc.pth   saved whenever validation subject-level AUC improves.
    - best_severity_f1.pth  saved whenever validation severity macro-F1 improves.

    Early Stopping
    --------------
    Monitors ``combined_auc_weight * bin_auc + combined_f1_weight * sev_f1`` on validation.

    Returns
    -------
    dict with keys: best_bin_auc, best_sev_f1, best_combined
    """
    os.makedirs(results_dir, exist_ok=True)
    auc_path = Path(results_dir) / best_auc_filename
    f1_path  = Path(results_dir) / best_f1_filename

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_bin_auc  = -1.0
    best_sev_f1   = -1.0
    best_combined = -1.0
    patience_counter = 0

    model.freeze_backbone()
    print(f"Backbone frozen for first {freeze_epochs} epochs.")

    for epoch in range(1, epochs + 1):

        # Unfreeze backbone after warm-up
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            print(f"\nEpoch {epoch}: backbone unfrozen (fine-tuning at LR {optimizer.param_groups[0]['lr']:.1e}).")

        # ------------------------------------------------------------------
        # Training pass
        # ------------------------------------------------------------------
        model.train()
        epoch_losses = []

        for x, y_bin, ge1, ge2, _y_sev, _sid in train_loader:
            x    = x.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            ge1  = ge1.to(device, non_blocking=True)
            ge2  = ge2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logit_bin, logit_ge1, logit_ge2 = model(x)
                loss = combined_loss(logit_bin, logit_ge1, logit_ge2, y_bin, ge1, ge2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        df_val = get_predictions(model, val_loader, device)
        g = aggregate_by_subject(df_val)
        val_bin_auc = roc_auc_score(g["y_bin"].to_numpy(), g["p_bin"].to_numpy())
        _, val_sev_f1 = report_severity(df_val, name=f"VAL Epoch {epoch}")

        combined = combined_auc_weight * val_bin_auc + combined_f1_weight * val_sev_f1

        print(
            f"\nEpoch {epoch:03d}/{epochs}  |  loss={mean_loss:.4f}  "
            f"|  val_bin_AUC={val_bin_auc:.4f}  |  val_sev_F1={val_sev_f1:.4f}  "
            f"|  combined={combined:.4f}"
        )

        # Dual checkpointing
        if val_bin_auc > best_bin_auc:
            best_bin_auc = val_bin_auc
            torch.save(model.state_dict(), auc_path)
            print(f"  Saved best binary-AUC checkpoint -> {auc_path}  [{best_bin_auc:.4f}]")

        if val_sev_f1 > best_sev_f1:
            best_sev_f1 = val_sev_f1
            torch.save(model.state_dict(), f1_path)
            print(f"  Saved best severity-F1 checkpoint -> {f1_path}  [{best_sev_f1:.4f}]")

        # Early stopping on combined score
        if combined > best_combined:
            best_combined = combined
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No combined improvement. Patience {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    return {"best_bin_auc": best_bin_auc, "best_sev_f1": best_sev_f1, "best_combined": best_combined}
