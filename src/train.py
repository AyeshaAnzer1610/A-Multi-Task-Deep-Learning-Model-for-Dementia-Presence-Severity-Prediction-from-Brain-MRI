"""
src/train.py  —  Two-phase training loop (paper §3.7)

*** AMP FIX ***
The model forward pass runs inside autocast (float16 = fast).
The loss is computed OUTSIDE autocast (float32 = safe for BCE).
This is the correct pattern for AMP + BCE loss.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.losses   import MTLoss, compute_pos_weight
from src.evaluate import (collect_predictions, tune_thresholds,
                           compute_subject_level_metrics)


def _head_params(model):
    return (list(model.projection.parameters()) +
            list(model.se.parameters()) +
            list(model.head_binary.parameters()) +
            list(model.head_ord_ge1.parameters()) +
            list(model.head_ord_ge2.parameters()))


def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, device, grad_clip=1.0):
    model.train()
    totals = {"total": 0., "binary": 0., "ordinal": 0., "consistency": 0.}
    n = 0

    for batch in loader:
        imgs  = batch["image"].to(device, non_blocking=True)
        y_bin = batch["binary"].to(device,  non_blocking=True)
        y_ord = batch["ordinal"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # ── AMP: forward in float16 (fast), loss in float32 (safe) ────────
            with torch.amp.autocast(device_type="cuda"):
                out = model(imgs)              # float16 activations

            # Loss is computed OUTSIDE autocast — always float32
            # (F.binary_cross_entropy crashes in float16)
            losses = criterion(out, y_bin, y_ord)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU / non-AMP path
            out    = model(imgs)
            losses = criterion(out, y_bin, y_ord)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = imgs.size(0)
        for k in totals:
            totals[k] += losses[k].item() * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


def save_checkpoint(model, optimizer, epoch, thresholds, path):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "thresholds":      thresholds,
    }, path)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    thr  = ckpt.get("thresholds", {"t1": 0.45, "t2": 0.55, "tbin": 0.48})
    print(f"  Loaded epoch {ckpt['epoch']} | thresholds: {thr}")
    return thr


def train(cfg, model, train_loader, val_loader,
          train_samples, val_samples, device, results_dir):

    os.makedirs(results_dir, exist_ok=True)
    use_amp = (device.type == "cuda")

    # Loss
    criterion = MTLoss(
        pos_weight         = compute_pos_weight(train_samples),
        alpha              = cfg["loss_alpha"],
        consistency_weight = cfg.get("consistency_weight", 0.05),
    ).to(device)

    # Phase 1: backbone frozen
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        _head_params(model),
        lr=cfg["head_lr"],
        weight_decay=cfg["weight_decay"],
    )
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])

    best_composite = best_bin_auc = best_sev_f1 = -1.0
    patience_count = 0
    t1   = cfg.get("t1",   0.45)
    t2   = cfg.get("t2",   0.55)
    tbin = cfg.get("tbin", 0.48)
    history = {
        "train_loss": [], "val_bin_auc": [],
        "val_sev_f1": [], "composite":  [],
    }

    for epoch in range(1, cfg["max_epochs"] + 1):

        # Switch to Phase 2
        if epoch == cfg["freeze_epochs"] + 1:
            print(f"\n[Epoch {epoch}] Switching to Phase 2: full fine-tuning")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone_features.parameters(),
                     "lr": cfg["backbone_lr"]},
                    {"params": _head_params(model),
                     "lr": cfg["head_lr"]},
                ],
                weight_decay=cfg["weight_decay"],
            )
            scaler    = torch.cuda.amp.GradScaler() if use_amp else None
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg["max_epochs"] - cfg["freeze_epochs"],
            )

        t0  = time.time()
        lss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg.get("grad_clip_norm", 1.0)
        )
        scheduler.step()

        # Validation
        val_preds    = collect_predictions(model, val_loader, device)
        t1, t2, tbin = tune_thresholds(val_preds, val_samples)
        sl           = compute_subject_level_metrics(
                           val_preds, val_samples, t1, t2, tbin)
        bin_auc      = sl["binary"]["auc"]
        sev_f1       = sl["severity"]["macro_f1"]
        composite    = 0.6 * bin_auc + 0.4 * sev_f1

        history["train_loss"].append(lss["total"])
        history["val_bin_auc"].append(bin_auc)
        history["val_sev_f1"].append(sev_f1)
        history["composite"].append(composite)

        print(
            f"Ep {epoch:3d}/{cfg['max_epochs']} | "
            f"Loss {lss['total']:.4f} "
            f"(bin {lss['binary']:.4f} ord {lss['ordinal']:.4f}) | "
            f"BinAUC {bin_auc:.4f}  SevF1 {sev_f1:.4f}  "
            f"Comp {composite:.4f} | {time.time()-t0:.1f}s"
        )

        thr = {"t1": t1, "t2": t2, "tbin": tbin}

        if bin_auc > best_bin_auc:
            best_bin_auc = bin_auc
            save_checkpoint(model, optimizer, epoch, thr,
                            os.path.join(results_dir, "best_binary_auc.pth"))
            print(f"  ✓ best_binary_auc.pth  (BinAUC={bin_auc:.4f})")

        if sev_f1 > best_sev_f1:
            best_sev_f1 = sev_f1
            save_checkpoint(model, optimizer, epoch, thr,
                            os.path.join(results_dir, "best_severity_f1.pth"))
            print(f"  ✓ best_severity_f1.pth (SevF1={sev_f1:.4f})")

        if composite > best_composite:
            best_composite = composite
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["early_stop_patience"]:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    save_checkpoint(model, optimizer, epoch, thr,
                    os.path.join(results_dir, "final_model.pth"))
    print(f"\n✓ Training complete. Checkpoints in: {results_dir}/")
    return history
