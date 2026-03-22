"""losses.py — Loss functions for multi-task binary + ordinal training."""

import torch
import torch.nn as nn
import pandas as pd


def build_loss_functions(
    train_df: pd.DataFrame,
    device: torch.device,
    binary_loss_weight: float = 0.6,
    ordinal_loss_weight: float = 0.4,
):
    """
    Construct the combined loss callable used during training.

    The binary cross-entropy uses a class-frequency pos_weight to handle
    imbalance: pos_weight = neg_count / pos_count when positives are the
    minority class, otherwise 1.0.

    Args:
        train_df:            Training DataFrame with a 'y_bin' column.
        device:              Torch device.
        binary_loss_weight:  Weight applied to the binary BCE term.
        ordinal_loss_weight: Weight applied to the ordinal BCE term.

    Returns:
        combined_loss (callable), loss_bin, loss_ord
    """
    neg = int((train_df["y_bin"] == 0).sum())
    pos = int((train_df["y_bin"] == 1).sum())

    pos_weight_value = (neg / max(pos, 1)) if pos < neg else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    print(f"Train binary counts: neg={neg}, pos={pos}  ->  pos_weight={pos_weight.item():.4f}")

    loss_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_ord = nn.BCEWithLogitsLoss()

    def combined_loss(
        logit_bin: torch.Tensor,
        logit_ge1: torch.Tensor,
        logit_ge2: torch.Tensor,
        y_bin:     torch.Tensor,
        sev_ge1:   torch.Tensor,
        sev_ge2:   torch.Tensor,
    ) -> torch.Tensor:
        l_bin = loss_bin(logit_bin, y_bin)
        l_ord = 0.5 * loss_ord(logit_ge1, sev_ge1) + 0.5 * loss_ord(logit_ge2, sev_ge2)
        return binary_loss_weight * l_bin + ordinal_loss_weight * l_ord

    return combined_loss, loss_bin, loss_ord
