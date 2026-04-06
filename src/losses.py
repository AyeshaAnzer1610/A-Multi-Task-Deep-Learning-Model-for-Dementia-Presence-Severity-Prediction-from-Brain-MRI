"""
src/losses.py  —  Joint loss function (paper §3.6)

    L = alpha * L_bin + (1-alpha) * L_ord + w_cons * L_cons

*** AMP FIX ***
F.binary_cross_entropy is UNSAFE inside torch.amp.autocast (crashes with float16).
All BCE inputs are explicitly cast to .float() (float32) before every BCE call.
This is the correct PyTorch-recommended pattern.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTLoss(nn.Module):
    def __init__(self, pos_weight=1.0, alpha=0.6, consistency_weight=0.05):
        super().__init__()
        self.alpha              = alpha
        self.consistency_weight = consistency_weight
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, outputs, binary_targets, ordinal_targets):
        # ── CRITICAL: cast to float32 before any BCE call ─────────────────────
        # F.binary_cross_entropy is not safe in AMP autocast (float16).
        # These .float() casts ensure float32 regardless of AMP context.
        p_bin = outputs["binary"].float()
        p_ge1 = outputs["ordinal1"].float()
        p_ge2 = outputs["ordinal2"].float()
        b_tgt = binary_targets.float()
        pw    = self.pos_weight.float()

        # Binary: class-weighted BCE
        w     = b_tgt * (pw - 1.0) + 1.0
        l_bin = F.binary_cross_entropy(p_bin, b_tgt, weight=w)

        # Ordinal: two cumulative BCE terms
        y_ge1 = (ordinal_targets >= 1).float()
        y_ge2 = (ordinal_targets >= 2).float()
        l_ord = 0.5 * F.binary_cross_entropy(p_ge1, y_ge1) + \
                0.5 * F.binary_cross_entropy(p_ge2, y_ge2)

        # Soft ordinal-consistency penalty
        l_cons = torch.clamp(p_ge2 - p_ge1, min=0.0).mean()

        l_total = (self.alpha * l_bin
                   + (1.0 - self.alpha) * l_ord
                   + self.consistency_weight * l_cons)

        return {
            "total":       l_total,
            "binary":      l_bin,
            "ordinal":     l_ord,
            "consistency": l_cons,
        }


def compute_pos_weight(samples):
    """N_neg / N_pos for binary BCE class weighting."""
    labels = np.array([s["binary"] for s in samples])
    n_pos  = max(float(labels.sum()), 1.0)
    n_neg  = float(len(labels)) - n_pos
    return n_neg / n_pos
