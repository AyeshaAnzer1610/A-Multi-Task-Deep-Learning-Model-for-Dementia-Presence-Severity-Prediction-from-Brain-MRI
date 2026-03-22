"""model.py — Multi-task EfficientNet-B0 with SE block and ordinal output heads."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block (1-D, applied to feature vectors)
# ---------------------------------------------------------------------------

class SEBlock1D(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation recalibration for 1-D feature vectors.

    Args:
        dim (int):        Input feature dimension.
        reduction (int):  Bottleneck reduction factor (default 8).
    """

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, dim // reduction)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


# ---------------------------------------------------------------------------
# Proposed model
# ---------------------------------------------------------------------------

class ProposedEffNetOrdinal(nn.Module):
    """
    Multi-task EfficientNet-B0 model for simultaneous dementia presence
    detection (binary) and clinical severity staging (3-class ordinal).

    Architecture
    ------------
    Backbone : EfficientNet-B0 (pretrained on ImageNet)
               -> AdaptiveAvgPool2d -> 1280-dim feature vector

    Shared head:
        Linear(1280 -> 256) + ReLU + Dropout + SEBlock1D(256)

    Task heads (three independent linear layers):
        head_bin  : P(Demented)          -- binary presence
        head_ge1  : P(severity >= 1)     -- ordinal: any dementia
        head_ge2  : P(severity >= 2)     -- ordinal: Mild or Moderate

    All heads output raw logits; sigmoid is applied externally for loss
    computation or during inference.
    """

    def __init__(self, dropout: float = 0.35, se_reduction: int = 8) -> None:
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            SEBlock1D(256, reduction=se_reduction),
        )

        self.head_bin = nn.Linear(256, 1)   # binary presence
        self.head_ge1 = nn.Linear(256, 1)   # P(severity >= Very Mild)
        self.head_ge2 = nn.Linear(256, 1)   # P(severity >= Mild/Moderate)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input batch of MRI images.

        Returns:
            Tuple of three (B,) logit tensors:
                logit_bin, logit_ge1, logit_ge2
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.shared(x)
        return (
            self.head_bin(x).squeeze(1),
            self.head_ge1(x).squeeze(1),
            self.head_ge2(x).squeeze(1),
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (used during warm-up phase)."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for full fine-tuning."""
        for p in self.features.parameters():
            p.requires_grad = True
