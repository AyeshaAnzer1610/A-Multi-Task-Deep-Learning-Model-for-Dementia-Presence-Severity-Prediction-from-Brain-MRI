"""
src/model.py  —  MT-CNN Architecture (paper §3.4)
EfficientNet-B0 + SE block + 3 task heads (binary, ordinal>=1, ordinal>=2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation on a 1-D feature vector (B, C).
    Input is already globally pooled — no spatial squeeze needed.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid      = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C)
        z = F.relu(self.fc1(x))         # (B, C//r)
        z = torch.sigmoid(self.fc2(z))  # (B, C) — per-channel gates
        return x * z                    # (B, C) — recalibrated


class MTCNNDementia(nn.Module):
    """
    Multi-Task CNN: binary dementia detection + ordinal severity staging.

    forward() → dict:
        binary   : (B,) P(Demented)
        ordinal1 : (B,) P(severity >= Very Mild)
        ordinal2 : (B,) P(severity >= Mild/Moderate)  [always <= ordinal1]
        features : (B, 256) shared features for Grad-CAM / t-SNE
    """
    def __init__(self, dropout=0.35, se_reduction=8,
                 pretrained=True, feature_dim=256):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        bb      = efficientnet_b0(weights=weights)

        self.backbone_features = bb.features
        self.avg_pool          = bb.avgpool

        self.projection = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.se           = SEBlock(channels=feature_dim, reduction=se_reduction)
        self.head_binary  = nn.Linear(feature_dim, 1)
        self.head_ord_ge1 = nn.Linear(feature_dim, 1)
        self.head_ord_ge2 = nn.Linear(feature_dim, 1)

        for m in [self.projection[0],
                  self.head_binary, self.head_ord_ge1, self.head_ord_ge2,
                  self.se.fc1, self.se.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for p in self.backbone_features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone_features.parameters():
            p.requires_grad = True

    def forward(self, x):
        feat     = self.backbone_features(x)       # (B,1280,7,7)
        feat     = self.avg_pool(feat).flatten(1)  # (B,1280)
        proj     = self.projection(feat)           # (B,256)
        se       = self.se(proj)                   # (B,256)

        p_binary = torch.sigmoid(self.head_binary(se)).squeeze(1)
        p_ge1    = torch.sigmoid(self.head_ord_ge1(se)).squeeze(1)
        p_ge2    = torch.sigmoid(self.head_ord_ge2(se)).squeeze(1)

        # Enforce P(>=2) <= P(>=1)  [paper Eq. 3]
        p_ge2 = torch.minimum(p_ge2, p_ge1)

        return {"binary": p_binary, "ordinal1": p_ge1,
                "ordinal2": p_ge2,  "features": se}

    @staticmethod
    def decode_ordinal(p_ge1, p_ge2, t1=0.45, t2=0.55):
        pred = torch.zeros_like(p_ge1, dtype=torch.long)
        pred[p_ge1 >= t1] = 1
        pred[(p_ge1 >= t1) & (p_ge2 >= t2)] = 2
        return pred

    def parameter_breakdown(self):
        parts = {
            "EfficientNet-B0": sum(p.numel() for p in self.backbone_features.parameters()),
            "Projection head": sum(p.numel() for p in self.projection.parameters()),
            "SE Block":        sum(p.numel() for p in self.se.parameters()),
            "Binary head":     sum(p.numel() for p in self.head_binary.parameters()),
            "Ord head >=1":    sum(p.numel() for p in self.head_ord_ge1.parameters()),
            "Ord head >=2":    sum(p.numel() for p in self.head_ord_ge2.parameters()),
        }
        total = sum(parts.values())
        print(f"\n{'Component':<25} {'Params':>10}")
        print("-" * 37)
        for k, v in parts.items():
            print(f"  {k:<23} {v:>10,}")
        print("-" * 37)
        print(f"  {'TOTAL':<23} {total:>10,}\n")
        return parts
