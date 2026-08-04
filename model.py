"""EfficientNet-B0 regression model for soil pH prediction."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SoilpHModel(nn.Module):
    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = False, n_extra: int = 0):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        in_features = backbone.classifier[1].in_features
        # Strip the classifier so forward() returns the 1280-dim pooled features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.n_extra = n_extra

        # Regression head; optionally accepts a side-input (e.g. farm one-hot)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features + n_extra, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x)  # (B, 1280)
        if self.n_extra > 0 and extra is not None:
            feat = torch.cat([feat, extra.float()], dim=1)
        return self.head(feat).squeeze(1)  # (B,)
