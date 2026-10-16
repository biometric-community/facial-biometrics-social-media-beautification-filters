"""ResNet18 face embedder → 512-D L2-normalized features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceEmbedder(nn.Module):
    def __init__(self, emb_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(in_features, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.bn(self.fc(feat))
        return F.normalize(emb, p=2, dim=1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
