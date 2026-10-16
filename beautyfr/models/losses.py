"""CosFace / ArcFace margin softmax heads (paper FR strand uses ArcFace-family losses)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """Large-margin cosine loss (AM-Softmax / CosFace)."""

    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class ArcFace(nn.Module):
    """Additive angular margin loss (ArcFace)."""

    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


def MarginHead(
    name: str,
    in_features: int,
    out_features: int,
    s: float = 64.0,
    m: float | None = None,
) -> nn.Module:
    key = name.lower().strip()
    if key in ("cosface", "amsoftmax", "am-softmax"):
        return CosFace(in_features, out_features, s=s, m=0.35 if m is None else m)
    if key in ("arcface", "arc"):
        return ArcFace(in_features, out_features, s=s, m=0.50 if m is None else m)
    raise ValueError(f"Unknown margin head '{name}' (use cosface or arcface)")
