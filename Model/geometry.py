"""104-feature landmark network and fixed geometric-prototype comparison."""

import torch
from torch import nn
from torch.nn import functional as F


class LandmarkMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(104, 192), nn.BatchNorm1d(192), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(192, 96), nn.BatchNorm1d(96), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(96, 7),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class PrototypeComparison(nn.Module):
    def __init__(self, prototypes: torch.Tensor, temperature: float = 1.8):
        super().__init__()
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        if prototypes.shape != (7, 16) or not torch.isfinite(prototypes).all():
            raise ValueError("Supply a finite (7,16) matrix of verified class prototypes")
        self.register_buffer("prototypes", prototypes.clone())
        self.theta = nn.Parameter(torch.zeros(16))
        self.temperature = temperature
        self.alpha_unclipped = nn.Parameter(torch.tensor(0.35))

    def forward(self, raw_cues: torch.Tensor):
        weights = F.softplus(self.theta).view(1, 1, 16)
        distance = (weights * (raw_cues.unsqueeze(1) - self.prototypes.unsqueeze(0)).square()).mean(-1)
        prototype_logits = -distance / self.temperature
        alpha = self.alpha_unclipped.clamp(0, 2)
        return prototype_logits, alpha
