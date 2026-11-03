"""Landmark classifier and fixed prototype matching."""

import torch
from torch import nn
from torch.nn import functional as F


class LandmarkMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(104, 192),
            nn.BatchNorm1d(192),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(96, 7),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 104:
            raise ValueError("Expected (batch, 104) landmark features")
        return self.layers(features)


class PrototypeComparison(nn.Module):
    """Compare measured cues to externally supplied, fixed class prototypes."""

    def __init__(self, prototypes: torch.Tensor, temperature: float = 1.8):
        super().__init__()
        template = torch.as_tensor(prototypes, dtype=torch.float32)
        if template.shape != (7, 16) or not bool(torch.isfinite(template).all()):
            raise ValueError("prototypes must be a finite (7, 16) array")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.register_buffer("prototypes", template.clone())
        self.theta = nn.Parameter(torch.zeros(16))
        self.alpha_unclipped = nn.Parameter(torch.tensor(0.35))
        self.temperature = temperature

    def forward(self, cues: torch.Tensor):
        if cues.ndim != 2 or cues.shape[1] != 16:
            raise ValueError("Expected (batch, 16) raw geometric cues")
        weights = F.softplus(self.theta).reshape(1, 1, 16)
        distance = (weights * (cues[:, None, :] - self.prototypes[None, :, :]).square()).mean(-1)
        logits = -distance / self.temperature
        return logits, self.alpha_unclipped.clamp(0, 2)
