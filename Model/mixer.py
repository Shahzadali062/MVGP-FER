"""Gated spatial-token mixer described in the manuscript."""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class GatedFacialSpatialMixer(nn.Module):
    def __init__(self, channels: int = 256, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Token-mixing kernel must be odd")
        self.norm = RMSNorm(channels)
        self.in_proj = nn.Linear(channels, channels * 2)
        self.mix = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2,
                             groups=channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B,N,C
        u, v = self.in_proj(self.norm(tokens)).chunk(2, dim=-1)
        mixed = F.silu(self.mix(u.transpose(1, 2)).transpose(1, 2))
        output = tokens + self.dropout(self.out_proj(mixed * torch.sigmoid(v)))
        return output.transpose(1, 2).reshape(b, c, h, w)
