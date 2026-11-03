"""Fused stem and inverted residual blocks."""

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, hidden: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Hardsigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(self.pool(x))


class FusedStem(nn.Module):
    """224x224 input -> 32x112x112."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            SqueezeExcitation(64, 8),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, expanded: int, out_channels: int,
                 stride: int, attention: nn.Module | None = None):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("Only strides 1 and 2 are supported")
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.Hardswish(),
            nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1,
                      groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.Hardswish(),
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = attention if attention is not None else nn.Identity()
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.attention(self.transform(x))
        return x + result if self.residual else result
