"""Multi-context attention for the visual branch."""

import torch
from torch import nn
from torch.nn import functional as F


class MultiContextCoordinateAttention(nn.Module):
    """Axis-wise attention plus a global context gate.

    This implements the operations described in the paper. The original
    attention module's exact parameterization has not been verified.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.height_map = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
            nn.Sigmoid(),
        )
        self.width_map = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
            nn.Sigmoid(),
        )
        reduced = max(8, channels // 8)
        self.global_map = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.Hardswish(),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vertical = self.height_map(x.mean(dim=3, keepdim=True))
        horizontal = self.width_map(x.mean(dim=2, keepdim=True))
        context = self.global_map(x)
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * vertical * horizontal * context
