"""Directional and global context attention, reference implementation.

The manuscript describes the three contexts but does not provide the precise
attention implementation. This is a conventional implementation of the stated
operations; it is not claimed to reproduce the authors' original code.
"""

import torch
from torch import nn


class MultiContextCoordinateAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
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
        hidden = max(8, channels // 8)
        self.global_map = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Hardswish(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = self.height_map(x.mean(dim=3, keepdim=True))  # B,C,H,1
        width = self.width_map(x.mean(dim=2, keepdim=True))   # B,C,1,W
        global_gate = self.global_map(x)                     # B,C,1,1
        return x * height * width * global_gate
