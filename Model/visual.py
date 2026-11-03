"""Visual branch shared by M1, M2, and M3."""

import torch
from torch import nn

from .attention import MultiContextCoordinateAttention
from .blocks import FusedStem, InvertedResidual
from .mixer import GatedFacialSpatialMixer

# Input, expanded, output, stride, repeats. Total: 11 inverted residual blocks.
STAGES = (
    (32, 48, 48, 2, 1),
    (48, 64, 64, 1, 1),
    (64, 128, 96, 2, 1),
    (96, 144, 96, 1, 1),
    (96, 192, 128, 2, 1),
    (128, 256, 128, 1, 2),
    (128, 320, 192, 1, 2),
    (192, 384, 256, 1, 2),
)


class VisualBranch(nn.Module):
    def __init__(self, use_attention: bool):
        super().__init__()
        self.stem = FusedStem()
        blocks = []
        for in_channels, expanded, out_channels, stride, repeat in STAGES:
            for index in range(repeat):
                channels_in = in_channels if index == 0 else out_channels
                block_stride = stride if index == 0 else 1
                attn = MultiContextCoordinateAttention(out_channels) if use_attention else None
                blocks.append(InvertedResidual(channels_in, expanded, out_channels,
                                               block_stride, attn))
        self.blocks = nn.Sequential(*blocks)
        self.mixer = GatedFacialSpatialMixer(256, kernel_size=9, dropout=0.1)
        self.projection = nn.Sequential(
            nn.Conv2d(256, 640, kernel_size=1, bias=False),
            nn.BatchNorm2d(640),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(640, 7)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mixer(self.blocks(self.stem(images)))
        features = self.projection(x)
        logits = self.classifier(features)
        return features, logits


class VisualOnly(nn.Module):
    def __init__(self, use_attention: bool):
        super().__init__()
        self.visual = VisualBranch(use_attention=use_attention)

    def forward(self, images: torch.Tensor, geometry=None, cues=None, valid=None):
        features, logits = self.visual(images)
        return {"logits": logits, "visual_logits": logits,
                "visual_features": features}
