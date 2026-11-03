"""Visual feature extractor and classifier."""

from collections.abc import Callable

import torch
from torch import nn

from .blocks import FusedStem, InvertedResidual
from .mixer import GatedFacialSpatialMixer

# Input channels, expansion channels, output channels, stride, repetitions.
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
    def __init__(self, attention_factory: Callable[[int], nn.Module] | None = None):
        super().__init__()
        self.stem = FusedStem()
        blocks = []

        for in_channels, expanded, out_channels, stride, repeats in STAGES:
            for index in range(repeats):
                input_channels = in_channels if index == 0 else out_channels
                block_stride = stride if index == 0 else 1
                attention = attention_factory(out_channels) if attention_factory else None
                blocks.append(
                    InvertedResidual(input_channels, expanded, out_channels, block_stride, attention)
                )

        self.blocks = nn.Sequential(*blocks)
        self.mixer = GatedFacialSpatialMixer(256, kernel_size=9, dropout=0.1)
        self.projection = nn.Sequential(
            nn.Conv2d(256, 640, 1, bias=False),
            nn.BatchNorm2d(640),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(640, 7)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.projection(self.mixer(self.blocks(self.stem(images))))
        return features, self.classifier(features)


class VisualOnly(nn.Module):
    """Runs the visual baseline without a geometric branch."""

    def __init__(self):
        super().__init__()
        self.visual = VisualBranch()

    def forward(self, images: torch.Tensor, geometry=None, cues=None, valid=None):
        features, logits = self.visual(images)
        return {
            "logits": logits,
            "visual_logits": logits,
            "visual_features": features,
        }
