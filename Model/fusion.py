"""Residual logit fusion with a sample-dependent gate."""

import torch
from torch import nn


class AdaptiveLogitFusion(nn.Module):
    def __init__(self, hidden: int = 128, maximum: float = 1.25):
        super().__init__()
        # Gate hidden width is not stated in the manuscript; 128 is a reference choice.
        self.gate = nn.Sequential(
            nn.Linear(640 + 7 + 7 + 7, hidden),
            nn.Hardswish(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.maximum = maximum

    def forward(self, visual_features, visual_logits, landmark_logits,
                prototype_logits, valid):
        gate_input = torch.cat((visual_features, visual_logits.detach(),
                                landmark_logits, prototype_logits), dim=1)
        g = self.maximum * self.gate(gate_input)
        g = g * valid.to(dtype=g.dtype).view(-1, 1)
        return visual_logits + g * landmark_logits, g
