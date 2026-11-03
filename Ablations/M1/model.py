"""M1: fused MBConv + inverted residual blocks + gated spatial mixer."""

from Model.visual import VisualOnly


class M1(VisualOnly):
    def __init__(self):
        super().__init__(use_attention=False)
