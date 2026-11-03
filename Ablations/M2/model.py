"""M2: M1 visual network with multi-context coordinate attention."""

from Model.visual import VisualOnly


class M2(VisualOnly):
    def __init__(self):
        super().__init__(use_attention=True)
