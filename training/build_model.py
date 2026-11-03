"""Build the visual baseline used by the public training script."""

from Model.visual import VisualOnly


def make_model(variant: str = "M1"):
    if variant != "M1":
        raise ValueError(
            "Only the M1 visual baseline is available. "
            "The attention and geometry modules needed for M2/M3 are not included."
        )
    return VisualOnly()
