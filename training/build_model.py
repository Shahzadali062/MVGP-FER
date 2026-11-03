"""Select the visual baseline, attention model, or geometry-guided model."""

from pathlib import Path

import numpy as np
import torch

from Model.mvgp_fer import MVGPFER
from Model.visual import VisualOnly


def load_prototypes(path: Path) -> torch.Tensor:
    """Read a 7-by-16 JSON array from an external, verified source."""
    import json

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    try:
        data = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prototype JSON must contain a numeric 7 x 16 array") from exc
    if data.shape != (7, 16) or not np.isfinite(data).all():
        raise ValueError("Prototype JSON must contain 7 rows of 16 finite values")
    return torch.from_numpy(data.copy())


def make_model(variant: str, prototypes: torch.Tensor | None = None):
    variant = variant.upper()
    if variant == "M1":
        return VisualOnly(use_attention=False)
    if variant == "M2":
        return VisualOnly(use_attention=True)
    if variant == "M3":
        if prototypes is None:
            raise ValueError("M3 requires the actual 7 x 16 class prototype matrix")
        return MVGPFER(prototypes)
    raise ValueError(f"Unknown model variant: {variant}")
