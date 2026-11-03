"""Construct one of the three ablation configurations."""

import json
from pathlib import Path

import numpy as np
import torch

from Ablations.M1.model import M1
from Ablations.M2.model import M2
from Ablations.M3.model import M3


def load_prototypes(path: Path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        matrix = np.asarray(json.loads(path.read_text(encoding="utf-8-sig")), dtype=np.float32)
    elif path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        raise ValueError("Prototypes must be a .json or .npy (7 x 16) matrix")
    if matrix.shape != (7, 16) or not np.isfinite(matrix).all():
        raise ValueError("Prototypes must be a finite (7,16) matrix")
    return torch.from_numpy(matrix)


def make_model(variant: str, prototypes: torch.Tensor | None = None):
    if variant == "M1":
        return M1()
    if variant == "M2":
        return M2()
    if variant == "M3":
        if prototypes is None:
            raise ValueError("M3 requires original verified 7x16 prototypes; no dummy values are supplied")
        return M3(prototypes)
    raise ValueError(f"Unknown variant: {variant}")
