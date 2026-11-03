"""Read the CSV and .npy files created by dataset_preprocess."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedFERDataset(Dataset):
    def __init__(self, manifest: Path, rows: list[dict],
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None):
        self.root = Path(manifest).resolve().parent
        self.rows = rows
        self.mean = mean
        self.std = std
        if (mean is None) != (std is None):
            raise ValueError("Mean and std must be provided together")

    def __len__(self):
        return len(self.rows)

    def _load(self, relative: str, expected: tuple[int, ...]):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Manifest paths cannot escape the data directory")
        arr = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if arr.shape != expected or not np.isfinite(arr).all():
            raise ValueError(f"Bad input shape or values: {path}, expected {expected}")
        return arr

    def _raw_geometry(self, row: dict):
        """Return raw (104,) features only when full real cues are available."""
        complete = row.get("geometry_104_npy", "").strip()
        if complete:
            return self._load(complete, (104,))
        coords = row.get("landmarks_88_npy", "").strip()
        cues = row.get("geometric_cues_16_npy", "").strip()
        if coords and cues:
            return np.concatenate((self._load(coords, (88,)), self._load(cues, (16,))))
        return None

    def __getitem__(self, index):
        row = self.rows[index]
        image = self._load(row["visual_npy"], (3, 224, 224))
        label = int(row["paper_label"])
        if not 0 <= label < 7:
            raise ValueError(f"Class index outside 0..6: {label}")
        raw = self._raw_geometry(row)
        valid = raw is not None and row.get("landmark_status", "valid").lower() == "valid"
        if valid:
            coords = raw if self.mean is None else (raw - self.mean) / self.std
            cues = raw[-16:]
        else:
            coords = np.zeros(104, dtype=np.float32)
            cues = np.zeros(16, dtype=np.float32)
        return (torch.from_numpy(image.copy()), torch.tensor(label, dtype=torch.long),
                torch.from_numpy(np.asarray(coords, dtype=np.float32).copy()),
                torch.from_numpy(np.asarray(cues, dtype=np.float32).copy()),
                torch.tensor(bool(valid), dtype=torch.bool))


def read_manifest(manifest: Path) -> list[dict]:
    with Path(manifest).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "paper_label", "visual_npy"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Manifest missing columns: {sorted(required)}")
        return list(reader)


def compute_train_statistics(manifest: Path, rows: list[dict]):
    """Fit standardization on VALID training rows only. Never use validation/test."""
    data = PreparedFERDataset(manifest, rows)
    values = []
    for row in rows:
        geom = data._raw_geometry(row)
        if geom is not None and row.get("landmark_status", "valid").lower() == "valid":
            values.append(geom)
    if not values:
        raise ValueError("No complete (88 + 16) training geometry found. M3 needs genuine cue arrays.")
    matrix = np.stack(values).astype(np.float64)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std
