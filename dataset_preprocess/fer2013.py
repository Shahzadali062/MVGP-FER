"""Read the original FER2013 CSV without changing the split."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import FER2013

SPLITS = {"Training", "PublicTest", "PrivateTest"}


def records(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"emotion", "pixels", "Usage"}.issubset(reader.fieldnames or []):
            raise ValueError("FER2013 needs emotion, pixels and Usage columns")

        for row_number, row in enumerate(reader, start=1):
            split = row["Usage"].strip()
            label = int(row["emotion"])
            if split not in SPLITS or label not in FER2013:
                raise ValueError(f"Invalid split or label at row {row_number}")

            pixels = np.fromstring(row["pixels"], dtype=np.int32, sep=" ")
            if pixels.size != 48 * 48 or np.any((pixels < 0) | (pixels > 255)):
                raise ValueError(f"Invalid pixel data at row {row_number}")

            image = Image.fromarray(pixels.astype(np.uint8).reshape(48, 48))
            yield f"{row_number:06d}", split, label, image
