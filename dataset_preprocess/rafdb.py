"""Read the RAF-DB basic-expression partition and aligned images."""

from pathlib import Path

from PIL import Image

from .labels import RAFDB


def records(root: Path):
    partition = root / "EmoLabel" / "list_patition_label.txt"
    aligned = root / "Image" / "aligned"

    with partition.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid RAF-DB partition line {line_number}")

            name, raw_label = parts
            if Path(name).name != name or Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Invalid RAF-DB image name: {name}")
            label = int(raw_label)
            if label not in RAFDB:
                raise ValueError(f"Invalid RAF-DB label: {label}")

            stem = Path(name).stem
            if stem.startswith("train_"):
                split = "train"
            elif stem.startswith("test_"):
                split = "test"
            else:
                raise ValueError(f"Unknown RAF-DB partition for {name}")

            image_path = aligned / f"{stem}_aligned.jpg"
            with Image.open(image_path) as source:
                image = source.copy()
            yield stem, split, label, image
