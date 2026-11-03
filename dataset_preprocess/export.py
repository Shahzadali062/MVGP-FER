"""Save preprocessed arrays and a dataset manifest."""

import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .fer2013 import records as fer2013_records
from .images import preprocess_image
from .labels import CLASSES, FER2013, RAFDB
from .landmarks import extract_landmarks, load_indices, open_landmarker
from .rafdb import records as rafdb_records

COLUMNS = (
    "sample_id", "split", "source_label", "paper_label", "paper_class",
    "visual_npy", "landmarks_88_npy", "landmark_status",
)


def check_output_path(path: Path) -> Path:
    target = path.expanduser().resolve()
    repository = Path(__file__).resolve().parent.parent
    if target == repository or target.is_relative_to(repository):
        raise ValueError("Save generated arrays outside the Git repository")
    return target


def prepare_dataset(dataset: str, source: Path, out: Path,
                    landmarker_model: Path | None = None,
                    landmark_indices: Path | None = None) -> int:
    if (landmarker_model is None) != (landmark_indices is None):
        raise ValueError("Provide both the landmarker model and 44-index JSON file")

    if dataset == "fer2013":
        rows, labels = fer2013_records(source), FER2013
    elif dataset == "rafdb":
        rows, labels = rafdb_records(source), RAFDB
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    out = check_output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    indices = load_indices(landmark_indices) if landmark_indices else None
    context = open_landmarker(landmarker_model) if landmarker_model else nullcontext(None)
    manifest = out / "manifest.csv"
    temporary = out / "manifest.csv.tmp"
    count = 0

    with context as landmarker, temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()

        for sample_id, split, source_label, image in rows:
            array, rgb = preprocess_image(image)
            visual_path = Path("visual") / split / f"{sample_id}.npy"
            (out / visual_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(out / visual_path, array, allow_pickle=False)

            landmark_file = ""
            status = "disabled"
            if landmarker is not None:
                coords = extract_landmarks(landmarker, rgb, indices)
                if coords is None:
                    status = "missing"
                else:
                    status = "valid"
                    path = Path("landmarks_88") / split / f"{sample_id}.npy"
                    (out / path).parent.mkdir(parents=True, exist_ok=True)
                    np.save(out / path, coords, allow_pickle=False)
                    landmark_file = path.as_posix()

            label = labels[source_label]
            writer.writerow({
                "sample_id": sample_id,
                "split": split,
                "source_label": source_label,
                "paper_label": label,
                "paper_class": CLASSES[label],
                "visual_npy": visual_path.as_posix(),
                "landmarks_88_npy": landmark_file,
                "landmark_status": status,
            })
            count += 1

    temporary.replace(manifest)
    return count
