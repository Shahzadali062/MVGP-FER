"""MediaPipe landmark detection and the manuscript's normalization equations."""

import json
from pathlib import Path

import numpy as np


def load_indices(path: Path) -> tuple[int, ...]:
    indices = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(indices, list) or len(indices) != 44:
        raise ValueError("Expected a JSON list of 44 landmark indices")
    if any(type(idx) is not int or idx < 0 or idx >= 468 for idx in indices):
        raise ValueError("Landmark indices must be integers from 0 to 467")
    if len(set(indices)) != 44:
        raise ValueError("Landmark indices must be unique")
    return tuple(indices)


def normalize_landmarks(mesh: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    points = np.asarray(mesh, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2:
        raise ValueError("Expected a full facial mesh with at least 468 (x, y) points")
    if len(indices) != 44 or len(set(indices)) != 44:
        raise ValueError("Expected 44 distinct selected landmarks")
    if min(indices) < 0 or max(indices) >= len(points) or not np.isfinite(points).all():
        raise ValueError("Invalid landmark indices or coordinates")

    center = (points[33] + points[263]) / 2
    scale = max(
        np.linalg.norm(points[33] - points[263]),
        0.35 * np.linalg.norm(points[10] - points[152]),
        1e-6,
    )
    return ((points[list(indices)] - center) / scale).reshape(88).astype(np.float32)


def open_landmarker(model_path: Path):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Install MediaPipe to extract landmarks: pip install mediapipe") from exc
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def extract_landmarks(landmarker, rgb: np.ndarray, indices: tuple[int, ...]):
    import mediapipe as mp

    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    detected = landmarker.detect(frame)
    if not detected.face_landmarks:
        return None

    mesh = np.array(
        [(p.x, p.y) for p in detected.face_landmarks[0]], dtype=np.float64
    )
    return normalize_landmarks(mesh, indices)
