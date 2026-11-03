"""Replicate a grayscale image to three channels and normalize it."""

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def replicate_channels(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 2:
        raise ValueError("Expected a grayscale image")
    return np.repeat(pixels[:, :, None], 3, axis=2)


def normalize_image(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB array (H, W, 3)")
    values = rgb.astype(np.float32) / 255.0
    values = (values - MEAN) / STD
    return np.ascontiguousarray(values.transpose(2, 0, 1))
