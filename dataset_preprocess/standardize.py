"""Training-only feature statistics for a completed geometric feature matrix."""

import numpy as np


def fit_training_stats(train_features: np.ndarray):
    samples = np.asarray(train_features, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("Expected a non-empty (samples, features) array")
    if not np.isfinite(samples).all():
        raise ValueError("Feature values must be finite")
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    if np.any(std == 0):
        raise ValueError("A feature has zero variance; check the source data")
    return mean, std


def apply_training_stats(features: np.ndarray, mean: np.ndarray, std: np.ndarray):
    if np.any(np.asarray(std) <= 0):
        raise ValueError("Standard deviations must be positive")
    return ((np.asarray(features) - mean) / std).astype(np.float32)
