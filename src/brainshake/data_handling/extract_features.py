"""
Signal feature extraction helpers
"""

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

FeatureDict = Dict[str, float]


def _per_channel_statistic(eeg_window: NDArray[Any], func: str) -> NDArray[Any]:
    return getattr(np, func)(eeg_window, axis=-1)


def extract_basic_features(eeg_window: NDArray[Any]) -> FeatureDict:
    """Return the statistics that best separate seizure from non-seizure windows."""

    channel_means = _per_channel_statistic(eeg_window, "mean")
    channel_stds = _per_channel_statistic(eeg_window, "std")
    channel_mins = _per_channel_statistic(eeg_window, "min")
    channel_maxs = _per_channel_statistic(eeg_window, "max")

    mean_val = float(np.mean(channel_means))
    std_val = float(np.mean(channel_stds))
    min_val = float(np.mean(channel_mins))
    max_val = float(np.mean(channel_maxs))
    range_val = max_val - min_val

    features: FeatureDict = {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "peak_to_peak": float(np.mean(channel_maxs - channel_mins)),
    }

    features["std_range_ratio"] = std_val / range_val if range_val else 0.0
    features["range_std_sum"] = range_val + std_val

    return features


def to_vector(feature_dict: FeatureDict, order: Dict[str, int]) -> np.ndarray:
    """Convert a feature dict into an ordered numpy vector for classifiers."""

    sorted_keys = sorted(order, key=lambda key: order[key])
    return np.array([feature_dict.get(key, 0.0) for key in sorted_keys], dtype=float)
