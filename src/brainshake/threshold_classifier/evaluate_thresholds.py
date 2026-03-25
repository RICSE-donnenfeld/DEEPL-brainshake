"""Patient-wise k-fold evaluation for the threshold classifier."""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple, cast

import numpy as np
from torch.utils.data import Subset
from torch import Tensor

from .classifier import ThresholdClassifier
from ..data_handling.load_data import EEGDataset
from ..data_handling.extract_features import FeatureDict, extract_basic_features


def extract_features_from_subset(subset: Subset) -> Tuple[List[FeatureDict], List[int]]:
    features: List[FeatureDict] = []
    labels: List[int] = []
    for i in range(len(subset)):
        item = subset[i]
        x, y = cast(Tuple[Tensor, Tensor], item)
        arr = x.numpy()
        features.append(extract_basic_features(arr))
        labels.append(int(y.item()))
    return features, labels


def compute_thresholds(
    features: List[FeatureDict], labels: Iterable[int]
) -> Tuple[float, float]:
    paired = list(zip(features, labels))
    seizure_std = [f["std"] for f, l in paired if l == 1]
    non_std = [f["std"] for f, l in paired if l == 0]
    seizure_range = [f["range"] for f, l in paired if l == 1]
    non_range = [f["range"] for f, l in paired if l == 0]

    std_threshold = (
        float((np.mean(non_std) + np.mean(seizure_std)) / 2)
        if non_std and seizure_std
        else 70.0
    )
    range_threshold = (
        float((np.mean(non_range) + np.mean(seizure_range)) / 2)
        if non_range and seizure_range
        else 220.0
    )
    return std_threshold, range_threshold


def evaluate_dataset(
    dataset: EEGDataset, n_splits: int = 5, random_state: int = 42
) -> None:
    accuracies: List[float] = []
    print("Starting patient-wise k-fold evaluation")
    for fold, train_subset, val_subset in dataset.k_fold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    ):
        train_features, train_labels = extract_features_from_subset(train_subset)
        val_features, val_labels = extract_features_from_subset(val_subset)

        std_thr, range_thr = compute_thresholds(train_features, train_labels)
        classifier = ThresholdClassifier(
            std_threshold=std_thr, range_threshold=range_thr
        )
        accuracy = classifier.evaluate(val_features, val_labels)

        print(
            f"Fold {fold}: std_thr={std_thr:.1f}, range_thr={range_thr:.1f}, "
            f"accuracy={accuracy:.4f}"
        )
        accuracies.append(accuracy)

    avg = np.mean(accuracies) if accuracies else 0.0
    print(f"K-fold ({n_splits}) average accuracy: {avg:.4f}")


def main() -> None:
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    dataset = EEGDataset(data_dir="data/Epilepsy", normalize=False)
    evaluate_dataset(dataset, n_splits=4, random_state=2026)


if __name__ == "__main__":
    main()
