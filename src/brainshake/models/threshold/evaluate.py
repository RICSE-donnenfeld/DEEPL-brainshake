"""Patient-wise k-fold evaluation for the threshold classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, cast

import numpy as np
from torch.utils.data import Subset
from torch import Tensor

from .classifier import ThresholdClassifier
from ...data_handling.load_data import EEGDataset
from ...data_handling.extract_features import FeatureDict, extract_basic_features


# project root defaults
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Epilepsy"
DEFAULT_N_SPLITS = 4
DEFAULT_RANDOM_STATE = 2026


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
) -> Tuple[float, float, Optional[float], Optional[float]]:
    paired = list(zip(features, labels))
    seizure_std = [f["std"] for f, l in paired if l == 1]
    non_std = [f["std"] for f, l in paired if l == 0]
    seizure_range = [f["range"] for f, l in paired if l == 1]
    non_range = [f["range"] for f, l in paired if l == 0]
    seizure_min = [f["min"] for f, l in paired if l == 1]
    non_min = [f["min"] for f, l in paired if l == 0]
    seizure_max = [f["max"] for f, l in paired if l == 1]
    non_max = [f["max"] for f, l in paired if l == 0]

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
    min_threshold = (
        float((np.mean(non_min) + np.mean(seizure_min)) / 2)
        if non_min and seizure_min
        else None
    )
    max_threshold = (
        float((np.mean(non_max) + np.mean(seizure_max)) / 2)
        if non_max and seizure_max
        else None
    )
    return std_threshold, range_threshold, min_threshold, max_threshold


def evaluate_dataset(
    dataset: EEGDataset, n_splits: int = 5, random_state: int = 42
) -> None:
    accuracies: List[float] = []
    results: dict = {"folds": [], "average_accuracy": None}
    print("Starting patient-wise k-fold evaluation")
    for fold, train_subset, val_subset in dataset.k_fold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    ):
        train_features, train_labels = extract_features_from_subset(train_subset)
        val_features, val_labels = extract_features_from_subset(val_subset)

        std_thr, range_thr, min_thr, max_thr = compute_thresholds(
            train_features, train_labels
        )
        classifier = ThresholdClassifier(
            std_threshold=std_thr,
            range_threshold=range_thr,
            min_threshold=min_thr,
            max_threshold=max_thr,
        )
        accuracy = classifier.evaluate(val_features, val_labels)

        threshold_parts = [
            f"std_thr={std_thr:.1f}",
            f"range_thr={range_thr:.1f}",
        ]
        if min_thr is not None:
            threshold_parts.append(f"min_thr={min_thr:.1f}")
        if max_thr is not None:
            threshold_parts.append(f"max_thr={max_thr:.1f}")
        print(f"Fold {fold}: {', '.join(threshold_parts)}, accuracy={accuracy:.4f}")
        accuracies.append(accuracy)
        results["folds"].append({
            "fold": fold,
            "std_threshold": std_thr,
            "range_threshold": range_thr,
            "min_threshold": min_thr,
            "max_threshold": max_thr,
            "accuracy": accuracy,
        })

    avg = np.mean(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(avg)
    print(f"K-fold ({n_splits}) average accuracy: {avg:.4f}")

    # write benchmarks JSON
    bench_dir = REPO_ROOT / "out" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "threshold.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmarks to {out_path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ThresholdClassifier patient-wise k-fold"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the CHB-MIT EEG data directory",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of patient-wise folds",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--patient-ids",
        type=int,
        nargs="+",
        help="Explicit patient IDs to load",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Smoke-test: load first 2 patients and use two folds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    if args.smoke_test:
        patient_ids = [1, 2]
        args.n_splits = min(args.n_splits, len(patient_ids))
    else:
        patient_ids = args.patient_ids
    dataset = EEGDataset(
        data_dir=args.data_dir, patient_ids=patient_ids, normalize=False
    )
    evaluate_dataset(
        dataset, n_splits=args.n_splits, random_state=args.random_state
    )


if __name__ == "__main__":
    main()
