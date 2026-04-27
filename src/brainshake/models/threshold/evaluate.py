"""Patient-wise k-fold evaluation for the threshold classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, cast, Sequence, Any

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Subset
    from torch import Tensor

    from ...data_handling.load_data import EEGDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    TORCH_AVAILABLE = False
    Subset = object  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment]
    EEGDataset = object  # type: ignore[assignment]

from .classifier import ThresholdClassifier
from ...data_handling.extract_features import FeatureDict, extract_basic_features


# project root defaults
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Epilepsy"
DEFAULT_N_SPLITS = 4
DEFAULT_RANDOM_STATE = 2026


def _confusion_and_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive_class: int = 1,
) -> tuple[dict, dict]:
    pos = int(positive_class)
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        yt_pos = int(yt) == pos
        yp_pos = int(yp) == pos
        if yp_pos and yt_pos:
            tp += 1
        elif yp_pos and not yt_pos:
            fp += 1
        elif (not yp_pos) and (not yt_pos):
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    confusion = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "support_pos": int(tp + fn),
        "support_neg": int(tn + fp),
    }
    return confusion, metrics


def extract_features_from_subset(subset: Any) -> Tuple[List[FeatureDict], List[int]]:
    features: List[FeatureDict] = []
    labels: List[int] = []
    for i in range(len(subset)):
        item = subset[i]
        x, y = cast(Tuple[Any, Any], item)
        arr = x.numpy()
        features.append(extract_basic_features(arr))
        labels.append(int(y.item()))
    return features, labels


def _patient_folds(
    patient_ids: Sequence[int],
    n_splits: int,
    random_state: int,
) -> List[List[int]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(patient_ids):
        raise ValueError("n_splits cannot exceed patient count")
    rng = np.random.default_rng(random_state)
    ordered = list(rng.permutation(list(patient_ids)))
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for idx, patient in enumerate(ordered):
        folds[idx % n_splits].append(int(patient))
    return folds


def _load_patient_arrays(data_dir: Path, patient_id: int) -> Tuple[np.ndarray, np.ndarray]:
    pid = f"chb{patient_id:02d}"
    npz_path = data_dir / f"{pid}_seizure_EEGwindow_1.npz"
    meta_path = data_dir / f"{pid}_seizure_metadata_1.parquet"
    npz = np.load(npz_path, allow_pickle=True)
    eeg = npz["EEG_win"].astype(np.float32)
    metadata = pd.read_parquet(meta_path)
    labels = metadata["class"].to_numpy(dtype=np.int64)
    return eeg, labels


def _extract_features_from_patients(
    data_dir: Path, patient_ids: Sequence[int]
) -> Tuple[List[FeatureDict], List[int]]:
    features: List[FeatureDict] = []
    labels: List[int] = []
    for pid in patient_ids:
        eeg, y = _load_patient_arrays(data_dir, int(pid))
        for window, label in zip(eeg, y):
            features.append(extract_basic_features(window))
            labels.append(int(label))
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


def evaluate_dataset(dataset: Any, n_splits: int = 5, random_state: int = 42) -> None:
    accuracies: List[float] = []
    confusion_total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    results: dict = {
        "meta": {
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "patient_ids": [int(pid) for pid in dataset.patient_ids],
            "positive_class": 1,
        },
        "folds": [],
        "average_accuracy": None,
        "aggregate": {},
    }
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
        predictions = classifier.predict_batch(val_features)
        accuracy = (
            sum(1 for p, l in zip(predictions, val_labels) if int(p) == int(l))
            / len(val_labels)
            if val_labels
            else 0.0
        )
        confusion, metrics = _confusion_and_metrics(
            y_true=val_labels, y_pred=predictions, positive_class=1
        )
        confusion_total["tp"] += confusion["tp"]
        confusion_total["fp"] += confusion["fp"]
        confusion_total["tn"] += confusion["tn"]
        confusion_total["fn"] += confusion["fn"]
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f1 = float(metrics["f1"])

        threshold_parts = [
            f"std_thr={std_thr:.1f}",
            f"range_thr={range_thr:.1f}",
        ]
        if min_thr is not None:
            threshold_parts.append(f"min_thr={min_thr:.1f}")
        if max_thr is not None:
            threshold_parts.append(f"max_thr={max_thr:.1f}")
        print(f"Fold {fold}: {', '.join(threshold_parts)}, accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        accuracies.append(accuracy)
        results["folds"].append({
            "fold": fold,
            "train_patients": getattr(train_subset, "patient_ids", None),
            "val_patients": getattr(val_subset, "patient_ids", None),
            "std_threshold": std_thr,
            "range_threshold": range_thr,
            "min_threshold": min_thr,
            "max_threshold": max_thr,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
            "metrics": metrics,
        })

    # Calculate average accuracy
    avg = np.mean(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(avg)
    print(f"K-fold ({n_splits}) average accuracy: {avg:.4f}")
    
    # Calculate average precision, recall, and f1 if they exist in results
    if results.get("folds") and len(results["folds"]) > 0:
        precisions = [fold.get("precision", 0.0) for fold in results["folds"]]
        recalls = [fold.get("recall", 0.0) for fold in results["folds"]]
        f1s = [fold.get("f1", 0.0) for fold in results["folds"]]
        
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1 = np.mean(f1s) if f1s else 0.0
        
        results["average_precision"] = float(avg_precision)
        results["average_recall"] = float(avg_recall)
        results["average_f1"] = float(avg_f1)
        
        print(f"K-fold average precision: {avg_precision:.4f}")
        print(f"K-fold average recall: {avg_recall:.4f}")
        print(f"K-fold average F1: {avg_f1:.4f}")

    tp = confusion_total["tp"]
    fp = confusion_total["fp"]
    tn = confusion_total["tn"]
    fn = confusion_total["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    results["aggregate"] = {
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_accuracy),
            "support_pos": int(tp + fn),
            "support_neg": int(tn + fp),
        },
    }

    # write benchmarks JSON
    bench_dir = REPO_ROOT / "out" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "threshold.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmarks to {out_path.relative_to(REPO_ROOT)}")


def evaluate_dataset_fallback(
    data_dir: Path,
    patient_ids: Sequence[int],
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    folds = _patient_folds(patient_ids, n_splits=n_splits, random_state=random_state)
    accuracies: List[float] = []
    confusion_total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    results: dict = {
        "meta": {
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "patient_ids": [int(pid) for pid in patient_ids],
            "torch_available": False,
            "positive_class": 1,
        },
        "folds": [],
        "average_accuracy": None,
        "aggregate": {},
    }

    print("Starting patient-wise k-fold evaluation (torch-free fallback)")
    for fold, val_patients in enumerate(folds):
        train_patients = [int(pid) for pid in patient_ids if int(pid) not in val_patients]
        train_features, train_labels = _extract_features_from_patients(data_dir, train_patients)
        val_features, val_labels = _extract_features_from_patients(data_dir, val_patients)

        std_thr, range_thr, min_thr, max_thr = compute_thresholds(train_features, train_labels)
        classifier = ThresholdClassifier(
            std_threshold=std_thr,
            range_threshold=range_thr,
            min_threshold=min_thr,
            max_threshold=max_thr,
        )
        predictions = classifier.predict_batch(val_features)
        accuracy = (
            sum(1 for p, l in zip(predictions, val_labels) if int(p) == int(l))
            / len(val_labels)
            if val_labels
            else 0.0
        )
        confusion, metrics = _confusion_and_metrics(
            y_true=val_labels, y_pred=predictions, positive_class=1
        )
        confusion_total["tp"] += confusion["tp"]
        confusion_total["fp"] += confusion["fp"]
        confusion_total["tn"] += confusion["tn"]
        confusion_total["fn"] += confusion["fn"]
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f1 = float(metrics["f1"])
        accuracies.append(accuracy)

        threshold_parts = [f"std_thr={std_thr:.1f}", f"range_thr={range_thr:.1f}"]
        if min_thr is not None:
            threshold_parts.append(f"min_thr={min_thr:.1f}")
        if max_thr is not None:
            threshold_parts.append(f"max_thr={max_thr:.1f}")
        print(f"Fold {fold}: {', '.join(threshold_parts)}, accuracy={accuracy:.4f}")

        results["folds"].append(
            {
                "fold": fold,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "std_threshold": std_thr,
                "range_threshold": range_thr,
                "min_threshold": min_thr,
                "max_threshold": max_thr,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion": confusion,
                "metrics": metrics,
            }
        )

    avg = np.mean(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(avg)
    print(f"K-fold ({n_splits}) average accuracy: {avg:.4f}")

    if results.get("folds") and len(results["folds"]) > 0:
        precisions = [fold.get("precision", 0.0) for fold in results["folds"]]
        recalls = [fold.get("recall", 0.0) for fold in results["folds"]]
        f1s = [fold.get("f1", 0.0) for fold in results["folds"]]

        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1 = np.mean(f1s) if f1s else 0.0

        results["average_precision"] = float(avg_precision)
        results["average_recall"] = float(avg_recall)
        results["average_f1"] = float(avg_f1)

        print(f"K-fold average precision: {avg_precision:.4f}")
        print(f"K-fold average recall: {avg_recall:.4f}")
        print(f"K-fold average F1: {avg_f1:.4f}")

    tp = confusion_total["tp"]
    fp = confusion_total["fp"]
    tn = confusion_total["tn"]
    fn = confusion_total["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    results["aggregate"] = {
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_accuracy),
            "support_pos": int(tp + fn),
            "support_neg": int(tn + fp),
        },
    }

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
    if TORCH_AVAILABLE:
        dataset = EEGDataset(
            data_dir=args.data_dir, patient_ids=patient_ids, normalize=False
        )
        evaluate_dataset(
            dataset, n_splits=args.n_splits, random_state=args.random_state
        )
    else:
        if patient_ids is None:
            patient_ids = list(range(1, 25))
        evaluate_dataset_fallback(
            data_dir=args.data_dir,
            patient_ids=patient_ids,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
