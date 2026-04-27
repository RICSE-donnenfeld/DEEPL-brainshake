"""Patient-wise k-fold evaluation for the random forest classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json
from typing import Iterable, List, Tuple, cast, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from typing import Any

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from torch import Tensor
    from torch.utils.data import Subset

    from ...data_handling.load_data import EEGDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    TORCH_AVAILABLE = False
    Tensor = object  # type: ignore[assignment]
    Subset = object  # type: ignore[assignment]
    EEGDataset = object  # type: ignore[assignment]

from .model import RandomForestSignalClassifier
from ...data_handling.extract_features import FeatureDict, extract_basic_features


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "out" / "models" / "randomforest"


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
    for idx in range(len(subset)):
        x, y = cast(Tuple[Any, Any], subset[idx])
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


def evaluate_dataset(
    data_dir: Path,
    model_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = None,
    patient_ids: Optional[Sequence[int]] = None,
) -> None:
    if TORCH_AVAILABLE:
        dataset = EEGDataset(data_dir=data_dir, patient_ids=patient_ids, normalize=False)
        dataset_patient_ids = [int(pid) for pid in dataset.patient_ids]
    else:
        dataset = None
        dataset_patient_ids = (
            [int(pid) for pid in patient_ids]
            if patient_ids is not None
            else list(range(1, 25))
        )
    accuracies: List[float] = []
    confusion_total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    results: dict = {
        "meta": {
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "patient_ids": dataset_patient_ids,
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth) if max_depth is not None else None,
            "torch_available": bool(TORCH_AVAILABLE),
            "positive_class": 1,
        },
        "folds": [],
        "average_accuracy": None,
        "aggregate": {},
    }
    model_dir.mkdir(parents=True, exist_ok=True)

    if TORCH_AVAILABLE:
        assert dataset is not None
        print("Starting patient-wise k-fold RandomForest evaluation")
        for fold, train_subset, val_subset in dataset.k_fold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        ):
            train_features, train_labels = extract_features_from_subset(train_subset)
            val_features, val_labels = extract_features_from_subset(val_subset)

            classifier = RandomForestSignalClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            classifier.fit(train_features, train_labels)
            predictions = classifier.predict(val_features)
            accuracy = float(accuracy_score(val_labels, predictions)) if val_labels else 0.0

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

            model_path = model_dir / f"model_fold_{fold:02d}.joblib"
            joblib.dump(classifier, model_path)
            try:
                display_path = model_path.relative_to(REPO_ROOT)
            except ValueError:
                display_path = model_path
            print(
                f"Fold {fold}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, saved_model={display_path}"
            )
            results["folds"].append(
                {
                    "fold": fold,
                    "train_patients": getattr(train_subset, "patient_ids", None),
                    "val_patients": getattr(val_subset, "patient_ids", None),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion": confusion,
                    "metrics": metrics,
                    "saved_model": str(display_path),
                }
            )
    else:
        print("Starting patient-wise k-fold RandomForest evaluation (torch-free fallback)")
        folds = _patient_folds(
            dataset_patient_ids, n_splits=n_splits, random_state=random_state
        )
        for fold, val_patients in enumerate(folds):
            train_patients = [
                int(pid) for pid in dataset_patient_ids if int(pid) not in val_patients
            ]
            train_features, train_labels = _extract_features_from_patients(
                data_dir, train_patients
            )
            val_features, val_labels = _extract_features_from_patients(
                data_dir, val_patients
            )

            classifier = RandomForestSignalClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            classifier.fit(train_features, train_labels)
            predictions = classifier.predict(val_features)
            accuracy = float(accuracy_score(val_labels, predictions)) if val_labels else 0.0

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

            model_path = model_dir / f"model_fold_{fold:02d}.joblib"
            joblib.dump(classifier, model_path)
            try:
                display_path = model_path.relative_to(REPO_ROOT)
            except ValueError:
                display_path = model_path

            print(
                f"Fold {fold}: accuracy={accuracy:.4f}, saved_model={display_path}"
            )
            results["folds"].append(
                {
                    "fold": fold,
                    "train_patients": train_patients,
                    "val_patients": val_patients,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion": confusion,
                    "metrics": metrics,
                    "saved_model": str(display_path),
                }
            )

    # Calculate average accuracy
    average = sum(accuracies) / len(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(average)
    print(f"K-fold ({n_splits}) average accuracy: {average:.4f}")
    
    # Calculate average precision, recall, and f1 if they exist in results
    if results.get("folds") and len(results["folds"]) > 0:
        precisions = [fold.get("precision", 0.0) for fold in results["folds"]]
        recalls = [fold.get("recall", 0.0) for fold in results["folds"]]
        f1s = [fold.get("f1", 0.0) for fold in results["folds"]]
        
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        
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
    out_path = bench_dir / "randomforest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmarks to {out_path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RandomForest on EEG features with patient-wise k-fold"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Epilepsy",
        help="Path to the CHB-MIT EEG data directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to persist trained random forest models",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of patient-wise folds"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for the random forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of each tree",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2026,
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.smoke_test:
        patient_ids = [1, 2]
        args.n_splits = min(args.n_splits, len(patient_ids))
    else:
        patient_ids = args.patient_ids
    evaluate_dataset(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        patient_ids=patient_ids,
    )


if __name__ == "__main__":
    main()
