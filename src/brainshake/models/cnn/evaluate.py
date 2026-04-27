"""Patient-wise k-fold evaluation for the CNN classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .model import SimpleEEGCNN, _make_loader, _evaluate_with_stats, train
from ...data_handling.load_data import EEGDataset

# project root for output paths
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "out" / "models" / "cnn"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Epilepsy"


def _persist_results(results: dict, accuracies: Sequence[float]) -> None:
    # Calculate average accuracy
    avg_acc = np.mean(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(avg_acc)
    print(f"K-fold average accuracy: {avg_acc:.4f}")

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

    bench_dir = REPO_ROOT / "out" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "cnn.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmarks to {out_path.relative_to(REPO_ROOT)}")


def evaluate_dataset(
    dataset: EEGDataset,
    model_dir: Path,
    n_splits: int = 5,
    epochs: int = 10,
    random_state: int = 42,
) -> None:
    accuracies: list[float] = []
    confusion_total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    results: dict = {
        "meta": {
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "epochs": int(epochs),
            "patient_ids": [int(pid) for pid in dataset.patient_ids],
            "normalize": True,
            "context_windows": int(getattr(dataset, "context_windows", 1)),
            "positive_class": 1,
        },
        "folds": [],
        "average_accuracy": None,
        "aggregate": {},
    }
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Starting patient-wise k-fold CNN evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, train_subset, val_subset in dataset.k_fold(
        n_splits=n_splits, shuffle=True, random_state=random_state, level="patient"
    ):
        model_path = model_dir / f"cnn_fold_{fold:02d}.pt"
        train(
            epochs=epochs,
            train_dataset=train_subset,
            val_dataset=val_subset,
            model_path=model_path,
            resume=False,
            seed=random_state,
        )

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        model = SimpleEEGCNN().to(device)
        model.load_state_dict(checkpoint.get("model_state", {}))

        loader = _make_loader(val_subset, shuffle=False, num_workers=1, seed=random_state)
        criterion = nn.CrossEntropyLoss()
        loss, accuracy, stats = _evaluate_with_stats(
            model, loader, criterion, device, positive_class=1
        )
        precision = float(stats["precision"])
        recall = float(stats["recall"])
        f1 = float(stats["f1"])
        accuracies.append(accuracy)

        confusion_total["tp"] += int(stats["tp"])
        confusion_total["fp"] += int(stats["fp"])
        confusion_total["tn"] += int(stats["tn"])
        confusion_total["fn"] += int(stats["fn"])

        try:
            display_path = model_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = model_path
        print(
            f"Fold {fold}: loss={loss:.4f}, accuracy={accuracy:.4f}, precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f}, saved_model={display_path}"
        )
        results["folds"].append(
            {
                "fold": fold,
                "train_patients": getattr(train_subset, "patient_ids", None),
                "val_patients": getattr(val_subset, "patient_ids", None),
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion": {
                    "tp": int(stats["tp"]),
                    "fp": int(stats["fp"]),
                    "tn": int(stats["tn"]),
                    "fn": int(stats["fn"]),
                },
                "metrics": {
                    "precision": float(stats["precision"]),
                    "recall": float(stats["recall"]),
                    "specificity": float(stats["specificity"]),
                    "f1": float(stats["f1"]),
                    "balanced_accuracy": float(stats["balanced_accuracy"]),
                    "support_pos": int(stats["support_pos"]),
                    "support_neg": int(stats["support_neg"]),
                },
                "saved_model": str(display_path),
            }
        )

    # Aggregate confusion and derived metrics over all validation windows.
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

    _persist_results(results, accuracies)


def evaluate_saved_models(
    dataset: EEGDataset,
    model_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    accuracies: list[float] = []
    confusion_total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    results: dict = {
        "meta": {
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "patient_ids": [int(pid) for pid in dataset.patient_ids],
            "normalize": True,
            "use_saved_models": True,
            "context_windows": int(getattr(dataset, "context_windows", 1)),
            "positive_class": 1,
        },
        "folds": [],
        "average_accuracy": None,
        "aggregate": {},
    }
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating existing CNN checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, train_subset, val_subset in dataset.k_fold(
        n_splits=n_splits, shuffle=True, random_state=random_state, level="patient"
    ):
        model_path = model_dir / f"cnn_fold_{fold:02d}.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected checkpoint not found: {model_path}"
            )
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        model = SimpleEEGCNN().to(device)
        model.load_state_dict(checkpoint.get("model_state", {}))

        loader = _make_loader(val_subset, shuffle=False, num_workers=1, seed=random_state)
        criterion = nn.CrossEntropyLoss()
        loss, accuracy, stats = _evaluate_with_stats(
            model, loader, criterion, device, positive_class=1
        )
        precision = float(stats["precision"])
        recall = float(stats["recall"])
        f1 = float(stats["f1"])
        accuracies.append(accuracy)

        confusion_total["tp"] += int(stats["tp"])
        confusion_total["fp"] += int(stats["fp"])
        confusion_total["tn"] += int(stats["tn"])
        confusion_total["fn"] += int(stats["fn"])

        try:
            display_path = model_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = model_path
        print(
            f"Fold {fold}: loss={loss:.4f}, accuracy={accuracy:.4f}, precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f}, loaded_model={display_path}"
        )
        results["folds"].append(
            {
                "fold": fold,
                "train_patients": getattr(train_subset, "patient_ids", None),
                "val_patients": getattr(val_subset, "patient_ids", None),
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion": {
                    "tp": int(stats["tp"]),
                    "fp": int(stats["fp"]),
                    "tn": int(stats["tn"]),
                    "fn": int(stats["fn"]),
                },
                "metrics": {
                    "precision": float(stats["precision"]),
                    "recall": float(stats["recall"]),
                    "specificity": float(stats["specificity"]),
                    "f1": float(stats["f1"]),
                    "balanced_accuracy": float(stats["balanced_accuracy"]),
                    "support_pos": int(stats["support_pos"]),
                    "support_neg": int(stats["support_neg"]),
                },
                "loaded_model": str(display_path),
            }
        )

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

    _persist_results(results, accuracies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CNN on raw EEG with patient-wise k-fold"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the CHB-MIT EEG data directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to persist CNN checkpoints",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of patient-wise folds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs per fold",
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
        "--context-windows",
        type=int,
        default=1,
        help="Number of consecutive 1-second windows to stack as context (default: 1).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Smoke-test: load first 2 patients and use two folds",
    )
    parser.add_argument(
        "--use-saved-models",
        action="store_true",
        help="Load the checkpoints previously saved by train-cnn instead of retraining.",
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
    dataset = EEGDataset(
        data_dir=args.data_dir,
        patient_ids=patient_ids,
        normalize=True,
        context_windows=args.context_windows,
    )
    if args.use_saved_models:
        evaluate_saved_models(
            dataset,
            model_dir=args.model_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
    else:
        evaluate_dataset(
            dataset,
            model_dir=args.model_dir,
            n_splits=args.n_splits,
            epochs=args.epochs,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
