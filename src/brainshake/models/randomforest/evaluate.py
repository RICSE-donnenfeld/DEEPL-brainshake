"""Patient-wise k-fold evaluation for the random forest classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json
from typing import Iterable, List, Tuple, cast, Optional, Sequence

import joblib
from torch import Tensor
from torch.utils.data import Subset

from .model import RandomForestSignalClassifier
from ...data_handling.extract_features import FeatureDict, extract_basic_features
from ...data_handling.load_data import EEGDataset


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "out" / "models" / "randomforest"


def extract_features_from_subset(subset: Subset) -> Tuple[List[FeatureDict], List[int]]:
    features: List[FeatureDict] = []
    labels: List[int] = []
    for idx in range(len(subset)):
        x, y = cast(Tuple[Tensor, Tensor], subset[idx])
        arr = x.numpy()
        features.append(extract_basic_features(arr))
        labels.append(int(y.item()))
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
    dataset = EEGDataset(data_dir=data_dir, patient_ids=patient_ids, normalize=False)
    accuracies: List[float] = []
    results: dict = {"folds": [], "average_accuracy": None}
    model_dir.mkdir(parents=True, exist_ok=True)

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

        correct = sum(
            1 for pred, label in zip(predictions, val_labels) if pred == label
        )
        accuracy = correct / len(val_labels) if val_labels else 0.0
        accuracies.append(accuracy)

        model_path = model_dir / f"model_fold_{fold:02d}.joblib"
        joblib.dump(classifier, model_path)
        try:
            display_path = model_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = model_path
        print(f"Fold {fold}: accuracy={accuracy:.4f}, saved_model={display_path}")
        results["folds"].append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "saved_model": str(display_path),
            }
        )

    average = sum(accuracies) / len(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(average)
    print(f"K-fold ({n_splits}) average accuracy: {average:.4f}")

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
