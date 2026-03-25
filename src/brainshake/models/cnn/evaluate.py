"""Patient-wise k-fold evaluation for the CNN classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .model import SimpleEEGCNN, _make_loader, _evaluate, train
from ...data_handling.load_data import EEGDataset

# project root for output paths
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "out" / "models" / "cnn"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Epilepsy"


def evaluate_dataset(
    data_dir: Path,
    model_dir: Path,
    n_splits: int = 5,
    epochs: int = 10,
    random_state: int = 42,
    patient_ids: Optional[Sequence[int]] = None,
) -> None:
    dataset = EEGDataset(data_dir=data_dir, patient_ids=patient_ids, normalize=False)
    accuracies: list[float] = []
    results: dict = {"folds": [], "average_accuracy": None}
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Starting patient-wise k-fold CNN evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, train_subset, val_subset in dataset.k_fold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    ):
        # train a fold-specific model
        model_path = model_dir / f"cnn_fold_{fold:02d}.pt"
        train(
            epochs=epochs,
            train_dataset=train_subset,
            val_dataset=val_subset,
            model_path=model_path,
            resume=False,
        )

        # load and evaluate
        checkpoint = torch.load(model_path, map_location=device)
        model = SimpleEEGCNN().to(device)
        model.load_state_dict(checkpoint.get("model_state", {}))

        loader = _make_loader(val_subset, shuffle=False, num_workers=1)
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = _evaluate(model, loader, criterion, device)
        accuracies.append(accuracy)

        try:
            display_path = model_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = model_path
        print(
            f"Fold {fold}: loss={loss:.4f}, accuracy={accuracy:.4f}, "
            f"saved_model={display_path}"
        )
        results["folds"].append({
            "fold": fold,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "saved_model": str(display_path),
        })

    avg = np.mean(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(avg)
    print(f"K-fold ({n_splits}) average accuracy: {avg:.4f}")

    # write benchmarks JSON
    bench_dir = REPO_ROOT / "out" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "cnn.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmarks to {out_path.relative_to(REPO_ROOT)}")


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
        epochs=args.epochs,
        random_state=args.random_state,
        patient_ids=patient_ids,
    )


if __name__ == "__main__":
    main()
