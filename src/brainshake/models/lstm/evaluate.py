"""
LSTM training and evaluation using seizure-level (or patient-level) k-fold splits.

The pipeline:
  1. Load EEGDataset
  2. Use k_fold(level="seizure") to get train/val splits (seizure episodes stay whole)
  3. Wrap each split in SeizureEpisodeDataset so the LSTM sees temporal sequences
  4. Train SeizureLSTM per-timestep (each window gets a prediction)
  5. Evaluate per-window accuracy, precision, recall, F1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ...data_handling.load_data import EEGDataset
from .dataset import SeizureEpisodeDataset, pad_collate
from .model import SeizureLSTM, episode_loss

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Epilepsy"
DEFAULT_MODEL_DIR = REPO_ROOT / "out" / "models" / "lstm"


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _extract_labels_from_subset(subset: Union[EEGDataset, Subset]) -> np.ndarray:
    if isinstance(subset, Subset):
        base = cast(EEGDataset, subset.dataset)
        return base.labels[subset.indices]
    return cast(EEGDataset, subset).labels


def _compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    total = len(labels)
    if n0 == 0 or n1 == 0:
        return torch.ones(2, dtype=torch.float32, device=device)
    w0 = total / (2 * n0)
    w1 = total / (2 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


# ------------------------------------------------------------------
# training
# ------------------------------------------------------------------

def train_fold(
    model: SeizureLSTM,
    train_ep: SeizureEpisodeDataset,
    val_ep: SeizureEpisodeDataset,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    model_path: Path | None = None,
    device: torch.device | None = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_train_labels = np.concatenate(
        [train_ep.base.labels[ep] for ep in train_ep.episodes]
    )
    class_weights = _compute_class_weights(all_train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        train_ep, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ep, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_windows = 0
        for feats, labels, lengths in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(feats, lengths)
            loss = episode_loss(logits, labels, lengths, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * lengths.sum().item()
            n_windows += lengths.sum().item()

        avg_loss = running_loss / max(n_windows, 1)
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{epochs} — train loss: {avg_loss:.4f}")

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, model_path)
        logger.info(f"Saved model to {model_path}")


# ------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------

def evaluate_fold(
    model: SeizureLSTM,
    val_ep: SeizureEpisodeDataset,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    val_loader = DataLoader(
        val_ep, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0
    )

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_windows = 0

    with torch.no_grad():
        for feats, labels, lengths in val_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(feats, lengths)
            loss = episode_loss(logits, labels, lengths, criterion)

            preds = logits.argmax(dim=-1)
            for i in range(feats.size(0)):
                ell = lengths[i].item()
                all_preds.extend(preds[i, :ell].cpu().numpy().tolist())
                all_labels.extend(labels[i, :ell].cpu().numpy().tolist())

            total_loss += loss.item() * lengths.sum().item()
            total_windows += lengths.sum().item()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() if len(all_labels) > 0 else 0.0

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(all_labels, all_preds, zero_division="warn")
    recall = recall_score(all_labels, all_preds, zero_division="warn")
    f1 = f1_score(all_labels, all_preds, zero_division="warn")
    avg_loss = total_loss / max(total_windows, 1)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ------------------------------------------------------------------
# k-fold driver
# ------------------------------------------------------------------

def evaluate_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    n_splits: int = 5,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    non_seizure_len: int = 10,
    pool: str = "std",
    context: int = 20,
    random_state: int = 42,
    patient_ids: list[int] | None = None,
    level: str = "seizure",
    suffix: str = "",
) -> None:
    dataset = EEGDataset(data_dir=data_dir, patient_ids=patient_ids, normalize=False)
    model_dir.mkdir(parents=True, exist_ok=True)

    input_size = {"std": 21, "mean": 21, "mean_std": 42, "none": 21 * 128, "conv_proj": 21}[pool]
    use_conv_proj = pool == "conv_proj"

    results: dict = {"folds": [], "average_accuracy": None}
    accuracies: list[float] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, train_sub, val_sub in dataset.k_fold(
        n_splits=n_splits, shuffle=True, random_state=random_state, level=level, context=context
    ):
        logger.info(f"Fold {fold + 1}/{n_splits} (level={level})")

        train_ep = SeizureEpisodeDataset(
            dataset, list(train_sub.indices), non_seizure_len=non_seizure_len, pool=pool
        )
        val_ep = SeizureEpisodeDataset(
            dataset, list(val_sub.indices), non_seizure_len=non_seizure_len, pool=pool
        )

        model = SeizureLSTM(input_size=input_size, hidden_size=128, num_layers=2, n_classes=2, dropout=0.3, conv_proj=use_conv_proj)
        model_path = model_dir / f"lstm_fold_{fold:02d}.pt"

        train_fold(
            model, train_ep, val_ep,
            epochs=epochs, lr=lr, batch_size=batch_size,
            model_path=model_path, device=device,
        )

        metrics = evaluate_fold(model, val_ep, batch_size=batch_size, device=device)
        accuracies.append(metrics["accuracy"])
        logger.info(
            f"Fold {fold}: loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )
        metrics["fold"] = fold
        results["folds"].append(metrics)

    avg_acc = float(np.mean(accuracies)) if accuracies else 0.0
    results["average_accuracy"] = avg_acc
    avg_prec = float(np.mean([f["precision"] for f in results["folds"]]))
    avg_rec = float(np.mean([f["recall"] for f in results["folds"]]))
    avg_f1 = float(np.mean([f["f1"] for f in results["folds"]]))
    results["average_precision"] = avg_prec
    results["average_recall"] = avg_rec
    results["average_f1"] = avg_f1

    bench_dir = REPO_ROOT / "out" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    results["pool"] = pool
    results["context"] = context
    results["non_seizure_len"] = non_seizure_len

    out_path = bench_dir / f"lstm{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmarks to {out_path.relative_to(REPO_ROOT)}")
    logger.info(
        f"Average: acc={avg_acc:.4f} prec={avg_prec:.4f} rec={avg_rec:.4f} f1={avg_f1:.4f}"
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM on EEG with episode-level k-fold")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--non-seizure-len", type=int, default=10, help="Length of non-seizure episode segments")
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--patient-ids", type=int, nargs="+", default=None)
    parser.add_argument("--level", type=str, default="seizure", choices=["patient", "window", "seizure"])
    parser.add_argument("--pool", type=str, default="std", choices=["mean", "std", "mean_std", "none", "conv_proj"],
                        help="Per-window pooling: std (amplitude), mean (risky), mean_std, none (flatten), conv_proj (Conv1d projection)")
    parser.add_argument("--context", type=int, default=20,
                        help="Number of non-seizure windows before/after each seizure episode included in val (seizure-level only)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to the benchmark output file (e.g. '_std_seizure' -> lstm_std_seizure.json)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    evaluate_dataset(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_splits=args.n_splits,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        non_seizure_len=args.non_seizure_len,
        pool=args.pool,
        context=args.context,
        random_state=args.random_state,
        patient_ids=args.patient_ids,
        level=args.level,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()