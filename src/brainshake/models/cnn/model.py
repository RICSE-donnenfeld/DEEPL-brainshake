from __future__ import annotations

from pathlib import Path
import os
import random
import functools

import argparse
import logging
from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score

from ...data_handling.load_data import EEGDataset

logger = logging.getLogger(__name__)

BATCH_SIZE = 32

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Epilepsy"
DEFAULT_MODEL_DIR = REPO_ROOT / "out" / "models" / "cnn"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = int(base_seed) + int(worker_id)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _make_loader(
    dataset: Union[EEGDataset, Subset],
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    worker_init_fn = None
    if seed is not None:
        worker_init_fn = functools.partial(seed_worker, base_seed=int(seed))

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def _extract_labels(dataset: Union[EEGDataset, Subset]) -> np.ndarray:
    if isinstance(dataset, Subset):
        if dataset.indices is None:
            raise ValueError("Subset indices must be provided")
        base_labels = cast(EEGDataset, dataset.dataset).labels
        return base_labels[dataset.indices]
    return cast(EEGDataset, dataset).labels


def _evaluate(
    model: SimpleEEGCNN,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            total += x_batch.size(0)

    model.train()
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Convert to numpy arrays for sklearn metrics
    import numpy as np
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division='warn')
    recall = recall_score(all_labels, all_preds, zero_division='warn')
    f1 = f1_score(all_labels, all_preds, zero_division='warn')
    
    avg_loss = total_loss / total
    return avg_loss, accuracy, precision, recall, f1


def _evaluate_with_stats(
    model: SimpleEEGCNN,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    positive_class: int = 1,
) -> tuple[float, float, dict]:
    """Evaluate and return loss/accuracy plus confusion-matrix stats.

    The stats assume binary classification with `positive_class` (default=1).
    """

    model.eval()
    total_loss = 0.0
    tp = fp = tn = fn = 0
    total = 0

    pos = int(positive_class)
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            preds = torch.argmax(outputs, dim=1)
            total += x_batch.size(0)

            y_pos = y_batch == pos
            p_pos = preds == pos
            tp += (p_pos & y_pos).sum().item()
            fp += (p_pos & ~y_pos).sum().item()
            tn += (~p_pos & ~y_pos).sum().item()
            fn += (~p_pos & y_pos).sum().item()

    model.train()
    if total == 0:
        return 0.0, 0.0, {
            "positive_class": pos,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "support_pos": 0,
            "support_neg": 0,
        }

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    stats = {
        "positive_class": pos,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "support_pos": int(tp + fn),
        "support_neg": int(tn + fp),
    }
    return total_loss / total, float(accuracy), stats


def _save_checkpoint(
    model: SimpleEEGCNN,
    optimizer: torch.optim.Optimizer,
    path: Path,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    logger.info(f"Saved checkpoint to {path}")


class SimpleEEGCNN(nn.Module):
    def __init__(self, in_channels: int = 21, n_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=32, kernel_size=5, padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Support stacked temporal context (e.g. 5x128 samples) by forcing
            # a fixed feature length. For the original 1-second windows this is
            # effectively a no-op because the pooled length is already 16.
            nn.AdaptiveAvgPool1d(16),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(
    epochs: int,
    train_dataset: Optional[Union[EEGDataset, Subset]] = None,
    val_dataset: Optional[Union[EEGDataset, Subset]] = None,
    model_path: Optional[Path] = None,
    resume: bool = False,
    seed: int | None = None,
    context_windows: int = 1,
) -> None:
    if seed is not None:
        seed_everything(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using data directory: {DEFAULT_DATA_DIR}")

    if train_dataset is None:
        dataset = EEGDataset(
            data_dir=DEFAULT_DATA_DIR,
            normalize=True,
            context_windows=int(context_windows),
        )
    else:
        dataset = train_dataset

    # Respect Slurm CPU allocation to avoid oversubscribing DataLoader workers.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpu_budget = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else (os.cpu_count() or 1)
    # Use at most 4 workers (PyTorch often warns beyond that on shared systems).
    num_workers = 0 if cpu_budget <= 1 else min(4, cpu_budget - 1)

    loader = _make_loader(dataset, shuffle=True, num_workers=num_workers, seed=seed)
    val_loader = (
        _make_loader(val_dataset, shuffle=False, num_workers=num_workers, seed=seed)
        if val_dataset is not None
        else None
    )

    model = SimpleEEGCNN().to(device)

    # simple class weighting because seizure class is smaller
    subset_labels = _extract_labels(dataset)
    n0 = (subset_labels == 0).sum()
    n1 = (subset_labels == 1).sum()
    total_labels = len(subset_labels)
    class_weights = torch.tensor(
        [total_labels / (2 * n0), total_labels / (2 * n1)],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    if resume and model_path is not None and model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", {}))
        optimizer.load_state_dict(checkpoint.get("optimizer_state", {}))
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resuming training from {model_path} (epoch {start_epoch})")

    model.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        running_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == y_batch).float().mean().item()
                logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Acc: {acc:.4f}"
                )

        current_epoch = epoch + 1
        logger.info(
            f"Epoch {current_epoch} finished. Avg loss: {running_loss / len(loader):.4f}"
        )
        if val_loader is not None:
            val_loss, val_acc, val_precision, val_recall, val_f1 = _evaluate(model, val_loader, criterion, device)
            logger.info(
                f"Epoch {current_epoch} validation: Loss {val_loss:.4f}, Acc {val_acc:.4f}, Precision {val_precision:.4f}, Recall {val_recall:.4f}, F1 {val_f1:.4f}"
            )
        if model_path is not None:
            _save_checkpoint(model, optimizer, model_path, current_epoch)


def main():
    print("Starting...")
    parser = argparse.ArgumentParser(description="Brainshake")
    parser.add_argument(
        "-c", "--command", type=str, help="Command", required=False, default="train"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="Epochs", required=False, default="10"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity level"
    )
    parser.add_argument(
        "-k",
        "--kfolds",
        type=int,
        default=1,
        help="Number of folds for cross-validation (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffle in K-fold (default: None)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save/load checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint at --model-path if it exists",
    )
    parser.add_argument(
        "--context-windows",
        type=int,
        default=1,
        help="Number of consecutive 1-second windows to stack as context (default: 1).",
    )
    args = vars(parser.parse_args())

    log_level = (
        logging.DEBUG
        if args["verbose"] >= 2
        else logging.INFO
        if args["verbose"] == 1
        else logging.WARNING
    )

    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    logger.info(f"CNN module launched with args : {args}")

    if args["command"] == "train":
        kfolds = args["kfolds"]
        seed = args.get("seed", None)
        context_windows = int(args.get("context_windows", 1))
        model_path_arg = args.get("model_path")
        base_model_path = Path(model_path_arg) if model_path_arg else None
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if kfolds > 1:
            dataset = EEGDataset(
                data_dir=DEFAULT_DATA_DIR,
                normalize=True,
                context_windows=context_windows,
            )
            for fold, train_ds, val_ds in dataset.k_fold(
                n_splits=kfolds, shuffle=True, random_state=seed
            ):
                logger.info(f"Starting fold {fold + 1}/{kfolds}")
                if base_model_path is not None:
                    fold_model_path = base_model_path.with_name(
                        f"{base_model_path.stem}_fold_{fold:02d}{base_model_path.suffix}"
                    )
                else:
                    fold_model_path = DEFAULT_MODEL_DIR / f"cnn_fold_{fold:02d}.pt"
                train(
                    args["epochs"],
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    model_path=fold_model_path,
                    resume=args["resume"],
                    seed=seed,
                )
        else:
            target_path = base_model_path or (DEFAULT_MODEL_DIR / "cnn_final.pt")
            train(
                args["epochs"],
                model_path=target_path,
                resume=args["resume"],
                seed=seed,
                context_windows=context_windows,
            )
    else:
        logger.error(f"Unrecognized command {args['command']}")


if __name__ == "__main__":
    main()
