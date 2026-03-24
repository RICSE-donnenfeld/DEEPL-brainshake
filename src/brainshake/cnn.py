from __future__ import annotations

from pathlib import Path
import os

import argparse
import logging
from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import EEGDataset

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _make_loader(
    dataset: Union[EEGDataset, Subset], shuffle: bool, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
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
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += x_batch.size(0)

    model.train()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


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
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "Epilepsy"

    logger.info(f"Using data directory: {data_dir}")

    if train_dataset is None:
        dataset = EEGDataset(data_dir=data_dir, patient_ids=[1, 2, 3], normalize=False)
    else:
        dataset = train_dataset
    cpu_count = os.cpu_count() or 1
    num_workers = min(8, cpu_count)
    loader = _make_loader(dataset, shuffle=True, num_workers=num_workers)
    val_loader = (
        _make_loader(val_dataset, shuffle=False, num_workers=num_workers)
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
        logger.info(
            f"Resuming training from {model_path} (epoch {start_epoch})"
        )

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
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            logger.info(
                f"Epoch {current_epoch} validation: Loss {val_loss:.4f}, Acc {val_acc:.4f}"
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
        model_path_arg = args.get("model_path")
        base_model_path = Path(model_path_arg) if model_path_arg else None
        if kfolds > 1:
            for fold, train_ds, val_ds in EEGDataset(
                data_dir=Path(__file__).resolve().parents[2] / "data" / "Epilepsy"
            ).k_fold(n_splits=kfolds, shuffle=True, random_state=seed):
                logger.info(f"Starting fold {fold + 1}/{kfolds}")
                fold_model_path = None
                if base_model_path is not None:
                    fold_model_path = base_model_path.with_name(
                        f"{base_model_path.stem}_fold{fold + 1}{base_model_path.suffix}"
                    )
                train(
                    args["epochs"],
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    model_path=fold_model_path,
                    resume=args["resume"],
                )
        else:
            train(
                args["epochs"],
                model_path=base_model_path,
                resume=args["resume"],
            )
    else:
        logger.error(f"Unrecognized command {args['command']}")


if __name__ == "__main__":
    main()
