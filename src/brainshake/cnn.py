from pathlib import Path

import torch
import argparse
import torch.nn as nn
import logging
from torch.utils.data import DataLoader

from .data import EEGDataset

logger = logging.getLogger(__name__)


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


def train(epochs, train_dataset=None):
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "Epilepsy"

    logger.info(f"Using data directory: {data_dir}")

    if train_dataset is None:
        dataset = EEGDataset(data_dir=data_dir, patient_ids=[1, 2, 3], normalize=False)
    else:
        dataset = train_dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleEEGCNN()

    # simple class weighting because seizure class is smaller
    n0 = (dataset.labels == 0).sum()
    n1 = (dataset.labels == 1).sum()
    class_weights = torch.tensor(
        [len(dataset.labels) / (2 * n0), len(dataset.labels) / (2 * n1)],
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad()

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

        logger.info(
            f"Epoch {epoch + 1} finished. Avg loss: {running_loss / len(loader):.4f}"
        )


def main():
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
        if kfolds > 1:
            for fold, train_ds, val_ds in EEGDataset(
                data_dir=Path(__file__).resolve().parents[2] / "data" / "Epilepsy"
            ).k_fold(n_splits=kfolds, shuffle=True, random_state=seed):
                logger.info(f"Starting fold {fold + 1}/{kfolds}")
                train(args["epochs"], train_dataset=train_ds)
        else:
            train(args["epochs"])
    else:
        logger.error(f"Unrecognized command {args['command']}")


if __name__ == "__main__":
    main()
