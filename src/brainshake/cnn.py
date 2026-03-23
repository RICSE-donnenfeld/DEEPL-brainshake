from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import EEGDataset


class SimpleEEGCNN(nn.Module):
    def __init__(self, in_channels: int = 21, n_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
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


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "Epilepsy"

    print("Using data directory:", data_dir)

    dataset = EEGDataset(data_dir=data_dir, patient_ids=[1, 2, 3], normalize=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleEEGCNN()

    # simple class weighting because seizure class is smaller
    n0 = (dataset.labels == 0).sum()
    n1 = (dataset.labels == 1).sum()
    class_weights = torch.tensor(
        [len(dataset.labels) / (2 * n0), len(dataset.labels) / (2 * n1)],
        dtype=torch.float32
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(2):
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
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Acc: {acc:.4f}"
                )

        print(f"Epoch {epoch+1} finished. Avg loss: {running_loss / len(loader):.4f}")


if __name__ == "__main__":
    main()