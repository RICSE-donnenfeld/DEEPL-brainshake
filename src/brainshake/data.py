from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import logging
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """
    PyTorch Dataset for CHB-MIT EEG windows.

    Expected files in data_dir:
        chb01_seizure_EEGwindow_1.npz
        chb01_seizure_metadata_1.parquet
        ...
        chb24_seizure_EEGwindow_1.npz
        chb24_seizure_metadata_1.parquet

    Each NPZ file must contain:
        EEG_win -> shape [N, 21, 128]

    Each parquet file must contain:
        class -> shape [N]
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        patient_ids: Optional[Sequence[int]] = None,
        normalize: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patient_ids = (
            list(patient_ids) if patient_ids is not None else list(range(1, 25))
        )
        self.normalize = normalize

        self.data, self.labels, self.patient_index = self._load_all_patients()

        if self.normalize:
            mean = self.data.mean()
            std = self.data.std()
            if std == 0:
                raise ValueError(
                    "Standard deviation is zero, cannot normalize dataset."
                )
            self.data = (self.data - mean) / std

    def _load_all_patients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_data = []
        all_labels = []
        all_patients = []

        for patient_id in self.patient_ids:
            pid = f"chb{patient_id:02d}"

            npz_path = self.data_dir / f"{pid}_seizure_EEGwindow_1.npz"
            meta_path = self.data_dir / f"{pid}_seizure_metadata_1.parquet"

            if not npz_path.exists():
                logger.warning(f"Missing EEG file: {npz_path}")
                continue

            if not meta_path.exists():
                logger.warning(f"Missing metadata file: {meta_path}")
                continue

            npz_data = np.load(npz_path, allow_pickle=True)
            if "EEG_win" not in npz_data:
                raise KeyError(f"'EEG_win' not found in {npz_path}")

            eeg_windows = npz_data["EEG_win"].astype(np.float32)
            metadata = pd.read_parquet(meta_path)

            if "class" not in metadata.columns:
                raise KeyError(f"'class' column not found in {meta_path}")

            labels = metadata["class"].to_numpy(dtype=np.int64)

            if len(eeg_windows) != len(labels):
                raise ValueError(
                    f"Mismatch for {pid}: {len(eeg_windows)} windows but {len(labels)} labels"
                )

            all_data.append(eeg_windows)
            all_labels.append(labels)
            all_patients.append(np.full(len(labels), patient_id, dtype=np.int64))

            logger.info(f"Loaded {pid}: {eeg_windows.shape}, labels={labels.shape}")

        if not all_data:
            raise RuntimeError(
                "No patient data could be loaded. Check data_dir and file names."
            )

        data = np.concatenate(all_data, axis=0)  # [N_total, 21, 128]
        labels = np.concatenate(all_labels, axis=0)  # [N_total]
        patient_index = np.concatenate(all_patients, axis=0)

        return data, labels, patient_index

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[index], dtype=torch.float32)  # [21, 128]
        y = torch.tensor(self.labels[index], dtype=torch.long)  # scalar
        return x, y

    def summary(self) -> None:
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))

        logger.info("Dataset summary")
        logger.info(f"  data shape: {self.data.shape}")
        logger.info(f"  labels shape: {self.labels.shape}")
        logger.info(f"  class distribution: {class_distribution}")
        logger.info(f"  patients loaded: {sorted(set(self.patient_index.tolist()))}")

    def k_fold(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.data)):
            train_set = Subset(self, train_idx)
            val_set = Subset(self, val_idx)
            yield fold, train_set, val_set


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "Epilepsy"

    logger.info(f"Using data directory: {data_dir}")

    dataset = EEGDataset(data_dir=data_dir, patient_ids=[1, 2, 3], normalize=False)
    dataset.summary()

    x, y = dataset[0]
    logger.info(f"Single sample shape: {x.shape}")
    logger.info(f"Single label: {y.item()}")

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    x_batch, y_batch = next(iter(loader))
    logger.info(f"Batch sample shape: {x_batch.shape}")
    logger.info(f"Batch label shape: {y_batch.shape}")
    logger.info(f"Batch labels: {y_batch}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
