"""
Data Loader :
Exports a pytorch Dataset class with integrated patient-level k-folding
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import logging

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
        context_windows: int = 1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patient_ids = (
            list(patient_ids) if patient_ids is not None else list(range(1, 25))
        )
        self.normalize = normalize

        if int(context_windows) < 1:
            raise ValueError("context_windows must be >= 1")
        self.context_windows = int(context_windows)

        # Raw arrays are always 1-second windows (N_raw, 21, 128).
        self.data, raw_labels, raw_patient_index = self._load_all_patients()
        self._raw_labels = raw_labels
        self._raw_patient_index = raw_patient_index

        # Map dataset index-space to valid raw start indices.
        self._start_indices = self._build_start_indices()
        self.labels = self._raw_labels[self._start_indices + (self.context_windows - 1)]
        self.patient_index = self._raw_patient_index[self._start_indices]

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
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self._start_indices[index])

        if self.context_windows == 1:
            window = self.data[start]
            if self.normalize:
                mean = float(window.mean())
                std = float(window.std())
                window = (window - mean) / std if std > 0 else (window - mean)
        else:
            segment = self.data[start : start + self.context_windows]
            if self.normalize:
                # Keep the original behaviour (per-window z-score), but applied
                # independently for each 1-second window before stacking.
                segment = segment.copy()
                for i in range(self.context_windows):
                    w = segment[i]
                    mean = float(w.mean())
                    std = float(w.std())
                    segment[i] = (w - mean) / std if std > 0 else (w - mean)
            # Stack along time: (K, 21, 128) -> (21, K*128)
            window = np.transpose(segment, (1, 0, 2)).reshape(
                segment.shape[1], segment.shape[0] * segment.shape[2]
            )

        x = torch.from_numpy(window).to(dtype=torch.float32)
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y

    def _build_start_indices(self) -> np.ndarray:
        """Return raw start indices for each sample in dataset index-space.

        For context_windows == 1, this is a direct 0..N-1 mapping.
        For context_windows > 1, we only keep starts that do not cross
        patient boundaries.
        """

        n_raw = int(self._raw_labels.shape[0])
        if self.context_windows == 1:
            return np.arange(n_raw, dtype=np.int64)

        starts: list[int] = []
        patients = self._raw_patient_index
        k = self.context_windows

        seg_start = 0
        while seg_start < n_raw:
            patient = patients[seg_start]
            seg_end = seg_start + 1
            while seg_end < n_raw and patients[seg_end] == patient:
                seg_end += 1

            seg_len = seg_end - seg_start
            if seg_len >= k:
                starts.extend(range(seg_start, seg_end - k + 1))

            seg_start = seg_end

        if not starts:
            raise RuntimeError(
                "No valid context windows could be constructed; "
                "check patient files or reduce context_windows."
            )

        return np.asarray(starts, dtype=np.int64)

    def summary(self) -> None:
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))

        logger.info("Dataset summary")
        logger.info(f"  raw data shape: {self.data.shape}")
        logger.info(f"  sample labels shape: {self.labels.shape}")
        logger.info(f"  context windows: {self.context_windows}")
        logger.info(f"  class distribution: {class_distribution}")
        logger.info(f"  patients loaded: {sorted(set(self.patient_index.tolist()))}")

    def k_fold(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        level: str = "patient",
    ):
        """
        k-fold split at different levels: "patient", "window", or "seizure".
        - patient: split by patient, no data leakage across patients.
        - window: split by individual windows; for seizure windows in validation, discard 4 neighbors.
        - seizure: split by contiguous seizure episodes (all windows of a seizure).
        """
        if level not in ("patient", "window", "seizure"):
            raise ValueError(f"Invalid level: {level}. Choose 'patient', 'window', or 'seizure'.")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        rng = np.random.default_rng(random_state)
        all_indices = np.arange(len(self.labels))

        if level == "patient":
            unique_patients = np.unique(self.patient_index)
            if n_splits > len(unique_patients):
                raise ValueError("n_splits cannot exceed unique patient count")
            patients = rng.permutation(unique_patients) if shuffle else unique_patients.copy()
            patient_to_idxs = {p: np.flatnonzero(self.patient_index == p) for p in unique_patients}
            folds = [[] for _ in range(n_splits)]
            for i, p in enumerate(patients):
                folds[i % n_splits].append(p)
            for fold, val_patients in enumerate(folds):
                val_idx = np.concatenate([patient_to_idxs[p] for p in val_patients])
                train_idx = np.setdiff1d(all_indices, val_idx)
                train_set = Subset(self, train_idx.tolist())
                val_set = Subset(self, val_idx.tolist())

                # Attach split metadata for reproducibility (best-effort; callers may ignore).
                train_patients = [int(p) for p in unique_patients if p not in val_patients]
                val_patients_list = [int(p) for p in val_patients]
                setattr(train_set, "patient_ids", train_patients)
                setattr(val_set, "patient_ids", val_patients_list)
                setattr(train_set, "fold", fold)
                setattr(val_set, "fold", fold)

                yield fold, train_set, val_set

        elif level == "window":
            idxs = rng.permutation(all_indices) if shuffle else all_indices.copy()
            # assign windows to folds in round-robin
            fold_bins = {i: idxs[i::n_splits] for i in range(n_splits)}
            for fold in range(n_splits):
                val_idx = np.array(fold_bins[fold], dtype=int)
                # remove neighbors around seizure windows in validation
                drop = set()
                radius = 4
                for idx in val_idx[self.labels[val_idx] == 1]:
                    pid = self.patient_index[idx]
                    patient_idxs = np.flatnonzero(self.patient_index == pid)
                    pos = np.searchsorted(patient_idxs, idx)
                    for offset in range(-radius, radius + 1):
                        j = pos + offset
                        if 0 <= j < len(patient_idxs):
                            drop.add(patient_idxs[j])
                val_idx = np.setdiff1d(val_idx, np.fromiter(drop, int))
                train_idx = np.setdiff1d(all_indices, val_idx)
                train_set = Subset(self, train_idx.tolist())
                val_set = Subset(self, val_idx.tolist())
                setattr(train_set, "fold", fold)
                setattr(val_set, "fold", fold)
                yield fold, train_set, val_set

        else:  # level == "seizure"
            # find contiguous seizure episodes per patient
            episodes: list[np.ndarray] = []
            for pid in np.unique(self.patient_index):
                pid_idxs = np.flatnonzero(self.patient_index == pid)
                labels = self.labels[pid_idxs]
                is_sz = labels == 1
                padded = np.concatenate(([0], is_sz.view(np.int8), [0]))
                diffs = np.diff(padded)
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                for s, e in zip(starts, ends):
                    episodes.append(pid_idxs[s:e])
            if len(episodes) < n_splits:
                raise ValueError("Number of seizure episodes less than n_splits")
            order = rng.permutation(len(episodes)) if shuffle else np.arange(len(episodes))
            folds = [[] for _ in range(n_splits)]
            for i, ep_i in enumerate(order):
                folds[i % n_splits].append(episodes[ep_i])
            for fold, eps in enumerate(folds):
                val_idx = np.concatenate(eps)
                train_idx = np.setdiff1d(all_indices, val_idx)
                train_set = Subset(self, train_idx.tolist())
                val_set = Subset(self, val_idx.tolist())
                setattr(train_set, "fold", fold)
                setattr(val_set, "fold", fold)
                yield fold, train_set, val_set


def main():
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data" / "Epilepsy"

    logger.info(f"Using data directory: {data_dir}")

    dataset = EEGDataset(data_dir=data_dir, patient_ids=[1, 2, 3], normalize=False)
    dataset.summary()

    x, y = dataset[0]
    logger.info(f"Single sample shape: {x.shape}")
    logger.info(f"Single label: {y.item()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
