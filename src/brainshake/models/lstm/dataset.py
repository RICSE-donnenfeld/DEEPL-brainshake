"""
SeizureEpisodeDataset: groups individual EEG windows into temporal sequences
(episodes) so an LSTM can exploit the sequential structure of seizure events.

Non-seizure windows are chunked into fixed-length segments.
Seizure windows that are contiguous within the same patient form a single episode.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ...data_handling.load_data import EEGDataset


class SeizureEpisodeDataset(Dataset):
    """
    Each item is a (features, labels) tuple where:
      features — [seq_len, n_features]  (pooled across time per window)
      labels   — [seq_len]               (0 or 1 per window)

    Episodes are variable-length; use ``pad_collate`` to batch them.

    Pooling options (per-window across the 128 timepoints):
      "std"      → standard deviation per channel  [seq_len, 21]  (default, preserves amplitude)
      "mean"     → mean per channel                 [seq_len, 21]  (risky — EEG oscillates around 0)
      "mean_std" → concatenation of mean and std     [seq_len, 42]
      "none"     → flatten the full window          [seq_len, 2688]
      "conv_proj" → keep raw window [seq_len, 21, 128] for Conv1d projection inside the model
    """

    def __init__(
        self,
        base: EEGDataset,
        indices: List[int] | np.ndarray,
        non_seizure_len: int = 10,
        pool: str = "std",
    ) -> None:
        self.base = base
        self.pool = pool
        self.non_seizure_len = non_seizure_len

        idx_arr = np.asarray(indices, dtype=np.int64)
        order = np.argsort(idx_arr)
        idx_arr = idx_arr[order]

        raw_episodes = self._group_episodes(idx_arr)
        self.episodes = self._chunk_non_seizure(raw_episodes)

    # ------------------------------------------------------------------
    # grouping
    # ------------------------------------------------------------------

    def _group_episodes(self, indices: np.ndarray) -> List[np.ndarray]:
        """Group contiguous, same-patient, same-label windows into episodes."""
        if len(indices) == 0:
            return []

        episodes: List[np.ndarray] = []
        start = 0

        for i in range(1, len(indices)):
            same_patient = (
                self.base.patient_index[indices[i]]
                == self.base.patient_index[indices[i - 1]]
            )
            same_label = (
                self.base.labels[indices[i]] == self.base.labels[indices[i - 1]]
            )
            contiguous = indices[i] == indices[i - 1] + 1

            if not (same_patient and same_label and contiguous):
                episodes.append(indices[start:i])
                start = i

        episodes.append(indices[start:])
        return episodes

    def _chunk_non_seizure(
        self, episodes: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Split long non-seizure episodes into shorter segments."""
        result: List[np.ndarray] = []
        for ep in episodes:
            if len(ep) == 0:
                continue
            is_sz = self.base.labels[ep[0]] == 1
            if is_sz:
                result.append(ep)
            else:
                chunk = self.non_seizure_len
                for start in range(0, len(ep), chunk):
                    seg = ep[start : start + chunk]
                    if len(seg) >= 2:
                        result.append(seg)
        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ep = self.episodes[idx]
        windows = self.base.data[ep]  # [seq_len, 21, 128]
        labels = self.base.labels[ep]  # [seq_len]

        if self.pool == "conv_proj":
            features = windows  # [seq_len, 21, 128] — kept raw for Conv1d projection
        elif self.pool == "mean":
            features = windows.mean(axis=2)  # [seq_len, 21]
        elif self.pool == "std":
            features = windows.std(axis=2)  # [seq_len, 21]
        elif self.pool == "mean_std":
            mean_pool = windows.mean(axis=2)   # [seq_len, 21]
            std_pool = windows.std(axis=2)      # [seq_len, 21]
            features = np.concatenate([mean_pool, std_pool], axis=-1)  # [seq_len, 42]
        elif self.pool == "none":
            features = windows.reshape(windows.shape[0], -1)  # [seq_len, 21*128]
        else:
            features = windows.mean(axis=2)

        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(labels.astype(np.int64)),
        )


def pad_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-length episodes into a padded batch.
    Handles both 2-D features [seq_len, n_features] and
    3-D features [seq_len, 21, 128] (conv_proj mode).

    Returns:
        padded_feats:  [batch, max_len, n_features] or [batch, max_len, 21, 128]
        padded_labels:  [batch, max_len]   (padding value = -1)
        lengths:        [batch]
    """
    feats = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)

    if feats[0].dim() == 3:
        # conv_proj mode: [seq_len, 21, 128] — pad along seq_len dimension
        max_len = max(f.size(0) for f in feats)
        c, t = feats[0].shape[1], feats[0].shape[2]
        padded_feats = feats[0].new_zeros(len(feats), max_len, c, t)
        for i, f in enumerate(feats):
            padded_feats[i, :f.size(0)] = f
    else:
        padded_feats = nn.utils.rnn.pad_sequence(feats, batch_first=True)

    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

    return padded_feats, padded_labels, lengths


# needed by pad_collate
from torch import nn  # noqa: E402