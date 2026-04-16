import sys, os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import numpy as np
import pytest  # type: ignore
from brainshake.data_handling.load_data import EEGDataset


@pytest.fixture
def synthetic_two_patients(monkeypatch):
    # Create synthetic dataset: 2 patients, 6 windows and 4 windows
    data = np.zeros((10, 1, 1), dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.int64)
    patient_index = np.array([1] * 6 + [2] * 4, dtype=np.int64)
    monkeypatch.setattr(
        EEGDataset, "_load_all_patients", lambda self: (data, labels, patient_index)
    )
    return EEGDataset(data_dir=".")


@pytest.fixture
def synthetic_one_patient(monkeypatch):
    # Single patient episodes: contiguous seizures at indices [2,3], [6], [8,9]
    data = np.zeros((12, 1, 1), dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    patient_index = np.ones(12, dtype=np.int64)
    monkeypatch.setattr(
        EEGDataset, "_load_all_patients", lambda self: (data, labels, patient_index)
    )
    return EEGDataset(data_dir=".")


def test_k_fold_patient_level(synthetic_two_patients):
    ds = synthetic_two_patients
    # 2 patients -> 2 splits, val sets of sizes 6 and 4
    folds = list(ds.k_fold(n_splits=2, shuffle=False, random_state=0, level="patient"))
    assert len(folds) == 2
    _, train0, val0 = folds[0]
    _, train1, val1 = folds[1]
    assert len(val0) == 6
    assert len(val1) == 4
    # Ensure train+val cover all indices
    all_idx = set(range(len(ds.labels)))
    assert set(train0.indices) | set(val0.indices) == all_idx
    assert set(train1.indices) | set(val1.indices) == all_idx


def test_k_fold_window_level_no_neighbor_leakage(synthetic_two_patients):
    ds = synthetic_two_patients
    folds = list(ds.k_fold(n_splits=2, shuffle=False, random_state=0, level="window"))
    # Check union of indices covers full dataset and no overlap
    for _, train, val in folds:
        assert set(train.indices) & set(val.indices) == set()
        assert set(train.indices) | set(val.indices) == set(range(len(ds.labels)))


def test_k_fold_seizure_level_episodes(synthetic_one_patient):
    ds = synthetic_one_patient
    folds = list(ds.k_fold(n_splits=3, shuffle=False, random_state=0, level="seizure"))
    # Expect 3 episodes: [2,3], [6], [8,9]
    expected = [{2, 3}, {6}, {8, 9}]
    for (fold, train, val), exp in zip(folds, expected):
        assert set(val.indices) == exp
        assert set(train.indices) | set(val.indices) == set(range(len(ds.labels)))
