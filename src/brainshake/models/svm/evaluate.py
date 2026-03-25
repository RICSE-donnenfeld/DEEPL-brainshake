import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, cast, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from .model import SVMSeizureClassifier
from ...data_handling.load_data import EEGDataset
from ...data_handling.extract_features import extract_basic_features, FeatureDict

# Path configuration
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "Epilepsy"

def extract_features_from_subset(subset: Subset) -> Tuple[List[FeatureDict], List[int]]:
    """Extract statistical features from a PyTorch data subset."""
    features, labels = [], []
    for i in range(len(subset)):
        x, y = cast(Tuple[torch.Tensor, torch.Tensor], subset[i])
        features.append(extract_basic_features(x.numpy()))
        labels.append(int(y.item()))
    return features, labels

def evaluate_dataset(data_dir, n_splits=5, random_state=2026, patient_ids=None):
    """Perform patient-wise k-fold evaluation for the SVM classifier."""
    dataset = EEGDataset(data_dir=data_dir, patient_ids=patient_ids, normalize=False)
    accuracies = []
    results = {"folds": [], "average_accuracy": None}

    print(f"--- Starting SVM evaluation (Folds: {n_splits}) ---")
    
    for fold, train_sub, val_sub in dataset.k_fold(n_splits=n_splits, random_state=random_state):
        # 1. Prepare data
        train_feat, train_labels = extract_features_from_subset(train_sub)
        val_feat, val_labels = extract_features_from_subset(val_sub)

        # 2. Train model
        clf = SVMSeizureClassifier()
        clf.fit(train_feat, train_labels)
        
        # 3. Evaluate accuracy
        preds = clf.predict(val_feat)
        acc = np.mean(np.array(preds) == np.array(val_labels))
        accuracies.append(float(acc))
        
        print(f"Fold {fold}: Accuracy = {acc:.4f}")
        results["folds"].append({"fold": fold, "accuracy": float(acc)})

    # 4. Save metrics
    avg = np.mean(accuracies) if accuracies else 0.0
    results["average_accuracy"] = float(avg)
    print(f"Average accuracy: {avg:.4f}")

    bench_dir = REPO_ROOT / "out" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "svm.json"
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmarks to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate SVM on EEG data")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick test with only 2 patients")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=2026)
    args = parser.parse_args()

    p_ids = [1, 2] if args.smoke_test else None
    
    evaluate_dataset(
        data_dir=DEFAULT_DATA_DIR,
        n_splits=args.n_splits,
        random_state=args.random_state,
        patient_ids=p_ids
    )

if __name__ == "__main__":
    main()