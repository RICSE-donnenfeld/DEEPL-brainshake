"""
Training script for EEG seizure detection.

Usage:
    python -m brainshake.train --model conv2d --split-mode cross_subject
    python -m brainshake.train --model conv1d --balance-mode balanced
    python -m brainshake.train --model lstm --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from brainshake.data import EEGDataset
from brainshake.cnn import (
    get_model, EEGTrainer, create_class_weights,
    EEGConv1D, EEGConv2D, EEGShallowConvNet, EEGDeepConvNet, EEGTemporalConvNet
)


log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG seizure detection model")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/Epilepsy",
                        help="Path to data directory")
    parser.add_argument("--split-mode", type=str, default="cross_subject",
                        choices=["cross_subject", "patient"],
                        help="Data split mode")
    parser.add_argument("--balance-mode", type=str, default="unbalanced",
                        choices=["balanced", "unbalanced"],
                        help="Class balance mode")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help="Train/val/test split ratios")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="conv2d",
                        choices=["conv1d", "conv2d", "shallow", "deep", "temporal", "lstm"],
                        help="Model architecture")
    parser.add_argument("--hidden-dims", type=int, nargs=3, default=[16, 32, 64],
                        help="Hidden dimensions for CNN")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early-stopping", type=int, default=7,
                        help="Early stopping patience")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    
    return parser.parse_args()


def setup_logging(verbosity: int):
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def set_seed(seed: int):
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_model(model_name: str, n_channels: int = 21, n_samples: int = 128,
                 n_classes: int = 2, dropout: float = 0.3, **kwargs):
    """Create model based on name."""
    
    # Check if it's an LSTM (not in get_model yet)
    if model_name == "lstm":
        from brainshake.cnn import EEGLSTM
        return EEGLSTM(n_channels, n_samples, n_classes, dropout=dropout, **kwargs)
    
    # Standard CNN models
    model_configs = {
        "conv1d": {"hidden_dims": kwargs.get("hidden_dims", (32, 64, 128))},
        "conv2d": {"hidden_dims": kwargs.get("hidden_dims", (16, 32, 64))},
        "shallow": {"dropout": dropout},
        "deep": {"dropout": dropout},
        "temporal": {"hidden_dim": kwargs.get("hidden_dims", [64])[0] if kwargs.get("hidden_dims") else 64},
    }
    
    return get_model(
        model_name,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        dropout=dropout,
        **model_configs.get(model_name, {})
    )


def main():
    args = parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    log.info("Loading dataset...")
    dataset = EEGDataset(
        data_dir=args.data_dir,
        split_mode=args.split_mode,
        balance_mode=args.balance_mode,
        split_ratio=tuple(args.split_ratio),
    )
    
    # Get splits
    train_set = dataset.get_split("train")
    val_set = dataset.get_split("val")
    test_set = dataset.get_split("test")
    
    log.info(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    # Create model
    log.info(f"Creating {args.model} model...")
    model = create_model(
        args.model,
        n_channels=21,
        n_samples=128,
        n_classes=2,
        dropout=args.dropout,
        hidden_dims=tuple(args.hidden_dims)
    )
    
    # Create class weights for imbalanced data
    if args.balance_mode == "unbalanced":
        class_weights = create_class_weights(train_set)
        log.info(f"Class weights: {class_weights.tolist()}")
    else:
        class_weights = None
    
    # Create trainer
    trainer = EEGTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
    )
    
    # Train
    log.info(f"Training for {args.epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
    )
    
    # Evaluate on test set
    log.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Split mode: {args.split_mode}")
    print(f"Balance mode: {args.balance_mode}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print("="*50)
    
    # Print classification report
    trainer.print_classification_report(test_loader)
    
    # Save results
    results = {
        "args": vars(args),
        "test_metrics": {k: v for k, v in test_metrics.items() 
                        if k not in ["predictions", "probabilities", "labels"]},
        "history": history,
    }
    
    results_path = output_dir / f"{args.model}_{args.split_mode}_{args.balance_mode}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info(f"Results saved to {results_path}")
    
    # Save model
    model_path = output_dir / f"{args.model}_{args.split_mode}_{args.balance_mode}_model.pt"
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
