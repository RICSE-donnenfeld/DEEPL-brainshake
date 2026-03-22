"""
CNN module
---
Defines CNN architectures for EEG-based seizure detection.

Two Channel Fusion Approaches (per Challenge_2_2026):
1. Input-level fusion: Treat [21 channels, 128 samples] as 2D input (like image)
2. Feature-level fusion: Extract features per channel, then combine

Reference: Chakrabarti et al. - LSTM Channel Fusion approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)


class EEGConv1D(nn.Module):
    """
    1D Convolutional EEG Classifier.
    
    Processes EEG as temporal sequences. Good for capturing temporal
    patterns within each channel. Uses feature-level channel fusion.
    
    Architecture:
    - Per-channel 1D convolutions to extract features
    - Global pooling across channels
    - Fully connected classifier
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 128,
        n_classes: int = 2,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
        # Per-channel feature extraction
        # Each channel gets its own conv layers (shared weights)
        self.channel_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dims[0], kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Calculate feature dimension after convolutions
        # After 2 MaxPool1d(2): 128 -> 64 -> 32
        # After AdaptiveAvgPool1d(1): becomes 1
        
        # Channel fusion layer
        self.channel_fusion = nn.Sequential(
            nn.Linear(hidden_dims[2], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, channels, samples]
            
        Returns:
            Class logits [batch, n_classes]
        """
        batch_size = x.shape[0]
        
        # Process each channel with shared conv weights
        # x: [batch, channels, samples] -> [batch*channels, 1, samples]
        x = x.view(batch_size * self.n_channels, 1, self.n_samples)
        
        # Shared conv for all channels
        x = self.channel_conv(x)  # [batch*channels, hidden_dims[-1], 1]
        x = x.squeeze(-1)  # [batch*channels, hidden_dims[-1]]
        
        # Reshape back to per-channel
        x = x.view(batch_size, self.n_channels, -1)  # [batch, channels, hidden]
        
        # Global channel pooling (mean across channels)
        x = x.mean(dim=1)  # [batch, hidden]
        
        # Channel fusion
        x = self.channel_fusion(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


class EEGConv2D(nn.Module):
    """
    2D Convolutional EEG Classifier.
    
    Treats EEG as a 2D "image" [channels, time] = [21, 128].
    Uses input-level channel fusion - all channels processed together.
    
    Architecture:
    - 2D convolutions capture spatial-temporal patterns
    - Multiple pooling layers reduce dimensionality
    - Fully connected classifier
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 128,
        n_classes: int = 2,
        hidden_dims: Tuple[int, ...] = (16, 32, 64),
        kernel_size: Tuple[int, int] = (3, 5),
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
        # Input-level channel fusion with 2D convolutions
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, hidden_dims[0], kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # Pool only in time dimension
            nn.Dropout2d(dropout),
            
            # Conv Block 2
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(dropout),
            
            # Conv Block 3
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
        )
        
        # Calculate: 128 samples -> after 2 pools -> 32 -> 16 -> AdaptivePool -> 1
        # So feature dim = hidden_dims[-1] * 1 * 1 = hidden_dims[-1]
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, channels, samples]
            
        Returns:
            Class logits [batch, n_classes]
        """
        # Add channel dimension for 2D conv: [batch, 1, channels, samples]
        x = x.unsqueeze(1)
        
        # Convolutions
        x = self.conv_layers(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


class EEGShallowConvNet(nn.Module):
    """
    Shallow ConvNet for EEG classification.
    
    Based on Schirrmeister et al. (2017). Simple architecture
    good for EEG with limited data.
    
    Uses temporal convolutions only (no spatial).
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 128,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # Temporal convolution
        self.conv_time = nn.Conv2d(1, 40, (1, 25))
        # Spatial convolution
        self.conv_spatial = nn.Conv2d(40, 40, (n_channels, 1))
        
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.batch_norm = nn.BatchNorm2d(40)
        
        # Use adaptive pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 4))
        
        # Feature size: 40 * 4 = 160
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40 * 4, n_classes),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [batch, 1, channels, samples]
        
        x = self.conv_time(x)
        x = self.conv_spatial(x)
        x = self.pool(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


class EEGDeepConvNet(nn.Module):
    """
    Deep ConvNet for EEG classification.
    
    Based on Schirrmeister et al. (2017). More complex
    with 3 conv blocks.
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 128,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (n_channels, 1)),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(dropout),
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(dropout),
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(dropout),
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(dropout),
        )
        
        # Use adaptive pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


class EEGTemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for EEG.
    
    Uses only temporal (1D) convolutions - treats each channel
    as independent initially, then fuses at the end.
    Good baseline for comparison.
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 128,
        n_classes: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
        # Temporal feature extraction per channel
        layers = []
        in_dim = 1
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        
        self.temporal_conv = nn.Sequential(*layers)
        
        # After 4 MaxPool1d(2): 128 -> 64 -> 32 -> 16 -> 8
        
        # Channel fusion + classifier
        self.fusion = nn.Sequential(
            nn.Linear(n_channels * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Process each channel independently
        x = x.view(batch_size * self.n_channels, 1, self.n_samples)
        x = self.temporal_conv(x)
        x = x.mean(dim=-1)  # Global average pooling
        
        # Reshape and fuse channels
        x = x.view(batch_size, -1)  # [batch, n_channels * hidden]
        
        return self.fusion(x)


class EEGLSTM(nn.Module):
    """
    LSTM-based EEG Classifier for temporal approach.
    
    Processes EEG sequences using LSTM to capture temporal dependencies.
    Based on Chakrabarti et al. (2020) LSTM Channel Fusion approach.
    
    Architecture:
    - Per-channel LSTM processing (feature-level fusion)
    - Bidirectional LSTM for capturing past and future context
    - Fully connected classifier
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 128,
        n_classes: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Per-channel feature extraction (1D convolutions before LSTM)
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # After 2 MaxPool1d(2): 128 -> 64 -> 32
        lstm_input_dim = 64
        
        # LSTM layer - processes temporal sequences
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, channels, samples]
            
        Returns:
            Class logits [batch, n_classes]
        """
        batch_size = x.shape[0]
        
        # Process each channel independently through encoder
        # [batch, channels, samples] -> [batch*channels, 1, samples]
        x = x.view(batch_size * self.n_channels, 1, self.n_samples)
        
        # Per-channel encoding
        x = self.channel_encoder(x)  # [batch*channels, 64, time]
        
        # Transpose for LSTM: [batch*channels, 64, time] -> [batch*channels, time, 64]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state from all layers
        # If bidirectional: concatenate forward and backward of last layer
        if self.bidirectional:
            # hidden: [num_layers * 2, batch*channels, hidden_dim]
            # Get last layer's forward and backward
            hidden_forward = hidden[-2]  # [batch*channels, hidden_dim]
            hidden_backward = hidden[-1]  # [batch*channels, hidden_dim]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_concat = hidden[-1]
        
        # Reshape: [batch*channels, hidden*2] -> [batch, channels*hidden*2]
        x = hidden_concat.view(batch_size, -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class EEGLSTMAttention(nn.Module):
    """
    LSTM with Attention for EEG classification.
    
    Adds attention mechanism to focus on important time steps.
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 128,
        n_classes: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
        # Per-channel encoder
        self.channel_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Per-channel encoding
        x = x.view(batch_size * self.n_channels, 1, self.n_samples)
        x = self.channel_encoder(x)
        x = x.transpose(1, 2)  # [batch*channels, time, features]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch*channels, time, hidden*2]
        
        # Attention
        attn_weights = self.attention(lstm_out)  # [batch*channels, time, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch*channels, hidden*2]
        
        # Reshape and classify
        x = context.view(batch_size, -1)
        return self.classifier(x)


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Factory function to get EEG model by name.
    
    Args:
        name: Model name ('conv1d', 'conv2d', 'shallow', 'deep', 'temporal')
        **kwargs: Model hyperparameters
        
    Returns:
        EEG model instance
    """
    models = {
        'conv1d': EEGConv1D,
        'conv2d': EEGConv2D,
        'shallow': EEGShallowConvNet,
        'deep': EEGDeepConvNet,
        'temporal': EEGTemporalConvNet,
    }
    
    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name.lower()](**kwargs)


class EEGTrainer:
    """
    Training and evaluation utilities for EEG models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss function with optional class weighting
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 30,
        early_stopping_patience: int = 7,
    ) -> Dict[str, list]:
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Log history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def get_confusion_matrix(self, data_loader: DataLoader) -> np.ndarray:
        """Get confusion matrix for a dataset."""
        metrics = self.evaluate(data_loader)
        return confusion_matrix(metrics['labels'], metrics['predictions'])
    
    def print_classification_report(self, data_loader: DataLoader):
        """Print sklearn classification report."""
        metrics = self.evaluate(data_loader)
        print("\nClassification Report:")
        print(classification_report(metrics['labels'], metrics['predictions'], 
                                    target_names=['Non-Seizure', 'Seizure']))


def create_class_weights(dataset) -> torch.Tensor:
    """
    Create class weights for imbalanced datasets.
    
    Args:
        dataset: PyTorch Dataset with labels
        
    Returns:
        Tensor of class weights [n_classes]
    """
    from collections import Counter
    
    # Count samples per class
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(counts)
    
    # Inverse frequency weighting
    weights = [n_samples / (n_classes * counts[i]) for i in range(n_classes)]
    
    return torch.tensor(weights, dtype=torch.float32)


def test_models():
    """Test all models with dummy input."""
    batch_size = 8
    n_channels = 21
    n_samples = 128
    x = torch.randn(batch_size, n_channels, n_samples)
    
    models = {
        'conv1d': EEGConv1D(n_channels, n_samples),
        'conv2d': EEGConv2D(n_channels, n_samples),
        'shallow': EEGShallowConvNet(n_channels, n_samples),
        'deep': EEGDeepConvNet(n_channels, n_samples),
        'temporal': EEGTemporalConvNet(n_channels, n_samples),
    }
    
    print("Testing CNN models...")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"  {name}: input {x.shape} -> output {out.shape}")
    
    print("\nAll model tests passed!")


if __name__ == "__main__":
    test_models()
