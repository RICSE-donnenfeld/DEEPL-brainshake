from __future__ import annotations

import torch
import torch.nn as nn


class SeizureLSTM(nn.Module):
    """
    LSTM that processes sequences of EEG windows (episodes).

    Input:  [batch, seq_len, input_size]  — one feature vector per window
    Output: [batch, seq_len, n_classes]   — per-timestep classification logits

    Typical input_size = 21 (one scalar per EEG channel after pooling time).
    """

    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        [batch, seq_len, input_size]
            lengths:  [batch]  optional, for packed-sequence processing
        Returns:
            logits:   [batch, seq_len, n_classes]
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        return self.classifier(out)


def episode_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """
    Compute loss masking out padding positions.

    Args:
        logits:   [batch, seq_len, n_classes]
        targets:  [batch, seq_len]
        lengths:  [batch]
        criterion: nn.CrossEntropyLoss (or similar)
    """
    batch_size, max_len, n_classes = logits.shape
    total = torch.tensor(0.0, device=logits.device)
    n = 0
    for i in range(batch_size):
        ell = lengths[i].item()
        if ell == 0:
            continue
        total = total + criterion(logits[i, :ell], targets[i, :ell])
        n += ell
    if n == 0:
        return total
    return total / n