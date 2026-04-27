from __future__ import annotations

import torch
import torch.nn as nn


class SeizureLSTM(nn.Module):
    """
    LSTM that processes sequences of EEG windows (episodes).

    Input:  [batch, seq_len, input_size]  — one feature vector per window
    Output: [batch, seq_len, n_classes]   — per-timestep classification logits

    When ``conv_proj`` is set, a 1-D convolution projects the raw window
    [21, 128] down to [input_size] *before* the LSTM, so even
    ``pool="none"`` (which feeds 21*128 features) stays fast.
    """

    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
        conv_proj: bool = False,
        conv_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv_proj = conv_proj
        self.input_size = input_size

        if conv_proj:
            self.proj: nn.Module | None = nn.Sequential(
                nn.Conv1d(21, conv_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            lstm_input = conv_channels
        else:
            self.proj = None
            lstm_input = input_size

        self.lstm = nn.LSTM(
            input_size=lstm_input,
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
            x:        [batch, seq_len, input_size]  or
                      [batch, seq_len, 21, 128] when conv_proj=True
            lengths:  [batch]  optional, for packed-sequence processing
        Returns:
            logits:   [batch, seq_len, n_classes]
        """
        if self.conv_proj:
            # x: [batch, seq_len, 21, 128] -> [batch, seq_len, conv_channels]
            b, s, c, t = x.shape
            x = x.reshape(b * s, c, t)           # [b*s, 21, 128]
            x = self.proj(x)                       # [b*s, conv_channels, 1]
            x = x.squeeze(-1).reshape(b, s, -1)   # [b, s, conv_channels]

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