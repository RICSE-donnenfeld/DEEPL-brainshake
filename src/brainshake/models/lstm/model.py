import torch
import torch.nn as nn

from lstmmodels.ModelWeightsInit import init_weights_xavier_normal


class SeizureLSTM(nn.Module):
    """
    LSTM for classifying EEG windows within seizure episodes.
    """
    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()
        print("Running class: ", self.__class__.__name__)

        self.inputmodule_params = inputmodule_params
        self.net_params = net_params
        self.outmodule_params = outmodule_params

        n_nodes = inputmodule_params["n_nodes"]
        Lstacks = net_params["Lstacks"]
        dropout = net_params["dropout"]
        hidden_size = net_params["hidden_size"]

        n_classes = outmodule_params["n_classes"]
        hd = outmodule_params["hd"]

        self.lstm = nn.LSTM(
            input_size=n_nodes,
            hidden_size=hidden_size,
            num_layers=Lstacks,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hd),
            nn.ReLU(),
            nn.Linear(hd, n_classes),
        )

    def init_weights(self):
        init_weights_xavier_normal(self)

    def forward(self, x):
        """
        Args:
            x: Tensor [batch, n_nodes, seq_len]
        Returns:
            logits: Tensor [batch, n_classes]
        """
        x = x.permute(0, 2, 1)  # [batch, seq_len, n_nodes]
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]  # final time-step output
        logits = self.fc(out)
        return logits
