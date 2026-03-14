"""
PyTorch LSTM for predicting rest_length at each epoch from the full trial history.
at each timestep t, predicts rest_length[t] given features [1..t].
Only calcualtes loss at 10th trial (at end of epoch guess)

Basic LSTM code
"""

import torch
import torch.nn as nn

class RestLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len, n_features)
        lengths: (batch,) actual sequence lengths (for packing)
        Returns: (batch, seq_len, 1)
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        return self.fc(out)