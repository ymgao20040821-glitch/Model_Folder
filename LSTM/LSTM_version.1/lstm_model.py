"""A reusable LSTM model for sequence-to-one prediction.

The structure follows the standard PyTorch LSTM pattern demonstrated in the
official ``pytorch/examples`` and ``pytorch/tutorials`` GitHub repositories:

    input sequence -> LSTM -> last time-step -> dropout -> linear output

Input shape:  (batch_size, sequence_length, input_size)
Output shape: (batch_size, output_size)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LSTMModel(nn.Module):
    """LSTM network for regression or sequence classification.

    Args:
        input_size: Number of features at each time step.
        hidden_size: Number of features in the LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        output_size: Number of predicted values or classes.
        dropout: Dropout probability between LSTM layers and before output.
        bidirectional: Whether to use a bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        if min(input_size, hidden_size, num_layers, output_size) <= 0:
            raise ValueError("input_size, hidden_size, num_layers and output_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch applies internal dropout only between stacked layers.
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass over a batch of sequences."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, sequence_length, input_size); "
                f"received {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_size:
            raise ValueError(f"expected {self.input_size} input features, received {x.size(-1)}")

        sequence_output, (hidden, _) = self.lstm(x)

        if self.num_directions == 1:
            features = sequence_output[:, -1, :]
        else:
            # At the final tensor position, the backward component has only seen
            # one item. Use h_n so both directions represent the full sequence.
            hidden = hidden.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)
            features = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=-1)

        return self.output_layer(self.dropout(features))


if __name__ == "__main__":
    # Minimal smoke test: 8 samples, 20 time steps, 3 features per step.
    model = LSTMModel(input_size=3, hidden_size=32, num_layers=1, output_size=1)
    sample_batch = torch.randn(8, 20, 3)
    prediction = model(sample_batch)
    print(model)
    print("Input shape: ", tuple(sample_batch.shape))
    print("Output shape:", tuple(prediction.shape))
