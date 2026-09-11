"""双向 LSTM 与时间注意力模块。"""

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """学习每个历史循环对当前 SOH 预测的重要程度。"""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1, bias=False),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """输入 ``(B, L, D)``，返回上下文 ``(B, D)`` 和权重 ``(B, L)``。"""
        logits = self.score(sequence).squeeze(-1)  # (B, L, D) -> (B, L)
        weights = torch.softmax(logits, dim=1)  # 每个样本的时间权重之和为 1
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)  # (B, D)
        return context, weights


class BiLSTMAttention(nn.Module):
    """用 BiLSTM 学习长期退化趋势，再用注意力汇总历史循环。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """输入 ``(B, L, input_dim)``，输出上下文和注意力权重。"""
        sequence, _ = self.bilstm(x)  # (B, L, input_dim) -> (B, L, 2*hidden_dim)
        return self.attention(sequence)
