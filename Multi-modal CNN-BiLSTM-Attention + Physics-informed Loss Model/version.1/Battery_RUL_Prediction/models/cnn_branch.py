"""一维卷积分支：提取单个传感器在历史循环窗口内的局部退化模式。"""

import torch
from torch import nn


class CNNBranch(nn.Module):
    """把形状为 ``(批量, 循环数, 1)`` 的单模态序列编码为逐循环特征。"""

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        # Conv1d 使用 (批量, 通道, 序列长度)，padding 保持循环数不变。
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 ``(B, L, 1)``，输出 ``(B, L, hidden_dim)``。"""
        x = x.transpose(1, 2)  # (B, L, 1) -> (B, 1, L)
        features = self.encoder(x)  # (B, hidden_dim, L)
        return features.transpose(1, 2)  # (B, hidden_dim, L) -> (B, L, hidden_dim)
