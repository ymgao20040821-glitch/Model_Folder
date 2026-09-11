"""多模态 CNN-BiLSTM-Attention SOH 预测模型。"""

import torch
from torch import nn

from models.bilstm_attention import BiLSTMAttention
from models.cnn_branch import CNNBranch


class MultiModalBatteryModel(nn.Module):
    """分别编码电压、电流、温度和容量，再融合预测 SOH。"""

    def __init__(self, cnn_dim: int = 32, capacity_dim: int = 24, lstm_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        # 三个 CNN 分支避免不同量纲的传感器信号在浅层互相干扰。
        self.voltage_branch = CNNBranch(cnn_dim, dropout)
        self.current_branch = CNNBranch(cnn_dim, dropout)
        self.temperature_branch = CNNBranch(cnn_dim, dropout)
        # 容量本身是核心退化状态，使用单向 LSTM 保留其因果时间演变。
        self.capacity_branch = nn.LSTM(1, capacity_dim, batch_first=True)
        fusion_dim = cnn_dim * 3 + capacity_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, lstm_dim),
            nn.LayerNorm(lstm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_model = BiLSTMAttention(lstm_dim, lstm_dim, dropout=dropout)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, 1),
            nn.Sigmoid(),  # SOH 归一化到 [0, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """输入特征顺序为 V、I、T、C，形状 ``(B, L, 4)``。"""
        voltage = self.voltage_branch(x[:, :, 0:1])  # (B, L, cnn_dim)
        current = self.current_branch(x[:, :, 1:2])  # (B, L, cnn_dim)
        temperature = self.temperature_branch(x[:, :, 2:3])  # (B, L, cnn_dim)
        capacity, _ = self.capacity_branch(x[:, :, 3:4])  # (B, L, capacity_dim)
        fused = torch.cat([voltage, current, temperature, capacity], dim=-1)
        fused = self.fusion(fused)  # (B, L, fusion_dim) -> (B, L, lstm_dim)
        context, attention = self.temporal_model(fused)  # (B, 2*lstm_dim), (B, L)
        soh = self.regressor(context)  # (B, 1)
        return soh, attention
