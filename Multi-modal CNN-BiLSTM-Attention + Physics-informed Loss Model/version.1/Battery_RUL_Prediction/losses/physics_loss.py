"""面向电池退化规律的物理约束损失。"""

import torch
from torch import nn


class PhysicsInformedLoss(nn.Module):
    """组合数据误差、SOH 单调性、衰减连续性和趋势约束。"""

    def __init__(self, physics_weight: float = 0.1, continuity_weight: float = 0.5) -> None:
        super().__init__()
        self.physics_weight = physics_weight
        self.continuity_weight = continuity_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        cycle: torch.Tensor,
        battery_id: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """同一批次中按电池和循环排序，计算有限差分物理约束。"""
        data_loss = self.mse(prediction, target)
        monotonic_terms: list[torch.Tensor] = []
        continuity_terms: list[torch.Tensor] = []
        trend_terms: list[torch.Tensor] = []
        # 按电池分别计算，避免把一块电池末尾与另一块电池开头错误连接。
        for identity in torch.unique(battery_id):
            mask = battery_id == identity
            if int(mask.sum()) < 2:
                continue
            local_cycle = cycle[mask]
            local_prediction = prediction[mask].flatten()
            local_target = target[mask].flatten()
            order = torch.argsort(local_cycle)
            local_cycle = local_cycle[order]
            local_prediction = local_prediction[order]
            local_target = local_target[order]
            delta_cycle = torch.diff(local_cycle).clamp_min(1.0)
            predicted_slope = torch.diff(local_prediction) / delta_cycle
            target_slope = torch.diff(local_target) / delta_cycle
            # 正斜率代表 SOH 随循环上升，违反整体退化规律。
            monotonic_terms.append(torch.relu(predicted_slope).pow(2).mean())
            # 约束相邻预测差分接近真实差分，使容量衰减连续且不过度跳变。
            continuity_terms.append(self.mse(predicted_slope, target_slope))
            # 当真实数据局部存在测量回弹时，仍约束预测的平均长期斜率不为正。
            trend_terms.append(torch.relu(predicted_slope.mean()).pow(2))
        zero = prediction.sum() * 0.0  # 保持设备、数据类型和梯度图一致
        monotonic = torch.stack(monotonic_terms).mean() if monotonic_terms else zero
        continuity = torch.stack(continuity_terms).mean() if continuity_terms else zero
        trend = torch.stack(trend_terms).mean() if trend_terms else zero
        physics = monotonic + self.continuity_weight * continuity + trend
        total = data_loss + self.physics_weight * physics
        return total, {
            "data": data_loss.detach(),
            "physics": physics.detach(),
            "monotonic": monotonic.detach(),
            "continuity": continuity.detach(),
            "trend": trend.detach(),
        }
