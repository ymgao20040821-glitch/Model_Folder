"""加载训练好的检查点，评估 SOH/RUL 并生成结果图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from models import MultiModalBatteryModel
from utils.preprocess import BatteryWindowDataset, StandardScaler, load_batteries, split_batteries
from utils.visualization import plot_attention, plot_rul, plot_soh


def parse_args() -> argparse.Namespace:
    """读取测试路径、批量大小和运行设备。"""
    parser = argparse.ArgumentParser(description="测试电池 SOH/RUL 模型")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """在留出的测试电池上推理并保存数值结果和四类图形。"""
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint_path = args.output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 {checkpoint_path}，请先运行 train.py")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    batteries = load_batteries(args.data_dir, synthetic=args.synthetic, seed=args.seed)
    _, _, test_batteries = split_batteries(batteries)
    scaler = StandardScaler.load(args.output_dir / "scaler.json")
    dataset = BatteryWindowDataset(test_batteries, int(checkpoint["window"]), scaler)
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    model = MultiModalBatteryModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    cycles: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    for batch in loader:
        prediction, attention = model(batch["x"].to(device))
        truths.append(batch["y"].numpy().reshape(-1))
        predictions.append(prediction.cpu().numpy().reshape(-1))
        cycles.append(batch["cycle"].numpy().reshape(-1))
        attentions.append(attention.cpu().numpy())
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    cycle = np.concatenate(cycles)
    attention = np.concatenate(attentions)
    metrics = {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
    }
    np.savez(args.output_dir / "predictions.npz", cycle=cycle, truth=truth, prediction=prediction, attention=attention)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_soh(cycle, truth, prediction, args.output_dir / "soh_prediction.png")
    plot_rul(cycle, truth, prediction, args.output_dir / "rul_prediction.png")
    plot_attention(attention, args.output_dir / "attention_heatmap.png")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"预测数组与图形已保存到：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
