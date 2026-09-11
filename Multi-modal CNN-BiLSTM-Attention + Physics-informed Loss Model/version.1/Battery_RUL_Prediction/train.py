"""训练 Multi-modal CNN-BiLSTM-Attention + Physics-informed Loss 模型。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from losses import PhysicsInformedLoss
from models import MultiModalBatteryModel
from utils.preprocess import BatteryWindowDataset, fit_scaler, load_batteries, split_batteries
from utils.visualization import plot_loss


def parse_args() -> argparse.Namespace:
    """集中定义命令行参数，便于复现实验。"""
    parser = argparse.ArgumentParser(description="训练多模态物理约束电池 SOH 模型")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="使用内置合成数据完成流程验证")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机状态，降低重复实验差异。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    """自动优先使用 CUDA，也允许用户显式选择 CPU。"""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 未检测到可用 GPU")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def run_epoch(
    model: MultiModalBatteryModel,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, dict[str, float]]:
    """执行一轮训练或验证；传入 optimizer 时启用反向传播。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    component_sums = {name: 0.0 for name in ("data", "physics", "monotonic", "continuity", "trend")}
    sample_count = 0
    for batch in loader:
        x = batch["x"].to(device)  # (B, window, 4)
        y = batch["y"].to(device)  # (B, 1)
        cycle = batch["cycle"].to(device)  # (B,)
        battery_id = batch["battery_id"].to(device)  # (B,)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction, _ = model(x)
            loss, components = criterion(prediction, y, cycle, battery_id)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = x.size(0)
        total_loss += float(loss.detach()) * batch_size
        for name, value in components.items():
            component_sums[name] += float(value) * batch_size
        sample_count += batch_size
    return total_loss / sample_count, {name: value / sample_count for name, value in component_sums.items()}


@torch.no_grad()
def evaluate_metrics(model: MultiModalBatteryModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """计算独立于训练损失的 MAE、RMSE 和 R²。"""
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in loader:
        prediction, _ = model(batch["x"].to(device))
        truths.append(batch["y"].numpy().reshape(-1))
        predictions.append(prediction.cpu().numpy().reshape(-1))
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
    }


def main() -> None:
    """完成数据准备、训练、早停、最优模型保存和最终测试。"""
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batteries = load_batteries(args.data_dir, synthetic=args.synthetic, seed=args.seed)
    train_batteries, validation_batteries, test_batteries = split_batteries(batteries)
    scaler = fit_scaler(train_batteries)
    scaler.save(args.output_dir / "scaler.json")
    train_data = BatteryWindowDataset(train_batteries, args.window, scaler)
    validation_data = BatteryWindowDataset(validation_batteries, args.window, scaler)
    test_data = BatteryWindowDataset(test_batteries, args.window, scaler)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False)
    model_config = {"cnn_dim": 32, "capacity_dim": 24, "lstm_dim": 64, "dropout": 0.1}
    model = MultiModalBatteryModel(**model_config).to(device)
    criterion = PhysicsInformedLoss(physics_weight=args.physics_weight)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_parts = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_parts = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(validation_loss)
        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} | val={validation_loss:.6f} "
            f"| data={validation_parts['data']:.6f} | physics={validation_parts['physics']:.6f}"
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "window": args.window,
                    "physics_weight": args.physics_weight,
                    "synthetic": args.synthetic,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"验证损失连续 {args.patience} 轮未改善，提前停止。")
                break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_loss(history, args.output_dir / "loss_curve.png")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_metrics(model, test_loader, device)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test | MAE={metrics['mae']:.6f} | RMSE={metrics['rmse']:.6f} | R2={metrics['r2']:.6f}")
    print(f"最优模型与训练结果已保存到：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
