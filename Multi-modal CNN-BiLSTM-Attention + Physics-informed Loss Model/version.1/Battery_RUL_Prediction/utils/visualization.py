"""训练与预测结果可视化。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history: dict[str, list[float]], output: str | Path) -> None:
    """绘制训练和验证总损失。"""
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_soh(cycles: np.ndarray, truth: np.ndarray, prediction: np.ndarray, output: str | Path) -> None:
    """绘制真实与预测 SOH 曲线。"""
    order = np.argsort(cycles)
    plt.figure(figsize=(8, 4))
    plt.plot(cycles[order], truth[order], label="True SOH")
    plt.plot(cycles[order], prediction[order], label="Predicted SOH")
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _estimate_rul(soh: np.ndarray, cycles: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """根据 SOH 首次到达阈值的循环估算每个观测点的剩余寿命。"""
    order = np.argsort(cycles)
    sorted_soh = soh[order]
    sorted_cycles = cycles[order]
    reached = np.flatnonzero(sorted_soh <= threshold)
    end_cycle = sorted_cycles[reached[0]] if reached.size else sorted_cycles[-1]
    rul = np.maximum(end_cycle - cycles, 0.0)
    return rul


def plot_rul(cycles: np.ndarray, truth: np.ndarray, prediction: np.ndarray, output: str | Path) -> None:
    """以 SOH=0.7 为失效阈值，将 SOH 序列转换成 RUL 曲线。"""
    order = np.argsort(cycles)
    true_rul = _estimate_rul(truth, cycles)
    predicted_rul = _estimate_rul(prediction, cycles)
    plt.figure(figsize=(8, 4))
    plt.plot(cycles[order], true_rul[order], label="True RUL")
    plt.plot(cycles[order], predicted_rul[order], label="Predicted RUL")
    plt.xlabel("Cycle")
    plt.ylabel("Remaining cycles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_attention(attention: np.ndarray, output: str | Path, max_samples: int = 100) -> None:
    """绘制最多 100 个测试样本的历史循环注意力热力图。"""
    shown = attention[:max_samples]
    plt.figure(figsize=(9, 5))
    plt.imshow(shown, aspect="auto", origin="lower", cmap="viridis")
    plt.xlabel("Position in history window")
    plt.ylabel("Test sample")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_rul(cycles: np.ndarray, truth: np.ndarray, prediction: np.ndarray, output: str | Path, threshold: float = 0.7) -> None:
    """根据 SOH 首次低于阈值的位置换算剩余循环数并绘图。"""
    order = np.argsort(cycles)
    cycles, truth, prediction = cycles[order], truth[order], prediction[order]
    true_crossings = np.flatnonzero(truth <= threshold)
    predicted_crossings = np.flatnonzero(prediction <= threshold)
    true_eol = cycles[true_crossings[0]] if true_crossings.size else cycles[-1]
    predicted_eol = cycles[predicted_crossings[0]] if predicted_crossings.size else cycles[-1]
    plt.figure(figsize=(8, 4))
    plt.plot(cycles, np.maximum(true_eol - cycles, 0), label="True RUL")
    plt.plot(cycles, np.maximum(predicted_eol - cycles, 0), label="Predicted RUL")
    plt.xlabel("Cycle")
    plt.ylabel("Remaining cycles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_attention(weights: np.ndarray, output: str | Path) -> None:
    """绘制若干测试样本的时间注意力热力图。"""
    plt.figure(figsize=(9, 4))
    plt.imshow(weights[: min(50, len(weights))], aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Position in historical window")
    plt.ylabel("Test sample")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
