"""NASA 电池数据解析、特征构造、标准化与滑动窗口数据集。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


FEATURE_NAMES = ("voltage", "current", "temperature", "capacity")


@dataclass
class StandardScaler:
    """保存训练集统计量，保证验证集和测试集不发生信息泄漏。"""

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean.tolist(), std.tolist())

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean)) / np.asarray(self.std)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _field(record: object, name: str, default: object = None) -> object:
    """兼容 scipy 读取 MATLAB struct 后的属性和结构化数组访问方式。"""
    if hasattr(record, name):
        return getattr(record, name)
    if isinstance(record, np.void) and record.dtype.names and name in record.dtype.names:
        return record[name]
    return default


def _mean_numeric(value: object, default: float = 0.0) -> float:
    """把一个循环内的曲线压缩为均值；原始循环动态可在后续版本扩展为双尺度输入。"""
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        return float(finite.mean()) if finite.size else default
    except (TypeError, ValueError):
        return default


def load_nasa_mat(path: str | Path) -> dict[str, np.ndarray]:
    """读取 NASA PCoE 常见 B0005/B0006/B0007/B0018 `.mat` 结构中的放电循环。"""
    path = Path(path)
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    root = raw.get(path.stem)
    if root is None:
        candidates = [value for key, value in raw.items() if not key.startswith("__")]
        if not candidates:
            raise ValueError(f"{path} 中未找到 MATLAB 数据变量")
        root = candidates[0]
    cycles = np.atleast_1d(_field(root, "cycle", []))
    rows: list[list[float]] = []
    for cycle_index, cycle in enumerate(cycles):
        if str(_field(cycle, "type", "")).lower() != "discharge":
            continue
        data = _field(cycle, "data")
        if data is None:
            continue
        capacity = _mean_numeric(_field(data, "Capacity"), np.nan)
        if not np.isfinite(capacity):
            continue
        voltage = _mean_numeric(_field(data, "Voltage_measured"))
        current = _mean_numeric(_field(data, "Current_measured"))
        temperature = _mean_numeric(_field(data, "Temperature_measured"))
        rows.append([voltage, current, temperature, capacity, float(cycle_index + 1)])
    if len(rows) < 2:
        raise ValueError(f"{path} 未解析出足够的放电循环，请确认文件来自 NASA PCoE 数据集")
    values = np.asarray(rows, dtype=np.float32)
    initial_capacity = float(np.max(values[: min(5, len(values)), 3]))
    soh = values[:, 3] / max(initial_capacity, 1e-8)
    return {"features": values[:, :4], "cycle": values[:, 4], "soh": soh.astype(np.float32)}


def generate_synthetic_batteries(count: int = 8, cycles: int = 180, seed: int = 42) -> list[dict[str, np.ndarray]]:
    """生成带温度、负载和随机测量扰动的退化曲线，用于安装后的冒烟测试。"""
    rng = np.random.default_rng(seed)
    batteries: list[dict[str, np.ndarray]] = []
    for battery_index in range(count):
        cycle = np.arange(1, cycles + 1, dtype=np.float32)
        rate = rng.uniform(0.0010, 0.0018)
        nonlinear = rng.uniform(0.08, 0.16) * (cycle / cycles) ** 2
        rebound = 0.005 * np.sin(cycle / 9.0 + battery_index)
        soh = np.clip(1.0 - rate * cycle - nonlinear + rebound + rng.normal(0, 0.002, cycles), 0.55, 1.02)
        capacity = rng.uniform(1.9, 2.1) * soh
        temperature = 24.0 + 12.0 * (1.0 - soh) + rng.normal(0, 0.6, cycles)
        voltage = 3.9 - 0.35 * (1.0 - soh) + rng.normal(0, 0.01, cycles)
        current = -2.0 + rng.normal(0, 0.04, cycles)
        features = np.stack([voltage, current, temperature, capacity], axis=1).astype(np.float32)
        batteries.append({"features": features, "cycle": cycle, "soh": soh.astype(np.float32)})
    return batteries


def load_batteries(data_dir: str | Path, synthetic: bool = False, seed: int = 42) -> list[dict[str, np.ndarray]]:
    """读取目录内全部 `.mat`；显式指定 synthetic 时生成演示数据。"""
    if synthetic:
        return generate_synthetic_batteries(seed=seed)
    paths = sorted(Path(data_dir).glob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"{data_dir} 中没有 .mat 文件；可加 --synthetic 先运行演示")
    return [load_nasa_mat(path) for path in paths]


def split_batteries(batteries: list[dict[str, np.ndarray]]) -> tuple[list, list, list]:
    """按电池划分以避免相邻窗口泄漏；至少三块电池时使用约 60/20/20 比例。"""
    if len(batteries) < 3:
        raise ValueError("至少需要 3 块电池，才能按电池划分训练、验证和测试集")
    train_end = max(1, int(len(batteries) * 0.6))
    validation_end = max(train_end + 1, int(len(batteries) * 0.8))
    validation_end = min(validation_end, len(batteries) - 1)
    return batteries[:train_end], batteries[train_end:validation_end], batteries[validation_end:]


class BatteryWindowDataset(Dataset):
    """将每块电池转换成 ``历史 window 个循环 -> 当前 SOH`` 样本。"""

    def __init__(self, batteries: list[dict[str, np.ndarray]], window: int, scaler: StandardScaler) -> None:
        self.samples: list[tuple[np.ndarray, float, float, int]] = []
        for battery_id, battery in enumerate(batteries):
            features = scaler.transform(battery["features"]).astype(np.float32)
            for end in range(window, len(features)):
                self.samples.append((features[end - window : end], battery["soh"][end], battery["cycle"][end], battery_id))
        if not self.samples:
            raise ValueError(f"窗口长度 {window} 大于或等于所有电池的有效循环数")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        features, soh, cycle, battery_id = self.samples[index]
        return {
            "x": torch.from_numpy(features),  # (window, 4)
            "y": torch.tensor([soh], dtype=torch.float32),  # (1,)
            "cycle": torch.tensor(cycle, dtype=torch.float32),
            "battery_id": torch.tensor(battery_id, dtype=torch.long),
        }


def fit_scaler(batteries: list[dict[str, np.ndarray]]) -> StandardScaler:
    """仅从训练电池拟合四个输入特征的标准化参数。"""
    return StandardScaler.fit(np.concatenate([battery["features"] for battery in batteries], axis=0))
