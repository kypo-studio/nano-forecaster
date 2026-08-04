"""Leak-free temporal splits, train-only scaling and sliding windows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        return cls(mean, np.where(std < 1e-8, 1.0, std))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_target(self, values: np.ndarray, target_index: int) -> np.ndarray:
        return values * self.std[target_index] + self.mean[target_index]


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, starts: np.ndarray, input_len: int,
                 horizon: int, target_index: int):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.input_len = input_len
        self.horizon = horizon
        self.target_index = target_index

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        boundary = start + self.input_len
        x = self.values[start:boundary]
        y = self.values[boundary:boundary + self.horizon, self.target_index]
        return x, y


def load_etth1(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    if "date" not in frame or "OT" not in frame:
        raise ValueError("ETTh1 must contain date and OT columns")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("date")))
    return dates, frame.to_numpy(dtype=np.float32), frame.columns.tolist()


def temporal_boundaries(n: int, ratios: Sequence[float]) -> tuple[int, int]:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split_ratios must contain three values summing to 1")
    return int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))


def window_starts(n: int, input_len: int, horizon: int, train_end: int,
                  val_end: int) -> dict[str, np.ndarray]:
    starts = np.arange(0, n - input_len - horizon + 1)
    target_start = starts + input_len
    target_end = target_start + horizon
    return {
        "train": starts[target_end <= train_end],
        "val": starts[(target_start >= train_end) & (target_end <= val_end)],
        "test": starts[(target_start >= val_end) & (target_end <= n)],
    }


def prepare_data(path: Path, input_len: int, horizon: int, ratios: Sequence[float]):
    dates, raw, columns = load_etth1(path)
    train_end, val_end = temporal_boundaries(len(raw), ratios)
    scaler = Standardizer.fit(raw[:train_end])
    scaled = scaler.transform(raw).astype(np.float32)
    starts = window_starts(len(raw), input_len, horizon, train_end, val_end)
    target_index = columns.index("OT")
    datasets = {name: WindowDataset(scaled, split_starts, input_len, horizon, target_index)
                for name, split_starts in starts.items()}
    return dates, raw, scaled, columns, scaler, datasets, (train_end, val_end)

