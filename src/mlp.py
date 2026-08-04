from __future__ import annotations

import torch
from torch import nn


class MLPForecaster(nn.Module):
    def __init__(self, n_features: int, input_len: int, horizon: int, hidden: list[int]):
        super().__init__()
        sizes = [n_features * input_len, *hidden, horizon]
        layers = []
        for source, target in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([nn.Linear(source, target), nn.ReLU()])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(1))

