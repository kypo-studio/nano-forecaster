from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    epsilon = 1e-8
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(100 * np.mean(np.abs(error) / np.maximum(np.abs(y_true), epsilon))),
        "sMAPE": float(200 * np.mean(np.abs(error) / np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon))),
    }


def result_table(targets: np.ndarray, predictions: dict[str, np.ndarray], horizons: list[int]):
    rows = []
    for model, forecast in predictions.items():
        for horizon in horizons:
            rows.append({"model": model, "horizon": horizon,
                         **metrics(targets[:, :horizon], forecast[:, :horizon])})
    return pd.DataFrame(rows).sort_values(["horizon", "MAE", "model"]).reset_index(drop=True)

