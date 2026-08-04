#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import prepare_data
from src.baselines import (arima_forecasts, fit_xgboost, naive_last,
                           predict_xgboost, seasonal_naive)
from src.evaluate import result_table
from src.mlp import MLPForecaster
from src.model import NanoForecaster, parameter_count
from src.plots import plot_errors, plot_forecasts, plot_losses
from src.train import limited, predict, set_seed, train_model


def arrays(dataset):
    xs, ys = zip(*(dataset[i] for i in range(len(dataset))))
    return torch.stack(xs).numpy(), torch.stack(ys).numpy()


def main(config_path: Path):
    config = yaml.safe_load(config_path.read_text())
    set_seed(config["seed"])
    data_cfg, train_cfg, base_cfg = config["data"], config["training"], config["baselines"]
    horizon = max(data_cfg["horizons"])
    dates, raw, scaled, columns, scaler, datasets, boundaries = prepare_data(
        ROOT / data_cfg["path"], data_cfg["input_len"], horizon, data_cfg["split_ratios"])
    train_ds = limited(datasets["train"], data_cfg["max_train_windows"])
    val_ds = limited(datasets["val"], data_cfg["max_eval_windows"])
    test_ds = limited(datasets["test"], data_cfg["max_eval_windows"])
    n_features, target_index = len(columns), columns.index(data_cfg["target"])
    output = ROOT / config["output_dir"]; output.mkdir(parents=True, exist_ok=True)
    predictions, histories, timings, counts = {}, {}, {}, {}

    transformer = NanoForecaster(n_features, data_cfg["input_len"], horizon, **config["model"])
    history, elapsed, device = train_model(transformer, train_ds, val_ds, train_cfg, config["seed"])
    pred, targets, inputs = predict(transformer, test_ds, train_cfg["batch_size"], device)
    predictions["Transformer"] = pred; histories["Transformer"] = history
    timings["Transformer"] = elapsed; counts["Transformer"] = parameter_count(transformer)

    mlp = MLPForecaster(n_features, data_cfg["input_len"], horizon, base_cfg["mlp_hidden"])
    history, elapsed, mlp_device = train_model(mlp, train_ds, val_ds, train_cfg, config["seed"])
    pred, _, _ = predict(mlp, test_ds, train_cfg["batch_size"], mlp_device)
    predictions["MLP"] = pred; histories["MLP"] = history
    timings["MLP"] = elapsed; counts["MLP"] = parameter_count(mlp)

    train_x, train_y = arrays(train_ds)
    started = time.perf_counter()
    xgb_models = fit_xgboost(train_x, train_y, base_cfg["xgboost_estimators"],
                             base_cfg["xgboost_max_depth"], base_cfg["xgboost_learning_rate"], config["seed"])
    predictions["XGBoost"] = predict_xgboost(xgb_models, inputs)
    timings["XGBoost"] = time.perf_counter() - started

    predictions["Naive"] = naive_last(inputs, target_index, horizon)
    predictions["SeasonalNaive"] = seasonal_naive(inputs, target_index, horizon)

    origins = datasets["test"].starts[:len(test_ds)] + data_cfg["input_len"]
    series = scaled[:, target_index]
    for name, seasonal in [("ARIMA", None), ("SARIMA", tuple(base_cfg["sarima_seasonal_order"]))]:
        started = time.perf_counter()
        predictions[name] = arima_forecasts(series, origins, horizon,
                                             tuple(base_cfg["arima_order"]), seasonal,
                                             base_cfg["arima_history"])
        timings[name] = time.perf_counter() - started

    targets_real = scaler.inverse_target(targets, target_index)
    predictions_real = {name: scaler.inverse_target(values, target_index)
                        for name, values in predictions.items()}
    table = result_table(targets_real, predictions_real, data_cfg["horizons"])
    table.to_csv(output / "metrics.csv", index=False, float_format="%.6f")
    headers = table.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.itertuples(index=False):
        values = [str(row[0]), str(row[1]), *[f"{value:.4f}" for value in row[2:]]]
        lines.append("| " + " | ".join(values) + " |")
    (output / "metrics.md").write_text("\n".join(lines) + "\n")
    plot_losses(histories, output / "loss.png")
    plot_forecasts(targets_real, predictions_real, output / "forecast.png")
    plot_errors(table, output / "error_by_horizon.png")
    metadata = {
        "config": str(config_path.relative_to(ROOT)), "seed": config["seed"],
        "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
        "rows": len(raw), "splits": {"train_end": boundaries[0], "val_end": boundaries[1]},
        "windows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "parameters": counts, "training_seconds": timings,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(table.to_string(index=False)); print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args(); main(args.config.resolve())
