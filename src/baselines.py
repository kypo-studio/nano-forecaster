"""Classical baselines sharing the exact same forecast origins."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def naive_last(inputs: np.ndarray, target_index: int, horizon: int) -> np.ndarray:
    return np.repeat(inputs[:, -1, target_index, None], horizon, axis=1)


def seasonal_naive(inputs: np.ndarray, target_index: int, horizon: int, season: int = 24) -> np.ndarray:
    history = inputs[:, :, target_index]
    offsets = (np.arange(horizon) % season) - season
    return history[:, offsets]


def lag_calendar_features(inputs: np.ndarray, dates: pd.DatetimeIndex,
                          origins: np.ndarray) -> np.ndarray:
    """Flatten lags and add cyclical calendar state known at forecast origin."""
    origin_dates = dates[origins]
    cycles = [(origin_dates.hour.to_numpy(), 24),
              (origin_dates.dayofweek.to_numpy(), 7),
              (origin_dates.month.to_numpy() - 1, 12)]
    calendar = []
    for values, period in cycles:
        angle = 2 * np.pi * values / period
        calendar.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack([inputs.reshape(len(inputs), -1), *calendar])


def fit_xgboost(train_x: np.ndarray, train_y: np.ndarray, estimators: int,
                max_depth: int, learning_rate: float, seed: int):
    # Lazy import avoids loading XGBoost's OpenMP runtime during torch-only tests.
    from xgboost import XGBRegressor
    # Multi-output direct forecasting: one independent boosted tree ensemble per step.
    models = []
    features = train_x.reshape(len(train_x), -1) if train_x.ndim > 2 else train_x
    for step in range(train_y.shape[1]):
        model = XGBRegressor(n_estimators=estimators, max_depth=max_depth,
                             learning_rate=learning_rate, subsample=0.9,
                             colsample_bytree=0.9, n_jobs=1, random_state=seed)
        model.fit(features, train_y[:, step])
        models.append(model)
    return models


def predict_xgboost(models, inputs: np.ndarray) -> np.ndarray:
    features = inputs.reshape(len(inputs), -1) if inputs.ndim > 2 else inputs
    return np.column_stack([model.predict(features) for model in models])


def arima_forecasts(series: np.ndarray, origins: np.ndarray, horizon: int,
                    order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int] | None,
                    history: int) -> np.ndarray:
    predictions = []
    for origin in origins:
        sample = series[max(0, origin - history):origin]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(sample, order=order, seasonal_order=seasonal_order or (0, 0, 0, 0),
                            trend="c", enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=50)
        predictions.append(fitted.forecast(horizon))
    return np.asarray(predictions)
