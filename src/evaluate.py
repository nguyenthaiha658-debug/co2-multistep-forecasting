"""Per-horizon MAE / RMSE / R2 and simple baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def walk_forward_origins(
    values: np.ndarray,
    train_len: int,
    window_size: int,
    horizon: int,
) -> list[int]:
    """Origins t such that t is the last observed index and t+1..t+horizon exist.

    The first origin is train_len - 1 (end of train), so every predicted
    timestamp falls in the test period. History may include training values,
    which is the usual setup for a genuine out-of-sample forecast.
    """
    n = len(values)
    last_origin = n - 1 - horizon
    first_origin = max(window_size - 1, train_len - 1)
    if last_origin < first_origin:
        raise ValueError("Not enough test points for the requested horizon")
    return list(range(first_origin, last_origin + 1))


def collect_horizon_predictions(
    predict_from_window,
    values: np.ndarray,
    origins: list[int],
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of shape (n_origins, horizon) for y_true and y_pred."""
    y_true = np.zeros((len(origins), horizon), dtype=float)
    y_pred = np.zeros((len(origins), horizon), dtype=float)
    for i, origin in enumerate(origins):
        window = values[origin - window_size + 1 : origin + 1]
        y_true[i] = values[origin + 1 : origin + 1 + horizon]
        y_pred[i] = predict_from_window(window, horizon)
    return y_true, y_pred


def metrics_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for h in range(y_true.shape[1]):
        rows.append(
            {
                "horizon": h + 1,
                "mae": float(mean_absolute_error(y_true[:, h], y_pred[:, h])),
                "rmse": float(root_mean_squared_error(y_true[:, h], y_pred[:, h])),
                "r2": float(r2_score(y_true[:, h], y_pred[:, h])),
            }
        )
    return pd.DataFrame(rows)


def persistence_predict(window: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every lead time."""
    last = float(np.asarray(window, dtype=float).reshape(-1)[-1])
    return np.full(horizon, last, dtype=float)


def seasonal_naive_predict_factory(values: np.ndarray, period: int):
    """yhat(t+h) = y(t+h-period). Uses the actual series (standard seasonal naive)."""

    def predict(window: np.ndarray, horizon: int, origin: int) -> np.ndarray:
        preds = []
        for h in range(1, horizon + 1):
            idx = origin + h - period
            if idx < 0:
                raise ValueError("seasonal naive needs a longer history than the period")
            preds.append(float(values[idx]))
        return np.asarray(preds, dtype=float)

    return predict


def collect_seasonal_naive(
    values: np.ndarray,
    origins: list[int],
    horizon: int,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.zeros((len(origins), horizon), dtype=float)
    y_pred = np.zeros((len(origins), horizon), dtype=float)
    for i, origin in enumerate(origins):
        y_true[i] = values[origin + 1 : origin + 1 + horizon]
        for h in range(1, horizon + 1):
            idx = origin + h - period
            y_pred[i, h - 1] = float(values[idx]) if idx >= 0 else np.nan
    return y_true, y_pred
