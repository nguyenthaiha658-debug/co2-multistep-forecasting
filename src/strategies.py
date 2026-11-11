"""Recursive and direct multi-step forecasting strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone


@dataclass
class FittedRecursive:
    """One model trained to predict one step ahead, rolled forward at test time."""

    model: BaseEstimator
    window_size: int

    def predict_from_window(self, window: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(window, dtype=float).reshape(-1).tolist()
        if len(history) < self.window_size:
            raise ValueError("window is shorter than window_size")
        preds = []
        for _ in range(horizon):
            features = np.asarray(history[-self.window_size :], dtype=float).reshape(1, -1)
            yhat = float(self.model.predict(features)[0])
            preds.append(yhat)
            history.append(yhat)
        return np.asarray(preds, dtype=float)


@dataclass
class FittedDirect:
    """One model per horizon, each mapping the same window to y_{t+h}."""

    models: list[BaseEstimator]
    window_size: int

    @property
    def horizon(self) -> int:
        return len(self.models)

    def predict_from_window(self, window: np.ndarray, horizon: int | None = None) -> np.ndarray:
        features = np.asarray(window, dtype=float).reshape(-1)
        if features.size != self.window_size:
            # allow a longer history buffer, use the last window_size points
            if features.size < self.window_size:
                raise ValueError("window is shorter than window_size")
            features = features[-self.window_size :]
        features = features.reshape(1, -1)
        h = self.horizon if horizon is None else horizon
        if h > self.horizon:
            raise ValueError(f"requested horizon {h} exceeds trained horizon {self.horizon}")
        return np.asarray(
            [float(model.predict(features)[0]) for model in self.models[:h]],
            dtype=float,
        )


def train_recursive(
    estimator: BaseEstimator,
    train_values: np.ndarray,
    window_size: int,
) -> FittedRecursive:
    """Fit a single one-step model on windows built only from the training series."""
    from src.data import make_supervised

    X, Y = make_supervised(train_values, window_size=window_size, horizon=1)
    model = clone(estimator)
    model.fit(X, Y.ravel())
    return FittedRecursive(model=model, window_size=window_size)


def train_direct(
    estimator: BaseEstimator,
    train_values: np.ndarray,
    window_size: int,
    horizon: int,
) -> FittedDirect:
    """Fit `horizon` independent models, one for each lead time."""
    from src.data import make_supervised

    X, Y = make_supervised(train_values, window_size=window_size, horizon=horizon)
    models = []
    for h in range(horizon):
        model = clone(estimator)
        model.fit(X, Y[:, h])
        models.append(model)
    return FittedDirect(models=models, window_size=window_size)


@dataclass
class FittedDifferenced:
    """Forecast first differences, then integrate back to levels.

    Tree models cannot predict values outside the range of the training
    targets. A trending series such as Mauna Loa CO2 therefore breaks a
    Random Forest trained on raw levels. Training on first differences,
    and adding the predicted changes back onto the last observed level,
    sidesteps that limitation.
    """

    inner: FittedRecursive | FittedDirect
    level_window_size: int

    def predict_from_window(self, window: np.ndarray, horizon: int) -> np.ndarray:
        levels = np.asarray(window, dtype=float).reshape(-1)
        if levels.size < self.level_window_size:
            raise ValueError("window is shorter than level_window_size")
        levels = levels[-self.level_window_size :]
        diffs = np.diff(levels)
        delta = self.inner.predict_from_window(diffs, horizon)
        return levels[-1] + np.cumsum(delta)


def train_recursive_on_differences(
    estimator: BaseEstimator,
    train_values: np.ndarray,
    level_window_size: int,
) -> FittedDifferenced:
    if level_window_size < 2:
        raise ValueError("level_window_size must be >= 2 to form differences")
    train_diff = np.diff(np.asarray(train_values, dtype=float))
    inner = train_recursive(estimator, train_diff, window_size=level_window_size - 1)
    return FittedDifferenced(inner=inner, level_window_size=level_window_size)


def train_direct_on_differences(
    estimator: BaseEstimator,
    train_values: np.ndarray,
    level_window_size: int,
    horizon: int,
) -> FittedDifferenced:
    if level_window_size < 2:
        raise ValueError("level_window_size must be >= 2 to form differences")
    train_diff = np.diff(np.asarray(train_values, dtype=float))
    inner = train_direct(
        estimator, train_diff, window_size=level_window_size - 1, horizon=horizon
    )
    return FittedDifferenced(inner=inner, level_window_size=level_window_size)
