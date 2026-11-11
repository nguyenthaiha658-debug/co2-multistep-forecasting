"""Load series and build supervised windows without mutating the input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR


def load_co2(path: Path | None = None) -> pd.Series:
    """Weekly atmospheric CO2. Linear interpolation fills 59 missing weeks."""
    path = path or (DATA_DIR / "co2.csv")
    frame = pd.read_csv(path)
    time = pd.to_datetime(frame["time"])
    values = pd.to_numeric(frame["co2"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=time, name="co2")
    # interpolate on a copy of the values; the CSV on disk is left alone
    series = series.interpolate(method="linear")
    if series.isna().any():
        series = series.bfill().ffill()
    return series


def load_bike_hourly(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "bike_sharing.csv")
    frame = pd.read_csv(path)
    frame = frame.copy()
    frame["date_time"] = pd.to_datetime(frame["date_time"])
    return frame


def load_bike_daily(path: Path | None = None) -> pd.Series:
    """Sum hourly rentals to a daily total. A genuine extra dataset vs the course CO2 demo."""
    hourly = load_bike_hourly(path)
    daily = (
        hourly.set_index("date_time")["users"]
        .resample("D")
        .sum()
        .rename("users")
        .asfreq("D")
    )
    if daily.isna().any():
        daily = daily.interpolate(method="linear")
    return daily


def time_split(series: pd.Series, train_ratio: float = 0.8) -> tuple[pd.Series, pd.Series]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    cut = int(len(series) * train_ratio)
    if cut < 1 or cut >= len(series):
        raise ValueError("train_ratio leaves an empty train or test split")
    return series.iloc[:cut].copy(), series.iloc[cut:].copy()


def make_supervised(
    values: np.ndarray | pd.Series,
    window_size: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window features and targets, leaving the input unchanged.

    For row i (starting at the beginning of the window):
        X[i] = values[i : i + window_size]
        Y[i, h] = values[i + window_size + h]   for h = 0 .. horizon-1
    so column h is the (h+1)-step-ahead target.

    The course helpers assigned new columns onto the *same* DataFrame
    they were given (`data['co2_1'] = data['co2'].shift(-1)`, ...),
    which silently mutated the caller's object. This function only
    reads `values` and always returns new arrays.
    """
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window_size < 1 or horizon < 1:
        raise ValueError("window_size and horizon must be >= 1")
    n = len(vector)
    n_samples = n - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"series length {n} is too short for window={window_size} horizon={horizon}"
        )
    windows = np.lib.stride_tricks.sliding_window_view(vector, window_size)
    # windows[i] covers indices [i, i+window_size); we need i + window_size + horizon - 1 < n
    X = np.ascontiguousarray(windows[:n_samples])
    Y = np.column_stack(
        [vector[window_size + h : window_size + h + n_samples] for h in range(horizon)]
    )
    return X, Y


def assert_input_not_mutated(series: pd.Series) -> None:
    """Guard used by the experiment runner: windowing must not touch `series`."""
    before_index = series.index.copy()
    before_values = series.to_numpy().copy()
    make_supervised(series, window_size=5, horizon=3)
    if not before_index.equals(series.index):
        raise AssertionError("make_supervised mutated the series index")
    if not np.array_equal(before_values, series.to_numpy()):
        raise AssertionError("make_supervised mutated the series values")
