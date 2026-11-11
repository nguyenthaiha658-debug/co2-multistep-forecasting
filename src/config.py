"""Shared settings for the forecasting experiments."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
TRAIN_RATIO = 0.8

# Weekly Mauna Loa CO2: a 12-week window (~one season quarter) and a
# 10-week horizon, matching the course recursive demo length.
CO2 = {
    "name": "co2",
    "value_col": "co2",
    "time_col": "time",
    "window_size": 12,
    "horizon": 10,
    "seasonal_period": 52,  # yearly seasonality in weekly data
}

# Daily bike-sharing totals (aggregated from the hourly series).
BIKE = {
    "name": "bike_sharing_daily",
    "value_col": "users",
    "time_col": "date",
    "window_size": 14,
    "horizon": 14,
    "seasonal_period": 7,
}
