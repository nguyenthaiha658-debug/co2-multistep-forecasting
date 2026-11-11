"""Error-vs-horizon and forecast plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error_vs_horizon(
    metrics: pd.DataFrame,
    path: Path,
    title: str,
    metric: str = "mae",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for method, group in metrics.groupby("method"):
        ax.plot(group["horizon"], group[metric], marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="best")
    # integer ticks if the horizon is short
    horizons = sorted(metrics["horizon"].unique())
    ax.set_xticks(horizons)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_from_origin(
    index,
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    origin: int,
    train_len: int,
    path: Path,
    title: str,
    ylabel: str,
    context: int = 40,
) -> None:
    """Plot actual series around the origin plus each method's multi-step forecast."""
    start = max(0, origin - context)
    end = min(len(actual), origin + 1 + max(len(v) for v in forecasts.values()))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(index[start:train_len], actual[start:train_len], color="#4a4a4a", label="Train (actual)")
    ax.plot(
        index[max(start, train_len) : end],
        actual[max(start, train_len) : end],
        color="black",
        label="Test (actual)",
    )
    origin_time = index[origin]
    ax.axvline(origin_time, color="grey", linestyle="--", linewidth=1, label="Forecast origin")
    for name, preds in forecasts.items():
        times = index[origin + 1 : origin + 1 + len(preds)]
        ax.plot(times, preds, marker="o", markersize=3.5, linewidth=1.6, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
