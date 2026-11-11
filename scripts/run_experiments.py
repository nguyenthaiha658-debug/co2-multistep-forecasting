#!/usr/bin/env python3
"""Compare recursive vs direct multi-step forecasts on CO2 and daily bike sharing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BIKE, CO2, RANDOM_STATE, RESULTS_DIR, TRAIN_RATIO
from src.data import (
    assert_input_not_mutated,
    load_bike_daily,
    load_co2,
    time_split,
)
from src.evaluate import (
    collect_horizon_predictions,
    collect_seasonal_naive,
    metrics_per_horizon,
    persistence_predict,
    walk_forward_origins,
)
from src.plots import plot_error_vs_horizon, plot_forecast_from_origin
from src.strategies import (
    train_direct,
    train_direct_on_differences,
    train_recursive,
    train_recursive_on_differences,
)


def estimators() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def run_dataset(series: pd.Series, spec: dict) -> pd.DataFrame:
    name = spec["name"]
    window_size = spec["window_size"]
    horizon = spec["horizon"]
    period = spec["seasonal_period"]
    ylabel = spec["value_col"]

    assert_input_not_mutated(series)
    train, test = time_split(series, TRAIN_RATIO)
    values = series.to_numpy(dtype=float)
    train_values = train.to_numpy(dtype=float)
    train_len = len(train)
    print(f"\n{'=' * 72}")
    print(
        f"{name}: n={len(series)}  train={train_len}  test={len(test)}  "
        f"window={window_size}  horizon={horizon}"
    )

    origins = walk_forward_origins(values, train_len, window_size, horizon)
    print(f"Walk-forward origins: {len(origins)}")

    methods: dict[str, callable] = {}
    fitted = {}

    # Baselines
    methods["Persistence"] = persistence_predict

    fitted_models = {}
    for est_name, est in estimators().items():
        rec = train_recursive(est, train_values, window_size)
        dir_ = train_direct(est, train_values, window_size, horizon)
        fitted_models[f"Recursive / {est_name}"] = rec
        fitted_models[f"Direct / {est_name}"] = dir_
        methods[f"Recursive / {est_name}"] = rec.predict_from_window
        methods[f"Direct / {est_name}"] = dir_.predict_from_window
        # Trees cannot extrapolate a trend in levels; the differenced
        # models are the follow-up experiment that addresses that.
        if est_name == "RandomForest":
            rec_d = train_recursive_on_differences(est, train_values, window_size)
            dir_d = train_direct_on_differences(est, train_values, window_size, horizon)
            fitted_models[f"Recursive-diff / {est_name}"] = rec_d
            fitted_models[f"Direct-diff / {est_name}"] = dir_d
            methods[f"Recursive-diff / {est_name}"] = rec_d.predict_from_window
            methods[f"Direct-diff / {est_name}"] = dir_d.predict_from_window
            print(f"Fitted recursive + direct + differenced models for {est_name}")
        else:
            print(f"Fitted recursive + direct models for {est_name}")

    frames = []
    origin_zero_forecasts = {}
    first_origin = origins[0]

    # Seasonal naive is slightly different (needs the origin index).
    y_true_sn, y_pred_sn = collect_seasonal_naive(values, origins, horizon, period)
    sn_metrics = metrics_per_horizon(y_true_sn, y_pred_sn)
    sn_metrics.insert(0, "method", f"Seasonal naive (period={period})")
    sn_metrics.insert(0, "dataset", name)
    frames.append(sn_metrics)
    origin_zero_forecasts[f"Seasonal naive (period={period})"] = y_pred_sn[0]

    for method_name, predict in methods.items():
        y_true, y_pred = collect_horizon_predictions(
            predict, values, origins, window_size, horizon
        )
        metrics = metrics_per_horizon(y_true, y_pred)
        metrics.insert(0, "method", method_name)
        metrics.insert(0, "dataset", name)
        frames.append(metrics)
        origin_zero_forecasts[method_name] = y_pred[0]
        last = metrics.iloc[-1]
        print(
            f"  {method_name:32s}  h=1 MAE={metrics.iloc[0]['mae']:.4f}  "
            f"h={horizon} MAE={last['mae']:.4f}  h={horizon} RMSE={last['rmse']:.4f}"
        )

    table = pd.concat(frames, ignore_index=True)

    # Error-vs-horizon plots (MAE and RMSE)
    plot_error_vs_horizon(
        table,
        RESULTS_DIR / f"{name}_mae_vs_horizon.png",
        title=f"{name}: MAE vs forecast horizon",
        metric="mae",
    )
    plot_error_vs_horizon(
        table,
        RESULTS_DIR / f"{name}_rmse_vs_horizon.png",
        title=f"{name}: RMSE vs forecast horizon",
        metric="rmse",
    )
    # Raw-level Random Forest on trending CO2 sits far above the other
    # curves; a zoomed plot makes the remaining comparison readable.
    if name == "co2":
        zoom = table[~table["method"].isin({"Direct / RandomForest", "Recursive / RandomForest"})]
        plot_error_vs_horizon(
            zoom,
            RESULTS_DIR / f"{name}_mae_vs_horizon_zoom.png",
            title=f"{name}: MAE vs forecast horizon (without raw-level Random Forest)",
            metric="mae",
        )

    # Forecast plot from the end of the training set.
    # Drop persistence and seasonal naive from the plot if it gets too busy?
    # Keep all — 6 series + actual is readable with a legend.
    plot_forecast_from_origin(
        index=series.index,
        actual=values,
        forecasts=origin_zero_forecasts,
        origin=first_origin,
        train_len=train_len,
        path=RESULTS_DIR / f"{name}_forecast.png",
        title=f"{name}: {horizon}-step forecast from the end of the training set",
        ylabel=ylabel,
    )
    # A later origin, where CO2 has already walked outside the training
    # range, makes the tree-extrapolation issue visible.
    late_origin = origins[max(0, int(0.75 * (len(origins) - 1)))]
    late_forecasts = {}
    for method_name, predict in methods.items():
        window = values[late_origin - window_size + 1 : late_origin + 1]
        late_forecasts[method_name] = predict(window, horizon)
    # include seasonal naive at the late origin too
    sn_name = f"Seasonal naive (period={period})"
    late_forecasts[sn_name] = collect_seasonal_naive(
        values, [late_origin], horizon, period
    )[1][0]
    plot_forecast_from_origin(
        index=series.index,
        actual=values,
        forecasts=late_forecasts,
        origin=late_origin,
        train_len=train_len,
        path=RESULTS_DIR / f"{name}_forecast_late.png",
        title=f"{name}: {horizon}-step forecast from a later test origin",
        ylabel=ylabel,
    )
    # Keep fitted unused-warning free
    _ = fitted_models
    return table


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    co2 = load_co2()
    bike = load_bike_daily()

    tables = [
        run_dataset(co2, CO2),
        run_dataset(bike, BIKE),
    ]
    all_metrics = pd.concat(tables, ignore_index=True)
    csv_path = RESULTS_DIR / "metrics.csv"
    all_metrics.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Compact pivot for the README: MAE by method x horizon, per dataset
    summary = {}
    for dataset, group in all_metrics.groupby("dataset"):
        mae_pivot = group.pivot(index="method", columns="horizon", values="mae")
        rmse_pivot = group.pivot(index="method", columns="horizon", values="rmse")
        r2_pivot = group.pivot(index="method", columns="horizon", values="r2")
        summary[dataset] = {
            "mae": mae_pivot.round(4).to_dict(),
            "rmse": rmse_pivot.round(4).to_dict(),
            "r2": r2_pivot.round(4).to_dict(),
        }
        print(f"\n===== {dataset} MAE =====")
        print(mae_pivot.round(3).to_string())
        print(f"\n===== {dataset} RMSE =====")
        print(rmse_pivot.round(3).to_string())
        print(f"\n===== {dataset} R2 =====")
        print(r2_pivot.round(3).to_string())

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
