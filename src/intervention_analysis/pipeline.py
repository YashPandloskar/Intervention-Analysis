"""End-to-end experiment: detect interventions, fit the six models with and without them, evaluate."""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .changepoint import (
    describe_interventions,
    detect_change_points,
    estimable_columns,
    make_intervention_matrix,
)
from .data import load_prices
from .metrics import regression_metrics
from .models.base import Forecast, Problem
from .models.deep import run_deep
from .models.statistical import fit_prophet, fit_sarimax, select_order

log = logging.getLogger(__name__)

# Model name -> family, in the order used by the paper's tables.
MODELS = {
    "PROPHET": "Time Series Analysis",
    "ARIMA": "Time Series Analysis",
    "SARIMA": "Time Series Analysis",
    "RNN": "Deep Learning",
    "MLP": "Deep Learning",
    "LSTM": "Deep Learning",
}
VARIANTS = {"baseline": "Without intervention", "intervention": "With intervention"}

# "paper": time-series models forecast the whole test window from the end of training,
#          deep models predict one step ahead from a window of true past prices.
# "one_step": every model predicts one step ahead from the true past (like-for-like).
PROTOCOLS = ("paper", "one_step")


def run_ticker(ticker: str, cfg: dict, out_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Run the full experiment for one ticker and return the long-form metrics table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cfg, deep_cfg = cfg["data"], cfg["deep"]

    prices = load_prices(ticker, data_cfg["years"], data_cfg["end"], data_cfg["cache_dir"], refresh)
    series = prices[data_cfg["target"]]
    n_train = int(len(series) * cfg["split"]["train_frac"])
    log.info("%s: %d observations, %s -> %s (train %d / test %d)", ticker, len(series),
             series.index[0].date(), series.index[-1].date(), n_train, len(series) - n_train)

    # --- Intervention analysis -------------------------------------------------------
    icfg = cfg["intervention"]
    dates = detect_change_points(
        series, width=icfg["width"], pen=icfg["pen"], n_bkps=icfg["n_bkps"],
        bandwidth=icfg["bandwidth"], min_size=icfg["min_size"],
    )
    matrix = estimable_columns(make_intervention_matrix(series.index, dates, icfg["kind"]), n_train)
    table = describe_interventions(series, dates, n_train)
    table.to_csv(out_dir / "interventions.csv", index=False)
    log.info("%s: %d interventions detected, %d usable as regressors: %s", ticker, len(dates),
             matrix.shape[1], [d.date().isoformat() for d in dates])
    plots.plot_interventions(series, dates, n_train, ticker, out_dir / "interventions.png")

    base = Problem(
        y_train=series.iloc[:n_train].to_numpy(), y_test=series.iloc[n_train:].to_numpy(),
        dates_train=series.index[:n_train], dates_test=series.index[n_train:],
    )
    problems = {"baseline": base}
    if matrix.shape[1]:
        problems["intervention"] = replace(
            base, d_train=matrix.iloc[:n_train].to_numpy(), d_test=matrix.iloc[n_train:].to_numpy()
        )
    else:
        log.warning("%s: no intervention falls inside the training window; skipping the 'with intervention' runs", ticker)

    # --- Fit every model under every variant -----------------------------------------
    acfg = cfg["arima"]
    order = select_order(base.y_train, acfg["d"], acfg["max_p"], acfg["max_q"])
    seasonal = tuple(cfg["sarima"]["seasonal_order"])
    log.info("%s: ARIMA order (p,d,q) = %s, SARIMA seasonal order = %s", ticker, order, seasonal)

    runners = {
        "PROPHET": lambda p: fit_prophet(p, cfg["prophet"]["country_holidays"], cfg["prophet"]["yearly_seasonality"]),
        "ARIMA": lambda p: fit_sarimax(p, order),
        "SARIMA": lambda p: fit_sarimax(p, order, seasonal),
        "RNN": lambda p: run_deep("rnn", p, deep_cfg),
        "MLP": lambda p: run_deep("mlp", p, deep_cfg),
        "LSTM": lambda p: run_deep("lstm", p, deep_cfg),
    }
    forecasts: dict[tuple[str, str], list[Forecast]] = {}
    for variant, problem in problems.items():
        for model, run in runners.items():
            log.info("%s: fitting %s (%s)", ticker, model, VARIANTS[variant])
            forecasts[(model, variant)] = run(problem)

    # --- Evaluate --------------------------------------------------------------------
    rows = []
    for (model, variant), fcs in forecasts.items():
        for protocol in PROTOCOLS:
            per_seed = pd.DataFrame(
                [regression_metrics(base.y_test, _select(fc, protocol)) for fc in fcs]
            )
            rows.append({
                "ticker": ticker, "family": MODELS[model], "model": model, "variant": variant,
                "protocol": protocol, "n_runs": len(fcs),
                **per_seed.mean().to_dict(),
                **{f"{m}_std": s for m, s in per_seed.std(ddof=0).items()},
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    # Predictions of the first run, for inspection and plotting.
    predictions = pd.DataFrame({"date": base.dates_test, "actual": base.y_test})
    for (model, variant), fcs in forecasts.items():
        predictions[f"{model}_{variant}"] = _select(fcs[0], "paper")
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    for variant in problems:
        plots.plot_forecasts(
            base.dates_test, base.y_test,
            {m: _select(forecasts[(m, variant)][0], "paper") for m in MODELS},
            f"{ticker} - {VARIANTS[variant].lower()}", out_dir / f"forecast_{variant}.png",
        )
    return metrics


def _select(forecast: Forecast, protocol: str) -> np.ndarray:
    if protocol == "paper" and forecast.multi_step is not None:
        return forecast.multi_step
    return forecast.one_step


def write_summary(metrics: pd.DataFrame, path: Path) -> None:
    """Write paper-style tables (MSE / RMSE / MAE per model) as Markdown."""
    lines = ["# Results", ""]
    for protocol in PROTOCOLS:
        title = {
            "paper": "Paper protocol (time-series models: multi-step forecast of the test window; "
                     "deep models: one-step-ahead)",
            "one_step": "Like-for-like protocol (every model: one-step-ahead)",
        }[protocol]
        lines += [f"## {title}", ""]
        for ticker, group in metrics[metrics["protocol"] == protocol].groupby("ticker", sort=False):
            for variant, label in VARIANTS.items():
                part = group[group["variant"] == variant]
                if part.empty:
                    continue
                lines += [f"### {ticker} - {label}", "", "| Family | Model | MSE | RMSE | MAE |", "|---|---|---:|---:|---:|"]
                best = part["RMSE"].idxmin()
                for idx, r in part.iterrows():
                    mark = "**" if idx == best else ""
                    lines.append(
                        f"| {r['family']} | {mark}{r['model']}{mark} | {r['MSE']:.3f} | {r['RMSE']:.3f} | {r['MAE']:.3f} |"
                    )
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
