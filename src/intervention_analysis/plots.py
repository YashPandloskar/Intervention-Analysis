"""Figures: detected interventions and true-vs-predicted forecasts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_interventions(series: pd.Series, dates: list[pd.Timestamp], n_train: int, ticker: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(series.index, series.values, color="#1f4e79", lw=1.4, label=series.name)
    ax.axvspan(series.index[n_train], series.index[-1], color="0.9", label="Test window")
    for i, date in enumerate(dates):
        ax.axvline(date, color="#c0392b", ls="--", lw=1.1, label="Intervention" if i == 0 else None)
        ax.annotate(date.strftime("%d %b %Y"), (date, series.max()), rotation=90,
                    va="top", ha="right", fontsize=8, color="#c0392b")
    ax.set(title=f"{ticker}: detected intervention points", ylabel="Price (USD)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecasts(dates: pd.DatetimeIndex, actual: np.ndarray, predicted: dict[str, np.ndarray],
                   title: str, path: Path) -> None:
    n = len(predicted)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, (name, values) in zip(axes.ravel(), predicted.items()):
        rmse = float(np.sqrt(np.mean((actual - values) ** 2)))
        ax.plot(dates, actual, color="0.2", lw=1.3, label="Actual")
        ax.plot(dates, values, color="#c0392b", lw=1.3, label="Predicted")
        ax.set_title(f"{name} (RMSE {rmse:.2f})", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend()
    fig.suptitle(f"{title}: true vs predicted (test window)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
