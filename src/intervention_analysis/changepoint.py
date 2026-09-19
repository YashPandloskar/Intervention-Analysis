"""Intervention-point detection and construction of the intervention variable D_t.

Detection follows Section III-D of the paper: a sliding-window search over a
kernel cost. The fit term is the RBF-kernel cost (Eq. 15), a penalty term is
added (C_k = F_k + P_k) and the segmentation with the lowest cost is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt


def detect_change_points(
    prices: pd.Series,
    width: int = 60,
    pen: float | None = 2.0,
    n_bkps: int | None = None,
    bandwidth: float | None = None,
    min_size: int = 10,
) -> list[pd.Timestamp]:
    """Return the dates on which a new regime starts.

    Parameters
    ----------
    width:     sliding-window width (trading days).
    pen:       penalty P_k added to the fit term; larger => fewer change points.
    n_bkps:    if given, ignore ``pen`` and return exactly this many points.
    bandwidth: RBF bandwidth gamma of Eq. 15 on the z-scored series.
               ``None`` uses the median heuristic.
    """
    values = prices.to_numpy(dtype=float)
    signal = ((values - values.mean()) / values.std()).reshape(-1, 1)

    # ruptures parametrises the kernel as exp(-g * ||x - y||^2), i.e. g = 1 / (2 * bandwidth^2).
    params = {} if bandwidth is None else {"gamma": 1.0 / (2.0 * bandwidth**2)}
    algo = rpt.Window(width=width, model="rbf", params=params, min_size=min_size, jump=1).fit(signal)
    bkps = algo.predict(n_bkps=n_bkps) if n_bkps else algo.predict(pen=pen)

    # The last breakpoint is always the series length.
    return [prices.index[i] for i in bkps[:-1]]


def make_intervention_matrix(
    index: pd.DatetimeIndex, dates: list[pd.Timestamp], kind: str = "step"
) -> pd.DataFrame:
    """Build the intervention variable(s) D_t, one column per intervention.

    ``step``  : D_t = 1 from the intervention date onwards (permanent level shift).
    ``pulse`` : D_t = 1 only on the intervention date (transient shock).
    """
    if kind not in {"step", "pulse"}:
        raise ValueError(f"Unknown intervention kind: {kind!r}")
    columns = {}
    for k, date in enumerate(dates, start=1):
        active = index >= date if kind == "step" else index == date
        columns[f"D{k}"] = active.astype(float)
    return pd.DataFrame(columns, index=index)


def estimable_columns(matrix: pd.DataFrame, n_train: int) -> pd.DataFrame:
    """Keep only interventions whose effect can be learned from the training data.

    A column that is constant over the training window (e.g. an intervention that
    happens inside the test period) carries no information for fitting delta.
    """
    keep = [c for c in matrix.columns if matrix[c].iloc[:n_train].nunique() > 1]
    return matrix[keep]


def describe_interventions(
    prices: pd.Series, dates: list[pd.Timestamp], n_train: int, window: int = 20
) -> pd.DataFrame:
    """Tabulate each detected intervention so it can be matched to real-world events."""
    returns = prices.pct_change()
    rows = []
    for date in dates:
        i = prices.index.get_loc(date)
        before = prices.iloc[max(0, i - window) : i].mean()
        after = prices.iloc[i : i + window].mean()
        rows.append(
            {
                "date": date.date().isoformat(),
                "close": round(float(prices.iloc[i]), 2),
                "one_day_return_pct": round(float(returns.iloc[i] * 100), 2),
                "mean_close_prior_20d": round(float(before), 2),
                "mean_close_next_20d": round(float(after), 2),
                "level_shift_pct": round(float((after / before - 1) * 100), 2) if before else np.nan,
                "in_training_window": i < n_train,
                "event": "",
            }
        )
    return pd.DataFrame(rows)
