"""Download and cache daily OHLCV price data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def load_prices(
    ticker: str,
    years: float = 2,
    end: str | None = None,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily prices for ``ticker`` covering the ``years`` before ``end``.

    The window is cached as ``<cache_dir>/<TICKER>.csv`` so results can be
    reproduced without re-downloading; pass ``refresh=True`` to pull it again.
    ``end`` defaults to today.
    """
    path = Path(cache_dir) / f"{ticker}.csv"
    if path.exists() and not refresh:
        return _read_csv(path)

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.DateOffset(years=years)
    # yfinance treats ``end`` as exclusive.
    raw = yf.download(
        ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[PRICE_COLUMNS].dropna().sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[PRICE_COLUMNS].sort_index()
