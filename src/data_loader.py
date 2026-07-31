"""
data_loader.py
---------------
Fetches historical daily OHLCV data and returns a clean, standardized
DataFrame indexed by date with columns: open, high, low, close, adj_close, volume.

Three source modes (see config.DATA_SOURCE):
  - 'yfinance'   : live pull from Yahoo Finance (requires outbound internet)
  - 'local_csv'  : read a CSV from disk (see config.LOCAL_CSV_PATH)
  - 'auto'       : try yfinance first, transparently fall back to local_csv

This module intentionally does NOT silently invent data. If yfinance is
unavailable (no package, no network) it says so and falls back; it never
fabricates prices.
"""

from __future__ import annotations

import sys
import warnings
import pandas as pd

from . import config


class DataLoadError(RuntimeError):
    pass


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever schema we got into lowercase open/high/low/close/adj_close/volume."""
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in ("adj_close", "adjclose", "adjusted_close", "adjusted"):
            rename_map[col] = "adj_close"
        elif key in ("open", "high", "low", "close", "volume"):
            rename_map[col] = key
    df = df.rename(columns=rename_map)

    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()

    # If there's no adjusted close, use raw close (fine for split/dividend-free demo windows)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    return df


def _clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    before = len(df)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    dropped = before - len(df)
    if dropped:
        warnings.warn(f"[{ticker}] dropped {dropped} rows with missing/non-positive close price")

    if len(df) < 100:
        warnings.warn(
            f"[{ticker}] only {len(df)} usable rows of price history -- "
            "results (especially the Monte Carlo bands) will be wide. "
            "This is itself a real-world lesson: financial ML has small samples."
        )
    return df


def _load_yfinance(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise DataLoadError("yfinance is not installed") from e

    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as e:  # network errors, HTTP errors, etc.
        raise DataLoadError(f"yfinance download failed: {e}") from e

    if raw is None or raw.empty:
        raise DataLoadError(f"yfinance returned no data for {ticker}")

    # yfinance sometimes returns MultiIndex columns (ticker, field) for single downloads
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return _standardize_columns(raw)


def _load_local_csv(path, ticker_prefix: str | None) -> pd.DataFrame:
    raw = pd.read_csv(path)

    date_col = next((c for c in raw.columns if str(c).strip().lower() == "date"), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.set_index(date_col)

    if ticker_prefix:
        stripped = {c: c[len(ticker_prefix):] for c in raw.columns if str(c).startswith(ticker_prefix)}
        if stripped:
            raw = raw.rename(columns=stripped)

    return _standardize_columns(raw)


def load_price_data(
    ticker: str = None,
    start: str = None,
    end: str = None,
    source: str = None,
) -> pd.DataFrame:
    """
    Returns a clean daily OHLCV DataFrame, indexed by date (ascending),
    with columns: open, high, low, close, adj_close, volume.

    Parameters mirror config.py defaults when not supplied explicitly.
    """
    ticker = ticker or config.TICKER
    start = start or config.START_DATE
    end = end or config.END_DATE
    source = source or config.DATA_SOURCE

    if source == "yfinance":
        df = _load_yfinance(ticker, start, end)
        used_source = "yfinance (live)"

    elif source == "local_csv":
        df = _load_local_csv(config.LOCAL_CSV_PATH, config.LOCAL_CSV_TICKER_PREFIX)
        used_source = f"local CSV ({config.LOCAL_CSV_PATH.name})"

    elif source == "auto":
        try:
            df = _load_yfinance(ticker, start, end)
            used_source = "yfinance (live)"
        except DataLoadError as e:
            print(f"[data_loader] yfinance unavailable ({e}); falling back to local CSV.", file=sys.stderr)
            df = _load_local_csv(config.LOCAL_CSV_PATH, config.LOCAL_CSV_TICKER_PREFIX)
            used_source = f"local CSV fallback ({config.LOCAL_CSV_PATH.name})"
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {source}")

    df = _clean(df, ticker)

    print(
        f"[data_loader] {ticker}: loaded {len(df)} rows "
        f"({df.index.min().date()} -> {df.index.max().date()}) via {used_source}"
    )
    df.attrs["source"] = used_source
    df.attrs["ticker"] = ticker
    return df


if __name__ == "__main__":
    data = load_price_data()
    print(data.head())
    print(data.tail())
