"""
features.py
------------
Turns a raw OHLCV DataFrame into a model-ready feature matrix.

LEAKAGE DISCIPLINE (this is the part most tutorials get wrong -- see
Lopez de Prado's critique of naive financial ML):
  - Every feature at row t uses ONLY information available at or before the
    close of day t.
  - The target at row t is the return from close(t) to close(t+1) -- i.e.
    it is deliberately shifted forward, and is unusable as a feature.
  - Rolling/EWM windows use min_periods so partially-filled windows at the
    start of the series come out as NaN and get dropped, rather than
    silently computed on fewer days than intended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line / close, signal_line / close, hist / close  # normalized by price


def _bollinger(close: pd.Series, window: int, n_std: float):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    percent_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return percent_b, bandwidth


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Parameters
    ----------
    df : DataFrame with columns open, high, low, close, adj_close, volume (date index, ascending)

    Returns
    -------
    (feature_df, feature_cols)
      feature_df has all engineered columns + 'close' + 'daily_return' + 'target_next_return',
      with NaN warm-up / tail rows dropped.
      feature_cols is the explicit list of column names to feed the model (excludes
      close/daily_return/target/raw price-volume columns).
    """
    out = pd.DataFrame(index=df.index)
    close = df["adj_close"]  # use adjusted close throughout so splits/divs don't masquerade as returns
    out["close"] = close

    daily_return = close.pct_change()
    out["daily_return"] = daily_return

    feature_cols: list[str] = []

    # --- Moving averages: price relative to SMA, plus a fast/slow crossover ---
    for w in config.SMA_WINDOWS:
        sma = close.rolling(w, min_periods=w).mean()
        col = f"sma_{w}_ratio"
        out[col] = close / sma - 1.0
        feature_cols.append(col)

    if len(config.SMA_WINDOWS) >= 2:
        fast_w, slow_w = min(config.SMA_WINDOWS), max(config.SMA_WINDOWS)
        sma_fast = close.rolling(fast_w, min_periods=fast_w).mean()
        sma_slow = close.rolling(slow_w, min_periods=slow_w).mean()
        out["sma_crossover"] = (sma_fast - sma_slow) / close
        feature_cols.append("sma_crossover")

    # --- Lagged daily returns ---
    for lag in config.LAG_WINDOWS:
        col = f"return_lag_{lag}"
        out[col] = daily_return.shift(lag - 1)  # lag=1 -> today's already-realized return
        feature_cols.append(col)

    # --- RSI ---
    out["rsi"] = _rsi(close, config.RSI_PERIOD)
    feature_cols.append("rsi")

    # --- Rolling volatility (realized, annualized) ---
    for w in config.VOL_WINDOWS:
        col = f"vol_{w}"
        out[col] = daily_return.rolling(w, min_periods=w).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        feature_cols.append(col)

    # --- MACD ---
    macd_line, macd_signal, macd_hist = _macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    out["macd_line"], out["macd_signal"], out["macd_hist"] = macd_line, macd_signal, macd_hist
    feature_cols += ["macd_line", "macd_signal", "macd_hist"]

    # --- Bollinger Bands ---
    pct_b, bandwidth = _bollinger(close, config.BB_WINDOW, config.BB_STD)
    out["bb_percent_b"], out["bb_bandwidth"] = pct_b, bandwidth
    feature_cols += ["bb_percent_b", "bb_bandwidth"]

    # --- Momentum (N-day cumulative return) ---
    for w in config.MOMENTUM_WINDOWS:
        col = f"momentum_{w}"
        out[col] = close.pct_change(w)
        feature_cols.append(col)

    # --- Volume ---
    if "volume" in df.columns and df["volume"].notna().any():
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(20, min_periods=20).mean()
        out["volume_ratio"] = vol / vol_ma
        out["volume_change"] = vol.pct_change()
        feature_cols += ["volume_ratio", "volume_change"]

    # --- Calendar (cyclical day-of-week encoding; 5 trading days/week) ---
    dow = df.index.dayofweek.values.astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    feature_cols += ["dow_sin", "dow_cos"]

    # --- Target: next-day simple return (NOT a feature) ---
    out["target_next_return"] = daily_return.shift(-1)

    # Replace inf (can appear in ratios if a denominator hits ~0) then drop warm-up/tail NaNs
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(out)
    out = out.dropna(subset=feature_cols + ["target_next_return"])
    print(f"[features] engineered {len(feature_cols)} features; "
          f"{before} -> {len(out)} rows after dropping warm-up/tail NaNs")

    return out, feature_cols


if __name__ == "__main__":
    from . import data_loader

    raw = data_loader.load_price_data()
    feat_df, cols = engineer_features(raw)
    print(feat_df[cols].describe().T[["mean", "std", "min", "max"]])
