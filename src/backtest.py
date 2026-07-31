"""
backtest.py
------------
Turns model predictions into a long/short trading strategy and evaluates it
against buy-and-hold on the SAME test-period dates.

Mechanics
---------
On the close of day t (using only information available through day t), the
model predicts the return from t to t+1. We take a position of:
    +1 (long)  if predicted_return >  deadband
    -1 (short) if predicted_return < -deadband
     0 (flat)  otherwise
and hold it for the t -> t+1 period, earning pos_t * actual_return_t.
Transaction costs are charged on every unit of position change (so flipping
from long to short costs 2x a flat-to-long trade, matching round-trip
notional turnover).

This module deliberately produces ONE realized equity curve. That single
path is exactly what monte_carlo.py argues you should not trust on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_positions(predicted_returns: pd.Series, deadband: float = None) -> pd.Series:
    deadband = config.SIGNAL_DEADBAND if deadband is None else deadband
    pos = pd.Series(0, index=predicted_returns.index, dtype=int)
    pos[predicted_returns > deadband] = 1
    pos[predicted_returns < -deadband] = -1
    return pos


def run_backtest(
    dates: pd.DatetimeIndex,
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost_bps: float = None,
    deadband: float = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by `dates` with columns:
      predicted, actual, position, turnover, cost, strategy_return, strategy_equity,
      buy_hold_return, buy_hold_equity
    """
    transaction_cost_bps = config.TRANSACTION_COST_BPS if transaction_cost_bps is None else transaction_cost_bps
    cost_rate = transaction_cost_bps / 10_000.0

    df = pd.DataFrame({
        "predicted": np.asarray(predicted_returns, dtype=float),
        "actual": np.asarray(actual_returns, dtype=float),
    }, index=pd.DatetimeIndex(dates))

    df["position"] = generate_positions(df["predicted"], deadband)
    prev_position = df["position"].shift(1).fillna(0)
    df["turnover"] = (df["position"] - prev_position).abs()
    df["cost"] = df["turnover"] * cost_rate
    df["strategy_return"] = df["position"] * df["actual"] - df["cost"]
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    df["buy_hold_return"] = df["actual"]
    bh_cost = pd.Series(0.0, index=df.index)
    bh_cost.iloc[0] = cost_rate  # one-time entry cost, for a fair apples-to-apples comparison
    df["buy_hold_return"] = df["buy_hold_return"] - bh_cost
    df["buy_hold_equity"] = (1 + df["buy_hold_return"]).cumprod()

    n_trades = int((df["turnover"] > 0).sum())
    print(f"[backtest] {len(df)} test days, {n_trades} position changes, "
          f"{transaction_cost_bps:.1f}bps/unit turnover cost, deadband={deadband or config.SIGNAL_DEADBAND}")
    print(f"[backtest] final strategy equity={df['strategy_equity'].iloc[-1]:.4f}  "
          f"final buy&hold equity={df['buy_hold_equity'].iloc[-1]:.4f}")

    return df


def performance_metrics(returns: pd.Series, freq: int = None, rf_annual: float = None) -> dict:
    """Standard backtest tear-sheet numbers for a daily return series."""
    freq = freq or config.TRADING_DAYS_PER_YEAR
    rf_annual = config.RISK_FREE_RATE_ANNUAL if rf_annual is None else rf_annual

    returns = pd.Series(returns).dropna()
    n = len(returns)
    if n == 0:
        return {}

    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    ann_return = float((1 + total_return) ** (freq / n) - 1)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(freq)) if n > 1 else float("nan")

    rf_daily = (1 + rf_annual) ** (1 / freq) - 1
    excess = returns - rf_daily
    sharpe = float(excess.mean() / returns.std(ddof=1) * np.sqrt(freq)) if returns.std(ddof=1) > 0 else float("nan")

    downside = returns[returns < 0]
    sortino = (float(excess.mean() / downside.std(ddof=1) * np.sqrt(freq))
               if len(downside) > 1 and downside.std(ddof=1) > 0 else float("nan"))

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else float("nan")

    win_rate = float((returns > 0).mean())

    return dict(
        n_days=n, total_return=total_return, annualized_return=ann_return,
        annualized_vol=ann_vol, sharpe=sharpe, sortino=sortino,
        max_drawdown=max_dd, calmar=calmar, win_rate=win_rate,
    )


def summarize(bt_df: pd.DataFrame) -> pd.DataFrame:
    strat = performance_metrics(bt_df["strategy_return"])
    bh = performance_metrics(bt_df["buy_hold_return"])
    summary = pd.DataFrame({"strategy": strat, "buy_and_hold": bh}).T
    return summary


if __name__ == "__main__":
    from . import data_loader, features, model

    raw = data_loader.load_price_data()
    feat_df, cols = features.engineer_features(raw)
    X_train, y_train, X_test, y_test, train_df, test_df = model.chronological_split(feat_df, cols)

    rf, params, _ = model.train_random_forest(X_train, y_train)
    preds = rf.predict(X_test)

    bt = run_backtest(test_df.index, preds, y_test.values)
    print(summarize(bt))
