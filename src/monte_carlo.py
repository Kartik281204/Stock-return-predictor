"""
monte_carlo.py
----------------
A single backtest equity curve is one draw from a distribution -- it tells
you almost nothing about how repeatable the result is (this is the crux of
Lopez de Prado's "backtest overfitting" critique). This module replaces
that single curve with two complementary Monte Carlo analyses:

1. BLOCK BOOTSTRAP
   Resample the realized daily (strategy, buy-and-hold) return PAIRS in
   contiguous blocks (to preserve autocorrelation / volatility clustering,
   unlike an i.i.d. bootstrap), many thousands of times, and look at the
   resulting distribution of outcomes -- not just the one path that
   happened to occur.

2. PERMUTATION / NULL TEST
   Randomly reshuffle the actual sequence of long/short/flat CALLS the
   model made (same calls, random order) and recompute performance. This
   answers: "does the *timing* of our signal carry information, or would
   any random ordering of the same bets have done just as well?" This is
   the same spirit as the classic warning that a coin-flip forecaster can
   post a >50% hit rate by chance alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# 1. Block bootstrap
# ---------------------------------------------------------------------------
def _circular_block_indices(n: int, block_size: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """(n_sims, n) matrix of resampled indices, built from circular blocks of length block_size."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_sims, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_sims, -1)[:, :n]


def _sharpe(returns_2d: np.ndarray, freq: int) -> np.ndarray:
    mu = returns_2d.mean(axis=1)
    sd = returns_2d.std(axis=1, ddof=1)
    out = np.full(len(mu), np.nan)
    mask = sd > 0
    out[mask] = mu[mask] / sd[mask] * np.sqrt(freq)
    return out


def _max_drawdown(equity_2d: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity_2d, axis=1)
    dd = equity_2d / running_max - 1
    return dd.min(axis=1)


def block_bootstrap(
    strategy_returns: np.ndarray,
    buy_hold_returns: np.ndarray,
    n_sims: int = None,
    block_size: int = None,
    seed: int = None,
):
    """
    Returns (summary_df, strategy_equity_paths, buy_hold_equity_paths).
    summary_df has one row per simulation with total return / Sharpe / max drawdown
    for both strategy and buy-and-hold, plus their excess. Equity paths are
    (n_sims, n_days) arrays for fan-chart plotting.
    """
    n_sims = n_sims or config.N_BOOTSTRAP_SIMS
    block_size = block_size or config.BLOCK_SIZE
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
    freq = config.TRADING_DAYS_PER_YEAR

    strat = np.asarray(strategy_returns, dtype=float)
    bh = np.asarray(buy_hold_returns, dtype=float)
    n = len(strat)
    if len(bh) != n:
        raise ValueError("strategy_returns and buy_hold_returns must be the same length")

    # Same block indices applied to BOTH series -> preserves the day-by-day pairing,
    # so "excess return" per simulation is a meaningful paired comparison.
    idx = _circular_block_indices(n, block_size, n_sims, rng)
    strat_paths = strat[idx]
    bh_paths = bh[idx]

    strat_equity = np.cumprod(1 + strat_paths, axis=1)
    bh_equity = np.cumprod(1 + bh_paths, axis=1)

    strat_total = strat_equity[:, -1] - 1
    bh_total = bh_equity[:, -1] - 1

    summary = pd.DataFrame({
        "strategy_total_return": strat_total,
        "buy_hold_total_return": bh_total,
        "excess_total_return": strat_total - bh_total,
        "strategy_sharpe": _sharpe(strat_paths, freq),
        "buy_hold_sharpe": _sharpe(bh_paths, freq),
        "strategy_max_dd": _max_drawdown(strat_equity),
        "buy_hold_max_dd": _max_drawdown(bh_equity),
    })

    prob_positive = float((strat_total > 0).mean())
    prob_beat_bh = float((summary["excess_total_return"] > 0).mean())
    print(f"[monte_carlo] block bootstrap: {n_sims} sims, block_size={block_size} days, horizon={n} days")
    print(f"[monte_carlo] P(strategy total return > 0)  = {prob_positive:.3f}")
    print(f"[monte_carlo] P(strategy beats buy & hold)  = {prob_beat_bh:.3f}")

    return summary, strat_equity, bh_equity


def mc_percentiles(summary: pd.DataFrame, cols=None, pcts=(5, 25, 50, 75, 95)) -> pd.DataFrame:
    cols = cols or list(summary.columns)
    return summary[cols].quantile([p / 100 for p in pcts]).set_axis([f"p{p}" for p in pcts])


# ---------------------------------------------------------------------------
# 2. Permutation / null test
# ---------------------------------------------------------------------------
def permutation_test(
    position: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost_bps: float = None,
    n_sims: int = None,
    seed: int = None,
):
    """
    Reshuffles the model's own realized position sequence (same number of long
    / short / flat calls, random order) to build a null distribution of
    performance, then reports where the ACTUAL (correctly-ordered) result
    ranks within that null distribution.
    """
    transaction_cost_bps = config.TRANSACTION_COST_BPS if transaction_cost_bps is None else transaction_cost_bps
    cost_rate = transaction_cost_bps / 10_000.0
    n_sims = n_sims or config.N_PERMUTATION_SIMS
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED + 1)
    freq = config.TRADING_DAYS_PER_YEAR

    pos = np.asarray(position, dtype=float)
    ret = np.asarray(actual_returns, dtype=float)
    n = len(pos)

    def _turnover(p2d):
        prev = np.zeros_like(p2d)
        prev[:, 1:] = p2d[:, :-1]
        return np.abs(p2d - prev)

    # Actual (correctly ordered) performance
    actual_2d = pos.reshape(1, -1)
    actual_returns_series = (actual_2d * ret.reshape(1, -1) - _turnover(actual_2d) * cost_rate)[0]
    actual_sharpe = float(_sharpe(actual_returns_series.reshape(1, -1), freq)[0])
    actual_total_return = float(np.prod(1 + actual_returns_series) - 1)

    # Null distribution: reshuffle the SAME calls into random order, many times
    rand_vals = rng.random((n_sims, n))
    perm_idx = np.argsort(rand_vals, axis=1)
    perm_pos = pos[perm_idx]
    perm_returns = perm_pos * ret[None, :] - _turnover(perm_pos) * cost_rate

    null_sharpe = _sharpe(perm_returns, freq)
    null_total_return = np.cumprod(1 + perm_returns, axis=1)[:, -1] - 1

    pctile_sharpe = float(np.nanmean(null_sharpe < actual_sharpe))
    pctile_return = float(np.nanmean(null_total_return < actual_total_return))

    print(f"[monte_carlo] permutation test: {n_sims} reshuffles of the realized {n}-day signal")
    print(f"[monte_carlo] actual Sharpe={actual_sharpe:.3f} -> {pctile_sharpe:.1%} percentile of random reshuffles "
          f"(50% = indistinguishable from a random ordering of the same bets)")

    return dict(
        actual_sharpe=actual_sharpe,
        actual_total_return=actual_total_return,
        null_sharpe=null_sharpe,
        null_total_return=null_total_return,
        percentile_sharpe=pctile_sharpe,
        percentile_return=pctile_return,
    )


if __name__ == "__main__":
    from . import data_loader, features, model, backtest

    raw = data_loader.load_price_data()
    feat_df, cols = features.engineer_features(raw)
    X_train, y_train, X_test, y_test, train_df, test_df = model.chronological_split(feat_df, cols)
    rf, params, _ = model.train_random_forest(X_train, y_train)
    preds = rf.predict(X_test)

    bt = backtest.run_backtest(test_df.index, preds, y_test.values)
    summary, strat_eq, bh_eq = block_bootstrap(bt["strategy_return"].values, bt["buy_hold_return"].values)
    print(mc_percentiles(summary))

    perm = permutation_test(bt["position"].values, bt["actual"].values)
