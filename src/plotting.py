"""
plotting.py
------------
All chart generation lives here so main.py stays about orchestration, not
matplotlib boilerplate. One consistent visual style throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLOR_STRATEGY = "#2E5EAA"     # deep blue
COLOR_BUYHOLD = "#E8873A"      # warm amber
COLOR_ACCENT_POS = "#2A9D6F"   # green
COLOR_ACCENT_NEG = "#C0392B"   # muted red
COLOR_GRID = "#E4E4E4"
COLOR_TEXT = "#2B2B2B"
COLOR_MUTED = "#9099A2"
BG = "#FCFCFB"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#CFCFCF",
    "axes.labelcolor": COLOR_TEXT,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def _pct_fmt(ax, axis="y"):
    fmt = mticker.PercentFormatter(xmax=1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[plotting] saved {save_path}")


# ---------------------------------------------------------------------------
def plot_price_overview(price_df: pd.DataFrame, ticker: str, save_path):
    close = price_df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = price_df["rsi"] if "rsi" in price_df.columns else None

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.4, 1]})

    ax = axes[0]
    ax.plot(close.index, close.values, color=COLOR_TEXT, lw=1.3, label="Adj. close")
    ax.plot(sma20.index, sma20.values, color=COLOR_STRATEGY, lw=1.1, label="SMA 20")
    ax.plot(sma50.index, sma50.values, color=COLOR_BUYHOLD, lw=1.1, label="SMA 50")
    ax.set_title(f"{ticker} -- price history and trend features")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left", ncol=3)

    ax2 = axes[1]
    if rsi is not None:
        ax2.plot(rsi.index, rsi.values, color="#6B4FA0", lw=1.1)
        ax2.axhline(70, color=COLOR_ACCENT_NEG, lw=0.8, ls="--", alpha=0.7)
        ax2.axhline(30, color=COLOR_ACCENT_POS, lw=0.8, ls="--", alpha=0.7)
        ax2.fill_between(rsi.index, 30, 70, color=COLOR_GRID, alpha=0.4)
        ax2.set_ylabel("RSI(14)")
        ax2.set_ylim(0, 100)
    ax2.set_xlabel("Date")

    _save(fig, save_path)


def plot_feature_importance(importance: pd.Series, save_path, top_n: int = 15):
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = [COLOR_STRATEGY if v == top.max() else "#7C9FD1" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.set_title("Random Forest -- feature importance (top {})".format(top_n))
    ax.set_xlabel("Mean decrease in impurity")
    _save(fig, save_path)


def plot_predictions_vs_actual(dates, y_true, y_pred_rf, y_pred_lstm=None, save_path=None):
    n_panels = 2 if y_pred_lstm is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    axes = np.atleast_1d(axes)

    def _panel(ax, pred, label):
        ax.scatter(y_true, pred, s=14, alpha=0.55, color=COLOR_STRATEGY, edgecolor="none")
        lim = max(np.abs(y_true).max(), np.abs(pred).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=COLOR_MUTED, lw=1, ls="--", label="perfect prediction")
        ax.axhline(0, color=COLOR_GRID, lw=1)
        ax.axvline(0, color=COLOR_GRID, lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        _pct_fmt(ax, "x")
        _pct_fmt(ax, "y")
        ax.set_xlabel("Actual next-day return")
        ax.set_ylabel("Predicted next-day return")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=9)

    _panel(axes[0], y_pred_rf, "Random Forest -- test set")
    if y_pred_lstm is not None:
        _panel(axes[1], y_pred_lstm, "LSTM -- test set")

    fig.suptitle("Predicted vs. actual next-day return (out-of-sample)", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, save_path)


def plot_equity_curve(bt_df: pd.DataFrame, model_name: str, save_path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(bt_df.index, bt_df["buy_hold_equity"], color=COLOR_BUYHOLD, lw=1.8, label="Buy & hold")
    ax.plot(bt_df.index, bt_df["strategy_equity"], color=COLOR_STRATEGY, lw=1.8, label=f"{model_name} long/short")
    ax.axhline(1.0, color=COLOR_MUTED, lw=0.8, ls=":")

    long_mask = bt_df["position"] == 1
    short_mask = bt_df["position"] == -1
    ymin, ymax = ax.get_ylim()
    ax.fill_between(bt_df.index, ymin, ymax, where=short_mask, color=COLOR_ACCENT_NEG, alpha=0.06, step="mid")
    ax.set_ylim(ymin, ymax)

    ax.set_title(f"Realized equity curve (single path, test period) -- {model_name} vs. buy & hold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.text(0.5, -0.02, "shaded bands = days the model was short", ha="center", fontsize=8.5, color=COLOR_MUTED)
    _save(fig, save_path)


def plot_monte_carlo_fan(dates, strat_paths, bh_paths, actual_strat_equity, actual_bh_equity, save_path):
    pcts = [5, 25, 50, 75, 95]
    strat_q = np.percentile(strat_paths, pcts, axis=0)
    bh_q = np.percentile(bh_paths, pcts, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    def _panel(ax, q, actual, color, label):
        ax.fill_between(dates, q[0], q[4], color=color, alpha=0.12, label="5th-95th pct (bootstrap)")
        ax.fill_between(dates, q[1], q[3], color=color, alpha=0.25, label="25th-75th pct (bootstrap)")
        ax.plot(dates, q[2], color=color, lw=1.2, ls="--", alpha=0.9, label="bootstrap median")
        ax.plot(dates, actual, color=COLOR_TEXT, lw=2.0, label="actual realized path")
        ax.axhline(1.0, color=COLOR_MUTED, lw=0.8, ls=":")
        ax.set_title(label)
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", fontsize=8.5)

    _panel(axes[0], strat_q, actual_strat_equity, COLOR_STRATEGY, "Strategy: block-bootstrapped equity paths")
    _panel(axes[1], bh_q, actual_bh_equity, COLOR_BUYHOLD, "Buy & hold: block-bootstrapped equity paths")
    axes[0].set_ylabel("Growth of $1")

    fig.suptitle(f"Monte Carlo resampling ({strat_paths.shape[0]:,} block-bootstrap simulations) "
                 "-- the realized path is one draw among many plausible ones",
                 fontsize=12.5, fontweight="bold", y=1.03)
    _save(fig, save_path)


def plot_mc_distribution(summary: pd.DataFrame, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = 60
    ax.hist(summary["buy_hold_total_return"], bins=bins, color=COLOR_BUYHOLD, alpha=0.55, label="Buy & hold")
    ax.hist(summary["strategy_total_return"], bins=bins, color=COLOR_STRATEGY, alpha=0.55, label="Strategy")
    ax.axvline(0, color=COLOR_MUTED, lw=1, ls=":")
    _pct_fmt(ax, "x")
    ax.set_xlabel("Simulated total return (test-period horizon)")
    ax.set_ylabel("Simulations")
    ax.set_title("Distribution of total return")
    ax.legend()

    ax2 = axes[1]
    ax2.hist(summary["excess_total_return"], bins=bins, color="#6B4FA0", alpha=0.7)
    ax2.axvline(0, color=COLOR_ACCENT_NEG, lw=1.4, ls="--", label="break-even vs. buy & hold")
    prob_beat = float((summary["excess_total_return"] > 0).mean())
    ax2.set_title(f"Strategy minus buy & hold  (P(beat) = {prob_beat:.1%})")
    _pct_fmt(ax2, "x")
    ax2.set_xlabel("Excess total return")
    ax2.legend()

    fig.suptitle("Monte Carlo outcome distributions (block bootstrap)", fontsize=12.5, fontweight="bold", y=1.03)
    _save(fig, save_path)


def plot_permutation_test(perm_result: dict, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    null = perm_result["null_sharpe"]
    null = null[np.isfinite(null)]
    ax.hist(null, bins=60, color=COLOR_MUTED, alpha=0.75,
            label="null: same calls, random order")
    ax.axvline(perm_result["actual_sharpe"], color=COLOR_ACCENT_NEG, lw=2.2,
               label=f"actual signal (Sharpe={perm_result['actual_sharpe']:.2f}, "
                     f"{perm_result['percentile_sharpe']:.0%} pct.)")
    ax.axvline(0, color=COLOR_TEXT, lw=0.8, ls=":")
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel("Reshuffled simulations")
    ax.set_title("Permutation test -- does the ORDER of the model's calls carry information?")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, save_path)
