"""
Central configuration for the Stock Return Predictor pipeline.

Edit these values to point at a different ticker, change feature windows,
or retune the model / backtest / Monte Carlo assumptions. Nothing else in
the codebase should hardcode these numbers -- they all read from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TICKER = "AAPL"

# 'yfinance'   -> pull live data (needs outbound internet access to Yahoo Finance)
# 'local_csv'  -> read LOCAL_CSV_PATH instead
# 'auto'       -> try yfinance, fall back to local_csv if that fails
DATA_SOURCE = "auto"
START_DATE = "2010-01-01"
END_DATE = None  # None = today

LOCAL_CSV_PATH = DATA_DIR / "aapl_plotly.csv"
LOCAL_CSV_TICKER_PREFIX = "AAPL."  # columns look like "AAPL.Close"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
SMA_WINDOWS = [5, 10, 20, 50]        # simple moving averages (days)
VOL_WINDOWS = [5, 10, 20]            # rolling realized-volatility windows (days)
LAG_WINDOWS = [1, 2, 3, 5, 10]       # lagged daily-return windows
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2            # Bollinger Bands
MOMENTUM_WINDOWS = [5, 10, 20]       # N-day price momentum

# ---------------------------------------------------------------------------
# Train / test split (chronological -- NEVER shuffled)
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70                # first 70% of rows -> train, rest -> test
CV_N_SPLITS = 5                      # expanding-window folds inside the TRAIN block only

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

RF_PARAM_GRID = {
    "n_estimators": [300],
    "max_depth": [3, 5, 8],
    "min_samples_leaf": [10, 20, 40],
    "max_features": ["sqrt", 0.5],
}

LSTM_LOOKBACK = 20        # trading days fed to the LSTM per sample
LSTM_HIDDEN_SIZE = 8      # small on purpose: ~300 training sequences can't support a big net
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.2
LSTM_WEIGHT_DECAY = 1e-2  # L2 -- chosen by internal validation MSE, not test performance
LSTM_EPOCHS = 80
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 32
LSTM_GRAD_CLIP = 1.0
LSTM_EARLY_STOP_PATIENCE = 12

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 5.0           # one-way cost in basis points, charged on position changes
SIGNAL_DEADBAND = 0.0                # |predicted return| must exceed this to take a position
RISK_FREE_RATE_ANNUAL = 0.0          # set >0 to compute excess-return Sharpe
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Monte Carlo evaluation
# ---------------------------------------------------------------------------
N_BOOTSTRAP_SIMS = 5000
BLOCK_SIZE = 15                      # ~3 trading weeks; preserves autocorrelation/vol clustering
N_PERMUTATION_SIMS = 2000            # null-hypothesis test: does the signal beat random signs?
