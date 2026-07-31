"""
model.py
---------
Two model families predicting next-day return, both trained with strict
chronological discipline (no shuffling, no test-set leakage):

  1. Random Forest  -- primary model. Hyperparameters are chosen by an
     expanding-window (sklearn TimeSeriesSplit) grid search *inside* the
     training block only; the test block is touched exactly once, after
     the model is frozen.
  2. LSTM (optional) -- secondary comparison model over sliding windows
     of the same engineered features. Requires torch; the pipeline
     degrades gracefully to RF-only if torch isn't installed.

Both models predict a continuous next-day return. Direction (sign) is what
backtest.py turns into a long/short position.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from . import config

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Chronological split (shared by both models)
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, feature_cols: list[str],
                         target_col: str = "target_next_return", train_fraction: float = None):
    train_fraction = train_fraction or config.TRAIN_FRACTION
    n = len(df)
    split_idx = int(n * train_fraction)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"[model] chronological split ({train_fraction:.0%} train): "
          f"train {train_df.index.min().date()}..{train_df.index.max().date()} ({len(train_df)} rows), "
          f"test {test_df.index.min().date()}..{test_df.index.max().date()} ({len(test_df)} rows)")

    return X_train, y_train, X_test, y_test, train_df, test_df


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Expanding-window grid search entirely within the training block.
    Returns (fitted_model, best_params, all_results).
    """
    tscv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS)
    grid = config.RF_PARAM_GRID
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            m = RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1, **params)
            m.fit(X_tr, y_tr)
            fold_scores.append(mean_squared_error(y_val, m.predict(X_val)))
        results.append((params, float(np.mean(fold_scores)), float(np.std(fold_scores))))

    results.sort(key=lambda r: r[1])
    best_params, best_mse, best_std = results[0]
    print(f"[model] RF expanding-window grid search: {len(combos)} combos x {config.CV_N_SPLITS} folds")
    print(f"[model] best params: {best_params}  (CV MSE={best_mse:.6e} +/- {best_std:.6e})")

    final_model = RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1, **best_params)
    final_model.fit(X_train, y_train)
    return final_model, best_params, results


def feature_importance(model: RandomForestRegressor, feature_cols: list[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# LSTM (optional secondary model)
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class LSTMRegressor(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def _build_sequences(scaled_df: pd.DataFrame, feature_cols: list[str], lookback: int,
                      target_col: str = "target_next_return"):
    X = scaled_df[feature_cols].to_numpy(dtype=np.float32)
    y = scaled_df[target_col].to_numpy(dtype=np.float32)
    seqs, targets, dates = [], [], []
    for i in range(lookback - 1, len(scaled_df)):
        seqs.append(X[i - lookback + 1: i + 1])
        targets.append(y[i])
        dates.append(scaled_df.index[i])
    return np.stack(seqs), np.array(targets, dtype=np.float32), pd.DatetimeIndex(dates)


def prepare_lstm_data(df: pd.DataFrame, feature_cols: list[str], split_date, lookback: int = None):
    """
    Scales features (fit on train rows only) then builds sliding-window sequences.
    Sequences whose *target date* falls before split_date are train; on/after -> test.
    A test-period sequence's INPUT WINDOW may reach back before split_date -- that is
    real, already-observed history, not leakage. Only the target must stay out-of-sample.
    """
    lookback = lookback or config.LSTM_LOOKBACK
    train_mask = df.index < split_date

    scaler = StandardScaler().fit(df.loc[train_mask, feature_cols])
    scaled = df.copy()
    scaled[feature_cols] = scaler.transform(df[feature_cols])

    X_seq, y_seq, idx_seq = _build_sequences(scaled, feature_cols, lookback)
    seq_train_mask = idx_seq < split_date

    return (X_seq[seq_train_mask], y_seq[seq_train_mask],
            X_seq[~seq_train_mask], y_seq[~seq_train_mask], idx_seq[~seq_train_mask], scaler)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray, n_features: int):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed -- LSTM path unavailable")

    torch.manual_seed(config.RANDOM_SEED)
    model = LSTMRegressor(n_features, config.LSTM_HIDDEN_SIZE, config.LSTM_NUM_LAYERS, config.LSTM_DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR, weight_decay=config.LSTM_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)

    # Chronological internal validation split for early stopping (no shuffling, ever)
    val_cut = int(len(X_t) * 0.85)
    X_tr, y_tr = X_t[:val_cut], y_t[:val_cut]
    X_val, y_val = X_t[val_cut:], y_t[val_cut:]

    best_val, best_state, bad_epochs, patience = float("inf"), None, 0, config.LSTM_EARLY_STOP_PATIENCE

    for epoch in range(config.LSTM_EPOCHS):
        model.train()
        for start in range(0, len(X_tr), config.LSTM_BATCH_SIZE):
            xb = X_tr[start:start + config.LSTM_BATCH_SIZE]
            yb = y_tr[start:start + config.LSTM_BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.LSTM_GRAD_CLIP)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val - 1e-9:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[model] LSTM early stopping at epoch {epoch + 1} (best val MSE={best_val:.6e})")
                break
    else:
        print(f"[model] LSTM trained full {config.LSTM_EPOCHS} epochs (best val MSE={best_val:.6e})")

    model.load_state_dict(best_state)
    return model


def predict_lstm(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Gu-Kelly-Xiu (2020) style out-of-sample R^2: benchmarked against a ZERO
    # forecast, not the historical mean (demeaning would leak look-ahead info).
    r2_oos_zero = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2))

    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_standard = float(1 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    hit_rate = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    metrics = dict(n=len(y_true), mse=mse, rmse=rmse, mae=mae,
                    r2_oos_zero_benchmark=r2_oos_zero, r2_standard=r2_standard, hit_rate=hit_rate)
    print(f"[model] {label} n={metrics['n']}  RMSE={rmse:.5f}  "
          f"R2_oos(0-benchmark)={r2_oos_zero:.5f}  HitRate={hit_rate:.3f}")
    return metrics


if __name__ == "__main__":
    from . import data_loader, features

    raw = data_loader.load_price_data()
    feat_df, cols = features.engineer_features(raw)
    X_train, y_train, X_test, y_test, train_df, test_df = chronological_split(feat_df, cols)

    rf, params, _ = train_random_forest(X_train, y_train)
    evaluate_predictions(y_test.values, rf.predict(X_test), label="RF test")
    print(feature_importance(rf, cols).head(10))
