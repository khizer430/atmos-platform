"""
src/data/features.py
====================
Phase 2: Feature engineering.

Builds the model-ready feature matrix from `data/raw_data.csv`:
  - Cyclical encodings for hour-of-day and month (sin/cos).
  - Lag features for PM2.5 at 1, 2, 3, 6, 12, 24, 48 hours.
  - Rolling means for PM2.5 at 3h and 24h windows.

Leakage discipline
------------------
Every feature at time t depends only on data with timestamp <= t-1.
  - Lags use .shift(k) with k >= 1.
  - Rolling windows are applied AFTER .shift(1), so the "current" row's
    target value is never included in its own rolling window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH: Path = Path("data/raw_data.csv")
FEATURES_DATA_PATH: Path = Path("data/features.csv")

TARGET_COL: str = "pm2_5"
LAG_HOURS: list[int] = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS: list[int] = [3, 24]


# ---------------------------------------------------------------------------
# Cyclical encoding
# ---------------------------------------------------------------------------
def add_cyclical_features(df: pd.DataFrame, ts_col: str = "time") -> pd.DataFrame:
    """Add sin/cos encodings for hour-of-day and month.

    Both are circular quantities: 23:00 -> 00:00 is a one-hour step, not a
    23-hour step. Linear integer encodings make tree models split awkwardly
    around the wrap-around. Projecting onto a unit circle fixes it:
        sin(2*pi * value / period), cos(2*pi * value / period)
    """
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])

    hour = ts.dt.hour
    month = ts.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: list[int] = LAG_HOURS,
) -> pd.DataFrame:
    """Add lagged values of the target at the specified hour offsets.

    Every lag k >= 1, so the feature at time t is target(t - k): strictly the
    past. Using .shift(0) would leak the current target into the features.
    """
    df = df.copy()
    for k in lags:
        df[f"{target_col}_lag_{k}h"] = df[target_col].shift(k)
    return df


# ---------------------------------------------------------------------------
# Rolling means
# ---------------------------------------------------------------------------
def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add rolling means of the target over past windows.

    The trick here: we shift(1) first so the rolling window at time t covers
    {t-1, t-2, ..., t-window}. Without the shift, a 3h mean at time t would
    include t itself -> leakage.
    """
    df = df.copy()
    shifted = df[target_col].shift(1)
    for w in windows:
        df[f"{target_col}_rollmean_{w}h"] = shifted.rolling(window=w, min_periods=w).mean()
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline end-to-end and drop rows with NaNs.

    Dropping NaN rows is necessary because the largest lag is 48h (and the
    largest rolling window is 24h applied after shift(1)). Those early rows
    have no valid history to look back on. We intentionally drop them rather
    than impute: imputing lags would fabricate autocorrelation structure that
    doesn't exist in the raw data.
    """
    df = raw_df.sort_values("time").reset_index(drop=True)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("Dropped %d warm-up rows with incomplete lag/rolling history.", before - len(df))

    return df


# ---------------------------------------------------------------------------
# Convenience: load + build in one call
# ---------------------------------------------------------------------------
def load_and_build(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: read raw CSV, build features, return DataFrame."""
    raw_df = pd.read_csv(raw_path, parse_dates=["time"])
    return build_features(raw_df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Building features from %s", RAW_DATA_PATH)
    features_df = load_and_build()
    FEATURES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_DATA_PATH, index=False)
    logger.info("Saved %d rows × %d cols to %s", *features_df.shape, FEATURES_DATA_PATH)
    print(features_df.head(3).to_string(index=False))