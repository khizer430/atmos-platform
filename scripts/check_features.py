"""
scripts/check_features.py
=========================
Sanity-check the Phase 2 feature pipeline. Verifies:
  1. Expected column set exists.
  2. Row count is raw_rows - 48 (the largest lag consumes 48 warm-up rows).
  3. No NaNs remain in the feature frame.
  4. Leakage guard: for a random row, the lag-1 feature must equal the
     previous row's target, not the current row's target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from project root when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import LAG_HOURS, ROLLING_WINDOWS, TARGET_COL, build_features

RAW_PATH = Path("data/raw_data.csv")


def main() -> int:
    raw_df = pd.read_csv(RAW_PATH, parse_dates=["time"])
    print(f"Raw shape: {raw_df.shape}")

    feat_df = build_features(raw_df)
    print(f"Feature shape: {feat_df.shape}")

    # 1. Expected columns
    expected_cols = (
        ["time", "temperature_2m", "wind_speed_10m", "pm2_5", "pm10"]
        + ["hour_sin", "hour_cos", "month_sin", "month_cos"]
        + [f"{TARGET_COL}_lag_{k}h" for k in LAG_HOURS]
        + [f"{TARGET_COL}_rollmean_{w}h" for w in ROLLING_WINDOWS]
    )
    missing = [c for c in expected_cols if c not in feat_df.columns]
    extra = [c for c in feat_df.columns if c not in expected_cols]
    assert not missing, f"Missing expected columns: {missing}"
    assert not extra, f"Unexpected extra columns: {extra}"
    print(f"[OK] Column set matches ({len(expected_cols)} columns).")

    # 2. Row count: max lag = 48, so 48 warm-up rows dropped.
    assert len(feat_df) == len(raw_df) - 48, (
        f"Expected {len(raw_df) - 48} rows after dropping warm-up, got {len(feat_df)}."
    )
    print(f"[OK] Row count = raw - 48 = {len(feat_df)}.")

    # 3. No NaNs
    assert feat_df.isna().sum().sum() == 0, "NaN values present after dropna."
    print("[OK] No NaNs in feature frame.")

    # 4. Leakage guard on lag_1: feature at index i == target at index i-1
    #    (feature frame already dropped the first 48 rows, so align by shift)
    n_leak_checked = 0
    for i in range(1, min(200, len(feat_df))):
        prev_target = feat_df.iloc[i - 1][TARGET_COL]
        lag1 = feat_df.iloc[i][f"{TARGET_COL}_lag_1h"]
        if not np.isclose(prev_target, lag1):
            raise AssertionError(
                f"Leakage detected at row {i}: lag_1h={lag1} != prev_target={prev_target}"
            )
        n_leak_checked += 1
    print(f"[OK] lag_1h matches previous target on {n_leak_checked} sampled rows.")

    # 5. A quick peek at the head
    print("\nFirst 3 rows (subset of columns):")
    cols = ["time", "pm2_5", "pm2_5_lag_1h", "pm2_5_lag_24h", "pm2_5_rollmean_24h"]
    print(feat_df[cols].head(3).to_string(index=False))

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())