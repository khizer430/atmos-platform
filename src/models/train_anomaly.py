"""
src/models/train_anomaly.py
===========================
Phase 4: Context-aware anomaly detection with IsolationForest.

Design rationale
----------------
Flagging raw PM2.5 alone would mark most of Beijing's winter as "anomalous" —
that is Tuesday, not an event. Instead the forest is fed context features:

  - pm2_5                : raw level (anchor)
  - pm2_5_dev_24h        : pm2_5 - pm2_5_lag_24h (sudden deviation)
  - pm2_5_fine_fraction  : pm2_5 / pm10 (secondary aerosol vs dust)
  - wind_speed_10m       : ventilation context
  - temperature_2m       : inversion / seasonal context
  - hour_sin, hour_cos   : diurnal context

A value of 150 ug/m3 under calm winter air with a large 24h jump is a
different event than 150 ug/m3 during a dust storm. IsolationForest sees
both the level and its context.

Fit discipline
--------------
The forest is trained on the FIT split only (Jan-Aug), mirroring a system
deployed in August 2023. The test period (winter heating onset) is genuinely
unseen. This intentionally produces a higher anomaly rate on the test set,
which is the correct behaviour for an early-warning system.

Contamination prior
-------------------
``contamination=0.02`` encodes a domain prior: roughly 2-3% of Beijing
winter hours are genuine smog events (PM2.5 > 200 ug/m3). sklearn 1.x maps
``contamination="auto"`` to 0.10, which flags far too many hours and
destroys signal-to-noise.

Artifacts
---------
    artifacts/iso_forest.pkl   fitted IsolationForest
    data/anomalies.csv         per-hour anomaly score + flag for the dashboard
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURES_PATH: Path = Path("data/features.csv")
ARTIFACTS_DIR: Path = Path("artifacts")
MODEL_PATH: Path = ARTIFACTS_DIR / "iso_forest.pkl"
ANOMALIES_PATH: Path = Path("data/anomalies.csv")

TARGET_COL: str = "pm2_5"
FIT_FRACTION: float = 0.64           # same split as Phase 3 for consistency
CONTAMINATION: float = 0.02
RANDOM_STATE: int = 42
EPS: float = 1e-6                    # guard against div-by-zero in the ratio

ANOMALY_FEATURE_COLS: list[str] = [
    "pm2_5",
    "pm2_5_dev_24h",
    "pm2_5_fine_fraction",
    "wind_speed_10m",
    "temperature_2m",
    "hour_sin",
    "hour_cos",
]


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the context features used by IsolationForest.

    This function is also imported by the serving layer (Phase 5) so that the
    training-time and serving-time feature contract cannot drift apart.
    """
    out = df.copy()
    out["pm2_5_dev_24h"] = out["pm2_5"] - out["pm2_5_lag_24h"]
    out["pm2_5_fine_fraction"] = out["pm2_5"] / (out["pm10"] + EPS)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_isolation_forest(fit_df: pd.DataFrame) -> IsolationForest:
    """Fit IsolationForest on the training split only.

    Note on scaling: IsolationForest is invariant to monotone feature
    transforms (its splits are per-feature thresholds), so StandardScaler
    is not required. This is documented deliberately so a reviewer doesn't
    assume the omission was an oversight.
    """
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=CONTAMINATION,
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(fit_df[ANOMALY_FEATURE_COLS])
    logger.info(
        "Fitted IsolationForest on %d rows | contamination=%.2f",
        len(fit_df), CONTAMINATION,
    )
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_dataset(
    model: IsolationForest,
    full_df: pd.DataFrame,
    fit_end_time: pd.Timestamp,
) -> pd.DataFrame:
    """Score every row and attach a 'split' column (fit / test) for reporting."""
    scores = model.score_samples(full_df[ANOMALY_FEATURE_COLS])   # higher = normal
    flags = model.predict(full_df[ANOMALY_FEATURE_COLS])          # -1 anomaly, +1 normal

    out = pd.DataFrame(
        {
            "time": full_df["time"].values,
            "pm2_5": full_df["pm2_5"].values,
            "pm10": full_df["pm10"].values,
            "wind_speed_10m": full_df["wind_speed_10m"].values,
            "temperature_2m": full_df["temperature_2m"].values,
            "pm2_5_dev_24h": full_df["pm2_5_dev_24h"].values,
            "pm2_5_fine_fraction": full_df["pm2_5_fine_fraction"].values,
            "anomaly_score": scores,
            "is_anomaly": (flags == -1).astype(int),
            "split": np.where(
                full_df["time"] <= fit_end_time, "fit", "test"
            ),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(anomalies_df: pd.DataFrame) -> None:
    """Print per-split anomaly rates and the top-10 highest-scoring events."""
    print("\n" + "=" * 72)
    print("Anomaly detection summary")
    print("=" * 72)
    for split in ["fit", "test"]:
        sub = anomalies_df[anomalies_df["split"] == split]
        rate = sub["is_anomaly"].mean()
        print(
            f"{split:<6} | rows={len(sub):>5} | "
            f"anomalies={int(sub['is_anomaly'].sum()):>4} | rate={rate:.3%}"
        )
    print("-" * 72)

    top = (
        anomalies_df.sort_values("anomaly_score")
        .head(10)
        .loc[
            :,
            [
                "time", "pm2_5", "wind_speed_10m",
                "temperature_2m", "pm2_5_dev_24h", "anomaly_score",
            ],
        ]
    )
    print("Top-10 most anomalous hours (lowest score_samples = most anomalous):")
    print(top.to_string(index=False))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logger.info("=== Phase 4: Context-aware anomaly detection ===")

    df = (
        pd.read_csv(FEATURES_PATH, parse_dates=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    logger.info("Loaded %d rows from %s", len(df), FEATURES_PATH)

    df = build_anomaly_features(df)

    # Fit on the earliest 64% only — same boundary as Phase 3.
    i_fit_end = int(FIT_FRACTION * len(df))
    fit_df = df.iloc[:i_fit_end].copy()
    fit_end_time = fit_df["time"].iloc[-1]
    logger.info(
        "Fit split: %s -> %s (%d rows)",
        fit_df["time"].iloc[0], fit_end_time, len(fit_df),
    )

    model = train_isolation_forest(fit_df)

    # Score the full dataset so the dashboard can render the entire year.
    anomalies_df = score_dataset(model, df, fit_end_time)

    # Persist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Saved IsolationForest -> %s", MODEL_PATH)

    ANOMALIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    anomalies_df.to_csv(ANOMALIES_PATH, index=False)
    logger.info("Saved scored anomalies -> %s", ANOMALIES_PATH)

    print_summary(anomalies_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())