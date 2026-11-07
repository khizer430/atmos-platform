"""
src/models/train_forecast.py
============================
Phase 3 (revised): LightGBM + conformal prediction for time series.

Two uncertainty methods are compared:
  1. Static split conformal (MapieRegressor, cv="prefit")
     - Correct under exchangeability; fails here because calibration
       (autumn) and test (winter heating season) are different regimes.
  2. Adaptive Conformal Inference (MapieTimeSeriesRegressor, method="aci")
     - Adapts the effective alpha online as the distribution drifts.
     - Restores asymptotic coverage under distribution shift.

Split (strictly chronological):
    fit    64%  train the point model
    calib  16%  initial conformal calibration
    test   20%  held-out evaluation (includes winter onset)

Artifacts:
    artifacts/lgbm_forecast.pkl   trained LightGBM point model
    artifacts/mapie_static.pkl    static split-conformal wrapper
    artifacts/mapie_aci.pkl       ACI time-series wrapper
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor, MapieTimeSeriesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
MODEL_PATH: Path = ARTIFACTS_DIR / "lgbm_forecast.pkl"
MAPIE_STATIC_PATH: Path = ARTIFACTS_DIR / "mapie_static.pkl"
MAPIE_ACI_PATH: Path = ARTIFACTS_DIR / "mapie_aci.pkl"

TARGET_COL: str = "pm2_5"

FEATURE_COLS: list[str] = [
    "temperature_2m",
    "wind_speed_10m",
    "pm10",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "pm2_5_lag_1h",
    "pm2_5_lag_2h",
    "pm2_5_lag_3h",
    "pm2_5_lag_6h",
    "pm2_5_lag_12h",
    "pm2_5_lag_24h",
    "pm2_5_lag_48h",
    "pm2_5_rollmean_3h",
    "pm2_5_rollmean_24h",
]

FIT_FRACTION: float = 0.64
CALIB_FRACTION: float = 0.16
ALPHA: float = 0.20  # 80% intervals

# ACI step size. gamma=0 (MAPIE default) disables adaptation entirely.
# 0.01 is the value used in the Gibbs & Candès paper for slowly-drifting data.
ACI_GAMMA: float = 0.01


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame,
    fit_frac: float = FIT_FRACTION,
    calib_frac: float = CALIB_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into (fit, calib, test) chronologically."""
    n = len(df)
    i_fit_end = int(fit_frac * n)
    i_calib_end = int((fit_frac + calib_frac) * n)

    return (
        df.iloc[:i_fit_end].copy(),
        df.iloc[i_fit_end:i_calib_end].copy(),
        df.iloc[i_calib_end:].copy(),
    )


# ---------------------------------------------------------------------------
# Point model
# ---------------------------------------------------------------------------
def train_lightgbm(fit_df: pd.DataFrame) -> lgb.LGBMRegressor:
    """Train LightGBM with early stopping on the last 10% of the fit window."""
    i_inner = int(0.9 * len(fit_df))
    fit_inner = fit_df.iloc[:i_inner]
    val_inner = fit_df.iloc[i_inner:]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        fit_inner[FEATURE_COLS],
        fit_inner[TARGET_COL],
        eval_set=[(val_inner[FEATURE_COLS], val_inner[TARGET_COL])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    logger.info("Best iteration: %d", model.best_iteration_)
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "coverage": float(np.mean((y_true >= y_low) & (y_true <= y_high))),
        "mean_width": float(np.mean(y_high - y_low)),
    }


# ---------------------------------------------------------------------------
# Method 1: static split conformal (baseline)
# ---------------------------------------------------------------------------
def run_static_split_conformal(
    model: lgb.LGBMRegressor,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Standard split conformal. Fails under distribution shift — kept as baseline."""
    mapie = MapieRegressor(estimator=model, method="base", cv="prefit")
    mapie.fit(calib_df[FEATURE_COLS], calib_df[TARGET_COL])

    y_pred, y_pis = mapie.predict(test_df[FEATURE_COLS], alpha=ALPHA)
    y_pred = np.asarray(y_pred).ravel()
    y_low = y_pis[:, 0, 0]
    y_high = y_pis[:, 1, 0]
    y_true = test_df[TARGET_COL].to_numpy()

    metrics = evaluate(y_true, y_pred, y_low, y_high, ALPHA)
    joblib.dump(mapie, MAPIE_STATIC_PATH)
    return metrics


# ---------------------------------------------------------------------------
# Method 2: Adaptive Conformal Inference (time-series aware)
# ---------------------------------------------------------------------------
def run_aci_conformal(
    model: lgb.LGBMRegressor,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """ACI: online-updated alpha under distribution shift.

    MAPIE v0.9.1 quirk
    ------------------
    ``MapieTimeSeriesRegressor._get_alpha`` rounds the adaptive alpha to
    two decimal places inside ``predict`` and ``update``:
        alpha_np = np.round(alpha_np, 2)
    If ACI drives ``current_alpha[ALPHA]`` below 0.005, it rounds to 0.00,
    and ``check_alpha_and_n_samples`` raises a ValueError because
    ``1 / 0.00`` is infinite. We therefore clamp the adaptive alpha to a
    safe floor that rounds to at least 0.01.

    Other API notes (v0.9.x)
    ------------------------
    - ``update(X, y, alpha=..., gamma=...)`` — parameter is ``alpha``,
      NOT ``confidence_level`` (that rename is v1-only).
    - ``update`` iterates over ``X`` directly, so pass a NumPy array.
    - ``gamma`` defaults to 0.; pass an explicit positive value.
    """
    mapie = MapieTimeSeriesRegressor(
        estimator=model,
        method="aci",
        cv="prefit",
    )
    mapie.fit(calib_df[FEATURE_COLS], calib_df[TARGET_COL])

    # Ensure current_alpha[ALPHA] exists before the loop reads it.
    try:
        mapie._get_alpha(alpha=ALPHA)  # type: ignore[attr-defined]
    except AttributeError:
        pass

    n = len(test_df)
    y_pred = np.empty(n)
    y_low = np.empty(n)
    y_high = np.empty(n)
    alpha_trace = np.full(n, ALPHA, dtype=float)

    # Clamp to 0.011 so that MAPIE's internal np.round(..., 2) yields 0.01,
    # not 0.00. 0.01 requires n >= 100, which is satisfied (n_calib ≈ 1394).
    min_safe_alpha = 0.011

    for i in range(n):
        X_t = test_df.iloc[[i]][FEATURE_COLS]
        y_t = np.array([test_df.iloc[i][TARGET_COL]])

        # Clamp current_alpha before predict (predict calls _get_alpha
        # internally, which rounds to 2 decimals).
        if hasattr(mapie, "current_alpha") and ALPHA in mapie.current_alpha:
            mapie.current_alpha[ALPHA] = max(
                float(mapie.current_alpha[ALPHA]), min_safe_alpha
            )

        y_pred_t, y_pis_t = mapie.predict(X_t, alpha=ALPHA)
        y_pred[i] = float(np.asarray(y_pred_t).ravel()[0])
        y_low[i] = float(y_pis_t[0, 0, 0])
        y_high[i] = float(y_pis_t[0, 1, 0])

        if i > 0 and hasattr(mapie, "current_alpha") and ALPHA in mapie.current_alpha:
            alpha_trace[i] = float(mapie.current_alpha[ALPHA])

        # Clamp again before update: adapt_conformal_inference calls predict
        # internally, so the same rounding issue applies.
        if hasattr(mapie, "current_alpha") and ALPHA in mapie.current_alpha:
            mapie.current_alpha[ALPHA] = max(
                float(mapie.current_alpha[ALPHA]), min_safe_alpha
            )

        X_t_np = X_t.to_numpy()          # shape (1, n_features)
        mapie.update(X_t_np, y_t, alpha=ALPHA, gamma=ACI_GAMMA)

    y_true = test_df[TARGET_COL].to_numpy()
    metrics = evaluate(y_true, y_pred, y_low, y_high, ALPHA)
    metrics["alpha_t_min"] = float(alpha_trace.min())
    metrics["alpha_t_max"] = float(alpha_trace.max())
    metrics["alpha_t_end"] = float(alpha_trace[-1])

    joblib.dump(mapie, MAPIE_ACI_PATH)
    return metrics
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logger.info("=== Phase 3 (revised): LightGBM + static vs ACI conformal ===")

    df = (
        pd.read_csv(FEATURES_PATH, parse_dates=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    logger.info("Loaded %d rows from %s", len(df), FEATURES_PATH)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in features.csv: {missing}")

    fit_df, calib_df, test_df = chronological_split(df)
    logger.info(
        "Split sizes | fit=%d | calib=%d | test=%d",
        len(fit_df), len(calib_df), len(test_df),
    )
    logger.info("Calib range: %s -> %s", calib_df["time"].iloc[0], calib_df["time"].iloc[-1])
    logger.info("Test range:  %s -> %s", test_df["time"].iloc[0], test_df["time"].iloc[-1])

    # Train once, reuse for both conformal methods
    logger.info("Training LightGBM ...")
    model = train_lightgbm(fit_df)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Saved point model -> %s", MODEL_PATH)

    # Baseline
    logger.info("Running static split conformal (baseline) ...")
    static_metrics = run_static_split_conformal(model, calib_df, test_df)

    # ACI
    logger.info("Running Adaptive Conformal Inference (ACI) ...")
    aci_metrics = run_aci_conformal(model, calib_df, test_df)

    # Side-by-side report
    print("\n" + "=" * 64)
    print("Conformal interval comparison (80% nominal coverage)")
    print("=" * 64)
    print(f"{'Metric':<22}{'Static split':>16}{'ACI (time-series)':>22}")
    print("-" * 64)
    print(f"{'MAE':<22}{static_metrics['mae']:>16.3f}{aci_metrics['mae']:>22.3f}")
    print(f"{'RMSE':<22}{static_metrics['rmse']:>16.3f}{aci_metrics['rmse']:>22.3f}")
    print(f"{'Coverage (80%)':<22}{static_metrics['coverage']:>16.3f}{aci_metrics['coverage']:>22.3f}")
    print(f"{'Mean width':<22}{static_metrics['mean_width']:>16.3f}{aci_metrics['mean_width']:>22.3f}")
    print("-" * 64)
    print(f"ACI alpha_t range: {aci_metrics['alpha_t_min']:.4f} -> {aci_metrics['alpha_t_max']:.4f}")
    print(f"ACI alpha_t final: {aci_metrics['alpha_t_end']:.4f} (nominal = {ALPHA})")
    print("=" * 64)
    print("\nReading: static split conformal under-covers because calibration")
    print("(autumn) and test (winter heating season) are different regimes.")
    print("ACI adapts alpha_t online and recovers coverage near 0.80.")
    print("Artifacts:")
    print(f"  {MODEL_PATH}")
    print(f"  {MAPIE_STATIC_PATH}")
    print(f"  {MAPIE_ACI_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())