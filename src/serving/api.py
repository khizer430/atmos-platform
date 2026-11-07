"""
src/serving/api.py
==================
Phase 5: FastAPI serving layer.

Endpoints
---------
GET  /health              Liveness check.
POST /forecast            Stateless: client sends recent history, gets a
                          point forecast + conformal interval.
POST /anomalies           Stateless: client sends recent history, gets an
                          IsolationForest anomaly flag + context.
GET  /timeline            Batch: pre-computed forecasts + anomalies across a
                          date range. Used by the Streamlit dashboard so it
                          does not have to fire thousands of POST requests.

Design notes
------------
- Forecast features are rebuilt by calling the *same* ``build_features``
  function used during training. This prevents train/serve skew.
- The full-year timeline is pre-computed in the lifespan startup hook
  (~1 s cold start), so ``/timeline`` is a filter over an in-memory frame.
- The MAPIE wrapper already contains the underlying LightGBM estimator;
  loading the raw LightGBM pickle is unnecessary for inference.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.data.features import build_features
from src.models.train_anomaly import (
    ANOMALY_FEATURE_COLS,
    build_anomaly_features,
)
from src.models.train_forecast import ALPHA, FEATURE_COLS, TARGET_COL
from src.serving.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    ForecastRequest,
    ForecastResponse,
    TimelinePoint,
    TimelineResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR: Path = Path("artifacts")
MAPIE_PATH: Path = ARTIFACTS_DIR / "mapie_aci.pkl"
ISOFOREST_PATH: Path = ARTIFACTS_DIR / "iso_forest.pkl"
FEATURES_PATH: Path = Path("data/features.csv")
ANOMALIES_PATH: Path = Path("data/anomalies.csv")


# ---------------------------------------------------------------------------
# Lifespan: load artifacts + precompute the full timeline
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading artifacts ...")
    for p in (MAPIE_PATH, ISOFOREST_PATH, FEATURES_PATH, ANOMALIES_PATH):
        if not p.exists():
            raise RuntimeError(f"Required artifact missing: {p}. Run Phases 3 and 4 first.")

    app.state.mapie = joblib.load(MAPIE_PATH)
    app.state.iforest = joblib.load(ISOFOREST_PATH)

    logger.info("Precomputing timeline predictions ...")
    features_df = pd.read_csv(FEATURES_PATH, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    anomalies_df = pd.read_csv(ANOMALIES_PATH, parse_dates=["time"])

    X = features_df[FEATURE_COLS]
    y_pred, y_pis = app.state.mapie.predict(X, alpha=ALPHA)
    y_pred = np.asarray(y_pred).ravel()

    timeline = pd.DataFrame(
        {
            "time": features_df["time"].values,
            "actual": features_df[TARGET_COL].values,
            "predicted": y_pred,
            "lower": y_pis[:, 0, 0],
            "upper": y_pis[:, 1, 0],
        }
    ).merge(
        anomalies_df[["time", "is_anomaly", "anomaly_score"]], on="time", how="left"
    )
    app.state.timeline_df = timeline
    logger.info("API ready. Timeline rows: %d", len(timeline))
    yield


app = FastAPI(
    title="atmos-platform API",
    description=(
        "Probabilistic PM2.5 forecasting with Adaptive Conformal Inference "
        "intervals and context-aware IsolationForest anomaly detection."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _history_to_features(history) -> pd.DataFrame:
    """Convert the request history into a feature frame via the training pipeline."""
    df = pd.DataFrame([h.model_dump() for h in history])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    feat_df = build_features(df)
    if len(feat_df) == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Not enough history after feature engineering. "
                "Provide at least 49 continuous hourly observations ending at the target hour."
            ),
        )
    return feat_df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    feat_df = _history_to_features(req.history)
    target_row = feat_df.iloc[[-1]]
    X = target_row[FEATURE_COLS]

    y_pred, y_pis = app.state.mapie.predict(X, alpha=req.alpha)
    point = float(np.asarray(y_pred).ravel()[0])
    lower = float(y_pis[0, 0, 0])
    upper = float(y_pis[0, 1, 0])

    return ForecastResponse(
        timestamp=target_row["time"].iloc[0],
        point=point,
        lower=lower,
        upper=upper,
        alpha=req.alpha,
        coverage_level=1.0 - req.alpha,
    )


@app.post("/anomalies", response_model=AnomalyResponse)
def anomalies(req: AnomalyRequest) -> AnomalyResponse:
    feat_df = _history_to_features(req.history)
    feat_df = build_anomaly_features(feat_df)
    target_row = feat_df.iloc[[-1]]
    X = target_row[ANOMALY_FEATURE_COLS]

    score = float(app.state.iforest.score_samples(X)[0])
    flag = int(app.state.iforest.predict(X)[0])

    return AnomalyResponse(
        timestamp=target_row["time"].iloc[0],
        is_anomaly=(flag == -1),
        score=score,
        context={
            "pm2_5": float(target_row["pm2_5"].iloc[0]),
            "pm10": float(target_row["pm10"].iloc[0]),
            "wind_speed_10m": float(target_row["wind_speed_10m"].iloc[0]),
            "temperature_2m": float(target_row["temperature_2m"].iloc[0]),
            "pm2_5_dev_24h": float(target_row["pm2_5_dev_24h"].iloc[0]),
            "pm2_5_fine_fraction": float(target_row["pm2_5_fine_fraction"].iloc[0]),
        },
    )


@app.get("/timeline", response_model=TimelineResponse)
def timeline(start: str | None = None, end: str | None = None) -> TimelineResponse:
    """Batch endpoint for the dashboard.

    Optional ``start`` and ``end`` ISO date filters. Rows come from the
    pre-computed in-memory frame built during startup.
    """
    df = app.state.timeline_df
    if start:
        df = df[df["time"] >= pd.to_datetime(start)]
    if end:
        df = df[df["time"] <= pd.to_datetime(end)]

    if len(df) == 0:
        raise HTTPException(status_code=404, detail="No rows in requested range.")

    points = [
        TimelinePoint(
            time=row.time,
            actual=float(row.actual),
            predicted=float(row.predicted),
            lower=float(row.lower),
            upper=float(row.upper),
            is_anomaly=int(row.is_anomaly),
            anomaly_score=float(row.anomaly_score),
        )
        for row in df.itertuples(index=False)
    ]
    return TimelineResponse(points=points)


# ---------------------------------------------------------------------------
# Entry point for `python -m src.serving.api`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.api:app", host="127.0.0.1", port=8000, reload=False)