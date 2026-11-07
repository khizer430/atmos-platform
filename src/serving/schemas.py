"""
src/serving/schemas.py
======================
Pydantic request/response models for the atmos-platform API.

Explicit schemas serve three purposes:
  1. Automatic OpenAPI docs at /docs.
  2. Type-safe validation before any inference runs.
  3. A readable contract that a dashboard developer can code against.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """One hourly observation of raw weather + air-quality data."""

    time: datetime
    temperature_2m: float
    wind_speed_10m: float
    pm2_5: float = Field(
        ...,
        description=(
            "PM2.5 in ug/m3. May be 0 on the *target* row "
            "(the last observation); it is not used as a feature."
        ),
    )
    pm10: float


class ForecastRequest(BaseModel):
    history: list[Observation] = Field(
        ...,
        min_length=49,
        description=(
            "At least 49 hourly observations, sorted ascending, ending at the "
            "target hour. 48 preceding hours are required to populate lag and "
            "rolling features; the 49th is the row being predicted."
        ),
    )
    alpha: float = Field(
        0.20,
        gt=0.0,
        lt=1.0,
        description="Miscoverage rate. 0.20 => 80% prediction intervals.",
    )


class ForecastResponse(BaseModel):
    timestamp: datetime
    point: float
    lower: float
    upper: float
    alpha: float
    coverage_level: float


class AnomalyRequest(BaseModel):
    history: list[Observation] = Field(..., min_length=49)


class AnomalyResponse(BaseModel):
    timestamp: datetime
    is_anomaly: bool
    score: float = Field(
        ...,
        description="IsolationForest score_samples — lower is more anomalous.",
    )
    context: dict[str, float]


class TimelinePoint(BaseModel):
    time: datetime
    actual: float
    predicted: float
    lower: float
    upper: float
    is_anomaly: int
    anomaly_score: float


class TimelineResponse(BaseModel):
    points: list[TimelinePoint]