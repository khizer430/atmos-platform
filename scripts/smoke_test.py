"""
scripts/smoke_test.py
=====================
End-to-end smoke test for the Phase 5 API.

Requires the API to be running at http://127.0.0.1:8000.

Exercises all four endpoints against real historical data from
``data/features.csv``:
    - GET  /health
    - POST /forecast    (built from 49 hours of history ending at a
                         known heavy-smog hour on 2023-12-19 00:00)
    - POST /anomalies   (same history)
    - GET  /timeline    (one-week window)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "http://127.0.0.1:8000"
FEATURES_PATH = Path("data/features.csv")
TARGET_TIME = pd.Timestamp("2023-12-19T00:00:00")
HISTORY_HOURS = 49


def _build_history() -> list[dict]:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["time"])
    hist = (
        df[df["time"] <= TARGET_TIME]
        .tail(HISTORY_HOURS)[
            ["time", "temperature_2m", "wind_speed_10m", "pm2_5", "pm10"]
        ]
        .copy()
    )
    hist["time"] = hist["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    # The target row's pm2_5 is the label, not a feature — set it to 0.
    hist.loc[hist.index[-1], "pm2_5"] = 0.0
    return hist.to_dict(orient="records")


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    # 1. /health
    _print_header("GET /health")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"HTTP {r.status_code}: {r.json()}")
    r.raise_for_status()

    history = _build_history()
    print(f"\nBuilt history payload: {len(history)} rows ending {history[-1]['time']}")

    # 2. /forecast
    _print_header("POST /forecast")
    r = requests.post(
        f"{BASE_URL}/forecast",
        json={"history": history, "alpha": 0.20},
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text)
        return 1
    fc = r.json()
    print(json.dumps(fc, indent=2))

    # 3. /anomalies
    _print_header("POST /anomalies")
    r = requests.post(
        f"{BASE_URL}/anomalies",
        json={"history": history},
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text)
        return 1
    an = r.json()
    print(json.dumps(an, indent=2))

    # 4. /timeline
    _print_header("GET /timeline (2023-12-15 -> 2023-12-21)")
    r = requests.get(
        f"{BASE_URL}/timeline",
        params={"start": "2023-12-15", "end": "2023-12-21"},
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    r.raise_for_status()
    points = r.json()["points"]
    print(f"Returned {len(points)} hourly points")
    print(f"First: {points[0]}")
    print(f"Last:  {points[-1]}")

    print("\nAll endpoints responded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())