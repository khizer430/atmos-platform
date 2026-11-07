"""
src/data/fetch_data.py
======================
Phase 1: Data ingestion.

Fetches Beijing 2023 hourly data from Open-Meteo:
  - Weather variables: temperature_2m, wind_speed_10m   (Historical Weather Archive API)
  - Air quality:       pm2_5, pm10                      (Air Quality API)

Two endpoints, two requests, merged on the timestamp column in pandas.
Output: data/raw_data.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BEIJING_LAT: float = 39.9042
BEIJING_LON: float = 116.4074
START_DATE: str = "2023-01-01"
END_DATE: str = "2023-12-31"

WEATHER_URL: str = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARIABLES: list[str] = ["temperature_2m", "wind_speed_10m"]
AIR_VARIABLES: list[str] = ["pm2_5", "pm10"]

REQUEST_TIMEOUT_SECONDS: int = 30
OUTPUT_PATH: Path = Path("data/raw_data.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def _request_hourly(
    url: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: list[str],
    timezone: str = "auto",
) -> dict:
    """Send a request to Open-Meteo and return the 'hourly' dict from the JSON.

    Parameters
    ----------
    url : str
        API endpoint (weather archive or air quality).
    latitude, longitude : float
        Beijing coordinates.
    start_date, end_date : str
        ISO date strings, YYYY-MM-DD.
    hourly_variables : list[str]
        Names of the hourly variables to request.
    timezone : str
        "auto" lets the API resolve the local timezone from the coordinates
        (Asia/Shanghai). Using "auto" instead of the default GMT is critical so
        both APIs share the same local-time axis.

    Returns
    -------
    dict
        {"time": [...], var1: [...], var2: [...], ...}
    """
    params: dict = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": timezone,
    }

    logger.info("Requesting %s | variables=%s", url, hourly_variables)
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    payload: dict = response.json()
    if "hourly" not in payload:
        raise KeyError(
            f"API response missing 'hourly' key. Keys received: {list(payload.keys())}"
        )

    hourly_data: dict = payload["hourly"]
    logger.info("Received %d hourly records", len(hourly_data.get("time", [])))
    return hourly_data


def fetch_weather() -> pd.DataFrame:
    """Fetch 2023 hourly temperature and wind speed for Beijing."""
    hourly = _request_hourly(
        url=WEATHER_URL,
        latitude=BEIJING_LAT,
        longitude=BEIJING_LON,
        start_date=START_DATE,
        end_date=END_DATE,
        hourly_variables=WEATHER_VARIABLES,
    )
    return pd.DataFrame(hourly)


def fetch_air_quality() -> pd.DataFrame:
    """Fetch 2023 hourly PM2.5 and PM10 for Beijing.

    Note: Open-Meteo's air quality API relies on the CAMS global model, which
    has coverage from August 2022 onward. 2023 is fully covered. If you ever
    query earlier dates, the PM variables will come back as null.
    """
    hourly = _request_hourly(
        url=AIR_QUALITY_URL,
        latitude=BEIJING_LAT,
        longitude=BEIJING_LON,
        start_date=START_DATE,
        end_date=END_DATE,
        hourly_variables=AIR_VARIABLES,
    )
    return pd.DataFrame(hourly)


def merge_datasets(weather_df: pd.DataFrame, air_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join weather and air quality on the time column, then tidy up.

    Inner join (not outer): both APIs should return an identical hourly axis
    for 2023 (365 days × 24 hours = 8760 rows). An inner join keeps only
    timestamps that exist in both sources, so a transient gap on one endpoint
    can't smuggle NaN rows into the feature pipeline.
    """
    weather_df = weather_df.copy()
    air_df = air_df.copy()
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    air_df["time"] = pd.to_datetime(air_df["time"])

    merged: pd.DataFrame = pd.merge(weather_df, air_df, on="time", how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)

    # Sanity checks
    expected_hours = 365 * 24
    if len(merged) != expected_hours:
        logger.warning(
            "Expected %d hourly rows for 2023, got %d. Check for missing data.",
            expected_hours,
            len(merged),
        )
    else:
        logger.info("Row count matches 2023 hourly expectation (%d).", expected_hours)

    null_counts = merged.isnull().sum()
    if null_counts.any():
        logger.warning("Null values detected:\n%s", null_counts[null_counts > 0])
    else:
        logger.info("No null values in merged dataset.")

    return merged


def save_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save to CSV, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows × %d cols to %s", *df.shape, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        logger.info("=== Phase 1: Fetching Beijing 2023 hourly data ===")
        weather_df = fetch_weather()
        air_df = fetch_air_quality()
        merged_df = merge_datasets(weather_df, air_df)
        save_to_csv(merged_df, OUTPUT_PATH)

        logger.info("Preview:")
        print(merged_df.head(3).to_string(index=False))
        logger.info("Done. Raw data ready at %s", OUTPUT_PATH)
        return 0
    except requests.RequestException as exc:
        logger.error("HTTP request failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())