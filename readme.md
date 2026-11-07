# atmos-platform 🌫️

**End-to-end probabilistic forecasting and anomaly detection for Beijing PM2.5.**

`atmos-platform` is a production-grade MLOps pipeline demonstrating time-series forecasting, uncertainty quantification via Adaptive Conformal Inference, context-aware anomaly detection, and full-stack deployment. 

The system ingests a year of hourly weather and air quality data from Open-Meteo, builds cyclical and lag-based features, trains a LightGBM model wrapped in MAPIE for conformal intervals, and serves the results alongside IsolationForest anomaly flags via a FastAPI backend and a Streamlit dashboard.

## 🏗️ Architecture

```text
[ Open-Meteo API ]
       │
       ▼
[ Data Ingestion ]  (requests, pandas)
       │
       ▼
[ Feature Engineering ]  (sin/cos cyclical, lags, rolling means)
       │
       ├─────────────────────────────┐
       ▼                             ▼
[ LightGBM Forecaster ]       [ IsolationForest ]
       │                             │
       ▼                             ▼
[ MAPIE ACI (80% intervals) ] [ Context-Aware Flags ]
       │                             │
       └─────────────┬───────────────┘
                     ▼
               [ FastAPI ]
                     │
                     ▼
           [ Streamlit Dashboard ]