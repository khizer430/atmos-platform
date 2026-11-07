"""
dashboard/app.py
================
Phase 6: Streamlit monitoring dashboard for atmos-platform.

Consumes the FastAPI backend at http://127.0.0.1:8000 (see Phase 5).
Renders a single-window view with:
  - Actual vs. forecasted PM2.5.
  - 80% Adaptive Conformal interval as a shaded band.
  - IsolationForest anomaly flags as red markers on the actual line.
  - Anomaly score trace below, so flagged points can be traced to their cause.
  - Window-level coverage metrics so the interval width is auditable.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30

# Data coverage (features.csv starts 2023-01-03 after warm-up drop)
MIN_DATE = date(2023, 1, 3)
MAX_DATE = date(2023, 12, 31)

# Dark-mode optimized colors
COLOR_ACTUAL = "#f3f4f6"       # Off-white
COLOR_FORECAST = "#60a5fa"     # Bright blue
COLOR_BAND = "rgba(96, 165, 250, 0.15)"
COLOR_ANOMALY = "#ef4444"      # Bright red
COLOR_SCORE = "#9ca3af"        # Light grey


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_timeline(start: str, end: str) -> pd.DataFrame:
    """Fetch the timeline window from the API. Cached for 60 s per (start, end)."""
    resp = requests.get(
        f"{API_BASE}/timeline",
        params={"start": start, "end": end},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["points"])
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def api_is_alive() -> bool:
    try:
        requests.get(f"{API_BASE}/health", timeout=3).raise_for_status()
        return True
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="atmos-platform · PM2.5 Monitoring",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-size: 1.55rem !important; margin-bottom: 0.15rem !important; }
    .subtitle { color: #9ca3af; font-size: 0.85rem; margin-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    hr { margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("atmos-platform · Beijing PM2.5 Monitoring")
st.markdown(
    '<div class="subtitle">LightGBM forecast · Adaptive Conformal 80% interval · '
    "Context-aware IsolationForest anomaly detection</div>",
    unsafe_allow_html=True,
)

if not api_is_alive():
    st.error(
        f"Cannot reach the API at **{API_BASE}**.\n\n"
        "Start it in another terminal:\n\n"
        "```\npython -m src.serving.api\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: date range + presets
# ---------------------------------------------------------------------------
st.sidebar.header("View")

if "start" not in st.session_state:
    st.session_state.start = date(2023, 12, 15)
    st.session_state.end = date(2023, 12, 31)


def _set_range(new_start: date, new_end: date) -> None:
    st.session_state.start = new_start
    st.session_state.end = new_end


st.sidebar.caption("Presets")
col_a, col_b, col_c = st.sidebar.columns(3)
if col_a.button("Winter", use_container_width=True):
    _set_range(date(2023, 11, 1), date(2023, 12, 31))
if col_b.button("Year", use_container_width=True):
    _set_range(MIN_DATE, MAX_DATE)
if col_c.button("Event", use_container_width=True):
    _set_range(date(2023, 12, 17), date(2023, 12, 21))

start_date = st.sidebar.date_input(
    "Start",
    value=st.session_state.start,
    min_value=MIN_DATE,
    max_value=MAX_DATE,
)
end_date = st.sidebar.date_input(
    "End",
    value=st.session_state.end,
    min_value=MIN_DATE,
    max_value=MAX_DATE,
)

if start_date > end_date:
    st.sidebar.error("Start must be before End.")
    st.stop()

if (end_date - start_date).days > 365:
    st.sidebar.error("Window too large (max 365 days).")
    st.stop()

if st.sidebar.button("Refresh now", use_container_width=True):
    fetch_timeline.clear()


# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
try:
    with st.spinner("Loading window from API ..."):
        df = fetch_timeline(start_date.isoformat(), end_date.isoformat())
except requests.RequestException as exc:
    st.error(f"API error: {exc}")
    st.stop()

if df.empty:
    st.warning("No data in the selected window.")
    st.stop()

# Display-side clip of the lower CI at 0 (physically non-negative target).
df["lower_display"] = df["lower"].clip(lower=0)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
coverage = ((df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])).mean()
below_ci = int((df["actual"] < df["lower"]).sum())
above_ci = int((df["actual"] > df["upper"]).sum())
n_anomalies = int(df["is_anomaly"].sum())
mean_width = float((df["upper"] - df["lower"]).mean())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hours", f"{len(df):,}")
c2.metric("Mean PM2.5 (obs)", f"{df['actual'].mean():.1f} µg/m³")
c3.metric("Mean PM2.5 (fcst)", f"{df['predicted'].mean():.1f} µg/m³")
c4.metric("80% coverage", f"{coverage * 100:.1f}%", help="Fraction of hours where actual fell inside the 80% conformal interval.")
c5.metric("Anomalies", f"{n_anomalies:,}", help="Hours flagged by the context-aware IsolationForest.")

st.caption(
    f"Interval width: **{mean_width:.1f} µg/m³** average · "
    f"Under-coverage breakdown: **{below_ci:,}** below · **{above_ci:,}** above."
)


# ---------------------------------------------------------------------------
# Main chart: PM2.5 + interval + anomalies
# ---------------------------------------------------------------------------
anomalies = df[df["is_anomaly"] == 1]

fig = go.Figure()

# Confidence band (upper line invisible, lower line fills to it)
fig.add_trace(
    go.Scatter(
        x=df["time"], y=df["upper"],
        mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    )
)
fig.add_trace(
    go.Scatter(
        x=df["time"], y=df["lower_display"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=COLOR_BAND,
        name="80% interval", hoverinfo="skip",
    )
)

# Actual
fig.add_trace(
    go.Scatter(
        x=df["time"], y=df["actual"],
        mode="lines", name="Actual",
        line=dict(color=COLOR_ACTUAL, width=1.5),
    )
)

# Forecast
fig.add_trace(
    go.Scatter(
        x=df["time"], y=df["predicted"],
        mode="lines", name="Forecast",
        line=dict(color=COLOR_FORECAST, width=1.8, dash="dash"),
    )
)

# Anomaly markers
if not anomalies.empty:
    fig.add_trace(
        go.Scatter(
            x=anomalies["time"], y=anomalies["actual"],
            mode="markers", name="Anomaly",
            marker=dict(
                color=COLOR_ANOMALY, size=9, symbol="circle",
                line=dict(color="#1e1e1e", width=1.5),
            ),
            hovertemplate="Anomaly<br>%{x|%Y-%m-%d %H:%M}<br>PM2.5 = %{y:.1f} µg/m³<extra></extra>",
        )
    )

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=470,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(
        title="PM2.5 (µg/m³)", showgrid=True,
        gridcolor="rgba(255,255,255,0.05)", rangemode="tozero",
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Anomaly score trace
# ---------------------------------------------------------------------------
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=df["time"], y=df["anomaly_score"],
        mode="lines", name="Score",
        line=dict(color=COLOR_SCORE, width=1.2),
    )
)
if not anomalies.empty:
    fig2.add_trace(
        go.Scatter(
            x=anomalies["time"], y=anomalies["anomaly_score"],
            mode="markers", name="Flagged",
            marker=dict(color=COLOR_ANOMALY, size=6),
        )
    )

fig2.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=220,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(
        title="Anomaly score (lower = anomalous)",
        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
    ),
)

st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Data table + download
# ---------------------------------------------------------------------------
with st.expander("Inspect raw data", expanded=False):
    display_cols = [
        "time", "actual", "predicted", "lower", "upper",
        "is_anomaly", "anomaly_score",
    ]
    st.dataframe(
        df[display_cols].sort_values("time", ascending=False),
        hide_index=True, use_container_width=True, height=320,
    )
    st.download_button(
        label="Download window as CSV",
        data=df[display_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"atmos_timeline_{start_date}_{end_date}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Footer note
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model: LightGBM point forecast + Adaptive Conformal 80% interval "
    "(MAPIE, `gamma=0.01`). Anomaly detection: IsolationForest on "
    "PM2.5 level, 24 h deviation, fine fraction, wind, temperature, "
    "and hour-of-day. The shaded band is clipped at 0 for display only."
)