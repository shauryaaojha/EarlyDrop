"""Shared loaders, Plotly theme and small UI pieces for the EarlyDrop dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from earlydrop import config as C  # noqa: E402

# Blackboard palette, validated with the dataviz checker on the board surface (categorical order fixed).
SERIES = ["#b3861a", "#3a8ad0", "#cf4b44", "#8f6fd0", "#2f9a74"]
STATE = ["#cf4b44", "#b3861a", "#3a8ad0"]   # Struggling / High risk · Stable / Coasting · Strong / Engaged
BG, PANEL, INK, MUTED, GRID = "#0f1a16", "#15231e", "#ecebe4", "#8fa197", "#26392f"

pio.templates["earlydrop"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#c9cfc6", size=13),
    title=dict(font=dict(family="Space Grotesk, sans-serif", size=17, color=INK), x=0.01),
    colorway=SERIES, margin=dict(l=10, r=10, t=48, b=10),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=PANEL, bordercolor="#e3c35a", font=dict(color=INK, family="IBM Plex Mono")),
    transition=dict(duration=500, easing="cubic-in-out"),
))
pio.templates.default = "earlydrop"


def header(title: str, eyebrow: str, lead: str = ""):
    st.markdown(f'<div class="ed-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    badge = ('<span class="ed-badge"><i></i>SYNTHETIC COHORT · n = 400 · calibrated on 15 real responses</span>'
             if C.DATA_IS_SYNTHETIC else "")
    st.markdown(f'{badge}{f"<p class=ed-lead style=margin-top:1rem>{lead}</p>" if lead else ""}',
                unsafe_allow_html=True)
    st.write("")


def kpi(col, value: float, label: str, sub: str = "", dec: int = 0, pre: str = "", suf: str = "", tag: str = ""):
    """Glass KPI tile; the number counts up on first render."""
    shown = f"{pre}{value:.{dec}f}{suf}"
    col.markdown(f'<div class="ed-kpi"><span class="ic">{tag}</span>'
                 f'<div class="v ed-count" data-to="{value}" data-dec="{dec}" data-pre="{pre}" data-suf="{suf}">{shown}</div>'
                 f'<div class="l">{label}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def card(col, title: str, body: str):
    col.markdown(f'<div class="ed-card"><b>{title}</b><div style="margin-top:.55rem;color:#c9cfc6;line-height:1.65">'
                 f'{body}</div></div>', unsafe_allow_html=True)


def ensure_results():
    if not C.METRICS_JSON.exists():
        st.warning("No results yet. Run the pipeline once (about two minutes on a laptop CPU).")
        if st.button("Run pipeline now", type="primary"):
            import subprocess
            with st.spinner("Running run_all.py ..."):
                subprocess.run([sys.executable, str(ROOT / "run_all.py")], cwd=ROOT, check=False)
            st.cache_data.clear()
            st.rerun()
        st.stop()


@st.cache_data
def metrics() -> dict:
    ensure_results()
    return json.loads(C.METRICS_JSON.read_text())


@st.cache_data
def clean() -> pd.DataFrame:
    return pd.read_csv(C.CLEAN_CSV, parse_dates=["timestamp"])


@st.cache_data
def history() -> pd.DataFrame:
    return pd.read_csv(C.SYN_HISTORY_CSV)


@st.cache_data
def real() -> pd.DataFrame | None:
    return pd.read_csv(C.REAL_CLEAN_CSV) if C.REAL_CLEAN_CSV.exists() else None


def table(rows, fmt="{:.3f}"):
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    return df.style.format({c: fmt for c in num}, na_rep="–")


def nice(name: str) -> str:
    return name.replace("_", " ")
