#!/usr/bin/env python3
"""
dashboard_signal_health.py — Signal Health Monitor for Varanus Neo-Flow P-Matrix Engine.

Real-time visualization of Variable P-Matrix (1x, 3x, 5x leverage) signal integrity,
XGB health gauges, percentile tracking, cooldown timers, and decay alerts.

Reads directly from the v1 bot state JSON, trade CSV, and scan log.

Usage:
    streamlit run dashboard_signal_health.py --server.address 0.0.0.0 --server.port 8503
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config.groups import ASSET_TO_GROUP, get_group

# ═══════════════════════════════════════════════════════════════════════════════
# Config — V1 P-Matrix Engine Paths
# ═══════════════════════════════════════════════════════════════════════════════

STATE_FILE = BASE_DIR / "live_extended_v1_state.json"
TRADES_FILE = BASE_DIR / "live_extended_v1_trades.csv"
SCAN_LOG_FILE = BASE_DIR / "logs" / "scan_results_v1.json"
XGB_MODEL_PATH = BASE_DIR / "models" / "meta_xgb.json"
DECAY_HISTORY_FILE = BASE_DIR / "logs" / "signal_decay_history.json"

BINANCE_PRICES_URL = "https://api.binance.com/api/v3/ticker/price"

# TF_MATRIX: single source of truth for leverage & percentile thresholds
TF_MATRIX = {
    "5m":  {"entry_p": 0.80, "exit_p": 0.30, "leverage": 1},
    "30m": {"entry_p": 0.75, "exit_p": 0.20, "leverage": 3},
    "1h":  {"entry_p": 0.70, "exit_p": 0.15, "leverage": 5},
    "4h":  {"entry_p": 0.70, "exit_p": 0.15, "leverage": 5},
}

TF_ENCODE = {"5m": 0, "30m": 1, "1h": 2, "4h": 3}

# Decay alert threshold: drop > 0.10 in 15 minutes
DECAY_DROP_THRESHOLD = 0.10
DECAY_WINDOW_MINUTES = 15

# ═══════════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Signal Health Monitor — Varanus P-Matrix",
    page_icon="\U0001F9EC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e14; }
    [data-testid="stHeader"] { background-color: #0a0e14; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827 0%, #0f1420 100%);
        border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px;
    }
    div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.8rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.35rem; }

    /* Status badges */
    .badge-green {
        background: #064e3b; color: #34d399; padding: 3px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.78rem; display: inline-block;
    }
    .badge-yellow {
        background: #451a03; color: #fbbf24; padding: 3px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.78rem; display: inline-block;
    }
    .badge-red {
        background: #450a0a; color: #f87171; padding: 3px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.78rem; display: inline-block;
    }
    .badge-gray {
        background: #1e293b; color: #94a3b8; padding: 3px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.78rem; display: inline-block;
    }

    /* Data lag warning */
    .data-lag {
        background: #451a03; border: 1px solid #92400e; border-radius: 8px;
        padding: 8px 14px; color: #fbbf24; font-weight: 600;
    }

    /* Zero-floor alert */
    .zero-floor {
        background: #450a0a; border: 1px solid #991b1b; border-radius: 8px;
        padding: 8px 14px; color: #f87171; font-weight: 600;
    }

    /* Toast notification */
    .toast-exit {
        position: fixed; top: 80px; right: 24px; z-index: 9999;
        background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #34d399;
        border-radius: 12px; padding: 16px 24px; color: #ecfdf5;
        font-weight: 600; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

    /* Cooldown countdown */
    .cd-timer {
        font-family: 'Courier New', monospace; font-size: 1.1rem;
        color: #f87171; font-weight: 700;
    }

    /* Gauge container */
    .gauge-card {
        background: #111827; border: 1px solid #1e293b; border-radius: 12px;
        padding: 12px; text-align: center;
    }

    /* Decay flash */
    .decay-flash {
        animation: decayPulse 1s ease-in-out infinite;
    }
    @keyframes decayPulse {
        0%, 100% { background-color: #111827; }
        50% { background-color: #451a03; }
    }

    /* Table styling */
    .audit-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .audit-table th {
        background: #1e293b; color: #94a3b8; padding: 10px 14px;
        text-align: left; font-weight: 600; border-bottom: 2px solid #334155;
    }
    .audit-table td {
        padding: 10px 14px; border-bottom: 1px solid #1e293b; color: #e2e8f0;
    }
    .audit-table tr:hover { background: #1e293b40; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Layer — Load, Validate, Transform
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=4)
def load_state() -> dict:
    """Load bot state with validation."""
    try:
        raw = json.loads(STATE_FILE.read_text())
        return raw
    except Exception:
        return {
            "positions": {}, "cooldowns": {}, "trade_counter": 0,
            "realized_pnl": 0.0, "initial_capital": 1000.0,
            "peak_equity": 1000.0, "circuit_breaker": False,
            "last_scan_ts": "", "total_scans": 0,
        }


@st.cache_data(ttl=8)
def load_trades() -> pd.DataFrame:
    """Load trade history CSV."""
    if TRADES_FILE.exists():
        try:
            df = pd.read_csv(TRADES_FILE)
            df.columns = [c.strip().lower() for c in df.columns]
            if "exit_ts" in df.columns:
                df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
            if "entry_ts" in df.columns:
                df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=4)
def load_scan_log() -> dict:
    """Load latest scan results."""
    try:
        return json.loads(SCAN_LOG_FILE.read_text())
    except Exception:
        return {"timestamp": "", "scan_number": 0, "signals": []}


@st.cache_data(ttl=10)
def fetch_prices() -> dict[str, float]:
    """Fetch live Binance prices."""
    try:
        resp = requests.get(BINANCE_PRICES_URL, timeout=10)
        resp.raise_for_status()
        return {item["symbol"]: float(item["price"]) for item in resp.json()}
    except Exception:
        return {}


def load_decay_history() -> dict:
    """Load signal decay tracking history (non-cached, always fresh)."""
    try:
        return json.loads(DECAY_HISTORY_FILE.read_text())
    except Exception:
        return {}


def save_decay_history(history: dict):
    """Persist signal decay history."""
    try:
        DECAY_HISTORY_FILE.write_text(json.dumps(history, indent=2))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Percentile Engine (Lightweight — reads from trade CSVs)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def build_percentile_distributions() -> dict[str, np.ndarray]:
    """Build XGB score distributions from historical trades, keyed by TF."""
    try:
        from ml.train_meta_model import load_model, predict_probability
    except ImportError:
        return {}

    try:
        model, _ = load_model(XGB_MODEL_PATH)
    except Exception:
        return {}

    csv_candidates = [
        BASE_DIR / "blind_test_trades.csv",
        BASE_DIR / "extended_trades_hybrid.csv",
        BASE_DIR / "extended_trades_physics.csv",
        BASE_DIR / "live_extended_v1_trades.csv",
    ]

    frames = []
    for path in csv_candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            required = {"confidence", "pvt_r", "best_tf", "best_period"}
            if required.issubset(set(df.columns)):
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in ["asset", "entry_ts", "exit_ts", "entry_price"] if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols)

    scores = []
    for _, row in combined.iterrows():
        try:
            s = predict_probability(
                model,
                confidence=float(row["confidence"]),
                pvt_r=float(row["pvt_r"]),
                best_tf=str(row["best_tf"]),
                best_period=int(row["best_period"]),
            )
            scores.append(s)
        except Exception:
            scores.append(np.nan)

    combined["xgb_score"] = scores
    combined = combined.dropna(subset=["xgb_score"])

    distributions: dict[str, np.ndarray] = {}
    for tf in ["5m", "30m", "1h", "4h"]:
        mask = combined["best_tf"] == tf
        tf_scores = combined.loc[mask, "xgb_score"].values
        if len(tf_scores) >= 10:
            distributions[tf] = np.sort(tf_scores)

    all_scores = combined["xgb_score"].values
    if len(all_scores) >= 10:
        distributions["GLOBAL"] = np.sort(all_scores)

    return distributions


def score_to_percentile(distributions: dict, tf: str, score: float) -> float:
    """Convert an XGB score to its percentile rank within TF distribution."""
    dist = distributions.get(tf, distributions.get("GLOBAL"))
    if dist is None or len(dist) == 0:
        return 0.0
    pct = float(np.searchsorted(dist, score, side="right")) / len(dist) * 100
    return min(pct, 100.0)  # Clamp to P100


def get_threshold(distributions: dict, tf: str, percentile: float) -> float:
    """Get the XGB score at a given percentile for a TF."""
    dist = distributions.get(tf, distributions.get("GLOBAL"))
    if dist is None or len(dist) == 0:
        return percentile
    return float(np.percentile(dist, percentile * 100))


# ═══════════════════════════════════════════════════════════════════════════════
# Data Validation & Integrity Checks
# ═══════════════════════════════════════════════════════════════════════════════

def validate_position(pos: dict, distributions: dict) -> dict:
    """
    Run integrity checks on a single position.
    Returns enriched dict with validation flags.
    """
    tf = pos.get("best_tf", "5m")
    xgb = pos.get("xgb_prob", 0.0)
    confidence = pos.get("confidence", 0.0)
    p_exit = pos.get("p_exit_threshold", 0.0)

    # Compute current percentile rank
    p_value = score_to_percentile(distributions, tf, xgb)

    # Matrix params for this TF
    matrix = TF_MATRIX.get(tf, TF_MATRIX["4h"])
    exit_threshold = get_threshold(distributions, tf, matrix["exit_p"])

    # Integrity flags
    flags = []

    # Clamp check: score above historical max → P100
    dist = distributions.get(tf, distributions.get("GLOBAL"))
    is_p100 = False
    if dist is not None and len(dist) > 0:
        if xgb >= dist[-1]:
            p_value = 100.0
            is_p100 = True

    # Missing data alert
    has_data_lag = (xgb <= 0.0 or confidence <= 0.0)
    if has_data_lag:
        flags.append("DATA_LAG")

    # Zero-floor detection
    has_zero_floor = (p_exit <= 0.001)
    if has_zero_floor:
        flags.append("ZERO_FLOOR")

    # Health status
    p_exit_pct = matrix["exit_p"] * 100  # e.g. 30 for P30
    p50_threshold = get_threshold(distributions, tf, 0.50)
    if has_data_lag:
        status = "DATA_LAG"
        status_color = "yellow"
    elif xgb < exit_threshold:
        status = "EXHAUSTED"
        status_color = "red"
    elif xgb < p50_threshold:
        status = "DECLINING"
        status_color = "yellow"
    else:
        status = "HEALTHY"
        status_color = "green"

    return {
        **pos,
        "p_value": round(p_value, 1),
        "is_p100": is_p100,
        "exit_threshold_actual": round(exit_threshold, 4),
        "status": status,
        "status_color": status_color,
        "flags": flags,
        "has_data_lag": has_data_lag,
        "has_zero_floor": has_zero_floor,
        "matrix": matrix,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Decay Tracker
# ═══════════════════════════════════════════════════════════════════════════════

def check_signal_decay(positions: dict) -> dict[str, dict]:
    """
    Track XGB score changes over time. Flag if drop > 0.10 in 15 min.
    Returns dict of {asset: {"decaying": bool, "drop": float, "prev": float}}.
    """
    now = datetime.now(timezone.utc)
    history = load_decay_history()
    results = {}

    for asset, pos in positions.items():
        xgb = pos.get("xgb_prob", 0.0)
        if xgb <= 0:
            continue

        asset_hist = history.get(asset, [])

        # Append current reading
        asset_hist.append({
            "ts": now.isoformat(),
            "score": xgb,
        })

        # Prune entries older than 30 minutes
        cutoff = (now - timedelta(minutes=30)).isoformat()
        asset_hist = [h for h in asset_hist if h["ts"] >= cutoff]
        history[asset] = asset_hist

        # Check for decay within window
        window_cutoff = (now - timedelta(minutes=DECAY_WINDOW_MINUTES)).isoformat()
        window_entries = [h for h in asset_hist if h["ts"] >= window_cutoff]

        decaying = False
        drop = 0.0
        peak_in_window = xgb
        if len(window_entries) >= 2:
            peak_in_window = max(h["score"] for h in window_entries)
            drop = peak_in_window - xgb
            if drop >= DECAY_DROP_THRESHOLD:
                decaying = True

        results[asset] = {
            "decaying": decaying,
            "drop": round(drop, 4),
            "peak": round(peak_in_window, 4),
            "current": round(xgb, 4),
        }

    # Clean up closed positions from history
    for asset in list(history.keys()):
        if asset not in positions:
            del history[asset]

    save_decay_history(history)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Visual Components
# ═══════════════════════════════════════════════════════════════════════════════

def render_health_gauge(asset: str, xgb_score: float, p_value: float,
                        exit_threshold: float, status: str, status_color: str,
                        tf: str, leverage: int, is_p100: bool) -> go.Figure:
    """Circular gauge showing current XGB score health."""
    # Color zones based on status
    if status_color == "red":
        bar_color = "#ef4444"
    elif status_color == "yellow":
        bar_color = "#f59e0b"
    else:
        bar_color = "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=xgb_score,
        number={"suffix": "", "font": {"size": 28, "color": "#e2e8f0"}},
        title={"text": f"<b>{asset}</b> ({tf} {leverage}x)", "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.7},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, exit_threshold], "color": "rgba(69,10,10,0.19)"},
                {"range": [exit_threshold, 0.5], "color": "rgba(69,26,3,0.19)"},
                {"range": [0.5, 1], "color": "rgba(6,78,59,0.19)"},
            ],
            "threshold": {
                "line": {"color": "#f87171", "width": 3},
                "thickness": 0.8,
                "value": exit_threshold,
            },
        },
    ))

    # Add P-value annotation
    label = "P100 (Max)" if is_p100 else f"P{p_value:.0f}"
    fig.add_annotation(
        text=f"<b>{label}</b>",
        x=0.5, y=0.18,
        showarrow=False,
        font={"size": 13, "color": bar_color},
    )

    fig.update_layout(
        height=220, margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


def render_cooldown_section(cooldowns: dict):
    """Render cooldown tracker with countdown timers."""
    now = datetime.now(timezone.utc)
    active_cds = []

    for asset, cd in cooldowns.items():
        until_str = cd.get("cooldown_until", "")
        losses = cd.get("consecutive_losses", 0)
        if not until_str:
            continue
        try:
            until = datetime.fromisoformat(until_str)
            if now < until:
                remaining = until - now
                mins_left = int(remaining.total_seconds() / 60)
                secs_left = int(remaining.total_seconds() % 60)
                step = [15, 45, 180][min(losses, 3) - 1] if losses > 0 else 0
                active_cds.append({
                    "asset": asset,
                    "losses": losses,
                    "mins_left": mins_left,
                    "secs_left": secs_left,
                    "step_minutes": step,
                    "until": until.strftime("%H:%M:%S UTC"),
                })
        except (ValueError, TypeError):
            continue

    if not active_cds:
        st.markdown(
            '<div style="text-align:center; color:#64748b; padding:20px;">'
            'No assets on cooldown</div>',
            unsafe_allow_html=True,
        )
        return

    for cd in active_cds:
        step_label = {15: "1st Loss (15m)", 45: "2nd Loss (45m)", 180: "3rd+ Loss (3h)"}.get(
            cd["step_minutes"], f"{cd['step_minutes']}m"
        )

        st.markdown(f"""
        <div style="background:#1e293b; border-left:4px solid #ef4444; border-radius:8px;
                    padding:14px 18px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:#f87171; font-weight:700; font-size:1.1rem;">
                        \U0001F534 {cd['asset']}USDT
                    </span>
                    <span class="badge-red" style="margin-left:8px;">
                        Streak: {cd['losses']}/3
                    </span>
                    <span class="badge-gray" style="margin-left:6px;">
                        {step_label}
                    </span>
                </div>
                <div class="cd-timer">
                    {cd['mins_left']:02d}:{cd['secs_left']:02d}
                </div>
            </div>
            <div style="color:#94a3b8; font-size:0.82rem; margin-top:6px;">
                Resumes at {cd['until']} &mdash; Market structure incompatible with signal math
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_exit_toast(trades_df: pd.DataFrame):
    """Show toast for recent P_FLOOR_EXIT trades (last 5 minutes)."""
    if trades_df.empty or "exit_ts" not in trades_df.columns:
        return

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=5)

    recent = trades_df[
        (trades_df["exit_ts"] >= cutoff)
        & (trades_df["exit_reason"] == "P_FLOOR_EXIT")
    ]

    for _, trade in recent.iterrows():
        asset = trade.get("asset", "?")
        pnl = trade.get("pnl_pct", 0.0)
        tf = trade.get("best_tf", "?")
        pnl_label = f"+{pnl:.2f}%" if pnl > 0 else f"{pnl:.2f}%"
        result_word = "Profit Secured" if pnl > 0 else "Loss Limited"

        st.markdown(f"""
        <div class="toast-exit">
            \U0001F3C1 <b>[{asset}]</b> Exit via Signal Exhaustion ({tf})
            &mdash; {pnl_label} &mdash; {result_word}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Layout
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    state = load_state()
    trades_df = load_trades()
    scan_log = load_scan_log()
    prices = fetch_prices()
    distributions = build_percentile_distributions()
    positions = state.get("positions", {})
    cooldowns = state.get("cooldowns", {})

    # Decay tracking
    decay_info = check_signal_decay(positions) if positions else {}

    # Toast for recent P-Floor exits
    render_exit_toast(trades_df)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        "## \U0001F9EC Signal Health Monitor"
    )
    st.caption("Variable P-Matrix (1x/3x/5x) | Real-Time Percentile Tracking | Statistical Exhaustion Detection")

    # ── Top Metrics Bar ───────────────────────────────────────────────────
    equity = state.get("initial_capital", 0) + state.get("realized_pnl", 0)
    n_open = len(positions)
    total_trades = state.get("trade_counter", 0)
    rpnl = state.get("realized_pnl", 0)
    scans = state.get("total_scans", 0)
    last_scan = state.get("last_scan_ts", "")

    # Compute mean XGB of open positions
    xgb_scores = [p.get("xgb_prob", 0) for p in positions.values() if p.get("xgb_prob", 0) > 0]
    mean_xgb = np.mean(xgb_scores) if xgb_scores else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Equity", f"${equity:,.2f}", f"${rpnl:+,.2f}")
    c2.metric("Open Positions", f"{n_open}/8")
    c3.metric("Total Trades", str(total_trades))
    c4.metric("Avg XGB (Open)", f"{mean_xgb:.4f}" if mean_xgb > 0 else "N/A")
    c5.metric("Scans", str(scans))

    if last_scan:
        try:
            scan_dt = datetime.fromisoformat(last_scan)
            age = (datetime.now(timezone.utc) - scan_dt).total_seconds()
            c6.metric("Last Scan", f"{int(age)}s ago")
        except Exception:
            c6.metric("Last Scan", "—")
    else:
        c6.metric("Last Scan", "—")

    st.divider()

    # ── Data Integrity Alerts ─────────────────────────────────────────────
    if not positions:
        st.info("No active positions. The signal health monitor will populate when trades are opened.")
        _render_cooldown_panel(cooldowns)
        _render_tf_reference(distributions)
        st.stop()

    # Validate all positions
    validated = {}
    for asset, pos in positions.items():
        validated[asset] = validate_position(pos, distributions)

    # Show critical alerts
    alerts_html = []
    for asset, vp in validated.items():
        if "DATA_LAG" in vp["flags"]:
            alerts_html.append(
                f'<div class="data-lag">'
                f'\u26A0\uFE0F <b>{asset}USDT</b> — DATA_LAG: XGB or Pearson R score is missing/zero. '
                f'Percentile values may be stale.</div>'
            )
        if "ZERO_FLOOR" in vp["flags"]:
            alerts_html.append(
                f'<div class="zero-floor">'
                f'\U0001F534 <b>{asset}USDT</b> — ZERO P-FLOOR: Exit protection threshold is 0.000. '
                f'This is a legacy position without P-Matrix exit guard.</div>'
            )

    # Decay alerts
    for asset, di in decay_info.items():
        if di["decaying"]:
            alerts_html.append(
                f'<div class="data-lag">'
                f'\u26A1 <b>{asset}USDT</b> — SIGNAL DECAY: '
                f'XGB dropped {di["drop"]:.4f} in {DECAY_WINDOW_MINUTES}min '
                f'(from {di["peak"]:.4f} to {di["current"]:.4f}). '
                f'Trend reversal risk.</div>'
            )

    if alerts_html:
        for alert in alerts_html:
            st.markdown(alert, unsafe_allow_html=True)
        st.markdown("")

    # ── Signal Health Gauges ──────────────────────────────────────────────
    st.markdown("### \U0001F3AF Signal Health Gauges")

    gauge_cols = st.columns(min(len(validated), 4))
    for i, (asset, vp) in enumerate(validated.items()):
        col = gauge_cols[i % len(gauge_cols)]
        with col:
            fig = render_health_gauge(
                asset=asset,
                xgb_score=vp.get("xgb_prob", 0),
                p_value=vp["p_value"],
                exit_threshold=vp["exit_threshold_actual"],
                status=vp["status"],
                status_color=vp["status_color"],
                tf=vp.get("best_tf", "?"),
                leverage=vp["matrix"].get("leverage", 0),
                is_p100=vp["is_p100"],
            )
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{asset}")

            # Status badge under gauge
            badge_class = f"badge-{vp['status_color']}"
            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="{badge_class}">{vp["status"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Live Audit Table ──────────────────────────────────────────────────
    st.markdown("### \U0001F4CB Live Audit Table")

    # Build table rows
    rows_html = []
    for asset, vp in validated.items():
        tf = vp.get("best_tf", "?")
        lev = vp["matrix"].get("leverage", 0)
        xgb = vp.get("xgb_prob", 0)
        r_score = vp.get("confidence", 0)
        p_val = vp["p_value"]
        p_floor = vp.get("p_exit_threshold", 0)
        status = vp["status"]
        sc = vp["status_color"]
        direction = "LONG" if vp.get("direction", 0) == 1 else "SHORT"
        entry_price = vp.get("entry_price", 0)

        # Live PnL from Binance price
        live_price = prices.get(f"{asset}USDT", 0)
        if live_price > 0 and entry_price > 0:
            dir_mult = 1 if vp.get("direction", 0) == 1 else -1
            unrealized_pnl = (live_price - entry_price) / entry_price * dir_mult * 100
            pnl_str = f"{unrealized_pnl:+.2f}%"
            pnl_color = "#10b981" if unrealized_pnl >= 0 else "#ef4444"
        else:
            pnl_str = "—"
            pnl_color = "#64748b"

        # Badge class
        badge = f"badge-{sc}"

        # P-Value display (clamped)
        p_label = "P100 (Max)" if vp["is_p100"] else f"P{p_val:.0f}"

        # XGB display
        if vp["has_data_lag"]:
            xgb_display = '<span style="color:#fbbf24;">\u26A0\uFE0F N/A</span>'
            r_display = '<span style="color:#fbbf24;">\u26A0\uFE0F N/A</span>'
        else:
            xgb_display = f"{xgb:.4f}"
            r_display = f"{r_score:.4f}"

        # P-Floor display (highlight zero in red)
        if vp["has_zero_floor"]:
            floor_display = '<span style="color:#f87171; font-weight:700;">0.000 \u26A0\uFE0F</span>'
        else:
            floor_display = f"{p_floor:.4f}"

        # Decay indicator
        di = decay_info.get(asset, {})
        decay_class = ' class="decay-flash"' if di.get("decaying", False) else ""

        rows_html.append(f"""
        <tr{decay_class}>
            <td><b>{asset}USDT</b><br><span style="color:#64748b;font-size:0.75rem;">{direction}</span></td>
            <td>{tf}</td>
            <td><b>{lev}x</b></td>
            <td>{xgb_display}</td>
            <td>{r_display}</td>
            <td><b>{p_label}</b></td>
            <td>{floor_display}</td>
            <td style="color:{pnl_color};">{pnl_str}</td>
            <td><span class="{badge}">{status}</span></td>
        </tr>
        """)

    table_html = f"""
    <table class="audit-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>TF</th>
                <th>Leverage</th>
                <th>Current XGB</th>
                <th>Current R</th>
                <th>P-Value</th>
                <th>P-Exit Floor</th>
                <th>Unrealized</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows_html)}
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.divider()

    # ── Cooldown Tracker ──────────────────────────────────────────────────
    _render_cooldown_panel(cooldowns)

    st.divider()

    # ── TF Matrix Reference ───────────────────────────────────────────────
    _render_tf_reference(distributions)

    # ── Auto-refresh ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center; color:#475569; font-size:0.75rem; margin-top:30px;">'
        f'Auto-refreshes every 5 seconds &mdash; '
        f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}'
        '</div>',
        unsafe_allow_html=True,
    )
    time.sleep(5)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Reusable Panels
# ═══════════════════════════════════════════════════════════════════════════════

def _render_cooldown_panel(cooldowns: dict):
    """Cooldown tracker panel."""
    st.markdown("### \U0001F534 Cooldown Tracker (Mola)")
    render_cooldown_section(cooldowns)


def _render_tf_reference(distributions: dict):
    """Show the TF-Matrix reference table with live percentile thresholds."""
    st.markdown("### \U0001F4D0 TF-Matrix Reference")

    ref_data = []
    for tf in ["5m", "30m", "1h", "4h"]:
        m = TF_MATRIX[tf]
        entry_t = get_threshold(distributions, tf, m["entry_p"])
        exit_t = get_threshold(distributions, tf, m["exit_p"])
        n_trades = len(distributions.get(tf, []))
        ref_data.append({
            "TF": tf,
            "Leverage": f"{m['leverage']}x",
            "Entry Gate": f"P{m['entry_p']*100:.0f} = {entry_t:.4f}",
            "Exit Floor": f"P{m['exit_p']*100:.0f} = {exit_t:.4f}",
            "Distribution Size": f"{n_trades} trades",
        })

    ref_df = pd.DataFrame(ref_data)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
