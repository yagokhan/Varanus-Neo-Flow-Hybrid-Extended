#!/usr/bin/env python3
"""
dashboard.py — Varanus Neo-Flow Hybrid Extended: Professional Monitoring Dashboard.

Read-only visual layer over the live bot. 33 assets, 3 groups, group-specific thresholds.

Usage:
    streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8502
"""

import sys
import json
import subprocess
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

from config.groups import (
    ALL_ASSETS, GROUP_A_MAJORS, GROUP_B_TECH_AI, GROUP_C_MOMENTUM_MEME,
    ASSET_TO_GROUP, GROUP_NAMES, get_group, get_thresholds, DEFAULT_THRESHOLDS,
    GroupThresholds,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

STATE_FILE = BASE_DIR / "live_extended_state.json"
TRADES_FILE = BASE_DIR / "live_extended_trades.csv"
SCAN_LOG_FILE = BASE_DIR / "logs" / "scan_results.json"
DASHBOARD_SCAN_CACHE = BASE_DIR / "logs" / "dashboard_scan_cache.json"
WFV_FILE = BASE_DIR / "wfv_results.json"
OPTIMIZED_THRESHOLDS_FILE = BASE_DIR / "config" / "optimized_thresholds.json"
LOG_FILE = BASE_DIR / "logs" / "live_extended_bot.log"
DATA_DIR = BASE_DIR / "data"

BINANCE_PRICES_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

ASSETS = ALL_ASSETS
SCAN_TIMEFRAMES = ["5m", "30m", "1h", "4h"]
BARS_7D = {"5m": 2016, "30m": 336, "1h": 168, "4h": 126}

GROUP_COLORS = {"A": "#39afd1", "B": "#f6c343", "C": "#e63757"}
GROUP_LABELS = {"A": "Majors", "B": "Tech/AI", "C": "Meme"}

def _load_group_thresholds() -> dict[str, GroupThresholds]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if OPTIMIZED_THRESHOLDS_FILE.exists():
        try:
            data = json.loads(OPTIMIZED_THRESHOLDS_FILE.read_text())
            for grp_key, grp_data in data.get("groups", {}).items():
                t = grp_data.get("thresholds", {})
                thresholds[grp_key] = GroupThresholds(
                    min_confidence=t.get("min_confidence", 0.80),
                    min_xgb_score=t.get("min_xgb_score", 0.55),
                    trail_buffer=t.get("trail_buffer", 0.50),
                    min_pvt_r=t.get("min_pvt_r", 0.75),
                    combined_gate=t.get("combined_gate", 0.80),
                    hard_sl_mult=t.get("hard_sl_mult", 2.50),
                    exhaust_r=t.get("exhaust_r", 0.475),
                    pos_frac=t.get("pos_frac", 0.05),
                )
        except Exception:
            pass
    return thresholds

GROUP_THRESHOLDS = _load_group_thresholds()

# ═══════════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Varanus Neo-Flow Extended",
    page_icon=":lizard:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192a 100%);
        border: 1px solid #2d3348; border-radius: 10px; padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b92a5 !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .pnl-pos { color: #00d97e; font-weight: 700; }
    .pnl-neg { color: #e63757; font-weight: 700; }
    .badge-high {
        background: linear-gradient(135deg, #00d97e, #0abf6b); color: #000;
        padding: 2px 10px; border-radius: 12px; font-weight: 700; font-size: 0.8rem;
    }
    .badge-pass {
        background: #1e3a2f; color: #00d97e;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;
    }
    .badge-fail {
        background: #3a1e2f; color: #e63757;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;
    }
    .badge-grp-a { background: #1e2e3a; color: #39afd1; padding: 1px 8px; border-radius: 8px; font-size: 0.75rem; }
    .badge-grp-b { background: #3a361e; color: #f6c343; padding: 1px 8px; border-radius: 8px; font-size: 0.75rem; }
    .badge-grp-c { background: #3a1e2f; color: #e63757; padding: 1px 8px; border-radius: 8px; font-size: 0.75rem; }
    .gate-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 0; border-bottom: 1px solid #1e2235; font-size: 0.9rem;
    }
    .trail-bar { background: #1e2235; border-radius: 6px; height: 22px; position: relative; overflow: hidden; }
    .trail-fill { height: 100%; border-radius: 6px; transition: width 0.3s; }
    @media (max-width: 768px) {
        div[data-testid="stMetric"] { padding: 8px 10px; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.1rem; }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def load_bot_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {
            "positions": {}, "trade_counter": 0, "realized_pnl": 0.0,
            "initial_capital": 1000.0, "peak_equity": 1000.0,
            "circuit_breaker": False, "last_scan_ts": "", "total_scans": 0,
        }

@st.cache_data(ttl=10)
def load_trades() -> pd.DataFrame:
    if TRADES_FILE.exists():
        try:
            df = pd.read_csv(TRADES_FILE)
            df["entry_ts"] = pd.to_datetime(df["entry_ts"])
            df["exit_ts"] = pd.to_datetime(df["exit_ts"])
            return df
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=5)
def load_scan_log() -> dict:
    try:
        return json.loads(SCAN_LOG_FILE.read_text())
    except Exception:
        return {"timestamp": "", "scan_number": 0, "signals": []}

@st.cache_data(ttl=300)
def load_wfv() -> dict:
    try:
        return json.loads(WFV_FILE.read_text())
    except Exception:
        return {}

@st.cache_data(ttl=15)
def fetch_binance_prices() -> dict:
    try:
        resp = requests.get(BINANCE_PRICES_URL, timeout=10)
        resp.raise_for_status()
        return {item["symbol"]: float(item["price"]) for item in resp.json()}
    except Exception:
        return {}

@st.cache_data(ttl=30)
def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    try:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def _fetch_live_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def run_dashboard_scan() -> list[dict]:
    """Run a READ-ONLY scan of all 33 assets using LIVE Binance data."""
    try:
        from neo_flow.adaptive_engine import (
            find_best_regression, compute_pvt_regression, check_pvt_alignment,
            get_htf_bias, check_combined_gate, trim_to_7d,
        )
        from ml.train_meta_model import predict_probability, load_model

        xgb_model = None
        xgb_path = BASE_DIR / "models" / "meta_xgb.json"
        if xgb_path.exists():
            xgb_model, _ = load_model(xgb_path)

        results = []
        for asset in ASSETS:
            symbol = f"{asset}USDT"
            group = get_group(asset)
            gt = GROUP_THRESHOLDS.get(group, DEFAULT_THRESHOLDS[group])

            scan_dfs = {}
            for tf in SCAN_TIMEFRAMES:
                n = BARS_7D[tf]
                df = _fetch_live_candles(symbol, tf, n)
                if not df.empty:
                    scan_dfs[tf] = df

            row = {
                "asset": asset, "group": group, "best_r": 0, "pvt_r": 0,
                "direction": 0, "best_tf": "-", "best_period": 0,
                "midline": 0, "std_dev": 0, "xgb_prob": 0,
                "gate_r": False, "gate_pvt": False, "gate_htf": False,
                "gate_combined": False, "gate_xgb": False,
                "reject_reason": "", "final_status": "no_signal",
                "entry_price": 0,
                "r_diff": 0.0, "pvt_diff": 0.0, "xgb_diff": 0.0,
                "htf_bias_str": "-", "pvt_detail": "", "combined_detail": "",
                "first_fail": "",
            }

            if not scan_dfs:
                row["reject_reason"] = "No live data"
                row["first_fail"] = "No Data"
                results.append(row)
                continue

            best, all_regs = find_best_regression(scan_dfs)
            if best is None:
                row["reject_reason"] = "No valid regression"
                row["first_fail"] = "R-Gate"
                results.append(row)
                continue

            abs_r = abs(best.pearson_r)
            direction = 1 if best.slope < 0 else -1
            row["best_r"] = round(abs_r, 4)
            row["direction"] = direction
            row["best_tf"] = best.timeframe
            row["best_period"] = best.period
            row["midline"] = round(best.midline, 6)
            row["std_dev"] = round(best.std_dev, 6)
            row["r_diff"] = round(abs_r - gt.min_confidence, 4)

            best_df = scan_dfs.get(best.timeframe)
            if best_df is not None and not best_df.empty:
                row["entry_price"] = float(best_df.iloc[-1]["close"])

            # Gate 1: Pearson R (group-specific)
            if abs_r >= gt.min_confidence:
                row["gate_r"] = True
            else:
                row["first_fail"] = row["first_fail"] or "R-Gate"
                row["reject_reason"] = f"|R| {abs_r:.4f} < {gt.min_confidence}"

            # Gate 2: PVT alignment (group-specific)
            pvt_result = None
            if best_df is not None and not best_df.empty:
                best_df_7d = trim_to_7d(best_df, best.timeframe)
                pvt_result = compute_pvt_regression(best_df_7d, best.period)
                pvt_abs_r = abs(pvt_result.pearson_r)
                row["pvt_r"] = round(pvt_abs_r, 4)
                row["pvt_diff"] = round(pvt_abs_r - gt.min_pvt_r, 4)
                pvt_dir_str = "RISING" if pvt_result.direction == 1 else ("FALLING" if pvt_result.direction == -1 else "FLAT")
                row["pvt_detail"] = pvt_dir_str

                if row["gate_r"]:
                    pvt_passes, pvt_reason = check_pvt_alignment(direction, abs_r, pvt_result)
                    if pvt_passes:
                        row["gate_pvt"] = True
                    else:
                        row["first_fail"] = row["first_fail"] or "PVT-Gate"
                        row["reject_reason"] = row["reject_reason"] or pvt_reason

            # Gate 3: 4H HTF bias
            htf_bias = 0
            df_4h = _fetch_live_candles(symbol, "4h", BARS_7D["4h"])
            htf_detail = ""
            if not df_4h.empty:
                htf_bias = get_htf_bias(df_4h)
                # Compute distance to MSS breakout for display
                try:
                    from neo_flow.adaptive_engine import detect_mss, compute_ema, MSS_LOOKBACK as _MSS_LB
                    if len(df_4h) >= _MSS_LB + 1:
                        _w = df_4h.iloc[-(_MSS_LB + 1):-1]
                        _cur = df_4h.iloc[-1]
                        _sh = float(_w["high"].max())
                        _sl = float(_w["low"].min())
                        _cp = float(_cur["close"])
                        _to_bull = ((_sh - _cp) / _cp) * 100
                        _to_bear = ((_cp - _sl) / _cp) * 100
                        _ema_f = float(compute_ema(df_4h["close"], 21).iloc[-1])
                        _ema_s = float(compute_ema(df_4h["close"], 55).iloc[-1])
                        _ema_str = "EMA:Bull" if _ema_f > _ema_s else "EMA:Bear"
                        htf_detail = f"{_ema_str} | +{_to_bull:.1f}%toHH -{_to_bear:.1f}%toLL"
                except Exception:
                    pass

            row["htf_bias_str"] = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}.get(htf_bias, "?")
            row["htf_detail"] = htf_detail
            if row["gate_r"] and row["gate_pvt"]:
                if htf_bias != 0 and htf_bias == direction:
                    row["gate_htf"] = True
                else:
                    row["first_fail"] = row["first_fail"] or "4H-Trend"
                    if htf_bias == 0:
                        row["reject_reason"] = row["reject_reason"] or f"4H neutral ({htf_detail})"
                    else:
                        dir_s = "LONG" if direction == 1 else "SHORT"
                        row["reject_reason"] = row["reject_reason"] or f"4H {row['htf_bias_str']} vs signal {dir_s}"

            # Gate 4: Combined gate (group-specific)
            combined_pass = check_combined_gate(direction, all_regs) if all_regs else True
            max_opposing_r = 0.0
            for r in all_regs:
                r_dir = -1 if r.slope > 0 else (1 if r.slope < 0 else 0)
                if r_dir != 0 and r_dir != direction:
                    max_opposing_r = max(max_opposing_r, abs(r.pearson_r))
            row["combined_detail"] = f"max_opp |R|={max_opposing_r:.4f}" if max_opposing_r > 0 else "no opposing"

            if row["gate_r"] and row["gate_pvt"] and row["gate_htf"]:
                if combined_pass:
                    row["gate_combined"] = True
                else:
                    row["first_fail"] = row["first_fail"] or "Combined"
                    row["reject_reason"] = row["reject_reason"] or f"Opposing |R| {max_opposing_r:.4f} > {gt.combined_gate}"

            # Gate 5: XGBoost (group-specific threshold)
            if xgb_model is not None and pvt_result is not None:
                xgb_prob = predict_probability(
                    xgb_model, confidence=abs_r, pvt_r=abs(pvt_result.pearson_r),
                    best_tf=best.timeframe, best_period=best.period, pnl_pct=0.0,
                )
                row["xgb_prob"] = round(xgb_prob, 4)
                row["xgb_diff"] = round(xgb_prob - gt.min_xgb_score, 4)

                if row["gate_r"] and row["gate_pvt"] and row["gate_htf"] and row["gate_combined"]:
                    if xgb_prob > gt.min_xgb_score:
                        row["gate_xgb"] = True
                        row["final_status"] = "qualified"
                    else:
                        row["first_fail"] = row["first_fail"] or "XGBoost"
                        row["final_status"] = "xgb_rejected"
                        row["reject_reason"] = row["reject_reason"] or f"XGB {xgb_prob:.4f} <= {gt.min_xgb_score}"

            if not row["first_fail"] and row["final_status"] != "qualified":
                row["final_status"] = "blocked"

            results.append(row)

        return results
    except Exception as e:
        return [{"asset": "ERROR", "group": "-", "reject_reason": str(e),
                 "final_status": "error", "best_r": 0, "pvt_r": 0, "direction": 0,
                 "best_tf": "-", "best_period": 0, "midline": 0, "std_dev": 0,
                 "xgb_prob": 0, "gate_r": False, "gate_pvt": False, "gate_htf": False,
                 "gate_combined": False, "gate_xgb": False, "entry_price": 0,
                 "r_diff": 0, "pvt_diff": 0, "xgb_diff": 0,
                 "htf_bias_str": "-", "pvt_detail": "", "combined_detail": "",
                 "first_fail": ""}]


def _detect_live_mode() -> bool:
    """Check if the running bot has --live flag."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "live_extended_bot.py"],
            capture_output=True, text=True, timeout=5,
        )
        return "--live" in result.stdout
    except Exception:
        return False


@st.cache_data(ttl=15)
def fetch_binance_account() -> dict | None:
    """Fetch Binance Futures account: balance + open positions."""
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(BASE_DIR / ".env")
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            return None
        import ccxt
        ex = ccxt.binanceusdm({
            "apiKey": api_key, "secret": api_secret,
            "options": {"defaultType": "future"}, "enableRateLimit": True,
        })
        balance = ex.fetch_balance()
        usdt = balance.get("USDT", {})
        positions = ex.fetch_positions()
        open_pos = [p for p in positions if abs(float(p.get("contracts", 0))) > 0]
        return {
            "balance_total": float(usdt.get("total", 0)),
            "balance_free": float(usdt.get("free", 0)),
            "balance_used": float(usdt.get("used", 0)),
            "positions": [{
                "symbol": p["symbol"],
                "side": p["side"],
                "contracts": float(p["contracts"]),
                "entry_price": float(p.get("entryPrice", 0)),
                "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                "leverage": p.get("leverage", "?"),
            } for p in open_pos],
        }
    except Exception:
        return None


def get_bot_pid() -> int | None:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_extended_bot.py"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        return int(pids[0]) if pids and pids[0] else None
    except Exception:
        return None

def parse_recent_logs(n_lines: int = 50) -> list[str]:
    try:
        result = subprocess.run(
            ["tail", "-n", str(n_lines), str(LOG_FILE)],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def pnl_color(val: float) -> str:
    return "pnl-pos" if val >= 0 else "pnl-neg"

def group_badge(group: str) -> str:
    return f'<span class="badge-grp-{group.lower()}">{group} {GROUP_LABELS.get(group, "")}</span>'

def trail_bar_html(pct: float) -> str:
    pct = max(0, min(100, pct))
    color = "#00d97e" if pct > 60 else "#f6c343" if pct > 30 else "#e63757"
    return f'<div class="trail-bar"><div class="trail-fill" style="width:{pct:.0f}%;background:{color};"></div></div>'

def time_since(iso_ts: str) -> str:
    if not iso_ts:
        return "Never"
    try:
        dt = datetime.fromisoformat(iso_ts)
        delta = datetime.now(timezone.utc) - dt
        secs = int(delta.total_seconds())
        if secs < 60: return f"{secs}s ago"
        if secs < 3600: return f"{secs // 60}m ago"
        return f"{secs // 3600}h {(secs % 3600) // 60}m ago"
    except Exception:
        return "Unknown"

def gate_badge(passed: bool, label: str) -> str:
    cls = "badge-pass" if passed else "badge-fail"
    icon = "&#10003;" if passed else "&#10007;"
    return f'<span class="{cls}">{icon} {label}</span>'

def _diff_cell(val: float, threshold: float, fmt: str = ".4f") -> str:
    diff = val - threshold
    if val == 0:
        return '<span style="color:#555">-</span>'
    color = "#00d97e" if diff >= 0 else "#e63757"
    sign = "+" if diff >= 0 else ""
    return f'<b>{val:{fmt}}</b> <span style="color:{color};font-size:0.8em">({sign}{diff:{fmt}})</span>'

def compute_regression_channel(close_arr: np.ndarray, period: int) -> dict | None:
    try:
        from neo_flow.adaptive_engine import calc_log_regression
        if len(close_arr) < period:
            return None
        std_dev, pearson_r, slope, intercept = calc_log_regression(close_arr, period)
        x = np.arange(period, dtype=np.float64)
        log_line = intercept + x * slope
        log_line = log_line[::-1]
        mid_line = np.exp(log_line)
        upper = np.exp(log_line + std_dev)
        lower = np.exp(log_line - std_dev)
        return {"midline": mid_line, "upper": upper, "lower": lower,
                "pearson_r": pearson_r, "slope": slope, "std_dev": std_dev, "period": period}
    except Exception:
        return None

def _find_best_period(close_arr: np.ndarray) -> int:
    try:
        from neo_flow.adaptive_engine import calc_log_regression
        best_r, best_p = 0, 80
        for p in [30, 50, 80, 100, 120, 150, 200]:
            if len(close_arr) < p: break
            _, r, _, _ = calc_log_regression(close_arr, p)
            if abs(r) > best_r:
                best_r = abs(r)
                best_p = p
        return best_p
    except Exception:
        return 80


# ═══════════════════════════════════════════════════════════════════════════════
# Live Ticker Bridge (Websocket)
# ═══════════════════════════════════════════════════════════════════════════════

class LiveTickerBridge:
    """Singleton bridge to store latest ticker prices from a background thread."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiveTickerBridge, cls).__new__(cls)
            cls._instance.prices = {}
            cls._instance.last_update = 0
            cls._instance._start_thread()
        return cls._instance

    def _start_thread(self):
        import threading
        def _run():
            import json
            from websocket import WebSocketApp
            def on_message(ws, msg):
                try:
                    data = json.loads(msg)
                    symbol = data["s"]
                    price = float(data["c"])
                    self.prices[symbol] = price
                    self.last_update = time.time()
                except: pass
            
            streams = [f"{s.lower()}@ticker" for s in ALL_ASSETS]
            url = f"wss://fstream.binance.com/ws/{'/'.join(streams)}"
            ws = WebSocketApp(url, on_message=on_message)
            ws.run_forever()
        
        t = threading.Thread(target=_run, daemon=True)
        t.start()

TICKER_BRIDGE = LiveTickerBridge()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Cockpit
# ═══════════════════════════════════════════════════════════════════════════════

def render_cockpit(state: dict, prices: dict):
    positions = state.get("positions", {})
    initial = state.get("initial_capital", 1000.0)
    realized = state.get("realized_pnl", 0.0)
    equity = initial + realized
    peak = state.get("peak_equity", initial)
    dd = (equity - peak) / peak * 100 if peak else 0
    cb = state.get("circuit_breaker", False)
    pid = get_bot_pid()
    is_live = _detect_live_mode()

    # Group counts
    grp_counts = {}
    for pos in positions.values():
        g = pos.get("group", "?")
        grp_counts[g] = grp_counts.get(g, 0) + 1
    grp_str = " | ".join(f"{g}:{c}" for g, c in sorted(grp_counts.items())) or "none"

    # Bot metrics
    mode_str = "LIVE" if is_live else "DRY-RUN"
    bot_status = "CB TRIPPED" if cb else ("RUNNING" if pid else "STOPPED")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Bot Equity", f"${equity:,.2f}", f"{realized:+,.2f}")
    c2.metric("Drawdown", f"{dd:.2f}%")
    c3.metric("Open / Max", f"{len(positions)} / 8")
    c4.metric("Groups", grp_str)
    c5.metric("Scans", f"{state.get('total_scans', 0)}")
    if is_live:
        c6.metric("Mode", f"{mode_str} — {bot_status}", delta="REAL MONEY", delta_color="inverse")
    else:
        c6.metric("Mode", f"{mode_str} — {bot_status}")

    # Binance live account section
    if is_live:
        binance = fetch_binance_account()
        if binance:
            st.markdown("---")
            st.subheader("Binance Account")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("USDT Balance", f"${binance['balance_total']:,.2f}")
            b2.metric("Free", f"${binance['balance_free']:,.2f}")
            b3.metric("In Positions", f"${binance['balance_used']:,.2f}")
            unrealized = sum(p["unrealized_pnl"] for p in binance["positions"])
            b4.metric("Unrealized PnL", f"${unrealized:+,.2f}")

            if binance["positions"]:
                st.caption("Binance Open Positions")
                for bp in binance["positions"]:
                    # Find matching bot position
                    asset = bp["symbol"].replace("/USDT:USDT", "")
                    bot_pos = positions.get(asset, {})
                    bot_entry = bot_pos.get("entry_price", 0) if bot_pos else 0

                    with st.container(border=True):
                        pc1, pc2, pc3 = st.columns(3)
                        with pc1:
                            side_emoji = ":green_circle:" if bp["side"] == "long" else ":red_circle:"
                            st.markdown(f"{side_emoji} **{asset}** `{bp['side'].upper()}` {bp['leverage']}x")
                        with pc2:
                            st.markdown(f"Binance entry: `{bp['entry_price']:.6f}`")
                            if bot_entry > 0:
                                st.markdown(f"Bot entry: `{bot_entry:.6f}`")
                                slip = abs(bp["entry_price"] - bot_entry) / bot_entry * 100
                                st.caption(f"Slippage: {slip:.3f}%")
                        with pc3:
                            pnl_cls = pnl_color(bp["unrealized_pnl"])
                            st.markdown(f'PnL: <span class="{pnl_cls}">${bp["unrealized_pnl"]:+.2f}</span>',
                                        unsafe_allow_html=True)
                            st.caption(f"Qty: {bp['contracts']}")
            elif not positions:
                st.info("No positions on Binance or bot.")

    st.markdown("---")

    if not positions:
        st.info("No active positions. Waiting for signals across 33 assets...")
        return

    st.subheader("Active Trades")

    for asset, pos in positions.items():
        symbol = pos.get("symbol", f"{asset}USDT")
        current_price = prices.get(symbol, 0)
        entry_price = pos["entry_price"]
        direction = pos["direction"]
        leverage = pos["leverage"]
        group = pos.get("group", get_group(asset))

        if current_price > 0 and entry_price > 0:
            live_pnl = (current_price - entry_price) / entry_price * direction * 100 * leverage
        else:
            live_pnl = 0.0

        trail_sl = pos["trail_sl"]
        hard_sl = pos["hard_sl"]
        if direction == 1 and current_price > 0:
            trail_pct = (current_price - trail_sl) / (current_price - hard_sl) * 100 if current_price > hard_sl else 0
        elif direction == -1 and current_price > 0:
            trail_pct = (trail_sl - current_price) / (hard_sl - current_price) * 100 if hard_sl > current_price else 0
        else:
            trail_pct = 50

        dir_str = "LONG" if direction == 1 else "SHORT"
        dir_emoji = ":chart_with_upwards_trend:" if direction == 1 else ":chart_with_downwards_trend:"

        with st.container(border=True):
            tc1, tc2, tc3, tc4 = st.columns([1.5, 2, 2, 1.5])
            with tc1:
                st.markdown(f"**{dir_emoji} {asset}** `{dir_str}`")
                st.markdown(f'{group_badge(group)}', unsafe_allow_html=True)
                st.caption(f"TF: {pos['best_tf']} | P: {pos['best_period']} | Lev: {leverage}x")
            with tc2:
                st.markdown(f"Entry: `{entry_price:.6f}`")
                st.markdown(f"Now: `{current_price:.6f}`" if current_price > 0 else "Now: `fetching...`")
                pnl_cls = pnl_color(live_pnl)
                st.markdown(f'PnL: <span class="{pnl_cls}">{live_pnl:+.2f}%</span>', unsafe_allow_html=True)
            with tc3:
                st.caption("Confidence")
                st.markdown(f"Pearson R: **{pos['confidence']:.4f}** | PVT R: **{pos['pvt_r']:.4f}**")
                st.markdown(f"XGB Prob: **{pos.get('xgb_prob', 0):.4f}**")
            with tc4:
                st.caption("Trail Stop Distance")
                st.markdown(trail_bar_html(trail_pct), unsafe_allow_html=True)
                st.caption(f"SL: {trail_sl:.6f} | Hard: {hard_sl:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Decision Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

def render_intelligence(state: dict, scan_log: dict):
    now = datetime.now(timezone.utc)

    # WebSocket Gear: 5-minute intervals
    next_5m = (now + timedelta(minutes=5)).replace(second=0, microsecond=0)
    next_5m = next_5m.replace(minute=(next_5m.minute // 5) * 5)
    rem_5m = (next_5m - now).total_seconds()
    m5, s5 = int(rem_5m // 60), int(rem_5m % 60)

    # Hourly Gear: Audit cycle
    next_1h = now.replace(minute=1, second=0, microsecond=0)
    if next_1h <= now:
        next_1h += timedelta(hours=1)
    rem_1h = (next_1h - now).total_seconds()
    m1h, s1h = int(rem_1h // 60), int(rem_1h % 60)

    c1, c2, c3 = st.columns(3)
    c1.metric("Next 5m Scan", f"{m5:02d}:{s5:02d}", delta="WS stream active")
    c2.metric("Last Activity", time_since(state.get("last_scan_ts", "")))
    c3.metric("Next Audit (1h)", f"{m1h:02d}:{s1h:02d}")
    st.markdown("---")

    # Group filter
    group_filter = st.radio("Filter by Group", ["All", "A (Majors)", "B (Tech/AI)", "C (Meme)"], horizontal=True)

    # Show group-specific thresholds
    with st.expander("Group Thresholds", expanded=False):
        th_cols = st.columns(3)
        for i, (g, gt) in enumerate(GROUP_THRESHOLDS.items()):
            with th_cols[i]:
                st.markdown(f"**Group {g} — {GROUP_LABELS.get(g, '')}**")
                st.caption(f"|R| >= {gt.min_confidence} | PVT >= {gt.min_pvt_r} | XGB > {gt.min_xgb_score}")
                st.caption(f"Combined <= {gt.combined_gate} | SL: {gt.hard_sl_mult}x | Trail: {gt.trail_buffer}")

    st.subheader(f"Live Vetting — {len(ASSETS)} Coins vs Group Thresholds")

    # Load cached scan or run new one on button click
    scan_results = None
    cached_ts = ""
    if DASHBOARD_SCAN_CACHE.exists():
        try:
            cache = json.loads(DASHBOARD_SCAN_CACHE.read_text())
            scan_results = cache.get("results", [])
            cached_ts = cache.get("timestamp", "")
        except Exception:
            pass

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        do_scan = st.button("Scan Now", type="primary", use_container_width=True)
    with info_col:
        if cached_ts:
            st.caption(f"Last dashboard scan: {cached_ts}")
        else:
            st.caption("No cached scan — click **Scan Now** to run")

    if do_scan:
        with st.spinner(f"Scanning {len(ASSETS)} assets through 5 gates..."):
            scan_results = run_dashboard_scan()
        if scan_results and not (len(scan_results) == 1 and scan_results[0].get("asset") == "ERROR"):
            DASHBOARD_SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "results": scan_results,
            }
            DASHBOARD_SCAN_CACHE.write_text(json.dumps(cache_data, indent=2))

    if not scan_results or (len(scan_results) == 1 and scan_results[0].get("asset") == "ERROR"):
        if scan_results:
            st.error(f"Scan error: {scan_results[0].get('reject_reason', 'Unknown')}")
        else:
            st.info("Click **Scan Now** to run a live scan of all 33 assets.")
        _render_bot_scan_log(scan_log)
        return

    # Apply group filter
    if group_filter.startswith("A"):
        scan_results = [s for s in scan_results if s.get("group") == "A"]
    elif group_filter.startswith("B"):
        scan_results = [s for s in scan_results if s.get("group") == "B"]
    elif group_filter.startswith("C"):
        scan_results = [s for s in scan_results if s.get("group") == "C"]

    qualified = [s for s in scan_results if s["final_status"] == "qualified"]
    others = [s for s in scan_results if s["final_status"] != "qualified"]
    n_xgb_rej = sum(1 for s in others if s["final_status"] == "xgb_rejected")
    n_physics = len(others) - n_xgb_rej

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Qualified", f"{len(qualified)}", delta="all 5 gates", delta_color="off")
    sc2.metric("XGB Rejected", f"{n_xgb_rej}", delta="passed physics", delta_color="off")
    sc3.metric("Physics Blocked", f"{n_physics}", delta="failed early", delta_color="off")
    sc4.metric("Total Scanned", f"{len(scan_results)}")

    st.markdown("---")
    st.subheader("Distance from Thresholds")

    # Sort by group first, then by gates/R within each group
    group_order = {"A": 0, "B": 1, "C": 2}
    sorted_results = sorted(
        scan_results,
        key=lambda s: (
            group_order.get(s.get("group", "?"), 9),
            -(s["final_status"] == "qualified"),
            -sum([s["gate_r"], s["gate_pvt"], s["gate_htf"], s["gate_combined"], s["gate_xgb"]]),
            -s["best_r"],
        ),
    )

    header = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        '<thead><tr style="border-bottom:2px solid #333;color:#8b92a5;">'
        '<th style="text-align:left;padding:5px;">Coin</th>'
        '<th style="text-align:left;padding:5px;">Grp</th>'
        '<th style="text-align:left;padding:5px;">Dir</th>'
        '<th style="text-align:left;padding:5px;">TF/P</th>'
        '<th style="text-align:center;padding:5px;">|R| (vs thr)</th>'
        '<th style="text-align:center;padding:5px;">PVT |R| (vs thr)</th>'
        '<th style="text-align:center;padding:5px;">4H</th>'
        '<th style="text-align:center;padding:5px;">Combined</th>'
        '<th style="text-align:center;padding:5px;">XGB (vs thr)</th>'
        '<th style="text-align:center;padding:5px;">Gates</th>'
        '<th style="text-align:left;padding:5px;">Blocked</th>'
        '</tr></thead><tbody>'
    )

    GROUP_BG_TINT = {
        "A": "rgba(57,175,209,0.04)",   # blue tint
        "B": "rgba(246,195,67,0.04)",    # yellow tint
        "C": "rgba(230,55,87,0.04)",     # red tint
    }
    GROUP_BORDER_LEFT = {
        "A": "border-left:3px solid #39afd1;",
        "B": "border-left:3px solid #f6c343;",
        "C": "border-left:3px solid #e63757;",
    }

    rows_html = ""
    prev_group = None
    for s in sorted_results:
        gates = sum([s["gate_r"], s["gate_pvt"], s["gate_htf"], s["gate_combined"], s["gate_xgb"]])
        is_qualified = s["final_status"] == "qualified"
        group = s.get("group", "?")
        gt = GROUP_THRESHOLDS.get(group, DEFAULT_THRESHOLDS.get(group, GroupThresholds()))

        # Group separator row
        if group != prev_group:
            grp_color = GROUP_COLORS.get(group, "#888")
            grp_label = GROUP_LABELS.get(group, "")
            grp_thr = f"|R|>={gt.min_confidence}  PVT>={gt.min_pvt_r}  XGB>{gt.min_xgb_score}"
            rows_html += (
                f'<tr style="background:{GROUP_BG_TINT.get(group, "transparent")};border-top:2px solid {grp_color};">'
                f'<td colspan="11" style="padding:6px 5px;font-weight:700;color:{grp_color};">'
                f'Group {group} — {grp_label} '
                f'<span style="font-weight:400;font-size:0.8em;color:#8b92a5;">({grp_thr})</span>'
                f'</td></tr>'
            )
            prev_group = group

        if is_qualified:
            bg = "background:rgba(0,217,126,0.08);"
        elif gates >= 4:
            bg = "background:rgba(246,195,67,0.06);"
        else:
            bg = f"background:{GROUP_BG_TINT.get(group, 'transparent')};"

        left_border = GROUP_BORDER_LEFT.get(group, "")

        dir_str = "LONG" if s["direction"] == 1 else ("SHORT" if s["direction"] == -1 else "-")
        dir_color = "#00d97e" if s["direction"] == 1 else ("#e63757" if s["direction"] == -1 else "#555")

        r_cell = _diff_cell(s["best_r"], gt.min_confidence)
        pvt_cell = _diff_cell(s["pvt_r"], gt.min_pvt_r)
        if s.get("pvt_detail"):
            pvt_color = "#00d97e" if s["pvt_detail"] == "RISING" else ("#e63757" if s["pvt_detail"] == "FALLING" else "#888")
            pvt_cell += f' <span style="color:{pvt_color};font-size:0.7em">{s["pvt_detail"]}</span>'

        htf = s.get("htf_bias_str", "-")
        htf_color = "#00d97e" if htf == "BULL" else ("#e63757" if htf == "BEAR" else "#888")
        htf_cell = f'<span style="color:{htf_color}">{htf}</span>'
        if s["direction"] != 0 and htf not in ("NEUTRAL", "-"):
            aligned = (s["direction"] == 1 and htf == "BULL") or (s["direction"] == -1 and htf == "BEAR")
            htf_cell += f' <span style="color:{"#00d97e" if aligned else "#e63757"}">{"&#10003;" if aligned else "&#10007;"}</span>'
        htf_detail = s.get("htf_detail", "")
        if htf_detail:
            htf_cell += f'<br><span style="color:#8b92a5;font-size:0.7em">{htf_detail}</span>'

        comb_detail = s.get("combined_detail", "")
        if s["gate_combined"]:
            comb_cell = f'<span style="color:#00d97e">&#10003;</span>'
        elif comb_detail:
            comb_cell = f'<span style="color:#e63757">&#10007;</span>'
        else:
            comb_cell = '<span style="color:#555">-</span>'

        xgb_cell = _diff_cell(s["xgb_prob"], gt.min_xgb_score)

        if gates == 5:
            gates_cell = '<span style="color:#00d97e;font-weight:700">5/5</span>'
        elif gates >= 4:
            gates_cell = f'<span style="color:#f6c343;font-weight:700">{gates}/5</span>'
        else:
            gates_cell = f'<span style="color:#e63757">{gates}/5</span>'

        coin_label = f"<b>{s['asset']}</b>"
        if is_qualified:
            coin_label += ' <span class="badge-pass">OK</span>'

        grp_color = GROUP_COLORS.get(group, "#888")
        grp_cell = f'<span style="color:{grp_color};font-weight:600">{group}</span>'

        first_fail = s.get("first_fail", "")
        blocked = f'<span style="color:#e63757">{first_fail}</span>' if first_fail and not is_qualified else '<span style="color:#555">-</span>'

        rows_html += (
            f'<tr style="border-bottom:1px solid #1e2235;{bg}{left_border}">'
            f'<td style="padding:5px;">{coin_label}</td>'
            f'<td style="padding:5px;">{grp_cell}</td>'
            f'<td style="padding:5px;color:{dir_color}">{dir_str}</td>'
            f'<td style="padding:5px;color:#aaa">{s["best_tf"]}/{s["best_period"]}</td>'
            f'<td style="text-align:center;padding:5px;">{r_cell}</td>'
            f'<td style="text-align:center;padding:5px;">{pvt_cell}</td>'
            f'<td style="text-align:center;padding:5px;">{htf_cell}</td>'
            f'<td style="text-align:center;padding:5px;">{comb_cell}</td>'
            f'<td style="text-align:center;padding:5px;">{xgb_cell}</td>'
            f'<td style="text-align:center;padding:5px;">{gates_cell}</td>'
            f'<td style="padding:5px;">{blocked}</td>'
            f'</tr>'
        )

    st.markdown(header + rows_html + '</tbody></table>', unsafe_allow_html=True)

    if qualified:
        st.markdown("---")
        st.markdown(f"### Qualified Signals ({len(qualified)})")
        for s in sorted(qualified, key=lambda x: x["xgb_prob"], reverse=True):
            group = s.get("group", "?")
            gt = GROUP_THRESHOLDS.get(group, GroupThresholds())
            dir_str = "LONG" if s["direction"] == 1 else "SHORT"
            dir_icon = ":chart_with_upwards_trend:" if s["direction"] == 1 else ":chart_with_downwards_trend:"
            with st.container(border=True):
                hc1, hc2, hc3 = st.columns([1.5, 3, 2])
                with hc1:
                    st.markdown(f"**{dir_icon} {s['asset']}** `{dir_str}`")
                    st.markdown(f'{group_badge(group)}', unsafe_allow_html=True)
                    st.caption(f"TF: {s['best_tf']} | Period: {s['best_period']}")
                with hc2:
                    pcol, mcol = st.columns(2)
                    with pcol:
                        st.metric("Pearson |R|", f"{s['best_r']:.4f}", f"{s['r_diff']:+.4f}")
                        st.metric("PVT |R|", f"{s['pvt_r']:.4f}", f"{s['pvt_diff']:+.4f}")
                    with mcol:
                        st.metric("XGB Probability", f"{s['xgb_prob']:.4f}", f"{s['xgb_diff']:+.4f}")
                        st.metric("Midline", f"{s['midline']:.6f}")
                with hc3:
                    st.caption("Gate Status")
                    gates_html = "".join([
                        f'<div class="gate-row">{gate_badge(s["gate_r"], "R-Gate")}</div>',
                        f'<div class="gate-row">{gate_badge(s["gate_pvt"], "PVT-Gate")}</div>',
                        f'<div class="gate-row">{gate_badge(s["gate_htf"], "4H-Trend")}</div>',
                        f'<div class="gate-row">{gate_badge(s["gate_combined"], "Combined")}</div>',
                        f'<div class="gate-row">{gate_badge(s["gate_xgb"], "XGBoost")}</div>',
                    ])
                    st.markdown(gates_html, unsafe_allow_html=True)

    st.markdown("---")
    _render_bot_scan_log(scan_log)


def _render_bot_scan_log(scan_log: dict):
    with st.expander("Bot's Last Scan Log", expanded=False):
        signals = scan_log.get("signals", [])
        ts = scan_log.get("timestamp", "")
        scan_num = scan_log.get("scan_number", 0)
        if ts:
            st.caption(f"Bot scan #{scan_num} at: {ts}")
        if signals:
            STATUS_COLORS = {
                "entered": "#00d97e", "xgb_rejected": "#f6c343",
                "physics_rejected": "#e63757", "in_position": "#39afd1",
                "just_closed": "#888", "no_data": "#555", "max_positions": "#888",
                "scanning": "#39afd1",
            }
            rows_html = ""
            for s in signals:
                status = s.get("status", "?")
                color = STATUS_COLORS.get(status, "#888")
                asset = s.get("asset", "?")
                group = s.get("group", ASSET_TO_GROUP.get(asset, "?"))
                detail_parts = []
                if s.get("direction"):
                    detail_parts.append("LONG" if s["direction"] == 1 else "SHORT")
                if s.get("confidence"):
                    detail_parts.append(f"|R|={s['confidence']:.4f}")
                if s.get("xgb_prob"):
                    detail_parts.append(f"XGB={s['xgb_prob']:.4f}")
                if s.get("tf"):
                    detail_parts.append(f"{s['tf']}/{s.get('period','')}")
                detail = " | ".join(detail_parts)
                grp_color = GROUP_COLORS.get(group, "#888")
                rows_html += (
                    f'<tr style="border-bottom:1px solid #1e2235;">'
                    f'<td style="padding:4px 6px;"><b>{asset}</b></td>'
                    f'<td style="padding:4px 6px;color:{grp_color}">{group}</td>'
                    f'<td style="padding:4px 6px;color:{color}">{status}</td>'
                    f'<td style="padding:4px 6px;color:#aaa;font-size:0.85em">{detail}</td>'
                    f'</tr>'
                )
            table = (
                '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
                '<thead><tr style="border-bottom:2px solid #333;color:#8b92a5;">'
                '<th style="text-align:left;padding:4px 6px;">Coin</th>'
                '<th style="text-align:left;padding:4px 6px;">Grp</th>'
                '<th style="text-align:left;padding:4px 6px;">Status</th>'
                '<th style="text-align:left;padding:4px 6px;">Details</th>'
                '</tr></thead><tbody>' + rows_html + '</tbody></table>'
            )
            st.markdown(table, unsafe_allow_html=True)
        else:
            st.caption("No scan data recorded yet.")

    with st.expander("Bot Log (last 30 lines)", expanded=False):
        log_lines = parse_recent_logs(30)
        if log_lines:
            st.code("\n".join(log_lines), language="log")
        else:
            st.caption("No log data available.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Live Charting
# ═══════════════════════════════════════════════════════════════════════════════

def render_charting(state: dict, trades_df: pd.DataFrame):
    # Live Ticker Integration
    positions = state.get("positions", {})
    open_assets = list(positions.keys())
    other_assets = [a for a in ASSETS if a not in open_assets]
    asset_list = open_assets + other_assets

    col_sel, col_tf, col_grp = st.columns([2, 1, 1])
    with col_sel:
        selected_asset = st.selectbox(
            "Select Asset",
            asset_list,
            format_func=lambda a: f"{a} [{get_group(a)}] {'(OPEN)' if a in positions else ''}",
        )
    
    # Get Live Price from WebSocket Bridge
    ws_price = TICKER_BRIDGE.prices.get(f"{selected_asset}USDT", 0)
    
    with col_tf:
        selected_tf = st.selectbox("Timeframe", SCAN_TIMEFRAMES, index=2)
    with col_grp:
        if selected_asset:
            if ws_price > 0:
                st.markdown(f"**Live Price: `${ws_price:.4f}`** :signal_strength:")
            else:
                st.markdown(f"**Group {get_group(selected_asset)}**")

    if not selected_asset:
        return

    symbol = f"{selected_asset}USDT"
    with st.spinner(f"Loading {symbol} {selected_tf}..."):
        df_candles = fetch_klines(symbol, selected_tf, limit=300)

    if df_candles.empty:
        st.error(f"Failed to fetch candle data for {symbol}")
        return

    # Update latest candle with WebSocket price for real-time feel
    if ws_price > 0:
        df_candles.iloc[-1, df_candles.columns.get_loc("close")] = ws_price
        if ws_price > df_candles.iloc[-1]["high"]:
            df_candles.iloc[-1, df_candles.columns.get_loc("high")] = ws_price
        if ws_price < df_candles.iloc[-1]["low"]:
            df_candles.iloc[-1, df_candles.columns.get_loc("low")] = ws_price

    close_arr = df_candles["close"].values.astype(np.float64)

    if selected_asset in positions:
        chart_period = positions[selected_asset]["best_period"]
    else:
        chart_period = _find_best_period(close_arr)

    channel = compute_regression_channel(close_arr, chart_period)

    # XGBoost sentiment
    xgb_prob = 0.0
    xgb_direction = 0
    group = get_group(selected_asset)
    gt = GROUP_THRESHOLDS.get(group, GroupThresholds())
    try:
        from neo_flow.adaptive_engine import calc_log_regression, compute_pvt, calc_linear_regression
        from ml.train_meta_model import predict_probability, load_model

        std_dev, pearson_r, slope, intercept = calc_log_regression(close_arr, chart_period)
        direction = 1 if slope < 0 else -1
        xgb_direction = direction

        pvt_r_val = 0.0
        if len(df_candles) >= chart_period:
            pvt_arr = compute_pvt(df_candles.iloc[-chart_period:])
            _, pvt_r, _, _ = calc_linear_regression(pvt_arr, min(chart_period, len(pvt_arr)))
            pvt_r_val = abs(pvt_r)

        xgb_path = BASE_DIR / "models" / "meta_xgb.json"
        if xgb_path.exists():
            model, _ = load_model(xgb_path)
            xgb_prob = predict_probability(
                model, confidence=abs(pearson_r), pvt_r=pvt_r_val,
                best_tf=selected_tf, best_period=chart_period, pnl_pct=0.0,
            )
    except Exception:
        pass

    # Build chart
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.20, 0.15], vertical_spacing=0.03,
    )

    timestamps = df_candles.index

    fig.add_trace(go.Candlestick(
        x=timestamps, open=df_candles["open"], high=df_candles["high"],
        low=df_candles["low"], close=df_candles["close"], name="Price",
        increasing_line_color="#00d97e", decreasing_line_color="#e63757",
        increasing_fillcolor="#00d97e", decreasing_fillcolor="#e63757",
    ), row=1, col=1)

    if channel is not None:
        period = channel["period"]
        ch_timestamps = timestamps[-period:]
        fig.add_trace(go.Scatter(x=ch_timestamps, y=channel["upper"], mode="lines",
            name="Upper (+1s)", line=dict(color="rgba(249,115,22,0.5)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ch_timestamps, y=channel["midline"], mode="lines",
            name="Midline", line=dict(color="#f59e0b", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ch_timestamps, y=channel["lower"], mode="lines",
            name="Lower (-1s)", line=dict(color="rgba(249,115,22,0.5)", width=1, dash="dot")), row=1, col=1)

    if selected_asset in positions:
        pos = positions[selected_asset]
        fig.add_hline(y=pos["trail_sl"], line_dash="dash", line_color="#f6c343",
            annotation_text=f"Trail SL: {pos['trail_sl']:.6f}", annotation_font_color="#f6c343", row=1, col=1)
        fig.add_hline(y=pos["hard_sl"], line_dash="dash", line_color="#e63757",
            annotation_text=f"Hard SL: {pos['hard_sl']:.6f}", annotation_font_color="#e63757", row=1, col=1)
        fig.add_hline(y=pos["entry_price"], line_dash="dot", line_color="#39afd1",
            annotation_text=f"Entry: {pos['entry_price']:.6f}", annotation_font_color="#39afd1", row=1, col=1)

    if not trades_df.empty:
        asset_trades = trades_df[trades_df["asset"] == selected_asset].copy()
        if not asset_trades.empty:
            for _, t in asset_trades.iterrows():
                color = "#00d97e" if t.get("direction") in [1, "LONG"] else "#e63757"
                marker_sym = "triangle-up" if t.get("direction") in [1, "LONG"] else "triangle-down"
                fig.add_trace(go.Scatter(x=[t["entry_ts"]], y=[t["entry_price"]], mode="markers",
                    marker=dict(symbol=marker_sym, size=14, color=color, line=dict(width=1, color="white")),
                    showlegend=False, hovertemplate=f"ENTRY {t['entry_price']:.6f}<extra></extra>"), row=1, col=1)
            for _, t in asset_trades.iterrows():
                pnl = t.get("pnl_pct", 0)
                color = "#00d97e" if pnl > 0 else "#e63757"
                fig.add_trace(go.Scatter(x=[t["exit_ts"]], y=[t["exit_price"]], mode="markers",
                    marker=dict(symbol="x", size=12, color=color, line=dict(width=2, color=color)),
                    showlegend=False, hovertemplate=f"EXIT {t['exit_price']:.6f} ({pnl:+.2f}%)<extra></extra>"), row=1, col=1)

    vol_colors = ["#00d97e" if c >= o else "#e63757" for c, o in zip(df_candles["close"], df_candles["open"])]
    fig.add_trace(go.Bar(x=timestamps, y=df_candles["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.5, showlegend=False), row=2, col=1)

    sentiment_label = "BULLISH" if xgb_direction == 1 else "BEARISH" if xgb_direction == -1 else "NEUTRAL"
    sent_color = f"rgba(0,217,126,{min(xgb_prob, 1.0):.2f})" if xgb_direction == 1 else f"rgba(230,55,87,{min(xgb_prob, 1.0):.2f})" if xgb_prob > 0 else "rgba(100,100,100,0.3)"
    fig.add_trace(go.Bar(x=timestamps[-50:], y=[xgb_prob] * min(50, len(timestamps)),
        name=f"XGB: {xgb_prob:.3f} ({sentiment_label})", marker_color=sent_color, showlegend=True), row=3, col=1)
    fig.add_hline(y=gt.min_xgb_score, line_dash="dash", line_color="#555",
        annotation_text=f"Threshold: {gt.min_xgb_score}", row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=750,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=60, r=20, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="XGB", row=3, col=1, range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

    if channel is not None:
        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        ic1.metric("Pearson |R|", f"{abs(channel['pearson_r']):.4f}")
        ic2.metric("Period", f"{channel['period']}")
        ic3.metric("Midline", f"{channel['midline'][-1]:.6f}")
        ic4.metric("XGB Prob", f"{xgb_prob:.4f}")
        ic5.metric("Sentiment", sentiment_label)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Performance Analytics
# ═══════════════════════════════════════════════════════════════════════════════

def render_analytics(state: dict, trades_df: pd.DataFrame, wfv: dict):
    initial_cap = state.get("initial_capital", 1000.0)

    if trades_df.empty:
        st.info("No completed trades yet. Analytics will populate as the bot trades.")
        blind = wfv.get("blind_test", {})
        if blind:
            st.markdown("---")
            st.subheader("Blind Test Reference (WFV)")
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            bc1.metric("Trades", f"{blind.get('trades', 0)}")
            bc2.metric("Win Rate", f"{blind.get('win_rate', 0):.1f}%")
            bc3.metric("PnL", f"{blind.get('pnl_pct', 0):+.1f}%")
            bc4.metric("Sharpe", f"{blind.get('sharpe', 0):.2f}")
            bc5.metric("Profit Factor", f"{blind.get('profit_factor', 0):.2f}")

        consensus = wfv.get("consensus_params", {})
        if consensus:
            st.markdown("---")
            st.subheader("Consensus Parameters")
            param_df = pd.DataFrame([consensus]).T
            param_df.columns = ["Value"]
            st.dataframe(param_df, use_container_width=True)
        return

    st.subheader("Equity Curve")
    df = trades_df.sort_values("exit_ts").copy()
    df["cumulative_pnl"] = df["pnl_usd"].cumsum()
    df["equity"] = initial_cap + df["cumulative_pnl"]

    fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig_eq.add_trace(go.Scatter(
        x=df["exit_ts"], y=df["equity"], mode="lines+markers", name="Equity",
        line=dict(color="#00d97e", width=2), marker=dict(size=3),
        fill="tozeroy", fillcolor="rgba(0,217,126,0.08)",
    ), row=1, col=1)
    fig_eq.add_hline(y=initial_cap, line_dash="dash", line_color="#555", annotation_text="Initial Capital", row=1, col=1)

    eq_arr = df["equity"].values
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100
    fig_eq.add_trace(go.Scatter(
        x=df["exit_ts"], y=dd_pct, fill="tozeroy", name="Drawdown",
        line=dict(color="#e63757", width=1.5), fillcolor="rgba(230,55,87,0.15)",
    ), row=2, col=1)
    fig_eq.update_layout(template="plotly_dark", height=450, showlegend=False,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", margin=dict(l=60, r=20, t=20, b=40))
    fig_eq.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig_eq.update_yaxes(title_text="DD %", row=2, col=1)
    st.plotly_chart(fig_eq, use_container_width=True)

    # Stats
    total_trades = len(df)
    winners = df[df["pnl_usd"] > 0]
    losers = df[df["pnl_usd"] <= 0]
    win_rate = len(winners) / total_trades * 100 if total_trades else 0
    total_pnl = df["pnl_usd"].sum()
    gross_profit = winners["pnl_usd"].sum() if len(winners) else 0
    gross_loss = abs(losers["pnl_usd"].sum()) if len(losers) else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    max_dd = float(dd_pct.min()) if len(dd_pct) else 0

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Total Trades", f"{total_trades}")
    sc2.metric("Win Rate", f"{win_rate:.1f}%")
    sc3.metric("Total PnL", f"${total_pnl:+,.2f}")
    sc4.metric("Profit Factor", f"{pf:.2f}")
    sc5.metric("Max DD", f"{max_dd:.2f}%")

    # Distribution charts
    st.markdown("---")
    st.subheader("Trade Distribution")
    pc1, pc2, pc3, pc4 = st.columns(4)

    with pc1:
        if "group" in df.columns:
            grp_counts = df["group"].value_counts()
            colors = [GROUP_COLORS.get(g, "#888") for g in grp_counts.index]
            fig_grp = go.Figure(data=[go.Pie(labels=grp_counts.index.tolist(), values=grp_counts.values.tolist(), hole=0.4, marker=dict(colors=colors))])
            fig_grp.update_layout(template="plotly_dark", height=280, title="By Group", paper_bgcolor="#0e1117", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_grp, use_container_width=True)

    with pc2:
        if "best_tf" in df.columns:
            tf_counts = df["best_tf"].value_counts()
            fig_tf = go.Figure(data=[go.Pie(labels=tf_counts.index.tolist(), values=tf_counts.values.tolist(), hole=0.4, marker=dict(colors=["#00d97e", "#39afd1", "#f6c343", "#e63757"]))])
            fig_tf.update_layout(template="plotly_dark", height=280, title="By Timeframe", paper_bgcolor="#0e1117", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_tf, use_container_width=True)

    with pc3:
        if "direction" in df.columns:
            dir_map = df["direction"].map({1: "LONG", -1: "SHORT", "LONG": "LONG", "SHORT": "SHORT"})
            dir_counts = dir_map.value_counts()
            fig_dir = go.Figure(data=[go.Pie(labels=dir_counts.index.tolist(), values=dir_counts.values.tolist(), hole=0.4, marker=dict(colors=["#00d97e", "#e63757"]))])
            fig_dir.update_layout(template="plotly_dark", height=280, title="By Direction", paper_bgcolor="#0e1117", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dir, use_container_width=True)

    with pc4:
        if "exit_reason" in df.columns:
            exit_counts = df["exit_reason"].value_counts()
            fig_exit = go.Figure(data=[go.Pie(labels=exit_counts.index.tolist(), values=exit_counts.values.tolist(), hole=0.4, marker=dict(colors=["#00d97e", "#e63757", "#f6c343", "#39afd1"]))])
            fig_exit.update_layout(template="plotly_dark", height=280, title="By Exit Reason", paper_bgcolor="#0e1117", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_exit, use_container_width=True)

    # Per-group performance table
    if "group" in df.columns:
        st.markdown("---")
        st.subheader("Per-Group Performance")
        grp_stats = []
        for g in ["A", "B", "C"]:
            gdf = df[df["group"] == g]
            if len(gdf) == 0: continue
            gw = gdf[gdf["pnl_usd"] > 0]
            grp_stats.append({
                "Group": f"{g} ({GROUP_LABELS.get(g, '')})",
                "Trades": len(gdf),
                "Win Rate": f"{len(gw)/len(gdf)*100:.1f}%",
                "PnL ($)": f"${gdf['pnl_usd'].sum():+,.2f}",
                "Avg Win": f"{gw['pnl_pct'].mean():.2f}%" if len(gw) else "-",
                "Avg Loss": f"{gdf[gdf['pnl_usd']<=0]['pnl_pct'].mean():.2f}%" if len(gdf[gdf['pnl_usd']<=0]) else "-",
            })
        if grp_stats:
            st.dataframe(pd.DataFrame(grp_stats), use_container_width=True, hide_index=True)

    # Full trade history
    st.markdown("---")
    with st.expander("Full Trade History", expanded=False):
        display_cols = ["trade_id", "asset", "group", "direction", "entry_ts", "exit_ts",
            "entry_price", "exit_price", "best_tf", "leverage", "exit_reason", "bars_held", "pnl_pct", "pnl_usd"]
        cols_present = [c for c in display_cols if c in df.columns]
        st.dataframe(df[cols_present].sort_values("exit_ts", ascending=False), use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Trade History
# ═══════════════════════════════════════════════════════════════════════════════

def render_trade_history(trades_df: pd.DataFrame):
    if trades_df.empty:
        st.info("No completed trades yet. History will populate as the bot closes positions.")
        return

    df = trades_df.copy()

    # ── Filters ───────────────────────────────────────────────────────
    st.subheader("Trade History")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)

    with fc1:
        groups = sorted(df["group"].unique().tolist()) if "group" in df.columns else []
        sel_groups = st.multiselect("Group", groups, default=groups, key="th_groups") if groups else groups
    with fc2:
        assets = sorted(df["asset"].unique().tolist())
        sel_assets = st.multiselect("Asset", assets, default=assets, key="th_assets")
    with fc3:
        dir_map = {"LONG": 1, "SHORT": -1}
        directions = df["direction"].unique().tolist()
        dir_labels = sorted(set("LONG" if d == 1 or str(d).upper() == "LONG" else "SHORT" for d in directions))
        sel_dirs = st.multiselect("Direction", dir_labels, default=dir_labels, key="th_dirs")
    with fc4:
        exit_reasons = sorted(df["exit_reason"].unique().tolist())
        sel_exits = st.multiselect("Exit Reason", exit_reasons, default=exit_reasons, key="th_exits")
    with fc5:
        outcome = st.selectbox("Outcome", ["All", "Winners", "Losers"], key="th_outcome")

    # Apply filters
    sel_dir_vals = [dir_map.get(d, d) for d in sel_dirs]
    if df["direction"].dtype == object:
        mask_dir = df["direction"].str.upper().isin([d.upper() for d in sel_dirs])
    else:
        mask_dir = df["direction"].isin(sel_dir_vals)

    mask = df["asset"].isin(sel_assets) & mask_dir & df["exit_reason"].isin(sel_exits)
    if "group" in df.columns and sel_groups:
        mask = mask & df["group"].isin(sel_groups)

    df = df[mask]

    if outcome == "Winners":
        df = df[df["pnl_pct"] > 0]
    elif outcome == "Losers":
        df = df[df["pnl_pct"] <= 0]

    if df.empty:
        st.warning("No trades match the selected filters.")
        return

    # ── Summary metrics ───────────────────────────────────────────────
    total = len(df)
    winners = (df["pnl_pct"] > 0).sum()
    win_rate = winners / total * 100 if total > 0 else 0
    total_pnl = df["pnl_usd"].sum()
    avg_pnl = df["pnl_pct"].mean()
    best_trade = df["pnl_pct"].max()
    worst_trade = df["pnl_pct"].min()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Trades", f"{total}")
    m2.metric("Win Rate", f"{win_rate:.1f}%")
    m3.metric("Total PnL", f"${total_pnl:+,.2f}")
    m4.metric("Avg PnL %", f"{avg_pnl:+.2f}%")
    m5.metric("Best Trade", f"{best_trade:+.2f}%")
    m6.metric("Worst Trade", f"{worst_trade:+.2f}%")

    st.markdown("---")

    # ── Per-group breakdown ───────────────────────────────────────────
    if "group" in df.columns and len(df["group"].unique()) > 1:
        st.subheader("Performance by Group")
        grp_stats = df.groupby("group").agg(
            trades=("pnl_pct", "count"),
            win_rate=("pnl_pct", lambda x: (x > 0).sum() / len(x) * 100),
            total_pnl=("pnl_usd", "sum"),
            avg_pnl=("pnl_pct", "mean"),
        ).round(2)
        st.dataframe(grp_stats, use_container_width=True)
        st.markdown("---")

    # ── Cumulative PnL chart ──────────────────────────────────────────
    df_sorted = df.sort_values("exit_ts").copy()
    df_sorted["cum_pnl"] = df_sorted["pnl_usd"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["exit_ts"],
        y=df_sorted["cum_pnl"],
        mode="lines+markers",
        line=dict(color="#00d97e", width=2),
        marker=dict(
            size=8,
            color=["#00d97e" if p > 0 else "#e63757" for p in df_sorted["pnl_usd"]],
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b> [%{customdata[4]}] %{customdata[1]}<br>"
            "PnL: %{customdata[2]:+.2f}% ($%{customdata[3]:+.2f})<br>"
            "Exit: %{customdata[5]}<br>"
            "Cum PnL: $%{y:+.2f}<extra></extra>"
        ),
        customdata=list(zip(
            df_sorted["asset"],
            df_sorted["direction"].apply(lambda d: "LONG" if d == 1 else "SHORT"),
            df_sorted["pnl_pct"],
            df_sorted["pnl_usd"],
            df_sorted["group"] if "group" in df_sorted.columns else [""] * len(df_sorted),
            df_sorted["exit_reason"],
        )),
    ))
    fig.update_layout(
        title="Cumulative PnL",
        xaxis_title="Time",
        yaxis_title="Cumulative PnL ($)",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=350,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── PnL per trade bar chart ───────────────────────────────────────
    df_sorted["trade_label"] = (
        df_sorted["trade_id"].astype(str) + " " +
        df_sorted["asset"] + " " +
        df_sorted["direction"].apply(lambda d: "L" if d == 1 else "S")
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_sorted["trade_label"],
        y=df_sorted["pnl_pct"],
        marker_color=["#00d97e" if p > 0 else "#e63757" for p in df_sorted["pnl_pct"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b> [%{customdata[3]}] %{customdata[1]}<br>"
            "PnL: %{y:+.2f}%<br>"
            "Exit: %{customdata[2]}<extra></extra>"
        ),
        customdata=list(zip(
            df_sorted["asset"],
            df_sorted["direction"].apply(lambda d: "LONG" if d == 1 else "SHORT"),
            df_sorted["exit_reason"],
            df_sorted["group"] if "group" in df_sorted.columns else [""] * len(df_sorted),
        )),
    ))
    fig2.update_layout(
        title="PnL per Trade (%)",
        xaxis_title="Trade",
        yaxis_title="PnL %",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Full trade table ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("All Trades")

    display_cols = [
        "trade_id", "asset", "group", "direction", "entry_ts", "exit_ts",
        "entry_price", "exit_price", "best_tf", "confidence", "pvt_r",
        "leverage", "exit_reason", "duration_hours", "pnl_pct", "pnl_usd",
    ]
    cols_present = [c for c in display_cols if c in df.columns]
    display_df = df[cols_present].sort_values("exit_ts", ascending=False).copy()

    if "direction" in display_df.columns:
        display_df["direction"] = display_df["direction"].apply(
            lambda d: "LONG" if d == 1 else "SHORT"
        )
    for col in ["entry_price", "exit_price", "confidence", "pvt_r", "pnl_pct", "pnl_usd", "duration_hours"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda v: round(v, 4) if isinstance(v, (int, float)) else v
            )

    st.dataframe(display_df, use_container_width=True, height=500)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Controls
# ═══════════════════════════════════════════════════════════════════════════════

def render_controls(state: dict):
    st.markdown("**Emergency controls.** Actions here affect the live bot.")
    st.markdown("---")

    pid = get_bot_pid()
    cb = state.get("circuit_breaker", False)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Bot Process")
        if pid:
            st.success(f"Bot is running (PID: {pid})")
        else:
            st.error("Bot is NOT running")

        st.markdown("#### Circuit Breaker")
        if cb:
            st.error("CIRCUIT BREAKER IS TRIPPED")
            if st.button("Reset Circuit Breaker", type="secondary"):
                try:
                    data = json.loads(STATE_FILE.read_text())
                    data["circuit_breaker"] = False
                    STATE_FILE.write_text(json.dumps(data, indent=2))
                    st.success("Circuit breaker reset.")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.success("Circuit Breaker: OK")

    with c2:
        st.markdown("#### Emergency Kill Switch")
        st.markdown("This will **immediately stop** the bot and trip the circuit breaker.")

        if "kill_confirm" not in st.session_state:
            st.session_state.kill_confirm = False

        if not st.session_state.kill_confirm:
            if st.button(":octagonal_sign: EMERGENCY STOP", type="primary", use_container_width=True):
                st.session_state.kill_confirm = True
                st.rerun()
        else:
            st.warning("Are you sure?")
            kc1, kc2 = st.columns(2)
            with kc1:
                if st.button("CONFIRM STOP", type="primary", use_container_width=True):
                    try:
                        data = json.loads(STATE_FILE.read_text())
                        data["circuit_breaker"] = True
                        STATE_FILE.write_text(json.dumps(data, indent=2))
                    except Exception:
                        pass
                    bot_pid = get_bot_pid()
                    if bot_pid:
                        try:
                            import signal as sig
                            import os
                            os.kill(bot_pid, sig.SIGINT)
                            st.success(f"SIGINT sent to PID {bot_pid}.")
                        except Exception as e:
                            st.error(f"Kill failed: {e}")
                    else:
                        st.info("Bot was not running.")
                    st.session_state.kill_confirm = False
                    time.sleep(2)
                    st.rerun()
            with kc2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.kill_confirm = False
                    st.rerun()

    st.markdown("---")
    st.markdown("#### Restart Bot")
    mode = st.radio("Mode", ["Dry-Run", "Live"], horizontal=True)
    capital = st.number_input("Capital ($)", value=1000.0, min_value=100.0, step=100.0)

    if st.button("Start Bot"):
        if get_bot_pid():
            st.warning("Bot is already running. Stop it first.")
        else:
            cmd = f"nohup python3 {BASE_DIR / 'live_extended_bot.py'} --capital {capital}"
            if mode == "Live":
                cmd += " --live"
            cmd += f" > {BASE_DIR / 'logs' / 'bot_stdout.log'} 2>&1 &"
            subprocess.Popen(cmd, shell=True, cwd=str(BASE_DIR))
            st.success(f"Bot started in {mode} mode with ${capital:,.0f}")
            time.sleep(2)
            st.rerun()

    st.markdown("---")
    st.markdown("#### System Health")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        n_parquet = len(list(DATA_DIR.glob("*.parquet"))) if DATA_DIR.exists() else 0
        st.metric("Parquet Files", f"{n_parquet}/132")
    with h2:
        st.metric("Assets", f"{len(ASSETS)}")
    with h3:
        consensus = load_wfv().get("consensus_params", {})
        st.metric("Min |R| (consensus)", f"{consensus.get('min_pearson_r', 'N/A')}")
    with h4:
        st.metric("Groups", "A/B/C")

    # Group thresholds display
    with st.expander("Optimized Group Thresholds", expanded=False):
        for g, gt in GROUP_THRESHOLDS.items():
            st.markdown(f"**Group {g} ({GROUP_LABELS.get(g, '')}):** "
                f"R>={gt.min_confidence} | PVT>={gt.min_pvt_r} | XGB>{gt.min_xgb_score} | "
                f"Trail={gt.trail_buffer} | SL={gt.hard_sl_mult}x | Exhaust={gt.exhaust_r} | Pos={gt.pos_frac}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("### :lizard: Varanus Neo-Flow")
        st.caption("Extended Dashboard v1.0")
        st.markdown("33 assets | 3 groups | 4 TFs")
        st.markdown("---")

        if st.button("Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        state = load_bot_state()
        equity = state.get("initial_capital", 1000.0) + state.get("realized_pnl", 0.0)
        pid = get_bot_pid()

        if pid:
            st.markdown(":green_circle: **Bot Running**")
            st.markdown(":antenna_bars: **Data Sync: Websocket**")
        else:
            st.markdown(":red_circle: **Bot Stopped**")
        if state.get("circuit_breaker"):
            st.markdown(":warning: **Circuit Breaker TRIPPED**")

        st.metric("Equity", f"${equity:,.2f}")
        st.metric("Open Positions", f"{len(state.get('positions', {}))}/8")
        st.metric("Last Scan", time_since(state.get("last_scan_ts", "")))

        # Group breakdown in sidebar
        positions = state.get("positions", {})
        if positions:
            st.markdown("---")
            st.caption("Open by Group")
            for g in ["A", "B", "C"]:
                count = sum(1 for p in positions.values() if p.get("group") == g)
                if count:
                    st.markdown(f'{group_badge(g)} **{count}** open', unsafe_allow_html=True)

    state = load_bot_state()
    prices = fetch_binance_prices()
    scan_log = load_scan_log()
    trades_df = load_trades()
    wfv = load_wfv()

    st.markdown("## :lizard: Varanus Neo-Flow Extended Dashboard")
    st.caption("33 assets | Groups A (Majors) / B (Tech-AI) / C (Meme) | 4 timeframes")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        ":dart: Cockpit",
        ":brain: Decision Intelligence",
        ":chart_with_upwards_trend: Charting",
        ":bar_chart: Analytics",
        ":scroll: Trade History",
        ":zap: Controls",
    ])

    with tab1:
        render_cockpit(state, prices)
    with tab2:
        render_intelligence(state, scan_log)
    with tab3:
        render_charting(state, trades_df)
    with tab4:
        render_analytics(state, trades_df, wfv)
    with tab5:
        render_trade_history(trades_df)
    with tab6:
        render_controls(state)

    # Sidebar Footer / Auto-Refresh
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    do_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    if do_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
