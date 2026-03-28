#!/usr/bin/env python3
"""
live_extended_v1.py — Varanus Neo-Flow Extended: TF-Based Percentile Execution Engine.

Timeframe-Specific Dynamic Percentile Optimization — risk scales with TF reliability.

TF Hierarchy (the only leverage/percentile authority):
  5m:   1x leverage | P90 entry | P20 exit   — Ultra-selective, noise elimination
  30m:  3x leverage | P80 entry | P20 exit   — Balanced risk, trend following
  1h:   5x leverage | P75 entry | P15 exit   — High confidence, trend riding
  4h:   5x leverage | P70 entry | P15 exit   — Macro moves, maximum patience

Group A (Majors): 5m/30m forced to P90 entry regardless of TF default.
LONG trades: require both XGBoost AND Pearson R above their P-thresholds.
4h positions: real-time projection via 5m snapshots (no-lag exit policy).

Core logic:
  1. Entry gate:  XGB score >= entry_p percentile for the signal's best_tf
  2. Exit gate:   XGB score < exit_p percentile → immediate MARKET close (Mathematical Exhaustion)
  3. Leverage:    Strictly TF-based (1x/3x/5x) — NOT group-based
  4. Hard SL:     Global 1.5% per position — overrides P-Floor if hit first
  5. Cooldown:    Escalating with 0.5% stress-test cost (15m/45m/3h)

Classes:
  PercentileCalculator — Maps P-values to actual numerical thresholds from trade history
  SignalAuditor        — Validates signals against physics gates + TF percentile entry gate
  RiskEngine           — P-Floor exits (Mathematical Exhaustion), 1.5% hard SL, circuit breaker
  CooldownManager      — Escalating cooldown with stress-test cost
  TradeManager         — Orchestrates scan → entry → 5min audit → exit lifecycle

Usage:
    python live_extended_v1.py --capital 1000          # Dry-run
    python live_extended_v1.py --live --capital 1000    # REAL MONEY
"""

import os
import sys
import json
import time
import fcntl
import signal
import logging
import argparse
import traceback
import threading
import requests as _requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import xgboost as xgb
import websocket
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# Setup Paths & Environment
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR))

from neo_flow.adaptive_engine import (
    calc_log_regression,
    compute_trail_sl,
    compute_hard_sl,
    get_leverage,
    scan_asset,
    find_best_regression,
    get_htf_bias,
    compute_pvt_regression,
    check_pvt_alignment,
    check_combined_gate,
    trim_to_7d,
    compute_atr,
    HIGH_VOL_ASSETS,
    SCAN_TIMEFRAMES,
)

from backtest.data_loader import (
    AssetData,
    load_all_assets,
    build_scan_dataframes,
    build_htf_dataframe,
    _compute_pvt,
    BARS_7D,
    ASSETS,
    TIMEFRAMES,
)

from config.groups import (
    ALL_ASSETS,
    ALL_SYMBOLS,
    ASSET_FROM_SYMBOL,
    ASSET_TO_GROUP,
    GROUP_NAMES,
    GroupThresholds,
    get_group,
    get_thresholds,
    DEFAULT_THRESHOLDS,
)

from ml.train_meta_model import (
    FEATURE_COLS,
    TF_ENCODE,
    predict_probability,
    load_model,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Load Standalone Config
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = BASE_DIR / "config_extended.json"


def _load_config() -> dict:
    """Load config_extended.json. Falls back to defaults if missing."""
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


CONFIG = _load_config()

# ═══════════════════════════════════════════════════════════════════════════════
# Config & Constants
# ═══════════════════════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

_infra = CONFIG.get("infrastructure", {})
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
STATE_FILE = BASE_DIR / _infra.get("state_file", "live_extended_v1_state.json")
TRADES_FILE = BASE_DIR / _infra.get("trades_file", "live_extended_v1_trades.csv")
SCAN_LOG_FILE = BASE_DIR / _infra.get("scan_log_file", "logs/scan_results_v1.json")
LOCK_FILE_PATH = BASE_DIR / _infra.get("lock_file", "live_extended_v1.lock")
WS_SETTLEMENT_BUFFER = _infra.get("websocket_settlement_buffer_seconds", 15)

XGB_MODEL_PATH = BASE_DIR / CONFIG.get("xgboost", {}).get("model_path", "models/meta_xgb.json")

_risk = CONFIG.get("risk", {})
MAX_CONCURRENT = _risk.get("max_concurrent_positions", 8)
MAX_DRAWDOWN_PCT = _risk.get("max_drawdown_pct", -15.0)
HIGH_VOL_SCALAR = _risk.get("high_vol_scalar", 0.75)

_scan_cfg = CONFIG.get("scan", {})
MAX_BARS_HELD = _scan_cfg.get("max_bars_held", 200)

_exec = CONFIG.get("execution", {})
STOP_LIMIT_BUFFER = _exec.get("stop_limit_buffer_pct", 0.05) / 100.0
LIMIT_BUFFER = _exec.get("limit_buffer_pct", 0.05) / 100.0

# P-Floor monitor interval (seconds) — check every 5 minutes
PFLOOR_MONITOR_INTERVAL = 300

# Global hard SL — 1.5% emergency circuit breaker per position
HARD_SL_PCT = 0.015


# ═══════════════════════════════════════════════════════════════════════════════
# THE BRAIN: TF-Based Leverage & Percentile Hierarchy
# ═══════════════════════════════════════════════════════════════════════════════
#
# Risk scales with TF reliability — NOT with asset group.
# Lower TFs = more noise = tighter gate + lower leverage.
# Higher TFs = stronger trends = more patience + higher leverage.
#
# v1 Extended Optimized (blind-test validated):
#   5m:  1x leverage | P90 entry (ultra-selective) | P20 exit
#   30m: 3x leverage | P80 entry                   | P20 exit
#   1h:  5x leverage | P75 entry (balanced)         | P15 exit
#   4h:  5x leverage | P70 entry (macro moves)      | P15 exit

TF_MATRIX = {
    "5m":  {"entry_p": 0.90, "exit_p": 0.20, "leverage": 1},
    "30m": {"entry_p": 0.80, "exit_p": 0.20, "leverage": 3},
    "1h":  {"entry_p": 0.75, "exit_p": 0.15, "leverage": 5},
    "4h":  {"entry_p": 0.70, "exit_p": 0.15, "leverage": 5},
}


def _get_matrix_params(group: str, tf: str) -> dict:
    """Lookup TF_MATRIX for a timeframe. Group A forces P90 on 5m/30m."""
    params = TF_MATRIX.get(tf)
    if params is None:
        # 1h+ rule: anything above 1h gets the macro-trend treatment
        params = {"entry_p": 0.70, "exit_p": 0.15, "leverage": 5}
    params = dict(params)  # copy to avoid mutating TF_MATRIX
    # Rule 2: Group A (Majors) — force P90 entry barrier on 5m and 30m
    if group == "A" and tf in ("5m", "30m"):
        params["entry_p"] = 0.90
    return params


# ═══════════════════════════════════════════════════════════════════════════════
# Load Group-Specific Thresholds (physics gates — preserved from v1)
# ═══════════════════════════════════════════════════════════════════════════════


def _load_group_thresholds() -> dict[str, GroupThresholds]:
    """Load optimized per-group thresholds from config_extended.json."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    groups_data = CONFIG.get("groups", {})
    for grp_key, grp_data in groups_data.items():
        t = grp_data.get("thresholds", {})
        if t:
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
    return thresholds


def _load_consensus_params() -> dict:
    """Load consensus params from config_extended.json."""
    consensus = CONFIG.get("consensus_params", {})
    return {
        "min_pearson_r": consensus.get("min_pearson_r", 0.92),
        "min_pvt_r": consensus.get("min_pvt_r", 0.65),
        "combined_gate": consensus.get("combined_gate", 0.80),
        "hard_sl_mult": consensus.get("hard_sl_mult", 2.50),
        "trail_buffer": consensus.get("trail_buffer", 0.50),
        "exhaust_r": consensus.get("exhaust_r", 0.625),
        "pos_frac": consensus.get("pos_frac", 0.03),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

import logging.handlers

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.handlers.RotatingFileHandler(
            LOGS_DIR / "live_extended_v1.log", maxBytes=10_000_000, backupCount=5,
        ),
    ],
)
logger = logging.getLogger("live_extended_v1")

for name in ["urllib3", "ccxt", "httpx", "httpcore", "telegram"]:
    logging.getLogger(name).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _tg_escape(text: str) -> str:
    """Escape HTML special chars OUTSIDE of <b> tags."""
    import re
    parts = re.split(r'(<b>.*?</b>)', text)
    result = []
    for p in parts:
        if p.startswith('<b>') and p.endswith('</b>'):
            result.append(p)
        else:
            result.append(p.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
    return ''.join(result)


def _mdv2(text) -> str:
    """Escape MarkdownV2 special characters in a plain-text value."""
    _SPECIAL = r'_*[]()~`>#+-=|{}.!'
    out = []
    for ch in str(text):
        if ch in _SPECIAL:
            out.append('\\')
        out.append(ch)
    return ''.join(out)


def tg_send(text: str, parse_mode: str = "HTML"):
    """Send a Telegram message. Non-blocking, swallows errors."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        safe_text = _tg_escape(text) if parse_mode == "HTML" else text
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = _requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": safe_text,
            "parse_mode": parse_mode,
        }, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Telegram send failed: %s", e)


# ── MarkdownV2 Notification Templates ────────────────────────────────────

def tg_entry_notify(symbol: str, side: str, tf: str, leverage: int,
                    xgb_score: float, entry_p: float, current_percentile: float,
                    exit_threshold: float, exit_p: float):
    """Enhanced entry notification — shows XGB Score, Percentile, Distance to P-Exit."""
    e = _mdv2
    distance = xgb_score - exit_threshold
    msg = (
        f"\U0001F3AF *NEW POSITION OPENED*\n"
        f"*Symbol:* {e(symbol)} \\({e(side)}\\)\n"
        f"*TF/Leverage:* {e(tf)} \\| {e(leverage)}x\n"
        f"*XGB Score:* {e(f'{xgb_score:.4f}')} \\(P{e(f'{current_percentile:.0f}')}\\)\n"
        f"*Entry Barrier:* P{e(f'{entry_p * 100:.0f}')} \\| "
        f"*P\\-Exit Floor:* P{e(f'{exit_p * 100:.0f}')} \\({e(f'{exit_threshold:.4f}')}\\)\n"
        f"*Distance to P\\-Exit:* {e(f'{distance:.4f}')}\n"
        f"*Hard SL:* 1\\.5% \\| *Strategy:* Mathematical Trend Milking"
    )
    tg_send(msg, parse_mode="MarkdownV2")


def tg_exhaustion_exit_notify(symbol: str, pnl_pct: float,
                              exit_score: float, exit_p: float,
                              current_percentile: float, exit_threshold: float):
    """Statistical Exhaustion exit — P-Floor triggered close."""
    e = _mdv2
    pnl_str = f"{pnl_pct:+.2f}"
    distance = exit_score - exit_threshold
    msg = (
        f"\U0001F3C1 *TRADE CLOSED \\(Signal Exhaustion\\)*\n"
        f"*Symbol:* {e(symbol)} \\| *Result:* {e(pnl_str)}%\n"
        f"*XGB Score:* {e(f'{exit_score:.4f}')} \\(P{e(f'{current_percentile:.0f}')}\\)\n"
        f"*P\\-Exit Floor:* P{e(f'{exit_p * 100:.0f}')} \\({e(f'{exit_threshold:.4f}')}\\) "
        f"\\| *Distance:* {e(f'{distance:.4f}')}\n"
        f"*Insight:* Trend momentum dissipated\\. "
        f"Profit secured via P\\-Exit\\."
    )
    tg_send(msg, parse_mode="MarkdownV2")


def tg_cooldown_notify(symbol: str, loss_count: int, pause_minutes: int):
    """Cooldown (Mola) alert — asset paused after stress-test loss."""
    e = _mdv2
    msg = (
        f"\U0001F534 *ASSET ON COOLDOWN*\n"
        f"*Symbol:* {e(symbol)} \\| *Loss Streak:* {e(loss_count)}/3\n"
        f"*Action:* No trades for the next {e(pause_minutes)} mins\\.\n"
        f"*Reason:* Market structure currently incompatible "
        f"with signal math\\."
    )
    tg_send(msg, parse_mode="MarkdownV2")


def tg_daily_snapshot(total_trades: int, wins: int, net_pnl: float,
                      mean_xgb: float, date_str: str):
    """End-of-day performance snapshot."""
    e = _mdv2
    wr = (wins / total_trades * 100) if total_trades > 0 else 0.0
    pnl_emoji = "\U0001F7E2" if net_pnl >= 0 else "\U0001F534"
    msg = (
        f"\U0001F4CA *DAILY PERFORMANCE SNAPSHOT*\n"
        f"*Date:* {e(date_str)}\n"
        f"*Total Trades:* {e(total_trades)}\n"
        f"*Win Rate:* {e(f'{wr:.1f}')}%\n"
        f"{pnl_emoji} *Net PnL:* {e(f'${net_pnl:+,.2f}')}\n"
        f"*Avg Entry Quality:* XGB {e(f'{mean_xgb:.4f}')}"
    )
    tg_send(msg, parse_mode="MarkdownV2")


# ═══════════════════════════════════════════════════════════════════════════════
# Live Data Engine
# ═══════════════════════════════════════════════════════════════════════════════

KLINES_URL = "https://api.binance.com/api/v3/klines"
_cached_data = None


def fetch_recent_klines(symbol: str, interval: str, start_ms: int = 0, limit: int = 1000) -> pd.DataFrame:
    """Fetch klines from Binance public API."""
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms > 0:
            params["startTime"] = start_ms
        resp = _requests.get(KLINES_URL, params=params, timeout=15)
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
    except Exception as e:
        logger.error("Failed to fetch %s %s: %s", symbol, interval, e)
        return pd.DataFrame()


def _load_cache() -> dict:
    """Load Parquet files (33 assets x 4 TFs)."""
    logger.info("Loading parquet cache (33 assets x 4 TFs)...")
    t0 = time.perf_counter()
    all_data = load_all_assets()
    elapsed = time.perf_counter() - t0
    total = sum(len(ad.close) for asset in all_data.values() for ad in asset.values())
    logger.info("Cache loaded in %.1fs -- %d total bars", elapsed, total)
    return all_data


def _append_new_bars(all_data: dict) -> dict:
    """Sync cache with Binance API."""
    updated = 0
    for asset in ASSETS:
        asset_data = all_data.get(asset, {})
        symbol = f"{asset}USDT"
        for tf in TIMEFRAMES:
            ad = asset_data.get(tf)
            if ad is None:
                continue
            last_ns = ad.timestamps[-1]
            cursor_ms = last_ns // 1_000_000 + 1
            all_close, all_high, all_low, all_open, all_vol, all_ts = [], [], [], [], [], []
            while True:
                df_new = fetch_recent_klines(symbol, tf, start_ms=cursor_ms, limit=1000)
                if df_new.empty:
                    break
                new_ts = df_new.index.values.astype("datetime64[ns]").astype("int64")
                mask = new_ts > last_ns
                if not mask.any():
                    break
                all_close.append(df_new["close"].values[mask].astype(np.float64))
                all_high.append(df_new["high"].values[mask].astype(np.float64))
                all_low.append(df_new["low"].values[mask].astype(np.float64))
                all_open.append(df_new["open"].values[mask].astype(np.float64))
                all_vol.append(df_new["volume"].values[mask].astype(np.float64))
                all_ts.append(new_ts[mask])
                last_ns = new_ts[-1]
                if len(df_new) < 1000:
                    break
                cursor_ms = int(new_ts[-1]) // 1_000_000 + 1
                time.sleep(0.05)
            if not all_close:
                continue
            ad.close = np.concatenate([ad.close, np.concatenate(all_close)])
            ad.high = np.concatenate([ad.high, np.concatenate(all_high)])
            ad.low = np.concatenate([ad.low, np.concatenate(all_low)])
            ad.open_ = np.concatenate([ad.open_, np.concatenate(all_open)])
            ad.volume = np.concatenate([ad.volume, np.concatenate(all_vol)])
            ad.timestamps = np.concatenate([ad.timestamps, np.concatenate(all_ts)])
            ad.pvt = _compute_pvt(ad.close, ad.volume)
            updated += 1
    if updated > 0:
        logger.info("Appended new bars from Binance for %d asset/TF pairs", updated)
    return all_data


def get_live_data():
    """Global data update."""
    global _cached_data
    if _cached_data is None:
        _cached_data = _load_cache()
    _cached_data = _append_new_bars(_cached_data)
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            ad = _cached_data.get(asset, {}).get(tf)
            if ad is not None:
                needed = BARS_7D[tf]
                if len(ad.close) < needed:
                    logger.warning("DATA GAP: %s %s has %d/%d bars", asset, tf, len(ad.close), needed)
    return _cached_data


def _append_ws_bar_to_cache(asset: str, open_: float, high: float, low: float,
                            close: float, volume: float, ts_ms: int):
    """Append a single closed 5m bar from websocket to the in-memory cache."""
    global _cached_data
    if _cached_data is None:
        return
    ad = _cached_data.get(asset, {}).get("5m")
    if ad is None:
        return
    ts_ns = int(ts_ms * 1_000_000)
    if ts_ns <= ad.timestamps[-1]:
        return
    ad.close = np.append(ad.close, close)
    ad.high = np.append(ad.high, high)
    ad.low = np.append(ad.low, low)
    ad.open_ = np.append(ad.open_, open_)
    ad.volume = np.append(ad.volume, volume)
    ad.timestamps = np.append(ad.timestamps, ts_ns)
    ad.pvt = _compute_pvt(ad.close, ad.volume)


def fetch_ticker_price(symbol: str) -> float | None:
    try:
        resp = _requests.get("https://api.binance.com/api/v3/ticker/price",
                             params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Execution Engine (Binance via ccxt)
# ═══════════════════════════════════════════════════════════════════════════════

_exchange = None


def _get_exchange():
    global _exchange
    if _exchange is None:
        import ccxt
        _exchange = ccxt.binanceusdm({
            "apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET,
            "options": {"defaultType": "future"}, "enableRateLimit": True,
        })
    return _exchange


def _fetch_order_commission(symbol: str, order_id: str) -> float:
    """Fetch total commission paid for an order from Binance trade fills."""
    try:
        ex = _get_exchange()
        trades = ex.fetch_my_trades(symbol, limit=50)
        total_commission = 0.0
        for t in trades:
            if str(t.get("order")) == str(order_id):
                fee = t.get("fee", {})
                cost = float(fee.get("cost", 0))
                if fee.get("currency") == "USDT":
                    total_commission += cost
                elif fee.get("currency") == "BNB":
                    bnb_price = fetch_ticker_price("BNBUSDT") or 600.0
                    total_commission += cost * bnb_price
                else:
                    total_commission += cost
        return total_commission
    except Exception as e:
        logger.error("Failed to fetch commission for %s order %s: %s", symbol, order_id, e)
        return 0.0


def place_market_order(symbol: str, side: str, amount_usd: float, leverage: int):
    try:
        ex = _get_exchange()
        ex.set_leverage(leverage, symbol)
        ticker = ex.fetch_ticker(symbol)
        price = ticker["last"]
        quantity = amount_usd / price
        quantity = float(ex.amount_to_precision(symbol, quantity))
        order = ex.create_market_order(symbol, side, quantity)
        logger.info("ORDER PLACED: %s %s %sx qty=%s @ ~%.4f", side, symbol, leverage, quantity, price)
        return order
    except Exception as e:
        logger.error("ORDER FAILED: %s %s -- %s", side, symbol, e)
        tg_send(f"<b>ORDER FAILED</b>\n{side.upper()} {symbol}\nError: {e}")
        return None


def _stop_limit_price(stop_price: float, side: str) -> float:
    """Compute limit price with buffer to ensure fill near the stop trigger."""
    if side == "sell":
        return stop_price * (1 - STOP_LIMIT_BUFFER)
    else:
        return stop_price * (1 + STOP_LIMIT_BUFFER)


def _cancel_all_algo_orders(symbol: str):
    """Cancel all algo/conditional orders for a symbol."""
    try:
        ex = _get_exchange()
        result = ex.fapiPrivateGetOpenAlgoOrders()
        orders = result if isinstance(result, list) else result.get("orders", [])
        cancelled = 0
        for o in orders:
            if o.get("symbol") == symbol.replace("/", "").replace(":USDT", ""):
                try:
                    ex.fapiPrivateDeleteAlgoOrder({"algoId": int(o["algoId"])})
                    cancelled += 1
                except Exception:
                    pass
        if cancelled:
            logger.info("Cancelled %d algo orders for %s", cancelled, symbol)
    except Exception as e:
        logger.error("Failed to cancel algo orders for %s: %s", symbol, e)


def _cancel_all_orders_full(symbol: str):
    """Cancel BOTH regular and algo/conditional orders for a symbol."""
    ex = _get_exchange()
    try:
        ex.cancel_all_orders(symbol)
    except Exception:
        pass
    _cancel_all_algo_orders(symbol)


def place_stop_loss(symbol: str, side: str, quantity: float, stop_price: float):
    try:
        ex = _get_exchange()
        stop_price = float(ex.price_to_precision(symbol, stop_price))
        limit_price = float(ex.price_to_precision(symbol, _stop_limit_price(stop_price, side)))
        order = ex.create_order(symbol, "STOP", side, quantity, limit_price,
                                params={"stopPrice": stop_price, "reduceOnly": True})
        logger.info("STOP SET: %s %s trigger=%.6f limit=%.6f", side, symbol, stop_price, limit_price)
        return order
    except Exception as e:
        logger.error("STOP FAILED: %s %s -- %s", side, symbol, e)
        return None


def close_position_order(symbol: str, side: str, quantity: float,
                         exit_price: float, reason: str):
    """Close position with MARKET order. Guarantees execution at signal death."""
    try:
        ex = _get_exchange()
        close_side = "sell" if side == "buy" else "buy"
        quantity = float(ex.amount_to_precision(symbol, quantity))
        order = ex.create_market_order(symbol, close_side, quantity,
                                       params={"reduceOnly": True})
        fill_price = float(order.get("average", 0) or 0)
        logger.info("CLOSED (MARKET/%s): %s %s qty=%s fill=%.6f",
                    reason, close_side, symbol, quantity, fill_price)
        return order
    except Exception as e:
        logger.error("CLOSE FAILED (%s): %s -- %s", reason, symbol, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# State Management
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LivePosition:
    trade_id: int
    asset: str
    group: str
    symbol: str
    direction: int
    entry_price: float
    entry_ts: str
    hard_sl: float
    trail_sl: float
    midline: float
    std_dev: float
    best_tf: str
    best_period: int
    confidence: float
    pvt_r: float
    leverage: int
    position_usd: float
    quantity: float = 0.0
    bars_held: int = 0
    peak_r: float = 0.0
    order_id: str = ""
    sl_order_id: str = ""
    last_bar_ns: int = 0
    xgb_prob: float = 0.0
    entry_commission: float = 0.0
    # P-Matrix fields
    p_entry_threshold: float = 0.0
    p_exit_threshold: float = 0.0
    matrix_leverage: int = 0


@dataclass
class CooldownState:
    consecutive_losses: int = 0
    cooldown_until: str = ""
    blacklist_until: str = ""  # 24h blacklist after 3rd consecutive loss


@dataclass
class LiveState:
    positions: dict[str, LivePosition] = field(default_factory=dict)
    cooldowns: dict[str, CooldownState] = field(default_factory=dict)
    trade_counter: int = 0
    realized_pnl: float = 0.0
    initial_capital: float = 1000.0
    peak_equity: float = 1000.0
    circuit_breaker: bool = False
    last_scan_ts: str = ""
    total_scans: int = 0
    current_scanning_asset: str = ""


def save_state(state: LiveState):
    data = {
        "trade_counter": state.trade_counter,
        "realized_pnl": state.realized_pnl,
        "initial_capital": state.initial_capital,
        "peak_equity": state.peak_equity,
        "circuit_breaker": state.circuit_breaker,
        "last_scan_ts": state.last_scan_ts,
        "total_scans": state.total_scans,
        "current_scanning_asset": state.current_scanning_asset,
        "positions": {a: asdict(p) for a, p in state.positions.items()},
        "cooldowns": {a: asdict(c) for a, c in state.cooldowns.items()},
    }
    STATE_FILE.write_text(json.dumps(data, indent=2, default=str))


def load_state() -> LiveState:
    state = LiveState()
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            state.trade_counter = data.get("trade_counter", 0)
            state.realized_pnl = data.get("realized_pnl", 0.0)
            state.initial_capital = data.get("initial_capital", 10_000.0)
            state.peak_equity = data.get("peak_equity", 10_000.0)
            state.circuit_breaker = data.get("circuit_breaker", False)
            state.last_scan_ts = data.get("last_scan_ts", "")
            state.total_scans = data.get("total_scans", 0)
            state.current_scanning_asset = data.get("current_scanning_asset", "")
            for asset, pos_data in data.get("positions", {}).items():
                # Handle legacy state files missing new P-Matrix fields
                for fld in ("p_entry_threshold", "p_exit_threshold", "matrix_leverage"):
                    pos_data.setdefault(fld, 0)
                state.positions[asset] = LivePosition(**pos_data)
            for asset, cd_data in data.get("cooldowns", {}).items():
                cd_data.setdefault("blacklist_until", "")
                state.cooldowns[asset] = CooldownState(**cd_data)
        except Exception as e:
            logger.error("Failed to load state: %s", e)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# PercentileCalculator — Maps P-values to Actual Numerical Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

class PercentileCalculator:
    """
    Analyze historical trade logs to compute XGB score distributions per TF.
    TF is the sole authority — distributions are built per timeframe across
    all asset groups for maximum sample size.

    Example: entry_p=0.80 for 5m → the 80th percentile of all XGB scores
    for trades entered on the 5m timeframe (regardless of asset group).
    """

    def __init__(self, xgb_model: xgb.XGBClassifier):
        self.xgb_model = xgb_model
        self._distributions: dict[str, np.ndarray] = {}  # key = "5m", "30m", etc.
        self._r_distributions: dict[str, np.ndarray] = {}  # Pearson R distributions per TF
        self._build_distributions()

    def _find_trade_csvs(self) -> list[Path]:
        """Find available trade log CSVs in priority order."""
        candidates = [
            BASE_DIR / "validation_trades.csv",
            BASE_DIR / "blind_test_trades.csv",
            BASE_DIR / "extended_trades_hybrid.csv",
            BASE_DIR / "extended_trades_physics.csv",
            BASE_DIR / "live_extended_v1_trades.csv",
            BASE_DIR / "live_extended_trades.csv",
        ]
        return [p for p in candidates if p.exists() and p.stat().st_size > 100]

    def _build_distributions(self):
        """Load trade logs and compute XGB score distributions per Group × TF."""
        csv_paths = self._find_trade_csvs()
        if not csv_paths:
            logger.warning("PercentileCalculator: No trade CSVs found. Using uniform fallback.")
            return

        frames = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip().lower() for c in df.columns]
                required = {"confidence", "pvt_r", "best_tf", "best_period", "group"}
                if required.issubset(set(df.columns)):
                    frames.append(df)
                    logger.info("PercentileCalculator: Loaded %s (%d trades)", path.name, len(df))
            except Exception as e:
                logger.warning("PercentileCalculator: Skipping %s: %s", path.name, e)

        if not frames:
            logger.warning("PercentileCalculator: No valid trade data. Using uniform fallback.")
            return

        combined = pd.concat(frames, ignore_index=True)
        # Deduplicate
        dedup_cols = [c for c in ["asset", "entry_ts", "exit_ts", "entry_price"]
                      if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols)

        logger.info("PercentileCalculator: %d trades total for distribution building", len(combined))

        # Compute XGB probability for each historical trade
        xgb_scores = []
        for _, row in combined.iterrows():
            try:
                score = predict_probability(
                    self.xgb_model,
                    confidence=float(row["confidence"]),
                    pvt_r=float(row["pvt_r"]),
                    best_tf=str(row["best_tf"]),
                    best_period=int(row["best_period"]),
                )
                xgb_scores.append(score)
            except Exception:
                xgb_scores.append(np.nan)

        combined["xgb_score"] = xgb_scores
        combined = combined.dropna(subset=["xgb_score"])

        # Build distributions per TF (across all groups — TF is sole authority)
        for tf in SCAN_TIMEFRAMES:
            mask = combined["best_tf"] == tf
            scores = combined.loc[mask, "xgb_score"].values
            if len(scores) >= 10:
                self._distributions[tf] = np.sort(scores)
                tf_params = TF_MATRIX.get(tf, {})
                logger.info("  TF %s: %d trades | P15=%.4f P20=%.4f P30=%.4f P70=%.4f P75=%.4f P80=%.4f",
                            tf, len(scores),
                            np.percentile(scores, 15),
                            np.percentile(scores, 20),
                            np.percentile(scores, 30),
                            np.percentile(scores, 70),
                            np.percentile(scores, 75),
                            np.percentile(scores, 80))
            else:
                logger.info("  TF %s: only %d trades (< 10), will use global fallback", tf, len(scores))

        # Build Pearson R distributions per TF (for LONG dual-gate confirmation)
        for tf in SCAN_TIMEFRAMES:
            mask = combined["best_tf"] == tf
            r_scores = combined.loc[mask, "confidence"].values
            if len(r_scores) >= 10:
                self._r_distributions[tf] = np.sort(r_scores)
                logger.info("  TF %s R-dist: %d trades | P70=%.4f P75=%.4f P80=%.4f P90=%.4f",
                            tf, len(r_scores),
                            np.percentile(r_scores, 70), np.percentile(r_scores, 75),
                            np.percentile(r_scores, 80), np.percentile(r_scores, 90))
        all_r = combined["confidence"].values
        if len(all_r) >= 10:
            self._r_distributions["GLOBAL"] = np.sort(all_r)

        # Global fallback (all TFs combined)
        all_scores = combined["xgb_score"].values
        if len(all_scores) >= 10:
            self._distributions["GLOBAL"] = np.sort(all_scores)
            logger.info("  GLOBAL fallback: %d trades", len(all_scores))

    def get_threshold(self, tf: str, percentile: float) -> float:
        """
        Get the actual XGB score threshold for a given percentile.

        Args:
            tf: "5m", "30m", "1h", "4h"
            percentile: 0.0-1.0 (e.g. 0.80 for P80)

        Returns:
            Numerical XGB score threshold.
        """
        # TF-specific distribution
        if tf in self._distributions:
            return float(np.percentile(self._distributions[tf], percentile * 100))

        # Fallback: global (all TFs combined)
        if "GLOBAL" in self._distributions:
            return float(np.percentile(self._distributions["GLOBAL"], percentile * 100))

        # No data at all: return the percentile itself as a direct probability threshold
        logger.warning("PercentileCalculator: No distribution for TF=%s, using raw percentile %.2f",
                        tf, percentile)
        return percentile

    def get_r_threshold(self, tf: str, percentile: float) -> float:
        """Get the Pearson R threshold at a given percentile for LONG dual-gate."""
        if tf in self._r_distributions:
            return float(np.percentile(self._r_distributions[tf], percentile * 100))
        if "GLOBAL" in self._r_distributions:
            return float(np.percentile(self._r_distributions["GLOBAL"], percentile * 100))
        return percentile

    def score_to_percentile(self, tf: str, score: float) -> float:
        """Convert an XGB score to its percentile rank (0-100) for a given TF."""
        dist = self._distributions.get(tf) or self._distributions.get("GLOBAL")
        if dist is None or len(dist) == 0:
            return score * 100
        idx = np.searchsorted(dist, score)
        return (idx / len(dist)) * 100

    def get_entry_threshold(self, group: str, tf: str) -> float:
        """Get the entry XGB threshold for a TF. Group A forces P90 on 5m/30m."""
        params = _get_matrix_params(group, tf)
        return self.get_threshold(tf, params["entry_p"])

    def get_exit_threshold(self, group: str, tf: str) -> float:
        """Get the exit XGB threshold (P-Floor) for a TF. Group A forces P90 on 5m/30m."""
        params = _get_matrix_params(group, tf)
        return self.get_threshold(tf, params["exit_p"])


# ═══════════════════════════════════════════════════════════════════════════════
# CooldownManager — Escalating Cooldown with Stress-Test Cost
# ═══════════════════════════════════════════════════════════════════════════════

class CooldownManager:
    """
    Adaptive recovery system for failed trades.

    Stress-test cost: net_pnl_pct = pnl_pct - 0.5%
    If net_pnl_pct <= 0:
        1st loss → 15 min cooldown
        2nd loss → 45 min cooldown
        3rd+ loss → 180 min blacklist (reset after 24h or manual /reset)
    If net_pnl_pct > 0.5%: reset loss counter (unless blacklisted)
    """

    STRESS_COST = 0.5  # percentage points
    STEPS = [15, 45, 180]  # minutes: 1st, 2nd, 3rd+
    BLACKLIST_HOURS = 24  # 3rd loss triggers 24h blacklist auto-reset

    def __init__(self, state: LiveState):
        self.state = state

    def record_exit(self, asset: str, pnl_pct: float):
        """Record a trade exit and update cooldown state."""
        net_pnl_pct = pnl_pct - self.STRESS_COST

        if asset not in self.state.cooldowns:
            self.state.cooldowns[asset] = CooldownState()

        cd = self.state.cooldowns[asset]

        # Check if existing blacklist has expired (24h auto-reset)
        if cd.blacklist_until:
            try:
                bl_until = datetime.fromisoformat(cd.blacklist_until)
                if datetime.now(timezone.utc) >= bl_until:
                    cd.consecutive_losses = 0
                    cd.blacklist_until = ""
                    cd.cooldown_until = ""
                    logger.info("BLACKLIST EXPIRED: %s — 24h auto-reset", asset)
            except (ValueError, TypeError):
                cd.blacklist_until = ""

        if net_pnl_pct > self.STRESS_COST:
            # Profitable enough to reset counter (unless 24h blacklisted)
            if not cd.blacklist_until:
                cd.consecutive_losses = 0
                cd.cooldown_until = ""
                logger.info("COOLDOWN RESET: %s — net_pnl=%.2f%% > %.1f%% threshold",
                            asset, net_pnl_pct, self.STRESS_COST)
            return

        if net_pnl_pct <= 0:
            cd.consecutive_losses += 1
            step_idx = min(cd.consecutive_losses, len(self.STEPS)) - 1
            pause_minutes = self.STEPS[step_idx]
            until = datetime.now(timezone.utc) + timedelta(minutes=pause_minutes)
            cd.cooldown_until = until.isoformat()

            # 3rd+ loss: set 24h blacklist (if not already blacklisted)
            if cd.consecutive_losses >= 3 and not cd.blacklist_until:
                bl_until = datetime.now(timezone.utc) + timedelta(hours=self.BLACKLIST_HOURS)
                cd.blacklist_until = bl_until.isoformat()
                logger.info("BLACKLIST SET: %s — 3rd loss, 24h blacklist until %s",
                            asset, bl_until.strftime("%Y-%m-%d %H:%M UTC"))

            step_name = ["1st", "2nd", "3rd+"][step_idx]
            logger.info("COOLDOWN: %s — %s loss (streak=%d), pausing %d min until %s",
                        asset, step_name, cd.consecutive_losses, pause_minutes,
                        until.strftime("%H:%M UTC"))
            tg_cooldown_notify(
                symbol=f"{asset}USDT",
                loss_count=cd.consecutive_losses,
                pause_minutes=pause_minutes,
            )

    def is_on_cooldown(self, asset: str) -> tuple[bool, str]:
        """Check if an asset is on cooldown or blacklisted. Returns (is_cooling, reason)."""
        cd = self.state.cooldowns.get(asset)
        if cd is None:
            return False, ""

        now = datetime.now(timezone.utc)

        # Check 24h blacklist first (overrides regular cooldown)
        if cd.blacklist_until:
            try:
                bl_until = datetime.fromisoformat(cd.blacklist_until)
                if now < bl_until:
                    remaining_h = (bl_until - now).total_seconds() / 3600
                    return True, f"blacklisted {remaining_h:.1f}h (losses={cd.consecutive_losses})"
                else:
                    # Expired — auto-reset
                    cd.consecutive_losses = 0
                    cd.blacklist_until = ""
                    cd.cooldown_until = ""
                    logger.info("BLACKLIST EXPIRED: %s — 24h auto-reset (checked in is_on_cooldown)", asset)
            except (ValueError, TypeError):
                cd.blacklist_until = ""

        # Check regular cooldown
        if not cd.cooldown_until:
            return False, ""

        try:
            until = datetime.fromisoformat(cd.cooldown_until)
            if now < until:
                remaining = (until - now).total_seconds() / 60
                return True, f"cooldown {remaining:.0f}m (losses={cd.consecutive_losses})"
            else:
                cd.cooldown_until = ""
                return False, ""
        except (ValueError, TypeError):
            cd.cooldown_until = ""
            return False, ""

    def reset_cooldown(self, asset: str):
        """Manual reset of cooldown and blacklist for an asset."""
        if asset in self.state.cooldowns:
            cd = self.state.cooldowns[asset]
            cd.consecutive_losses = 0
            cd.cooldown_until = ""
            cd.blacklist_until = ""
            logger.info("MANUAL RESET: %s — cooldown and blacklist cleared", asset)
            return True
        return False

    @property
    def states(self) -> dict[str, CooldownState]:
        return self.state.cooldowns


# ═══════════════════════════════════════════════════════════════════════════════
# SignalAuditor — Physics Gates + P-Matrix Sniper Gate
# ═══════════════════════════════════════════════════════════════════════════════

class SignalAuditor:
    """
    Validates candidate signals through two layers:
      1. Physics gates (Pearson R, PVT, 4H bias, combined gate) — preserved
      2. P-Matrix sniper gate: XGB score >= entry_p percentile threshold
    """

    def __init__(self, xgb_model: xgb.XGBClassifier, percentile_calc: PercentileCalculator,
                 group_thresholds: dict[str, GroupThresholds], consensus: dict):
        self.xgb_model = xgb_model
        self.percentile_calc = percentile_calc
        self.group_thresholds = group_thresholds
        self.consensus = consensus

    def _apply_group_params(self, asset: str):
        """Apply group-specific thresholds to engine globals before scanning."""
        import neo_flow.adaptive_engine as ae
        gt = self.group_thresholds.get(get_group(asset))
        if gt:
            ae.MIN_PEARSON_R = gt.min_confidence
            ae.MIN_PVT_PEARSON_R = gt.min_pvt_r
            ae.COMBINED_GATE_THRESHOLD = gt.combined_gate
            ae.HARD_SL_ATR_MULT = gt.hard_sl_mult
            ae.TRAIL_BUFFER_STD = gt.trail_buffer
            ae.TREND_EXHAUST_R = gt.exhaust_r

    def run_physics_gates(self, asset: str, scan_dfs: dict, df_4h,
                          gt: GroupThresholds) -> tuple:
        """
        Run physics pipeline gates.
        Returns (signal, near_miss_or_None, rejection_status).
        """
        group = get_group(asset)
        best, all_results = find_best_regression(scan_dfs)
        if not best:
            return None, None, "no_regression"

        abs_r = abs(best.pearson_r)
        direction = 1 if best.slope < 0 else -1
        min_r = gt.min_confidence if gt else self.consensus["min_pearson_r"]
        dir_str = "LONG" if direction == 1 else "SHORT"

        if abs_r < min_r:
            return None, None, "weak_r"

        best_df_7d = trim_to_7d(scan_dfs[best.timeframe], best.timeframe)
        pvt = compute_pvt_regression(best_df_7d, best.period)
        min_pvt_r = gt.min_pvt_r if gt else self.consensus["min_pvt_r"]
        pvt_passes, pvt_reason = check_pvt_alignment(direction, abs_r, pvt, min_pvt_r)
        pvt_r_val = abs(pvt.pearson_r) if pvt else 0.0

        if not pvt_passes:
            nm = {"asset": asset, "group": group, "dir": dir_str,
                  "r": abs_r, "pvt_r": pvt_r_val, "tf": best.timeframe, "period": best.period,
                  "rejected_at": "PVT", "detail": pvt_reason}
            return None, nm, "pvt_rejected"

        htf_bias = get_htf_bias(df_4h)
        if htf_bias == 0:
            nm = {"asset": asset, "group": group, "dir": dir_str,
                  "r": abs_r, "pvt_r": pvt_r_val, "tf": best.timeframe, "period": best.period,
                  "rejected_at": "HTF", "detail": "4H neutral"}
            return None, nm, "htf_neutral"
        if htf_bias != direction:
            nm = {"asset": asset, "group": group, "dir": dir_str,
                  "r": abs_r, "pvt_r": pvt_r_val, "tf": best.timeframe, "period": best.period,
                  "rejected_at": "HTF", "detail": f"conflict (sig={direction}, htf={htf_bias})"}
            return None, nm, "htf_conflict"

        cg_thresh = gt.combined_gate if gt else self.consensus["combined_gate"]
        if not check_combined_gate(direction, all_results, cg_thresh):
            nm = {"asset": asset, "group": group, "dir": dir_str,
                  "r": abs_r, "pvt_r": pvt_r_val, "tf": best.timeframe, "period": best.period,
                  "rejected_at": "GATE", "detail": f"opposing >{cg_thresh:.2f}"}
            return None, nm, "combined_gate_blocked"

        signal = scan_asset(asset, scan_dfs, df_4h,
                            min_pearson_r=min_r, min_pvt_r=min_pvt_r,
                            combined_gate_threshold=cg_thresh, group=group)
        return signal, None, "passed" if signal else "physics_rejected_final"

    def audit_entry(self, asset: str, signal, scan_dfs: dict, df_4h) -> dict:
        """
        Full audit: physics gates + P-Matrix sniper gate.

        Returns dict:
            passed: bool
            signal: ScanSignal or None
            xgb_prob: float
            entry_threshold: float
            exit_threshold: float
            matrix_leverage: int
            near_miss: dict or None
            reason: str
        """
        group = get_group(asset)
        gt = self.group_thresholds.get(group, DEFAULT_THRESHOLDS.get(group))
        self._apply_group_params(asset)

        signal_obj, near_miss, status = self.run_physics_gates(asset, scan_dfs, df_4h, gt)

        if signal_obj is None:
            return {
                "passed": False, "signal": None, "xgb_prob": 0.0,
                "entry_threshold": 0.0, "exit_threshold": 0.0, "matrix_leverage": 0,
                "near_miss": near_miss, "reason": status,
            }

        # Compute XGB probability
        xgb_prob = predict_probability(
            self.xgb_model,
            confidence=signal_obj.confidence,
            pvt_r=signal_obj.pvt_r,
            best_tf=signal_obj.best_tf,
            best_period=signal_obj.best_period,
        )

        # P-Matrix sniper gate: XGB score must exceed entry_p percentile
        entry_threshold = self.percentile_calc.get_entry_threshold(group, signal_obj.best_tf)
        exit_threshold = self.percentile_calc.get_exit_threshold(group, signal_obj.best_tf)
        matrix_params = _get_matrix_params(group, signal_obj.best_tf)
        matrix_leverage = matrix_params["leverage"]

        if xgb_prob < entry_threshold:
            near_miss = {
                "asset": asset, "group": group,
                "dir": "LONG" if signal_obj.direction == 1 else "SHORT",
                "r": round(signal_obj.confidence, 4), "pvt_r": round(signal_obj.pvt_r, 4),
                "tf": signal_obj.best_tf, "period": signal_obj.best_period,
                "rejected_at": "P-MATRIX",
                "detail": f"XGB={xgb_prob:.4f} < entry_P{matrix_params['entry_p']*100:.0f}={entry_threshold:.4f}",
                "xgb_prob": round(xgb_prob, 4),
            }
            return {
                "passed": False, "signal": signal_obj, "xgb_prob": xgb_prob,
                "entry_threshold": entry_threshold, "exit_threshold": exit_threshold,
                "matrix_leverage": matrix_leverage, "near_miss": near_miss,
                "reason": "p_matrix_rejected",
            }

        # Rule 2: LONG Confirmation — both XGB AND Pearson R must be above their
        # respective P-thresholds simultaneously to execute a LONG trade
        if signal_obj.direction == 1:  # LONG
            r_threshold = self.percentile_calc.get_r_threshold(
                signal_obj.best_tf, matrix_params["entry_p"]
            )
            if signal_obj.confidence < r_threshold:
                near_miss = {
                    "asset": asset, "group": group, "dir": "LONG",
                    "r": round(signal_obj.confidence, 4),
                    "pvt_r": round(signal_obj.pvt_r, 4),
                    "tf": signal_obj.best_tf, "period": signal_obj.best_period,
                    "rejected_at": "LONG-R-GATE",
                    "detail": (f"R={signal_obj.confidence:.4f} < "
                               f"R_P{matrix_params['entry_p']*100:.0f}={r_threshold:.4f}"),
                    "xgb_prob": round(xgb_prob, 4),
                }
                return {
                    "passed": False, "signal": signal_obj, "xgb_prob": xgb_prob,
                    "entry_threshold": entry_threshold, "exit_threshold": exit_threshold,
                    "matrix_leverage": matrix_leverage, "near_miss": near_miss,
                    "reason": "long_r_gate_rejected",
                }

        return {
            "passed": True, "signal": signal_obj, "xgb_prob": xgb_prob,
            "entry_threshold": entry_threshold, "exit_threshold": exit_threshold,
            "matrix_leverage": matrix_leverage, "near_miss": None,
            "reason": "passed",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RiskEngine — P-Floor Exits, Hard SL, Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Manages all exit logic via Mathematical Exhaustion:
      1. P-Floor exit: current XGB score < exit_p threshold → immediate MARKET close
         (Mathematical Exhaustion — overrides trailing stops)
      2. Hard SL: 1.5% adverse move from entry → MARKET close (overrides P-Floor)
      3. Circuit breaker: portfolio drawdown exceeds MAX_DRAWDOWN_PCT

    Priority: Hard SL > P-Floor > Time Barrier
    """

    def __init__(self, xgb_model: xgb.XGBClassifier, percentile_calc: PercentileCalculator):
        self.xgb_model = xgb_model
        self.percentile_calc = percentile_calc

    def compute_current_score(self, pos: LivePosition, close_arr: np.ndarray,
                              pvt_arr: np.ndarray | None = None) -> tuple[float, float, float]:
        """
        Recompute Pearson R and XGB score using latest data on the entry TF.

        Returns: (current_r, current_xgb_score, new_midline)
        """
        if len(close_arr) < pos.best_period:
            return pos.confidence, pos.xgb_prob, pos.midline

        std_dev, pearson_r, slope, intercept = calc_log_regression(close_arr, pos.best_period)
        midline = np.exp(intercept)
        current_r = abs(pearson_r)

        # Recompute XGB score with updated confidence
        current_xgb = predict_probability(
            self.xgb_model,
            confidence=current_r,
            pvt_r=pos.pvt_r,
            best_tf=pos.best_tf,
            best_period=pos.best_period,
        )

        return current_r, current_xgb, midline

    def check_p_floor_exit(self, pos: LivePosition, current_xgb_score: float) -> tuple[bool, str]:
        """
        Check if current XGB score has fallen below the P-Floor (exit_p threshold).
        Returns (should_exit, reason).
        """
        exit_threshold = pos.p_exit_threshold
        if exit_threshold <= 0:
            # Legacy position without P-Matrix thresholds — use dynamic lookup
            exit_threshold = self.percentile_calc.get_exit_threshold(pos.group, pos.best_tf)

        # Zero-Floor Protection: if threshold is effectively zero, re-sync or use fallback
        if exit_threshold <= 0.001:
            exit_threshold = self._zero_floor_fallback(pos)
            # Persist the recovered threshold back to the position
            pos.p_exit_threshold = exit_threshold

        if current_xgb_score < exit_threshold:
            return True, (
                f"P-FLOOR: XGB={current_xgb_score:.4f} < "
                f"exit_threshold={exit_threshold:.4f}"
            )
        return False, ""

    def _zero_floor_fallback(self, pos: LivePosition) -> float:
        """
        Zero-Floor Protection: re-sync with historical validation data
        or use P15 floor from last 100 successful trades.
        """
        # First: re-sync from PercentileCalculator (P15 for the TF)
        threshold = self.percentile_calc.get_threshold(pos.best_tf, 0.15)
        if threshold > 0.001:
            logger.warning("ZERO-FLOOR RESYNC: %s/%s → P15=%.4f from distributions",
                           pos.asset, pos.best_tf, threshold)
            return threshold

        # Second: compute P15 from last 100 successful trades
        try:
            if TRADES_FILE.exists():
                df = pd.read_csv(TRADES_FILE)
                df.columns = [c.strip().lower() for c in df.columns]
                if "pnl_usd" in df.columns and "confidence" in df.columns:
                    wins = df[df["pnl_usd"] > 0].tail(100)
                    if len(wins) >= 10:
                        xgb_scores = []
                        for _, row in wins.iterrows():
                            try:
                                score = predict_probability(
                                    self.xgb_model,
                                    confidence=float(row["confidence"]),
                                    pvt_r=float(row["pvt_r"]),
                                    best_tf=str(row["best_tf"]),
                                    best_period=int(row["best_period"]),
                                )
                                xgb_scores.append(score)
                            except Exception:
                                pass
                        if len(xgb_scores) >= 5:
                            p15 = float(np.percentile(xgb_scores, 15))
                            logger.warning("ZERO-FLOOR FALLBACK: %s → P15=%.4f from %d winning trades",
                                           pos.asset, p15, len(xgb_scores))
                            return p15
        except Exception as e:
            logger.error("ZERO-FLOOR fallback computation failed: %s", e)

        # Ultimate fallback: absolute P15 default
        logger.warning("ZERO-FLOOR DEFAULT: %s → using absolute fallback P15=0.15", pos.asset)
        return 0.15

    def check_hard_sl(self, pos: LivePosition, current_price: float) -> tuple[bool, str]:
        """
        Global 1.5% hard SL check — emergency circuit breaker per position.
        Returns (should_exit, reason).
        """
        if pos.direction == 1:
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            if pnl_pct <= -HARD_SL_PCT:
                return True, f"HARD_SL_1.5%: price={current_price:.6f} pnl={pnl_pct*100:.2f}%"
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            if pnl_pct <= -HARD_SL_PCT:
                return True, f"HARD_SL_1.5%: price={current_price:.6f} pnl={pnl_pct*100:.2f}%"
        return False, ""

    def check_circuit_breaker(self, state: LiveState) -> bool:
        """Check portfolio-level drawdown circuit breaker."""
        equity = state.initial_capital + state.realized_pnl
        if equity > state.peak_equity:
            state.peak_equity = equity
        dd = (equity - state.peak_equity) / state.peak_equity * 100
        if dd <= MAX_DRAWDOWN_PCT:
            logger.critical("CIRCUIT BREAKER: Max DD hit (%.2f%%)", dd)
            tg_send(f"<b>CIRCUIT BREAKER TRIPPED</b>\nDrawdown: {dd:.2f}%")
            state.circuit_breaker = True
            save_state(state)
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TradeManager — Orchestrates Full Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TradeManager:
    """
    Main trading orchestrator integrating:
      - SignalAuditor for entry validation
      - RiskEngine for exit monitoring
      - CooldownManager for recovery logic
      - Binance execution via ccxt
    """

    def __init__(self, initial_capital: float, live_mode: bool = False):
        self.live_mode = live_mode
        self.lock = threading.Lock()
        self.state = load_state()

        # Sync with Binance if live
        if self.live_mode:
            try:
                ex = _get_exchange()
                bal = ex.fetch_balance()
                actual_bal = float(bal.get("USDT", {}).get("total", 0))
                if actual_bal > 0:
                    logger.info("Syncing state with Binance: Live Balance = $%.2f", actual_bal)
                    self.state.initial_capital = actual_bal
                    self.state.realized_pnl = 0.0
                    if actual_bal > self.state.peak_equity:
                        self.state.peak_equity = actual_bal
                    save_state(self.state)
            except Exception as e:
                logger.error("Failed to sync balance with Binance: %s", e)

        if self.state.trade_counter == 0 and not self.live_mode:
            self.state.initial_capital = initial_capital
            self.state.peak_equity = initial_capital

        # Load XGBoost model
        self.xgb_model, _ = load_model(XGB_MODEL_PATH)

        # Load physics thresholds
        self.consensus = _load_consensus_params()
        self.group_thresholds = _load_group_thresholds()
        self._apply_consensus()

        # Initialize core components
        self.percentile_calc = PercentileCalculator(self.xgb_model)
        self.cooldown_mgr = CooldownManager(self.state)
        self.signal_auditor = SignalAuditor(
            self.xgb_model, self.percentile_calc,
            self.group_thresholds, self.consensus,
        )
        self.risk_engine = RiskEngine(self.xgb_model, self.percentile_calc)

        # WS scan batch accumulator
        self._ws_scan_batch = []
        self._ws_scan_batch_ts = ""

        logger.info("TradeManager initialized — mode=%s, capital=$%.0f, assets=%d",
                     "LIVE" if live_mode else "DRY-RUN", self.state.initial_capital, len(ASSETS))
        logger.info("TF-Based Percentile Engine active. Leverage is TF-only (not group-based).")

        # Log TF hierarchy thresholds
        for tf in ["5m", "30m", "1h", "4h"]:
            params = _get_matrix_params("", tf)
            entry_t = self.percentile_calc.get_entry_threshold("", tf)
            exit_t = self.percentile_calc.get_exit_threshold("", tf)
            logger.info("  TF %3s: %dx leverage | entry >= P%.0f (%.4f) | exit < P%.0f (%.4f)",
                        tf, params["leverage"], params["entry_p"]*100, entry_t,
                        params["exit_p"]*100, exit_t)

    def _apply_consensus(self):
        """Patch adaptive_engine globals to consensus defaults."""
        import neo_flow.adaptive_engine as ae
        ae.MIN_PEARSON_R = self.consensus["min_pearson_r"]
        ae.MIN_PVT_PEARSON_R = self.consensus["min_pvt_r"]
        ae.COMBINED_GATE_THRESHOLD = self.consensus["combined_gate"]
        ae.HARD_SL_ATR_MULT = self.consensus["hard_sl_mult"]
        ae.TRAIL_BUFFER_STD = self.consensus["trail_buffer"]
        ae.TREND_EXHAUST_R = self.consensus["exhaust_r"]

    # ── Trade Execution ────────────────────────────────────────────────────

    def _close_trade(self, pos: LivePosition, exit_price: float, reason: str):
        """Finalize trade: close on exchange, compute PnL, log, send alert, update cooldown."""
        now = datetime.now(timezone.utc)
        actual_exit_price = exit_price

        if self.live_mode:
            side = "buy" if pos.direction == 1 else "sell"
            try:
                _cancel_all_orders_full(pos.symbol)
                order = close_position_order(pos.symbol, side, pos.quantity,
                                             exit_price, reason)
                if order is None:
                    logger.error("CLOSE RETURNED NONE for %s — keeping position", pos.asset)
                    save_state(self.state)
                    return
                fill = float(order.get("average", 0) or 0)
                if fill > 0:
                    actual_exit_price = fill
            except Exception as e:
                logger.error("Exchange close failed for %s: %s — keeping position", pos.asset, e)
                save_state(self.state)
                return

        # PnL calculation
        if self.live_mode and pos.quantity > 0:
            pnl_usd = (actual_exit_price - pos.entry_price) * pos.direction * pos.quantity
            pnl_pct = pnl_usd / (pos.position_usd / pos.leverage) * 100.0 if pos.leverage > 0 else 0.0
        else:
            pnl_pct = (actual_exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100.0
            pnl_usd = pnl_pct / 100.0 * pos.position_usd

        # Fetch real commissions
        exit_commission = 0.0
        total_commission = 0.0
        if self.live_mode:
            exit_order_id = ""
            if order is not None:
                exit_order_id = str(order.get("id", ""))
            if exit_order_id:
                exit_commission = _fetch_order_commission(pos.symbol, exit_order_id)
            total_commission = pos.entry_commission + exit_commission
            pnl_usd -= total_commission
            logger.info("COMMISSION %s: entry=$%.4f exit=$%.4f total=$%.4f",
                        pos.asset, pos.entry_commission, exit_commission, total_commission)

        net_pnl_pct = pnl_usd / pos.position_usd * 100.0 if pos.position_usd > 0 else pnl_pct

        self.state.realized_pnl += pnl_usd
        del self.state.positions[pos.asset]

        # Update cooldown
        self.cooldown_mgr.record_exit(pos.asset, net_pnl_pct)
        save_state(self.state)

        entry_dt = datetime.fromisoformat(pos.entry_ts)
        dur = (now - entry_dt).total_seconds() / 3600

        trade_log = {
            "trade_id": pos.trade_id, "asset": pos.asset, "group": pos.group,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "entry_ts": pos.entry_ts, "exit_ts": now.isoformat(), "entry_price": pos.entry_price,
            "exit_price": actual_exit_price, "best_tf": pos.best_tf, "best_period": pos.best_period,
            "confidence": pos.confidence, "pvt_r": pos.pvt_r, "leverage": pos.leverage,
            "position_usd": pos.position_usd, "hard_sl": pos.hard_sl, "exit_reason": reason,
            "bars_held": pos.bars_held, "duration_hours": round(dur, 2), "pnl_pct": round(net_pnl_pct, 4),
            "pnl_usd": round(pnl_usd, 2), "peak_r": round(pos.peak_r, 4),
            "entry_commission": round(pos.entry_commission, 4),
            "exit_commission": round(exit_commission, 4),
            "total_commission": round(total_commission, 4),
        }
        df = pd.DataFrame([trade_log])
        df.to_csv(TRADES_FILE, mode="a", header=not TRADES_FILE.exists(), index=False)

        logger.info("CLOSED %s %s [%s]: Net PnL %+.2f%% ($%+.2f) | %s",
                     "LONG" if pos.direction == 1 else "SHORT", pos.asset, pos.group,
                     net_pnl_pct, pnl_usd, reason)

        # Compute current percentile for exit notification
        exit_pctl = self.percentile_calc.score_to_percentile(pos.best_tf, pos.xgb_prob)

        if reason in ("P_FLOOR_EXIT", "P_FLOOR_EXIT_4H_PROJ"):
            # Statistical Exhaustion exit — dedicated MarkdownV2 template
            matrix_p = _get_matrix_params(pos.group, pos.best_tf)
            tg_exhaustion_exit_notify(
                symbol=pos.symbol,
                pnl_pct=net_pnl_pct,
                exit_score=pos.xgb_prob,
                exit_p=matrix_p["exit_p"],
                current_percentile=exit_pctl,
                exit_threshold=pos.p_exit_threshold,
            )
        else:
            # Standard close notification (MarkdownV2) with XGB info
            _e = _mdv2
            fee_str = f" \\| Fees: {_e(f'${total_commission:.2f}')}" if self.live_mode else ""
            distance = pos.xgb_prob - pos.p_exit_threshold if pos.p_exit_threshold > 0 else 0.0
            reason_label = {
                "HARD_SL_1.5%": "\U0001F6A8 HARD SL 1\\.5%",
                "TIME_BARRIER": "\u23F0 TIME BARRIER",
                "ADAPTIVE_TRAIL_HIT": "\U0001F4C9 TRAIL STOP",
            }.get(reason, _e(reason))
            _pnl = f"{net_pnl_pct:+.2f}"
            _usd = f"${pnl_usd:+,.2f}"
            msg = (
                f"{reason_label} *\\- {_e(pos.asset)}*\n"
                f"*Result:* {_e(_pnl)}% \\({_e(_usd)}\\){fee_str}\n"
                f"*XGB:* {_e(f'{pos.xgb_prob:.4f}')} \\(P{_e(f'{exit_pctl:.0f}')}\\) "
                f"\\| *Dist to P\\-Exit:* {_e(f'{distance:.4f}')}\n"
                f"*Duration:* {_e(f'{dur:.1f}')}h \\| *TF:* {_e(pos.best_tf)} \\| *Lev:* {_e(pos.leverage)}x"
            )
            tg_send(msg, parse_mode="MarkdownV2")

    def _try_enter(self, asset: str, audit_result: dict,
                   scan_ns: int, now: datetime) -> dict | None:
        """
        Execute entry using P-Matrix leverage and thresholds.
        Returns entry dict or None.
        """
        signal = audit_result["signal"]
        xgb_prob = audit_result["xgb_prob"]
        entry_threshold = audit_result["entry_threshold"]
        exit_threshold = audit_result["exit_threshold"]
        matrix_leverage = audit_result["matrix_leverage"]

        group = get_group(asset)
        gt = self.group_thresholds.get(group, DEFAULT_THRESHOLDS.get(group))

        price = signal.entry_price
        if self.live_mode:
            price = fetch_ticker_price(f"{asset}USDT") or signal.entry_price

        # Leverage from RISK_MATRIX (not confidence tiers)
        lev = matrix_leverage
        vol_scalar = HIGH_VOL_SCALAR if asset in HIGH_VOL_ASSETS else 1.0
        pos_frac = gt.pos_frac if gt else self.consensus["pos_frac"]
        pos_usd = self.state.initial_capital * pos_frac * lev * vol_scalar
        if pos_usd <= 0 or lev == 0:
            return None

        # Hard SL: 1.5% from entry (global rule)
        if signal.direction == 1:
            hard_sl = price * (1 - HARD_SL_PCT)
        else:
            hard_sl = price * (1 + HARD_SL_PCT)

        # Adaptive trail initial (preserved for sub-bar tracking)
        trail_buffer = gt.trail_buffer if gt else self.consensus["trail_buffer"]
        std_price = signal.midline * signal.std_dev
        if signal.direction == 1:
            trail_sl = signal.midline - trail_buffer * std_price
            trail_sl = min(trail_sl, price * (1 - 0.0015))  # min 0.15%
        else:
            trail_sl = signal.midline + trail_buffer * std_price
            trail_sl = max(trail_sl, price * (1 + 0.0015))

        quantity, order_id, sl_order_id, entry_commission = 0.0, "", "", 0.0
        if self.live_mode:
            _cancel_all_orders_full(f"{asset}USDT")
            side = "buy" if signal.direction == 1 else "sell"
            order = place_market_order(f"{asset}USDT", side, pos_usd, lev)
            if not order:
                return None
            order_id = str(order.get("id", ""))
            quantity = float(order.get("filled", 0) or order.get("amount", 0))
            price = float(order.get("average", price) or price)
            if quantity <= 0:
                logger.error("ORDER FILLED qty=0 for %s — skipping", asset)
                return None
            entry_commission = _fetch_order_commission(f"{asset}USDT", order_id)
            logger.info("ENTRY COMMISSION %s: $%.4f", asset, entry_commission)

            # Recalculate hard SL with actual fill price
            if signal.direction == 1:
                hard_sl = price * (1 - HARD_SL_PCT)
            else:
                hard_sl = price * (1 + HARD_SL_PCT)

            # Place exchange stop loss
            sl_order = place_stop_loss(f"{asset}USDT",
                                       "sell" if signal.direction == 1 else "buy",
                                       quantity, hard_sl)
            if sl_order:
                sl_order_id = str(sl_order.get("id", ""))

        self.state.trade_counter += 1
        pos = LivePosition(
            trade_id=self.state.trade_counter, asset=asset, group=group,
            symbol=f"{asset}USDT",
            direction=signal.direction, entry_price=price, entry_ts=now.isoformat(),
            hard_sl=hard_sl, trail_sl=trail_sl, midline=signal.midline, std_dev=signal.std_dev,
            best_tf=signal.best_tf, best_period=signal.best_period, confidence=signal.confidence,
            pvt_r=signal.pvt_r, leverage=lev, position_usd=pos_usd, quantity=quantity,
            peak_r=signal.confidence, order_id=order_id, sl_order_id=sl_order_id,
            last_bar_ns=scan_ns, xgb_prob=xgb_prob, entry_commission=entry_commission,
            p_entry_threshold=entry_threshold,
            p_exit_threshold=exit_threshold,
            matrix_leverage=matrix_leverage,
        )
        self.state.positions[asset] = pos
        save_state(self.state)

        dir_str = "LONG" if signal.direction == 1 else "SHORT"
        matrix_p = _get_matrix_params(group, signal.best_tf)
        entry = {
            "asset": asset, "group": group, "dir": dir_str,
            "r": round(signal.confidence, 4), "pvt_r": round(signal.pvt_r, 4),
            "tf": signal.best_tf, "period": signal.best_period,
            "xgb_prob": round(xgb_prob, 4),
            "entry_p": f"P{matrix_p['entry_p']*100:.0f}={entry_threshold:.4f}",
            "exit_p": f"P{matrix_p['exit_p']*100:.0f}={exit_threshold:.4f}",
            "price": round(price, 4),
            "lev": lev, "size": round(pos_usd, 0), "sl": round(hard_sl, 4),
        }
        current_pctl = self.percentile_calc.score_to_percentile(signal.best_tf, xgb_prob)
        tg_entry_notify(
            symbol=f"{asset}USDT",
            side=dir_str,
            tf=signal.best_tf,
            leverage=lev,
            xgb_score=xgb_prob,
            entry_p=matrix_p["entry_p"],
            current_percentile=current_pctl,
            exit_threshold=exit_threshold,
            exit_p=matrix_p["exit_p"],
        )
        return entry

    # ── Position Monitoring (P-Floor + Hard SL) ───────────────────────────

    def _refresh_position_data(self):
        """Fetch latest bars from Binance for open positions' TFs."""
        global _cached_data
        if _cached_data is None:
            return
        for asset, pos in list(self.state.positions.items()):
            ad = _cached_data.get(asset, {}).get(pos.best_tf)
            if ad is None:
                continue
            symbol = f"{asset}USDT"
            last_ns = ad.timestamps[-1]
            cursor_ms = last_ns // 1_000_000 + 1
            df_new = fetch_recent_klines(symbol, pos.best_tf, start_ms=cursor_ms, limit=100)
            if df_new.empty:
                continue
            new_ts = df_new.index.values.astype("datetime64[ns]").astype("int64")
            mask = new_ts > last_ns
            if not mask.any():
                continue
            ad.close = np.concatenate([ad.close, df_new["close"].values[mask].astype(np.float64)])
            ad.high = np.concatenate([ad.high, df_new["high"].values[mask].astype(np.float64)])
            ad.low = np.concatenate([ad.low, df_new["low"].values[mask].astype(np.float64)])
            ad.open_ = np.concatenate([ad.open_, df_new["open"].values[mask].astype(np.float64)])
            ad.volume = np.concatenate([ad.volume, df_new["volume"].values[mask].astype(np.float64)])
            ad.timestamps = np.concatenate([ad.timestamps, new_ts[mask]])
            ad.pvt = _compute_pvt(ad.close, ad.volume)

    def monitor_active_trades(self):
        """
        Continuous Signal Audit — runs every 5 minutes (on each WS bar close).
        For each open trade:
          1. Hard SL check: 1.5% adverse → MARKET close (highest priority)
          2. Recompute Pearson R + XGB score on original entry_tf with latest data
          3. Mathematical Exhaustion: if current_score < exit_p_floor → MARKET close
             (overrides any trailing stop)
          4. Time barrier: safety net at MAX_BARS_HELD
        """
        global _cached_data
        if _cached_data is None:
            return

        now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        completed_cutoff = now_ns - (WS_SETTLEMENT_BUFFER * 10**9)

        for asset, pos in list(self.state.positions.items()):
            asset_cache = _cached_data.get(asset, {})
            ad = asset_cache.get(pos.best_tf)
            if ad is None:
                continue

            # Process new bars since last check
            mask = (ad.timestamps > pos.last_bar_ns) & (ad.timestamps <= completed_cutoff)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                # No new bars — still check hard SL via ticker
                price = fetch_ticker_price(f"{asset}USDT")
                if price:
                    hit, reason = self.risk_engine.check_hard_sl(pos, price)
                    if hit:
                        logger.warning("HARD SL TRIGGERED (ticker): %s — %s", asset, reason)
                        self._close_trade(pos, price, "HARD_SL_1.5%")
                        continue

                # Rule 3: Real-time 4h signal projection (No-Lag Policy)
                # For 4h positions, project live XGB using 5m data snapshots.
                # Do NOT wait for 4h candle close — exit immediately if sub-candle
                # math drops below P-Exit Floor.
                if pos.best_tf == "4h":
                    ad_5m = asset_cache.get("5m")
                    ad_4h = ad
                    if (ad_5m is not None and len(ad_5m.close) > 0
                            and len(ad_4h.close) >= pos.best_period):
                        last_5m_close = float(ad_5m.close[-1])
                        projected_close = np.append(ad_4h.close, last_5m_close)
                        _, proj_xgb, _ = self.risk_engine.compute_current_score(
                            pos, projected_close
                        )
                        p_exit, p_reason = self.risk_engine.check_p_floor_exit(pos, proj_xgb)
                        if p_exit:
                            exit_price = price if price else last_5m_close
                            logger.warning("4H PROJECTION P-EXIT: %s — proj_xgb=%.4f | %s",
                                           asset, proj_xgb, p_reason)
                            self._close_trade(pos, exit_price, "P_FLOOR_EXIT_4H_PROJ")
                            continue
                continue

            closed = False
            for idx in indices:
                pos.bars_held += 1
                close_arr = ad.close[:idx + 1]
                bar_high = float(ad.high[idx])
                bar_low = float(ad.low[idx])
                bar_close = float(ad.close[idx])

                # 1. Hard SL check (1.5% from entry)
                current_price = bar_close
                if pos.direction == 1:
                    current_price = bar_low
                else:
                    current_price = bar_high

                hit, reason = self.risk_engine.check_hard_sl(pos, current_price)
                if hit:
                    logger.warning("HARD SL TRIGGERED: %s — %s", asset, reason)
                    self._close_trade(pos, pos.hard_sl, "HARD_SL_1.5%")
                    closed = True
                    break

                # 2. Recompute Pearson R and XGB score on entry TF
                current_r, current_xgb, new_midline = self.risk_engine.compute_current_score(
                    pos, close_arr
                )
                pos.midline = new_midline
                pos.peak_r = max(pos.peak_r, current_r)

                # 3. P-Floor exit check: current XGB score vs exit_p threshold
                p_exit, p_reason = self.risk_engine.check_p_floor_exit(pos, current_xgb)
                if p_exit:
                    logger.warning("P-FLOOR EXIT: %s — %s", asset, p_reason)
                    self._close_trade(pos, bar_close, "P_FLOOR_EXIT")
                    closed = True
                    break

                # 4. Time barrier (safety net)
                if pos.bars_held >= MAX_BARS_HELD:
                    self._close_trade(pos, bar_close, "TIME_BARRIER")
                    closed = True
                    break

                pos.last_bar_ns = int(ad.timestamps[idx])

            if not closed and len(indices) > 0:
                pos.last_bar_ns = int(ad.timestamps[indices[-1]])
                save_state(self.state)

    # ── Scanning & Entry ──────────────────────────────────────────────────

    def _scan_and_enter(self, all_data: dict, skip_assets: set):
        """Scan all 33 assets, audit through SignalAuditor, enter qualifying."""
        if len(self.state.positions) >= MAX_CONCURRENT:
            return [], []

        now = datetime.now(timezone.utc)
        scan_ns = int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1_000_000_000)
        scan_log = {"timestamp": now.isoformat(), "scan_number": self.state.total_scans,
                     "signals": []}
        near_misses = []
        entries = []

        for asset in ASSETS:
            if asset in self.state.positions or asset in skip_assets:
                scan_log["signals"].append({
                    "asset": asset,
                    "status": "in_position" if asset in self.state.positions else "just_closed",
                })
                continue
            if len(self.state.positions) >= MAX_CONCURRENT:
                scan_log["signals"].append({"asset": asset, "status": "max_positions"})
                break

            # Cooldown check
            is_cd, cd_reason = self.cooldown_mgr.is_on_cooldown(asset)
            if is_cd:
                scan_log["signals"].append({"asset": asset, "status": "cooldown", "detail": cd_reason})
                continue

            asset_data = all_data.get(asset)
            if not asset_data:
                scan_log["signals"].append({"asset": asset, "status": "no_data"})
                continue
            scan_dfs = build_scan_dataframes(asset_data, scan_ns - 1)
            df_4h = build_htf_dataframe(asset_data, scan_ns - 14400 * 10**9)
            if not scan_dfs or df_4h is None:
                scan_log["signals"].append({"asset": asset, "status": "no_data"})
                continue

            # Full audit: physics gates + P-Matrix sniper gate
            audit = self.signal_auditor.audit_entry(asset, None, scan_dfs, df_4h)

            if audit["near_miss"]:
                near_misses.append(audit["near_miss"])

            if not audit["passed"]:
                scan_log["signals"].append({
                    "asset": asset, "status": audit["reason"],
                    "xgb_prob": round(audit["xgb_prob"], 4),
                })
                continue

            # Enter
            entry = self._try_enter(asset, audit, scan_ns, now)
            if entry:
                entries.append(entry)
                group = get_group(asset)
                scan_log["signals"].append({
                    "asset": asset, "group": group, "status": "entered",
                    "confidence": round(audit["signal"].confidence, 4),
                    "xgb_prob": round(audit["xgb_prob"], 4),
                    "tf": audit["signal"].best_tf, "period": audit["signal"].best_period,
                    "leverage": audit["matrix_leverage"],
                })

        try:
            SCAN_LOG_FILE.write_text(json.dumps(scan_log, indent=2))
        except Exception:
            pass
        return near_misses, entries

    # ── Main Cycle (Hourly) ───────────────────────────────────────────────

    def run_cycle(self):
        """Hourly scan cycle: monitor positions → scan → enter → report."""
        now = datetime.now(timezone.utc)
        all_data = get_live_data()
        with self.lock:
            self.state.last_scan_ts = now.isoformat()
            self.state.total_scans += 1
            logger.info("SCAN CYCLE #%d — %s | Open: %d/%d",
                        self.state.total_scans, now.strftime("%H:%M UTC"),
                        len(self.state.positions), MAX_CONCURRENT)

            if self.state.circuit_breaker or self.risk_engine.check_circuit_breaker(self.state):
                return

            before = set(self.state.positions.keys())
            self._refresh_position_data()
            self.monitor_active_trades()
            closed = before - set(self.state.positions.keys())
            near_misses, entries = self._scan_and_enter(all_data, skip_assets=closed)
            save_state(self.state)
            equity = self.state.initial_capital + self.state.realized_pnl
            n_open = len(self.state.positions)
            scan_num = self.state.total_scans
            rpnl = self.state.realized_pnl
            grp_counts = {}
            for pos in self.state.positions.values():
                grp_counts[pos.group] = grp_counts.get(pos.group, 0) + 1
            grp_str = " | ".join(f"{g}:{c}" for g, c in sorted(grp_counts.items())) or "none"
            pos_list = [(a, p.group, p.direction, p.entry_price, p.confidence,
                         p.xgb_prob, p.best_tf, p.best_period, p.leverage,
                         p.position_usd, p.p_exit_threshold)
                        for a, p in self.state.positions.items()]

        logger.info("Cycle complete — Equity: $%.2f | Open: %d/%d [%s] | PnL: $%.2f",
                     equity, n_open, MAX_CONCURRENT, grp_str, rpnl)

        # ── Build Telegram Scan Report (MarkdownV2) ──
        e = _mdv2
        pnl_emoji = "\U0001F4C8" if rpnl >= 0 else "\U0001F4C9"

        # ── 1. THE PULSE (Header) ──
        lines = [
            f"\U0001F50D *Scan \\#{e(scan_num)}* \\| {e(now.strftime('%H:%M UTC'))}",
            f"\U0001F3E6 *EQUITY:* {e(f'${equity:,.2f}')} \\| "
            f"{pnl_emoji} *PnL:* {e(f'${rpnl:+,.2f}')}",
            f"\U0001F916 *BOT STATUS:* {e(n_open)} Positions Open / {e(MAX_CONCURRENT)} Max",
        ]

        # ── 2. NEW ENTRIES ──
        if entries:
            lines.append(f"\n\U0001F3AF *NEW ENTRIES \\({e(len(entries))}\\)*")
            for ent in entries:
                _xgb = f"{ent['xgb_prob']:.4f}"
                _r = f"{ent['r']:.4f}"
                _pvt = f"{ent['pvt_r']:.4f}"
                _sz = f"${ent['size']:,.0f}"
                lines.append(
                    f"  *{e(ent['asset'])}* \\[{e(ent['group'])}\\] "
                    f"\\({e(ent['dir'])}\\) \\| {e(ent['tf'])} {e(ent['lev'])}x\n"
                    f"    XGB: {e(_xgb)} \\| R: {e(_r)} \\| PVT: {e(_pvt)}\n"
                    f"    @ {e(ent['price'])} \\| {e(_sz)}"
                )

        # ── 3. ACTIVE POSITIONS: The Battlefield ──
        if pos_list:
            lines.append(f"\n\u2694\uFE0F *ACTIVE BATTLES \\({e(len(pos_list))}\\)*")
            for a, grp, d, ep, conf, xp, tf, per, lv, sz, exit_t in pos_list:
                ds = "LONG" if d == 1 else "SHORT"
                pctl = self.percentile_calc.score_to_percentile(tf, xp)
                distance = xp - exit_t if exit_t > 0 else 0.0
                matrix_p = _get_matrix_params(grp, tf)
                # Dynamic status comment
                if exit_t > 0 and xp < exit_t:
                    status = "\u26A0\uFE0F Exhausted\\!"
                elif exit_t > 0 and distance < 0.05:
                    status = "\u26A0\uFE0F Approaching Exit\\!"
                elif pctl >= 75:
                    status = "\u2705 Strong"
                elif pctl >= 50:
                    status = "\U0001F7E1 Stable"
                else:
                    status = "\U0001F7E0 Weakening"
                exit_p_label = f"{matrix_p['exit_p']*100:.0f}"
                lines.append(
                    f"\n  \u2694\uFE0F *{e(a)}* \\[{e(grp)}\\] \\({e(ds)}\\) "
                    f"\\| TF: {e(tf)} \\| Lev: {e(lv)}x\n"
                    f"  \U0001F9EC HEALTH: {e(f'{xp:.4f}')} \\| "
                    f"PERCENTILE: P{e(f'{pctl:.0f}')}\n"
                    f"  \U0001F6E1 EXIT FLOOR: P{e(exit_p_label)} "
                    f"\\({e(f'{exit_t:.4f}')}\\) \\| HARD SL: \\-1\\.5%\n"
                    f"  Status: {status}"
                )

        # ── 4. REJECTED NOISE (grouped by reason) ──
        if near_misses:
            # Group near misses by rejection reason
            rejection_groups: dict[str, list] = {}
            for nm in near_misses:
                key = nm.get("rejected_at", "UNKNOWN")
                detail = nm.get("detail", "")
                # Build a human-readable reason key
                if key == "HTF" and "neutral" in detail.lower():
                    reason_key = "HTF_NEUTRAL"
                elif key == "HTF" and "conflict" in detail.lower():
                    reason_key = "HTF_CONFLICT"
                elif key == "PVT":
                    reason_key = "PVT_DIVERGENCE"
                elif key == "P-MATRIX":
                    reason_key = "P_MATRIX"
                elif key == "LONG-R-GATE":
                    reason_key = "LONG_R_GATE"
                elif key == "GATE":
                    reason_key = "COMBINED_GATE"
                else:
                    reason_key = key
                if reason_key not in rejection_groups:
                    rejection_groups[reason_key] = []
                rejection_groups[reason_key].append(nm)

            # Rejection reason labels and meanings
            reason_labels = {
                "HTF_NEUTRAL": (
                    "\U0001F6E1 *REJECTED: HTF NEUTRAL \\(4H Sideways\\)*",
                    "Trend strong on local TF, but 4H big picture is sideways\\. Stayed out for safety\\.",
                ),
                "HTF_CONFLICT": (
                    "\U0001F6E1 *REJECTED: HTF CONFLICT \\(4H Opposes\\)*",
                    "Signal direction conflicts with the 4H macro trend\\. No counter\\-trend trades\\.",
                ),
                "PVT_DIVERGENCE": (
                    "\U0001F4C9 *REJECTED: VOLUME DIVERGENCE \\(PVT\\)*",
                    "Price moving but Smart Money \\(Volume\\) not following\\. High fake\\-out risk\\.",
                ),
                "P_MATRIX": (
                    "\U0001F4CA *REJECTED: BELOW ENTRY BARRIER \\(P\\-Matrix\\)*",
                    "XGB score too low for the TF entry percentile\\. Signal quality insufficient\\.",
                ),
                "LONG_R_GATE": (
                    "\U0001F512 *REJECTED: LONG R\\-GATE*",
                    "LONG signal passed XGB but Pearson R below threshold\\. Dual\\-gate blocked\\.",
                ),
                "COMBINED_GATE": (
                    "\u26A1 *REJECTED: OPPOSING SIGNAL*",
                    "Strong opposing regression detected\\. Conflicting forces in the market\\.",
                ),
            }

            lines.append(f"\n\U0001F6AB *REJECTED NOISE \\({e(len(near_misses))}\\)*")
            for reason_key, nm_group in rejection_groups.items():
                label, meaning = reason_labels.get(reason_key, (
                    f"\U0001F6E1 *REJECTED: {e(reason_key)}*",
                    f"Did not pass the {e(reason_key)} filter\\.",
                ))
                assets_str = ", ".join(
                    f"{nm['asset']}\\[{nm.get('group', '?')}\\]"
                    for nm in nm_group
                )
                lines.append(
                    f"\n{label}\n"
                    f"  Assets: {assets_str}\n"
                    f"  _Why: {meaning}_"
                )

            # ── 5. ALPHA OPPORTUNITY (best rejected) ──
            best_nm = max(
                near_misses,
                key=lambda nm: nm.get("xgb_prob", 0) or nm.get("r", 0),
            )
            best_xgb = best_nm.get("xgb_prob", 0)
            best_r = best_nm.get("r", 0)
            best_reason = best_nm.get("rejected_at", "?")
            best_detail = best_nm.get("detail", "")
            # Suggest what to watch for
            watch_hints = {
                "HTF": "4H candle to align with signal direction",
                "PVT": "Volume to confirm price direction",
                "P-MATRIX": "XGB score to climb above entry barrier",
                "LONG-R-GATE": "Pearson R to exceed the P\\-threshold",
                "GATE": "Opposing regression to weaken",
            }
            watch = watch_hints.get(best_reason, "conditions to improve")
            lines.append(
                f"\n\U0001F48E *TOP SPEC:* {e(best_nm['asset'])} "
                f"\\[{e(best_nm.get('group', '?'))}\\] {e(best_nm['dir'])} "
                f"\\(P\\={e(best_nm.get('period', '?'))} \\| "
                f"R\\={e(f'{best_r:.4f}')}\\)\n"
                f"  Barrier: {e(best_reason)} \\- {e(best_detail)}\n"
                f"  _Watch for: {watch}_"
            )

        # ── Cooldowns ──
        cd_list = []
        for a, cd in self.state.cooldowns.items():
            is_cd, reason = self.cooldown_mgr.is_on_cooldown(a)
            if is_cd:
                cd_list.append((a, reason))
        if cd_list:
            lines.append(f"\n\U0001F6D1 *COOLDOWNS \\({e(len(cd_list))}\\)*")
            for a, reason in cd_list:
                lines.append(f"  {e(a)}: {e(reason)}")

        tg_send("\n".join(lines), parse_mode="MarkdownV2")

    # ── 5m Websocket Scanner ─────────────────────────────────────────────

    def _flush_ws_scan_log(self):
        """Write accumulated 5m scan batch to scan log for dashboard."""
        try:
            ws_log = {
                "timestamp": self._ws_scan_batch_ts,
                "scan_number": self.state.total_scans,
                "signals": list(self._ws_scan_batch),
            }
            SCAN_LOG_FILE.write_text(json.dumps(ws_log, indent=2))
        except Exception:
            pass

    def scan_5m_asset(self, asset: str):
        """Run full physics + P-Matrix pipeline for one asset on WS 5m bar close."""
        global _cached_data
        if _cached_data is None:
            return
        with self.lock:
            if self.state.circuit_breaker:
                return

            now = datetime.now(timezone.utc)
            now_str = f"{now.strftime('%H:%M:%S')} (WS-5m)"

            if self._ws_scan_batch_ts != now_str:
                self._ws_scan_batch = []
                self._ws_scan_batch_ts = now_str

            self.state.current_scanning_asset = asset
            self.state.last_scan_ts = now.isoformat()
            save_state(self.state)

            group = get_group(asset)

            def _done(status="", detail="", r=0.0, pvt_r=0.0, xgb=0.0):
                self._ws_scan_batch.append({
                    "asset": asset, "group": group, "status": status, "detail": detail,
                    "confidence": round(r, 4), "pvt_r": round(pvt_r, 4), "xgb_prob": round(xgb, 4),
                    "tf": "5m",
                })
                self.state.current_scanning_asset = ""
                save_state(self.state)
                self._flush_ws_scan_log()

            if asset in self.state.positions:
                _done("in_position")
                return

            if len(self.state.positions) >= MAX_CONCURRENT:
                _done("max_positions")
                return

            is_cd, cd_reason = self.cooldown_mgr.is_on_cooldown(asset)
            if is_cd:
                _done("cooldown", cd_reason)
                return

            asset_data = _cached_data.get(asset)
            if not asset_data:
                _done("no_data")
                return

            scan_ns = int(now.timestamp() * 1_000_000_000)
            scan_dfs = build_scan_dataframes(asset_data, scan_ns - 1, scan_tfs=["5m"])
            df_4h = build_htf_dataframe(asset_data, scan_ns - 14400 * 10**9)
            if not scan_dfs or df_4h is None:
                _done("no_data")
                return

            # Full audit: physics + P-Matrix
            audit = self.signal_auditor.audit_entry(asset, None, scan_dfs, df_4h)

            if not audit["passed"]:
                abs_r = 0.0
                pvt_r_val = 0.0
                xgb_val = audit["xgb_prob"]
                if audit["near_miss"]:
                    abs_r = audit["near_miss"].get("r", 0.0)
                    pvt_r_val = audit["near_miss"].get("pvt_r", 0.0)
                _done(audit["reason"], r=abs_r, pvt_r=pvt_r_val, xgb=xgb_val)
                return

            signal = audit["signal"]
            _done("entered", f"XGB={audit['xgb_prob']:.4f}",
                  r=signal.confidence, pvt_r=signal.pvt_r, xgb=audit["xgb_prob"])

            entry = self._try_enter(asset, audit, scan_ns, now)
            if entry:
                logger.info("[WS-5m] NEW %s — %s [Grp%s] | R:%.4f | XGB:%.4f | Lev:%dx | $%.0f",
                            entry["dir"], asset, group, signal.confidence,
                            audit["xgb_prob"], entry["lev"], entry["size"])

    # ── Status Report ─────────────────────────────────────────────────────

    def send_status(self):
        with self.lock:
            equity = self.state.initial_capital + self.state.realized_pnl
            dd = (equity - self.state.peak_equity) / self.state.peak_equity * 100 if self.state.peak_equity else 0.0
            mode = "LIVE" if self.live_mode else "DRY\\-RUN"
            cb = "TRIPPED" if self.state.circuit_breaker else "OK"
            rpnl = self.state.realized_pnl
            peak = self.state.peak_equity
            n_open = len(self.state.positions)
            scans = self.state.total_scans
            pos_snapshot = [(a, p.group, p.direction, p.entry_price, p.hard_sl,
                            p.confidence, p.xgb_prob, p.p_exit_threshold, p.leverage,
                            p.best_tf)
                           for a, p in self.state.positions.items()]
            cd_snapshot = []
            for a, cd in self.state.cooldowns.items():
                is_cd, reason = self.cooldown_mgr.is_on_cooldown(a)
                if is_cd:
                    cd_snapshot.append((a, cd.consecutive_losses, reason))

        e = _mdv2
        pnl_emoji = "\U0001F4C8" if rpnl >= 0 else "\U0001F4C9"

        # ── 1. THE PULSE ──
        lines = [
            f"\U0001F4CB *Varanus Neo\\-Flow Extended*",
            f"*Mode:* {mode} \\| *CB:* {e(cb)}",
            f"\U0001F3E6 *EQUITY:* {e(f'${equity:,.2f}')} \\| "
            f"{pnl_emoji} *PnL:* {e(f'${rpnl:+,.2f}')}",
            f"*Drawdown:* {e(f'{dd:.2f}')}% \\| *Peak:* {e(f'${peak:,.2f}')}",
            f"\U0001F916 *BOT STATUS:* {e(n_open)} Positions Open / {e(MAX_CONCURRENT)} Max",
            f"*Scans:* {e(scans)} \\| *Entry:* P90/P80/P75/P70",
        ]

        # ── 2. ACTIVE BATTLES ──
        if pos_snapshot:
            lines.append(f"\n\u2694\uFE0F *ACTIVE BATTLES \\({e(len(pos_snapshot))}\\)*")
            for asset, grp, direction, entry_px, sl, conf, xgb_p, exit_t, lev, tf in pos_snapshot:
                dir_str = "LONG" if direction == 1 else "SHORT"
                pctl = self.percentile_calc.score_to_percentile(tf, xgb_p)
                distance = xgb_p - exit_t if exit_t > 0 else 0.0
                matrix_p = _get_matrix_params(grp, tf)
                if exit_t > 0 and xgb_p < exit_t:
                    status = "\u26A0\uFE0F Exhausted\\!"
                elif exit_t > 0 and distance < 0.05:
                    status = "\u26A0\uFE0F Approaching Exit\\!"
                elif pctl >= 75:
                    status = "\u2705 Strong"
                elif pctl >= 50:
                    status = "\U0001F7E1 Stable"
                else:
                    status = "\U0001F7E0 Weakening"
                exit_p_label = f"{matrix_p['exit_p']*100:.0f}"
                _xgb_s = f"{xgb_p:.4f}"
                _pctl_s = f"{pctl:.0f}"
                _exit_t_s = f"{exit_t:.4f}"
                lines.append(
                    f"\n  \u2694\uFE0F *{e(asset)}* \\[{e(grp)}\\] \\({e(dir_str)}\\) "
                    f"\\| TF: {e(tf)} \\| Lev: {e(lev)}x\n"
                    f"  \U0001F9EC HEALTH: {e(_xgb_s)} \\| "
                    f"PERCENTILE: P{e(_pctl_s)}\n"
                    f"  \U0001F6E1 EXIT FLOOR: P{e(exit_p_label)} "
                    f"\\({e(_exit_t_s)}\\) \\| HARD SL: \\-1\\.5%\n"
                    f"  Status: {status}"
                )

        # ── Cooldowns ──
        if cd_snapshot:
            lines.append(f"\n\U0001F6D1 *COOLDOWNS \\({e(len(cd_snapshot))}\\)*")
            for a, losses, reason in cd_snapshot:
                lines.append(f"  {e(a)} \\| streak: {e(losses)} \\- {e(reason)}")

        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        lines.append(f"\n{e(ts)}")
        tg_send("\n".join(lines), parse_mode="MarkdownV2")

    # ── Daily Performance Snapshot ────────────────────────────────────────

    def send_daily_snapshot(self, target_date: str = ""):
        """
        Read today's closed trades from CSV and send a MarkdownV2 daily summary.
        target_date: 'YYYY-MM-DD' string; defaults to yesterday UTC.
        """
        if not target_date:
            target_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            if not TRADES_FILE.exists():
                logger.info("Daily snapshot skipped — no trade log file")
                return
            df = pd.read_csv(TRADES_FILE)
            df.columns = [c.strip().lower() for c in df.columns]
            if "exit_ts" not in df.columns or "pnl_usd" not in df.columns:
                return

            df["exit_date"] = pd.to_datetime(df["exit_ts"]).dt.strftime("%Y-%m-%d")
            day_df = df[df["exit_date"] == target_date]

            total = len(day_df)
            if total == 0:
                logger.info("Daily snapshot: no trades closed on %s", target_date)
                return

            wins = int((day_df["pnl_usd"] > 0).sum())
            net_pnl = float(day_df["pnl_usd"].sum())

            # Mean entry XGB score (confidence column is Pearson R; xgb_prob is
            # not in the CSV, so approximate with confidence as quality proxy)
            mean_xgb = 0.0
            if "confidence" in day_df.columns:
                mean_xgb = float(day_df["confidence"].mean())

            tg_daily_snapshot(total, wins, net_pnl, mean_xgb, target_date)
            logger.info("Daily snapshot sent for %s: %d trades, WR=%.1f%%, PnL=$%.2f",
                        target_date, total, wins / total * 100 if total else 0, net_pnl)
        except Exception as e:
            logger.error("Daily snapshot failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Websocket Engine (Real-Time 5m Alignment)
# ═══════════════════════════════════════════════════════════════════════════════

def start_websocket_stream(bot: TradeManager):
    """Listen to Binance Futures Kline streams for real-time bar updates."""
    import json as _json

    def on_message(ws, message):
        try:
            msg = _json.loads(message)
            data = msg.get("data", msg)
            k = data.get("k", {})
            if k.get("x"):
                asset = data["s"].replace("USDT", "")
                _append_ws_bar_to_cache(
                    asset=asset,
                    open_=float(k["o"]),
                    high=float(k["h"]),
                    low=float(k["l"]),
                    close=float(k["c"]),
                    volume=float(k["v"]),
                    ts_ms=int(k["t"])
                )
                logger.debug("WS BAR RECEIVED: %s %s @ %.4f", asset, k["i"], float(k["c"]))
                # Monitor active trades on every 5m bar close
                with bot.lock:
                    if bot.state.positions:
                        bot._refresh_position_data()
                        bot.monitor_active_trades()
        except Exception as e:
            logger.error("Websocket message error: %s", e)

    def on_error(ws, error):
        logger.error("Websocket error: %s", error)

    def on_close(ws, close_status_code, close_msg):
        logger.warning("Websocket closed. Reconnecting...")
        time.sleep(5)
        _run_ws()

    def _run_ws():
        streams = [f"{s.lower()}@kline_5m" for s in ALL_SYMBOLS]
        stream_url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        ws = websocket.WebSocketApp(stream_url, on_message=on_message,
                                     on_error=on_error, on_close=on_close)
        ws.run_forever()

    threading.Thread(target=_run_ws, daemon=True).start()
    logger.info("Websocket stream started for 33 assets (5m interval)")


# ═══════════════════════════════════════════════════════════════════════════════
# Threads: Command Listener & P-Floor Monitor
# ═══════════════════════════════════════════════════════════════════════════════

def start_telegram_listener(bot: TradeManager):
    def _poll():
        offset = 0
        while True:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
                resp = _requests.get(url, params={"offset": offset, "timeout": 30}, timeout=40)
                for update in resp.json().get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    if str(msg.get("chat", {}).get("id")) == TELEGRAM_CHAT_ID:
                        text = msg.get("text", "").lower()
                        if "/status" in text:
                            bot.send_status()
                        elif "/daily" in text:
                            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            bot.send_daily_snapshot(target_date=today)
                        elif text.startswith("/reset"):
                            # Manual cooldown/blacklist reset: /reset ASSET
                            parts = msg.get("text", "").split()
                            if len(parts) >= 2:
                                asset_arg = parts[1].upper().replace("USDT", "")
                                with bot.lock:
                                    if bot.cooldown_mgr.reset_cooldown(asset_arg):
                                        save_state(bot.state)
                                        tg_send(
                                            f"\u2705 *MANUAL RESET*\n"
                                            f"{_mdv2(asset_arg)}: cooldown and blacklist cleared\\.",
                                            parse_mode="MarkdownV2",
                                        )
                                    else:
                                        tg_send(
                                            f"\u26A0\uFE0F {_mdv2(asset_arg)} has no active cooldown\\.",
                                            parse_mode="MarkdownV2",
                                        )
                            else:
                                tg_send(
                                    "\u26A0\uFE0F Usage: `/reset ASSET` \\(e\\.g\\. `/reset BTC`\\)",
                                    parse_mode="MarkdownV2",
                                )
                        elif "/help" in text:
                            tg_send(
                                "\u2139\uFE0F *Commands:*\n"
                                "/status \\- Bot status with P\\-Matrix info\n"
                                "/daily \\- Today's performance snapshot\n"
                                "/reset ASSET \\- Manual cooldown/blacklist reset\n"
                                "/help \\- Show this help",
                                parse_mode="MarkdownV2",
                            )
            except Exception as e:
                logger.error("Telegram listener error: %s", e)
                time.sleep(5)
    threading.Thread(target=_poll, daemon=True).start()


def start_pfloor_monitor(bot: TradeManager):
    """
    Background P-Floor monitor: every 5 minutes.
    Fetches latest bars, recomputes XGB scores, checks P-Floor exits.
    Also checks 1.5% hard SL via ticker as safety net.
    """
    def _monitor():
        while True:
            try:
                with bot.lock:
                    bot.state.last_scan_ts = datetime.now(timezone.utc).isoformat()
                    save_state(bot.state)

                if bot.state.positions:
                    # Ticker price safety net for hard SL
                    prices = {}
                    for asset, pos in list(bot.state.positions.items()):
                        p = fetch_ticker_price(pos.symbol)
                        if p:
                            prices[asset] = p

                    with bot.lock:
                        bot._refresh_position_data()
                        bot.monitor_active_trades()

                        # Ticker-based hard SL fallback
                        for asset, price in prices.items():
                            if asset not in bot.state.positions:
                                continue
                            pos = bot.state.positions[asset]
                            hit, reason = bot.risk_engine.check_hard_sl(pos, price)
                            if hit:
                                logger.warning("HARD SL (ticker fallback): %s — %s", asset, reason)
                                bot._close_trade(pos, price, "HARD_SL_1.5%")

            except Exception as e:
                logger.error("P-Floor monitor error: %s", e)
            time.sleep(PFLOOR_MONITOR_INTERVAL)
    threading.Thread(target=_monitor, daemon=True).start()
    logger.info("P-Floor monitor started (interval=%ds)", PFLOOR_MONITOR_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════════════════════

_lock_fd = None


def _acquire_lock():
    """Acquire an exclusive file lock. Exits if another instance holds it."""
    global _lock_fd
    _lock_fd = open(LOCK_FILE_PATH, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.critical("Another v1 instance is already running (lock held). Aborting.")
        tg_send("<b>V1 STARTUP BLOCKED</b>\nAnother bot instance holds the lock file.")
        sys.exit(1)
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
    logger.info("Lock acquired — PID %d", os.getpid())


def _check_binance_sync(bot: TradeManager):
    """On startup, check if Binance has positions the bot doesn't know about."""
    try:
        ex = _get_exchange()
        ex.load_markets()
        bal = ex.fetch_balance()
        binance_positions = {
            p["symbol"].replace("USDT", ""): float(p["positionAmt"])
            for p in bal["info"]["positions"]
            if abs(float(p["positionAmt"])) > 0
        }
        bot_positions = set(bot.state.positions.keys())

        orphans = set(binance_positions.keys()) - bot_positions
        if orphans:
            msg = (f"<b>WARNING: Orphaned Binance positions</b>\n"
                   f"Binance has positions not tracked by bot:\n")
            for asset in orphans:
                msg += f"  {asset}USDT: qty={binance_positions[asset]}\n"
            logger.warning("Orphaned Binance positions: %s", orphans)
            tg_send(msg)

        for asset in list(bot_positions):
            if asset not in binance_positions:
                logger.warning("Bot has %s position but Binance does not — removing from state", asset)
                del bot.state.positions[asset]
                save_state(bot.state)
    except Exception as e:
        logger.error("Binance sync check failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Varanus Neo-Flow Extended — P-Matrix Engine")
    parser.add_argument("--live", action="store_true", help="Enable real trading on Binance")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital (USD)")
    parser.add_argument("--once", action="store_true", help="Run single scan cycle then exit")
    args = parser.parse_args()

    _acquire_lock()

    bot = TradeManager(args.capital, args.live)

    if args.live and not args.once:
        _check_binance_sync(bot)

    logger.info("Pre-loading data cache for immediate 5m WS scanning...")
    get_live_data()

    start_telegram_listener(bot)
    start_pfloor_monitor(bot)
    start_websocket_stream(bot)

    stop_event = threading.Event()

    def _shutdown(s, f):
        logger.info("Shutting down (signal %s)...", s)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.once:
        logger.info("Single scan mode (--once)")
        bot.run_cycle()
        return

    now = datetime.now(timezone.utc)
    next_trigger = now.replace(minute=1, second=0, microsecond=0) + timedelta(hours=1)
    wait_secs = int((next_trigger - now).total_seconds())
    logger.info("Waiting for next hourly scan: %s UTC (%d min %d sec)",
                next_trigger.strftime("%H:%M"), wait_secs // 60, wait_secs % 60)
    _e = _mdv2
    _mode = "LIVE" if args.live else "DRY\\-RUN"
    tg_send(
        f"\U0001F680 *v1 Extended Optimized Engine Started*\n"
        f"*Mode:* {_mode} \\| *Capital:* {_e(f'${args.capital:,.0f}')}\n"
        f"*Assets:* {_e(len(ASSETS))} \\| *Leverage:* 5m\\=1x 30m\\=3x 1h/4h\\=5x\n"
        f"*Entry:* P90/P80/P75/P70 \\| *Exit:* P20/P20/P15/P15 \\+ 1\\.5% Hard SL\n"
        f"*Group A:* P90 forced on 5m/30m \\| *LONG:* Dual\\-gate \\(XGB\\+R\\)\n"
        f"*4h:* Real\\-time projection \\(no\\-lag\\) \\| *Zero\\-Floor:* Protected\n"
        f"*Cooldown:* 15m/45m/180m\\+24h blacklist \\+ 0\\.5% stress cost\n"
        f"*Next scan:* {_e(next_trigger.strftime('%H:%M UTC'))} \\({_e(wait_secs // 60)}m\\)",
        parse_mode="MarkdownV2",
    )

    last_cycle_hour = -1
    last_daily_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        next_trigger = now.replace(minute=1, second=0, microsecond=0) + timedelta(hours=1)
        sleep_secs = max(1, int((next_trigger - now).total_seconds()))
        stop_event.wait(timeout=sleep_secs)
        if stop_event.is_set():
            break
        now = datetime.now(timezone.utc)

        # Daily performance snapshot at midnight UTC rollover
        today_str = now.strftime("%Y-%m-%d")
        if today_str != last_daily_date:
            try:
                bot.send_daily_snapshot(target_date=last_daily_date)
            except Exception as e:
                logger.error("Daily snapshot error: %s", e)
            last_daily_date = today_str

        current_hour = now.hour
        if current_hour == last_cycle_hour:
            logger.warning("Skipping duplicate cycle for hour %02d (already ran)", current_hour)
            continue
        last_cycle_hour = current_hour
        try:
            bot.run_cycle()
        except Exception as e:
            logger.error("Cycle error: %s\n%s", e, traceback.format_exc())


if __name__ == "__main__":
    main()
