#!/usr/bin/env python3
"""
live_extended_bot.py — Varanus Neo-Flow Hybrid Extended: 33 assets, 3 groups, Physics + XGBoost.

Architecture:
    1. Hourly Scan (Top of Hour): Fetches new bars from Binance, updates Parquet cache.
    2. Physics Gates: R-gate, PVT-gate, HTF-bias, Combined-gate (group-specific thresholds).
    3. ML Meta-Labeler: XGBoost classification vets signals (group-specific thresholds).
    4. Execution: Dry-Run or Live Binance Futures via CCXT.
    5. Monitoring: Continuous background check for SL/Trail (every 30s).
    6. Telegram: Full control & reporting via @varanusneoextended.

Usage:
    # Dry-run mode (safe):
    python live_extended_bot.py --capital 1000

    # Live mode (REAL MONEY):
    python live_extended_bot.py --live --capital 1000
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
import subprocess
import requests as _requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import xgboost as xgb
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
    HIGH_VOL_ASSETS,
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
# Config & Constants
# ═══════════════════════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
STATE_FILE = BASE_DIR / "live_extended_state.json"
TRADES_FILE = BASE_DIR / "live_extended_trades.csv"
SCAN_LOG_FILE = LOGS_DIR / "scan_results.json"

XGB_MODEL_PATH = BASE_DIR / "models" / "meta_xgb.json"
WFV_RESULTS_PATH = BASE_DIR / "wfv_results.json"
OPTIMIZED_THRESHOLDS_PATH = BASE_DIR / "config" / "optimized_thresholds.json"

MAX_CONCURRENT = 8
MAX_DRAWDOWN_PCT = -15.0

# ═══════════════════════════════════════════════════════════════════════════════
# Load Group-Specific Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

def _load_group_thresholds() -> dict[str, GroupThresholds]:
    """Load optimized per-group thresholds."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    if OPTIMIZED_THRESHOLDS_PATH.exists():
        try:
            data = json.loads(OPTIMIZED_THRESHOLDS_PATH.read_text())
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
        except Exception as e:
            logging.error("Failed to load optimized thresholds: %s", e)
    return thresholds

def _load_consensus_params() -> dict:
    """Load consensus params from wfv_results.json (fallback defaults)."""
    if WFV_RESULTS_PATH.exists():
        with open(WFV_RESULTS_PATH) as f:
            results = json.load(f)
        consensus = results.get("consensus_params", {})
    else:
        consensus = {}
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
            LOGS_DIR / "live_extended_bot.log", maxBytes=10_000_000, backupCount=5,
        ),
    ],
)
logger = logging.getLogger("live_extended")

for name in ["urllib3", "ccxt", "httpx", "httpcore", "telegram"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _tg_escape(text: str) -> str:
    """Escape HTML special chars OUTSIDE of <b> tags."""
    import re
    # Extract <b>...</b> tags, escape everything else, then restore tags
    parts = re.split(r'(<b>.*?</b>)', text)
    result = []
    for p in parts:
        if p.startswith('<b>') and p.endswith('</b>'):
            result.append(p)
        else:
            p = p.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            result.append(p)
    return ''.join(result)

def tg_send(text: str, parse_mode: str = "HTML"):
    """Send a Telegram message. Non-blocking, swallows errors."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping notification")
        return
    try:
        safe_text = _tg_escape(text) if parse_mode == "HTML" else text
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": safe_text,
            "parse_mode": parse_mode,
        }
        resp = _requests.post(url, json=payload, timeout=10)
        if resp.status_code == 400 and parse_mode == "HTML":
            # Retry without HTML if it fails
            payload["parse_mode"] = ""
            payload["text"] = text.replace("<b>","").replace("</b>","")
            resp = _requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Telegram send failed: %s", e)

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
    """Load Parquet files (33 assets × 4 TFs)."""
    logger.info("Loading parquet cache (33 assets × 4 TFs)...")
    t0 = time.perf_counter()
    all_data = load_all_assets()
    elapsed = time.perf_counter() - t0
    total = sum(len(ad.close) for asset in all_data.values() for ad in asset.values())
    logger.info("Cache loaded in %.1fs — %d total bars", elapsed, total)
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

def fetch_ticker_price(symbol: str) -> float | None:
    try:
        resp = _requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except:
        return None

def fetch_long_short_ratio(symbol: str) -> float | None:
    """Fetch the Long/Short Ratio for Top Traders (Accounts) from Binance Futures."""
    try:
        # Use direct requests to avoid CCXT version attribute issues
        url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"
        params = {"symbol": symbol.replace("/", "").replace(":USDT", "") + "USDT" if not symbol.endswith("USDT") else symbol.replace("/", ""), 
                  "period": "5m", "limit": 1}
        # Simplify symbol handling: ensure it ends with USDT and has no slashes
        clean_symbol = symbol.replace("/", "")
        if not clean_symbol.endswith("USDT"):
            clean_symbol += "USDT"
        params["symbol"] = clean_symbol
        
        resp = _requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) > 0:
            return float(data[0]["longShortRatio"])
    except Exception as e:
        logger.error("Failed to fetch L/S ratio for %s: %s", symbol, e)
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# Execution Engine (Binance)
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
        market = ex.market(symbol)
        quantity = ex.amount_to_precision(symbol, quantity)
        order = ex.create_market_order(symbol, side, float(quantity))
        logger.info("ORDER PLACED: %s %s %sx qty=%s @ ~%.4f", side, symbol, leverage, quantity, price)
        return order
    except Exception as e:
        logger.error("ORDER FAILED: %s %s — %s", side, symbol, e)
        tg_send(f"<b>ORDER FAILED</b>\n{side.upper()} {symbol}\nError: {e}")
        return None

# Slippage buffer for STOP_LIMIT orders (0.05% = 5 bps)
# sell-side limit is slightly below stop; buy-side limit is slightly above stop
STOP_LIMIT_BUFFER = 0.0005


def _stop_limit_price(stop_price: float, side: str) -> float:
    """Compute limit price with buffer to ensure fill near the stop trigger."""
    if side == "sell":
        return stop_price * (1 - STOP_LIMIT_BUFFER)
    else:
        return stop_price * (1 + STOP_LIMIT_BUFFER)


def _cancel_all_algo_orders(symbol: str):
    """Cancel all algo/conditional orders for a symbol (STOP, TP, etc.)."""
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
        logger.error("STOP FAILED: %s %s — %s", side, symbol, e)
        return None


def update_stop_loss(symbol: str, side: str, quantity: float, new_stop: float, old_sl_order_id: str = "") -> str:
    """Cancel old stop and place a new STOP_LIMIT at the tightened trail price. Returns new order ID."""
    try:
        ex = _get_exchange()
        _cancel_all_orders_full(symbol)
        new_stop = float(ex.price_to_precision(symbol, new_stop))
        limit_price = float(ex.price_to_precision(symbol, _stop_limit_price(new_stop, side)))
        order = ex.create_order(symbol, "STOP", side, quantity, limit_price,
                                params={"stopPrice": new_stop, "reduceOnly": True})
        logger.info("STOP UPDATED: %s %s trigger=%.6f limit=%.6f", side, symbol, new_stop, limit_price)
        return str(order.get("id", ""))
    except Exception as e:
        logger.error("STOP UPDATE FAILED: %s %s — %s", side, symbol, e)
        return old_sl_order_id


def close_position(symbol: str, side: str, quantity: float,
                   exit_price: float = 0.0, use_market: bool = False):
    """Close position. Market order for SL exits (guaranteed fill), limit for profit exits."""
    try:
        ex = _get_exchange()
        close_side = "sell" if side == "buy" else "buy"
        quantity = float(ex.amount_to_precision(symbol, quantity))
        if use_market or exit_price <= 0:
            order = ex.create_market_order(symbol, close_side, quantity, params={"reduceOnly": True})
            fill_price = float(order.get("average", 0) or 0)
            logger.info("POSITION CLOSED (MARKET): %s %s qty=%s fill=%.6f", close_side, symbol, quantity, fill_price)
        else:
            if close_side == "sell":
                limit = float(ex.price_to_precision(symbol, exit_price * (1 - STOP_LIMIT_BUFFER)))
            else:
                limit = float(ex.price_to_precision(symbol, exit_price * (1 + STOP_LIMIT_BUFFER)))
            order = ex.create_limit_order(symbol, close_side, quantity, limit, params={"reduceOnly": True})
            logger.info("POSITION CLOSED (LIMIT): %s %s qty=%s @ %.6f", close_side, symbol, quantity, limit)
        return order
    except Exception as e:
        logger.error("CLOSE FAILED: %s — %s", symbol, e)
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

@dataclass
class BotState:
    positions: dict[str, LivePosition] = field(default_factory=dict)
    trade_counter: int = 0
    realized_pnl: float = 0.0
    initial_capital: float = 10_000.0
    peak_equity: float = 10_000.0
    circuit_breaker: bool = False
    last_scan_ts: str = ""
    total_scans: int = 0
    current_scanning_asset: str = ""
    closed_cooldowns: dict[str, str] = field(default_factory=dict)

def save_state(state: BotState):
    data = {
        "trade_counter": state.trade_counter,
        "realized_pnl": state.realized_pnl,
        "initial_capital": state.initial_capital,
        "peak_equity": state.peak_equity,
        "circuit_breaker": state.circuit_breaker,
        "last_scan_ts": state.last_scan_ts,
        "total_scans": state.total_scans,
        "current_scanning_asset": state.current_scanning_asset,
        "closed_cooldowns": state.closed_cooldowns,
        "positions": {k: asdict(v) for k, v in state.positions.items()},
    }
    STATE_FILE.write_text(json.dumps(data, indent=2, default=str))

def load_state() -> BotState:
    state = BotState()
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
            state.closed_cooldowns = data.get("closed_cooldowns", {})
            for k, v in data.get("positions", {}).items():
                state.positions[k] = LivePosition(**v)
        except Exception as e:
            logger.error("Failed to load state: %s", e)
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# Extended Live Bot Core
# ═══════════════════════════════════════════════════════════════════════════════

class LiveExtendedBot:
    def __init__(self, initial_capital: float, live_mode: bool = False):
        self.live_mode = live_mode
        self.lock = threading.Lock()
        self.state = load_state()
        
        # If live mode, sync initial_capital with actual Binance balance
        if self.live_mode:
            try:
                ex = _get_exchange()
                bal = ex.fetch_balance()
                actual_bal = float(bal.get("USDT", {}).get("total", 0))
                if actual_bal > 0:
                    logger.info("Syncing state with Binance: Live Balance = $%.2f", actual_bal)
                    # We reset realized_pnl to 0 and set initial_capital to actual balance 
                    # so that Equity (initial + realized) matches reality on startup.
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
        
        self.xgb_model, _ = load_model(XGB_MODEL_PATH)
        self.consensus = _load_consensus_params()
        self.group_thresholds = _load_group_thresholds()
        self._apply_consensus()
        logger.info("Bot initialized — mode=%s, capital=$%.0f, assets=%d, groups=3",
                    "LIVE" if live_mode else "DRY-RUN", self.state.initial_capital, len(ASSETS))
        logger.info("Consensus params: %s", self.consensus)
        for g, gt in self.group_thresholds.items():
            logger.info("Group %s thresholds: confidence=%.2f xgb=%.2f trail=%.2f sl=%.2f exhaust=%.3f pos=%.2f",
                        g, gt.min_confidence, gt.min_xgb_score, gt.trail_buffer, gt.hard_sl_mult, gt.exhaust_r, gt.pos_frac)

    def _apply_consensus(self):
        """Patch adaptive_engine globals to consensus defaults (overridden per-asset by group thresholds)."""
        import neo_flow.adaptive_engine as ae
        ae.MIN_PEARSON_R = self.consensus["min_pearson_r"]
        ae.MIN_PVT_PEARSON_R = self.consensus["min_pvt_r"]
        ae.COMBINED_GATE_THRESHOLD = self.consensus["combined_gate"]
        ae.HARD_SL_ATR_MULT = self.consensus["hard_sl_mult"]
        ae.TRAIL_BUFFER_STD = self.consensus["trail_buffer"]
        ae.TREND_EXHAUST_R = self.consensus["exhaust_r"]

    def _apply_group_params(self, asset: str):
        """Apply group-specific thresholds to engine globals before scanning an asset."""
        import neo_flow.adaptive_engine as ae
        gt = self.group_thresholds.get(get_group(asset))
        if gt:
            ae.MIN_PEARSON_R = gt.min_confidence
            ae.MIN_PVT_PEARSON_R = gt.min_pvt_r
            ae.COMBINED_GATE_THRESHOLD = gt.combined_gate
            ae.HARD_SL_ATR_MULT = gt.hard_sl_mult
            ae.TRAIL_BUFFER_STD = gt.trail_buffer
            ae.TREND_EXHAUST_R = gt.exhaust_r

    def _check_circuit_breaker(self) -> bool:
        equity = self.state.initial_capital + self.state.realized_pnl
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        dd = (equity - self.state.peak_equity) / self.state.peak_equity * 100
        if dd <= MAX_DRAWDOWN_PCT:
            logger.critical("CIRCUIT BREAKER: Max DD hit (%.2f%%)", dd)
            tg_send(f"<b>CIRCUIT BREAKER TRIPPED</b>\nDrawdown: {dd:.2f}%")
            self.state.circuit_breaker = True
            save_state(self.state)
            return True
        return False

    def _close_trade(self, pos: LivePosition, exit_price: float, reason: str):
        """Finalize and record a trade with commission tracking."""
        now = datetime.now(timezone.utc)
        actual_exit_price = exit_price
        order = None
        if self.live_mode:
            side = "buy" if pos.direction == 1 else "sell"
            use_market = reason in ("HARD_SL_HIT", "ADAPTIVE_TRAIL_HIT")
            try:
                _cancel_all_orders_full(pos.symbol)
                order = close_position(pos.symbol, side, pos.quantity, exit_price, use_market=use_market)
                if order is None:
                    logger.error("CLOSE RETURNED NONE for %s — keeping position in state", pos.asset)
                    save_state(self.state)
                    return
                fill = float(order.get("average", 0) or 0)
                if fill > 0:
                    actual_exit_price = fill
            except Exception as e:
                logger.error("Exchange close failed for %s: %s", pos.asset, e)
                save_state(self.state)
                return
        
        pnl_pct = (actual_exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100.0 * pos.leverage
        pnl_usd = pnl_pct / 100.0 * pos.position_usd
        
        exit_commission = 0.0
        if self.live_mode and order:
            exit_order_id = str(order.get("id", ""))
            if exit_order_id:
                exit_commission = _fetch_order_commission(pos.symbol, exit_order_id)
            total_commission = pos.entry_commission + exit_commission
            pnl_usd -= total_commission
            logger.info("COMMISSION %s: entry=$%.4f exit=$%.4f total=$%.4f",
                        pos.asset, pos.entry_commission, exit_commission, total_commission)
        else:
            total_commission = 0.0
            
        net_pnl_pct = pnl_usd / pos.position_usd * 100.0 if pos.position_usd > 0 else pnl_pct
        self.state.realized_pnl += pnl_usd
        del self.state.positions[pos.asset]
        self.state.closed_cooldowns[pos.asset] = now.isoformat()
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
        
        fee_str = f" | Fees: ${total_commission:.2f}" if self.live_mode else ""
        outcome_emoji = "✅" if pnl_usd > 0 else "❌"
        msg = (
            f"<b>{outcome_emoji} CLOSED {pos.asset}</b> [Grp {pos.group}]\n"
            f"Reason: {reason}\n"
            f"PnL: <b>{net_pnl_pct:+.2f}%</b> (${pnl_usd:+,.2f}){fee_str}\n"
            f"Duration: {dur:.1f}h | TF: {pos.best_tf}"
        )
        logger.info("CLOSED %s %s [%s]: Net PnL %+.2f%% ($%+.2f) | Fees: $%.2f | %s",
                    "LONG" if pos.direction == 1 else "SHORT", pos.asset, pos.group, 
                    net_pnl_pct, pnl_usd, total_commission, reason)
        tg_send(msg)

    def _refresh_position_data(self):
        """Fetch latest bars from Binance for open positions' TFs only."""
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

    def _process_sub_bars(self):
        """Process ALL new completed sub-bars for open positions."""
        global _cached_data
        if _cached_data is None:
            return
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        # 15s settlement buffer for all timeframes (ensures data stability)
        completed_cutoff = now_ns - (15 * 10**9)

        for asset, pos in list(self.state.positions.items()):
            ad = _cached_data.get(asset, {}).get(pos.best_tf)
            if ad is None:
                continue
            gt = self.group_thresholds.get(pos.group)
            
            mask = (ad.timestamps > pos.last_bar_ns) & (ad.timestamps <= completed_cutoff)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            closed = False
            for idx in indices:
                pos.bars_held += 1
                close_arr = ad.close[:idx + 1]
                current_r = 0.0
                if len(close_arr) >= pos.best_period:
                    std_dev, pearson_r, slope, intercept = calc_log_regression(close_arr, pos.best_period)
                    midline = np.exp(intercept)
                    pos.midline = midline
                    pos.std_dev = std_dev
                    trail_buffer = gt.trail_buffer if gt else self.consensus["trail_buffer"]
                    old_trail = pos.trail_sl
                    pos.trail_sl = compute_trail_sl(pos.direction, midline, std_dev, pos.trail_sl, trail_buffer)
                    pos.peak_r = max(pos.peak_r, abs(pearson_r))
                    current_r = pearson_r

                    # Update Binance stop order when trail tightens
                    if self.live_mode and pos.trail_sl != old_trail and pos.quantity > 0:
                        sl_side = "sell" if pos.direction == 1 else "buy"
                        pos.sl_order_id = update_stop_loss(
                            pos.symbol, sl_side, pos.quantity, pos.trail_sl, pos.sl_order_id,
                        )

                bar_high = float(ad.high[idx])
                bar_low = float(ad.low[idx])
                bar_close = float(ad.close[idx])

                if pos.direction == 1 and bar_low <= pos.hard_sl:
                    self._close_trade(pos, pos.hard_sl, "HARD_SL_HIT")
                    closed = True
                    break
                if pos.direction == -1 and bar_high >= pos.hard_sl:
                    self._close_trade(pos, pos.hard_sl, "HARD_SL_HIT")
                    closed = True
                    break

                if pos.direction == 1 and bar_low <= pos.trail_sl:
                    self._close_trade(pos, pos.trail_sl, "ADAPTIVE_TRAIL_HIT")
                    closed = True
                    break
                if pos.direction == -1 and bar_high >= pos.trail_sl:
                    self._close_trade(pos, pos.trail_sl, "ADAPTIVE_TRAIL_HIT")
                    closed = True
                    break

                exhaust_r = gt.exhaust_r if gt else self.consensus["exhaust_r"]
                if abs(current_r) < exhaust_r:
                    self._close_trade(pos, bar_close, "TREND_EXHAUSTION")
                    closed = True
                    break

                if pos.bars_held >= 200:
                    self._close_trade(pos, bar_close, "TIME_BARRIER")
                    closed = True
                    break

                pos.last_bar_ns = int(ad.timestamps[idx])

            if not closed and len(indices) > 0:
                pos.last_bar_ns = int(ad.timestamps[indices[-1]])
                save_state(self.state)

    def _is_on_cooldown(self, asset: str) -> bool:
        """Check if asset is still in re-entry cooldown."""
        REENTRY_COOLDOWN_SECONDS = 3600
        ts_str = self.state.closed_cooldowns.get(asset)
        if not ts_str:
            return False
        try:
            closed_at = datetime.fromisoformat(ts_str)
            elapsed = (datetime.now(timezone.utc) - closed_at).total_seconds()
            if elapsed < REENTRY_COOLDOWN_SECONDS:
                return True
            del self.state.closed_cooldowns[asset]
        except Exception:
            pass
        return False

    def _scan_and_enter(self, all_data: dict, skip_assets: set):
        if len(self.state.positions) >= MAX_CONCURRENT:
            return
        now = datetime.now(timezone.utc)
        scan_ns = int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1_000_000_000)
        scan_log = {"timestamp": now.isoformat(), "scan_number": self.state.total_scans,
                     "signals": []}
        entries = []
        for asset in ASSETS:
            if asset in self.state.positions or asset in skip_assets:
                scan_log["signals"].append({"asset": asset, "status": "in_position" if asset in self.state.positions else "just_closed"})
                continue
            if self._is_on_cooldown(asset):
                scan_log["signals"].append({"asset": asset, "status": "cooldown"})
                continue
            if len(self.state.positions) >= MAX_CONCURRENT:
                scan_log["signals"].append({"asset": asset, "status": "max_positions"})
                break

            group = get_group(asset)
            gt = self.group_thresholds.get(group, DEFAULT_THRESHOLDS.get(group))
            self._apply_group_params(asset)

            asset_data = all_data.get(asset)
            if not asset_data:
                scan_log["signals"].append({"asset": asset, "status": "no_data"})
                continue
            scan_dfs = build_scan_dataframes(asset_data, scan_ns - 1)
            df_4h = build_htf_dataframe(asset_data, scan_ns - 14400*10**9)
            if not scan_dfs or df_4h is None:
                scan_log["signals"].append({"asset": asset, "status": "no_data"})
                continue

            best, all_results = find_best_regression(scan_dfs)
            if not best:
                scan_log["signals"].append({"asset": asset, "status": "no_regression"})
                continue

            abs_r = abs(best.pearson_r)
            direction = 1 if best.slope < 0 else -1
            min_r = gt.min_confidence if gt else self.consensus["min_pearson_r"]

            if abs_r < min_r:
                scan_log["signals"].append({"asset": asset, "status": "weak_r", "r": abs_r})
                continue

            best_df_7d = trim_to_7d(scan_dfs[best.timeframe], best.timeframe)
            pvt = compute_pvt_regression(best_df_7d, best.period)
            min_pvt_r = gt.min_pvt_r if gt else self.consensus["min_pvt_r"]
            pvt_passes, pvt_reason = check_pvt_alignment(direction, abs_r, pvt, min_pvt_r)
            if not pvt_passes:
                scan_log["signals"].append({"asset": asset, "status": "pvt_rejected", "reason": pvt_reason})
                continue

            # Step 3: HTF bias OR Sentiment Override
            htf_bias = get_htf_bias(df_4h)
            ls_ratio = fetch_long_short_ratio(f"{asset}USDT")
            sentiment_override = False
            if ls_ratio is not None:
                if direction == 1 and ls_ratio > 1.15:
                    sentiment_override = True
                elif direction == -1 and ls_ratio < 0.85:
                    sentiment_override = True

            if htf_bias == 0 and not sentiment_override:
                scan_log["signals"].append({"asset": asset, "status": "htf_neutral"})
                continue
            if htf_bias != 0 and htf_bias != direction and not sentiment_override:
                scan_log["signals"].append({"asset": asset, "status": "htf_conflict"})
                continue

            cg_thresh = gt.combined_gate if gt else self.consensus["combined_gate"]
            if not check_combined_gate(direction, all_results, cg_thresh):
                scan_log["signals"].append({"asset": asset, "status": "combined_gate_blocked"})
                continue

            signal = scan_asset(asset, scan_dfs, df_4h, 
                                min_pearson_r=min_r, min_pvt_r=min_pvt_r, 
                                combined_gate_threshold=cg_thresh, group=group)
            if not signal:
                continue

            xgb_prob = predict_probability(
                self.xgb_model, confidence=signal.confidence, pvt_r=signal.pvt_r,
                best_tf=signal.best_tf, best_period=signal.best_period, pnl_pct=0.0
            )
            xgb_threshold = gt.min_xgb_score if gt else 0.55
            if xgb_prob <= xgb_threshold:
                scan_log["signals"].append({"asset": asset, "status": "xgb_rejected", "xgb_prob": round(xgb_prob, 4)})
                continue

            price = signal.entry_price if not self.live_mode else (fetch_ticker_price(f"{asset}USDT") or signal.entry_price)
            lev = get_leverage(signal.confidence)
            vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
            pos_frac = gt.pos_frac if gt else self.consensus["pos_frac"]
            pos_usd = self.state.initial_capital * pos_frac * lev * vol_scalar
            if pos_usd <= 0 or lev == 0:
                continue

            hard_sl_mult = gt.hard_sl_mult if gt else self.consensus["hard_sl_mult"]
            hard_sl = compute_hard_sl(price, signal.atr, signal.direction, hard_sl_mult)
            trail_buffer = gt.trail_buffer if gt else self.consensus["trail_buffer"]
            std_price = signal.midline * signal.std_dev
            if signal.direction == 1:
                trail_sl = min(price, signal.midline - trail_buffer * std_price)
            else:
                trail_sl = max(price, signal.midline + trail_buffer * std_price)

            quantity, order_id, sl_order_id, entry_commission = 0.0, "", "", 0.0
            if self.live_mode:
                _cancel_all_orders_full(f"{asset}USDT")
                side = "buy" if signal.direction == 1 else "sell"
                order = place_market_order(f"{asset}USDT", side, pos_usd, lev)
                if not order:
                    continue
                order_id = str(order.get("id", ""))
                quantity = float(order.get("filled", 0) or order.get("amount", 0))
                price = float(order.get("average", price) or price)
                if quantity <= 0:
                    continue
                entry_commission = _fetch_order_commission(f"{asset}USDT", order_id)
                sl_order = place_stop_loss(f"{asset}USDT", "sell" if signal.direction == 1 else "buy", quantity, hard_sl)
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
            )
            self.state.positions[asset] = pos
            save_state(self.state)
            
            dir_str = "LONG" if signal.direction == 1 else "SHORT"
            sent_str = f"{ls_ratio:.2f}" if ls_ratio is not None else "N/A"
            ovr_tag = " (SENTIMENT OVERRIDE)" if sentiment_override else ""
            
            entries.append({
                "asset": asset, "group": group, "dir": dir_str, "r": round(signal.confidence, 4),
                "ls_ratio": sent_str, "htf_bias": htf_bias, "price": round(price, 4),
                "size": round(pos_usd, 0), "sl": round(hard_sl, 4), "tf": signal.best_tf, "period": signal.best_period,
                "xgb_prob": round(xgb_prob, 4), "lev": lev
            })
            
            tg_send(
                f"<b>🚀 NEW {dir_str} — {asset}</b> [Grp {group}]{ovr_tag}\n"
                f"XGB: {xgb_prob:.4f} | R: {signal.confidence:.4f}\n"
                f"L/S Ratio: {sent_str} | HTF: {htf_bias}\n"
                f"Size: ${pos_usd:,.0f} | Lev: {lev}x | SL: ${hard_sl:.4f}\n"
                f"TF: {signal.best_tf} | Period: {signal.best_period}"
            )
        try:
            SCAN_LOG_FILE.write_text(json.dumps(scan_log, indent=2))
        except Exception:
            pass
        return entries

    def run_cycle(self):
        now = datetime.now(timezone.utc)
        all_data = get_live_data()
        with self.lock:
            self.state.last_scan_ts = now.isoformat()
            self.state.total_scans += 1
            logger.info("SCAN CYCLE #%d — %s | Open: %d/%d",
                        self.state.total_scans, now.strftime("%H:%M UTC"),
                        len(self.state.positions), MAX_CONCURRENT)
            if self.state.circuit_breaker or self._check_circuit_breaker():
                return

            before = set(self.state.positions.keys())
            self._process_sub_bars()
            closed = before - set(self.state.positions.keys())
            entries = self._scan_and_enter(all_data, skip_assets=closed)
            save_state(self.state)
            
            equity = self.state.initial_capital + self.state.realized_pnl
            n_open = len(self.state.positions)
            scan_num = self.state.total_scans
            rpnl = self.state.realized_pnl
            grp_counts = {}
            for pos in self.state.positions.values():
                grp_counts[pos.group] = grp_counts.get(pos.group, 0) + 1
            grp_str = " | ".join(f"{g}:{c}" for g, c in sorted(grp_counts.items())) or "none"

        logger.info("Cycle complete — Equity: $%.2f | Open: %d/%d [%s] | PnL: $%.2f",
                     equity, n_open, MAX_CONCURRENT, grp_str, rpnl)
        
        lines = [
            f"<b>🔍 Scan #{scan_num} Complete</b>",
            f"Time: {now.strftime('%H:%M UTC')}",
            f"Equity: ${equity:,.2f} | Open: {n_open}/{MAX_CONCURRENT}",
            f"Groups: {grp_str}",
            f"Realized PnL: ${rpnl:+,.2f}",
        ]
        
        if entries:
            lines.append("\n<b>🟢 NEW ENTRIES</b>")
            for e in entries:
                ovr_note = " (SENTIMENT)" if "OVERRIDE" in str(e.get('ls_ratio','')) else ""
                lines.append(
                    f"• <b>{e['asset']}</b> {e['dir']} {ovr_note}\n"
                    f"  R:{e['r']} | XGB:{e['xgb_prob']}\n"
                    f"  L/S:{e['ls_ratio']} | HTF:{e['htf_bias']}\n"
                    f"  Price:${e['price']} | Size:${e['size']}"
                )
        
        tg_send("\n".join(lines))

    def send_status(self):
        with self.lock:
            equity = self.state.initial_capital + self.state.realized_pnl
            dd = (equity - self.state.peak_equity) / self.state.peak_equity * 100 if self.state.peak_equity else 0.0
            mode = "LIVE" if self.live_mode else "DRY-RUN"
            cb = "TRIPPED" if self.state.circuit_breaker else "OK"
            rpnl = self.state.realized_pnl
            peak = self.state.peak_equity
            n_open = len(self.state.positions)
            scans = self.state.total_scans
            pos_snapshot = [(a, p.group, p.direction, p.entry_price, p.hard_sl, p.confidence)
                           for a, p in self.state.positions.items()]
        now = datetime.now(timezone.utc)
        lines = [
            f"<b>🦎 Varanus Neo-Flow Extended</b>",
            f"Mode: {mode} | CB: {cb}",
            f"Frequency: <b>Hourly Scan</b>",
            f"Sentiment: <b>Enabled</b> (L/S Ratio)",
            f"Equity: ${equity:,.2f} | PnL: ${rpnl:+,.2f}",
            f"Drawdown: {dd:.2f}% | Peak: ${peak:,.2f}",
            f"Open: {n_open}/{MAX_CONCURRENT} | Scans: {scans}",
            f"Assets: {len(ASSETS)} | Groups: A/B/C",
        ]
        if pos_snapshot:
            lines.append("")
            lines.append("<b>Positions:</b>")
            for asset, grp, direction, entry, sl, conf in pos_snapshot:
                dir_str = "LONG" if direction == 1 else "SHORT"
                lines.append(f"  {asset}[{grp}] {dir_str} @{entry:.4f} | SL:{sl:.4f} | R:{conf:.3f}")
        lines.append(f"\n{now.strftime('%Y-%m-%d %H:%M UTC')}")
        tg_send("\n".join(lines))

# ═══════════════════════════════════════════════════════════════════════════════
# Threads: Command Listener & Price Monitor
# ═══════════════════════════════════════════════════════════════════════════════

def start_telegram_listener(bot: LiveExtendedBot):
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
                        elif "/help" in text:
                            tg_send(
                                "<b>Commands:</b>\n"
                                "/status - Check bot status\n"
                                "/help - Show this help"
                            )
            except Exception as e:
                logger.error("Telegram listener error: %s", e)
                time.sleep(5)
    threading.Thread(target=_poll, daemon=True).start()

def start_price_monitor(bot: LiveExtendedBot):
    """Background monitor: every 30s fetch new bars, process sub-bars, ticker safety-net."""
    def _monitor():
        while True:
            try:
                # Always update timestamp to show bot is ALIVE every 30s
                with bot.lock:
                    bot.state.last_scan_ts = datetime.now(timezone.utc).isoformat()
                    save_state(bot.state)

                if bot.state.positions:
                    n_pos = len(bot.state.positions)
                    logger.debug("Monitor heartbeat: Checking %d positions", n_pos)
                    # Fetch prices outside lock (network I/O)
                    prices = {}
                    for asset, pos in list(bot.state.positions.items()):
                        p = fetch_ticker_price(pos.symbol)
                        if p:
                            prices[asset] = p

                    # All state reads/writes inside a single lock
                    with bot.lock:
                        bot._refresh_position_data()
                        bot._process_sub_bars()

                        # Ticker safety-net: check hard SL with live prices
                        for asset, price in prices.items():
                            if asset not in bot.state.positions:
                                continue  # already closed by _process_sub_bars
                            pos = bot.state.positions[asset]
                            if (pos.direction == 1 and price <= pos.hard_sl) or \
                               (pos.direction == -1 and price >= pos.hard_sl):
                                bot._close_trade(pos, pos.hard_sl, "HARD_SL_HIT")
            except Exception as e:
                logger.error("Monitor error: %s", e)
            time.sleep(30)
    threading.Thread(target=_monitor, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════════════════════

LOCK_FILE = BASE_DIR / "live_extended_bot.lock"
_lock_fd = None  # keep reference so GC doesn't close it


def _acquire_lock():
    """Acquire an exclusive file lock. Exits if another instance holds it.
    The OS releases the lock automatically when the process dies (even kill -9)."""
    global _lock_fd
    _lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.critical("Another instance is already running (lock held). Aborting.")
        tg_send("<b>STARTUP BLOCKED</b>\nAnother bot instance holds the lock file.")
        sys.exit(1)
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
    logger.info("Lock acquired — PID %d", os.getpid())


def _check_binance_sync(bot: "LiveExtendedBot"):
    """On startup, check if Binance has positions the bot doesn't know about (orphans)."""
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
            msg += "These may be leftover from duplicate processes.\nBot will NOT open new positions in these assets."
            logger.warning("Orphaned Binance positions: %s", orphans)
            tg_send(msg)

        for asset in bot_positions:
            if asset not in binance_positions:
                logger.warning("Bot has %s position but Binance does not — removing from state", asset)
                del bot.state.positions[asset]
                save_state(bot.state)
    except Exception as e:
        logger.error("Binance sync check failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Varanus Neo-Flow Extended Live Bot")
    parser.add_argument("--live", action="store_true", help="Enable real trading on Binance")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital (USD)")
    parser.add_argument("--once", action="store_true", help="Run single scan cycle then exit")
    args = parser.parse_args()

    _acquire_lock()

    bot = LiveExtendedBot(args.capital, args.live)

    # On live mode, verify no orphaned Binance positions before starting
    if args.live and not args.once:
        _check_binance_sync(bot)

    start_telegram_listener(bot)
    start_price_monitor(bot)

    stop_event = threading.Event()
    def _shutdown(s, f):
        logger.info("Shutting down (signal %s)...", s)
        stop_event.set()  # wakes up Event.wait() immediately
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.once:
        logger.info("Single scan mode (--once)")
        bot.run_cycle()
        return

    now = datetime.now(timezone.utc)
    next_trigger = now.replace(minute=1, second=0, microsecond=0) + timedelta(hours=1)
    wait_secs = int((next_trigger - now).total_seconds())
    logger.info("Waiting for next top-of-hour scan (60s settlement): %s UTC (%d min %d sec)",
                next_trigger.strftime("%H:%M"), wait_secs // 60, wait_secs % 60)
    tg_send(
        f"<b>Extended Bot Started</b> — {now.strftime('%H:%M UTC')}\n"
        f"Mode: {'LIVE' if args.live else 'DRY-RUN'} | Capital: ${args.capital:,.0f}\n"
        f"Scan: <b>Hourly Frequency</b> (:01 UTC)\n"
        f"Sentiment: <b>Enabled</b> (Binance L/S Ratio Override)\n"
        f"Next scan: {next_trigger.strftime('%H:%M UTC')} ({wait_secs // 60}m)"
    )

    last_cycle_hour = -1  # track last executed hour to prevent double execution
    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        next_trigger = now.replace(minute=1, second=0, microsecond=0) + timedelta(hours=1)
        sleep_secs = max(1, int((next_trigger - now).total_seconds()))
        stop_event.wait(timeout=sleep_secs)  # wakes immediately on SIGTERM/SIGINT
        if stop_event.is_set():
            break
        now = datetime.now(timezone.utc)
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
