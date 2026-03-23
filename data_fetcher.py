#!/usr/bin/env python3
"""
data_fetcher.py — Historical OHLCV Fetcher for Varanus Neo-Flow Extended

Fetches 5m, 30m, 1h, 4h data from Binance public REST API for 33 assets
across training + blind test periods. Resumable.

Usage:
    python data_fetcher.py                     # Fetch all 33 × 4 TFs
    python data_fetcher.py --asset BTCUSDT     # Single asset
    python data_fetcher.py --tf 1h             # Single TF
    python data_fetcher.py --dry-run           # Show what would be fetched
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# 33-asset universe (3 groups)
ASSETS = [
    # Group A [Majors]
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT",
    # Group B [Tech & AI]
    "FETUSDT", "RENDERUSDT", "NEARUSDT", "ARUSDT", "GRTUSDT",
    "INJUSDT", "THETAUSDT", "FILUSDT", "ATOMUSDT", "ICPUSDT", "STXUSDT",
    # Group C [Momentum & Meme]
    "PEPEUSDT", "TIAUSDT", "WIFUSDT", "BONKUSDT", "SUIUSDT",
    "SEIUSDT", "APTUSDT", "SHIBUSDT", "DOGEUSDT", "FLOKIUSDT", "OPUSDT",
]

TIMEFRAMES = ["5m", "30m", "1h", "4h"]

TF_MS = {
    "5m":   5 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
}

GLOBAL_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
GLOBAL_END   = datetime(2026, 3, 21, 13, 0, 0, tzinfo=timezone.utc)

KLINES_URL  = "https://fapi.binance.com/fapi/v1/klines"    # Futures endpoint
SPOT_URL    = "https://api.binance.com/api/v3/klines"       # Spot fallback
MAX_CANDLES = 1000
RATE_LIMIT_S = 0.15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_fetcher")


def _ts_ms(dt):
    return int(dt.timestamp() * 1000)


def _parquet_path(symbol, tf):
    return DATA_DIR / f"{symbol}_{tf}.parquet"


def _load_existing(path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            return None
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _fetch_klines(symbol, interval, start_ms, end_ms):
    all_klines = []
    cursor = start_ms
    url = KLINES_URL

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": MAX_CANDLES,
        }
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    time.sleep(int(resp.headers.get("Retry-After", 30)))
                    continue
                if resp.status_code == 418:
                    time.sleep(120)
                    continue
                # If futures fails (e.g. symbol not found), try spot
                if resp.status_code == 400 and url == KLINES_URL:
                    url = SPOT_URL
                    continue
                resp.raise_for_status()
                batch = resp.json()
                break
            except requests.RequestException as exc:
                wait = 2 ** attempt
                logger.warning("Request error (attempt %d/5): %s", attempt + 1, exc)
                time.sleep(wait)
                if attempt == 4:
                    raise
        else:
            raise RuntimeError(f"Failed after 5 retries: {symbol} {interval}")

        if not batch:
            break
        all_klines.extend(batch)
        last_open_ms = int(batch[-1][0])
        next_cursor = last_open_ms + TF_MS[interval]
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(RATE_LIMIT_S)

    return all_klines


def _klines_to_df(raw):
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index()
    return df[~df.index.duplicated(keep="first")]


def fetch_symbol_tf(symbol, tf, dry_run=False):
    path = _parquet_path(symbol, tf)
    existing = _load_existing(path)
    global_start_ms = _ts_ms(GLOBAL_START)
    global_end_ms = _ts_ms(GLOBAL_END)

    if existing is not None and not existing.empty:
        first_ts = existing.index[0]
        last_ts = existing.index[-1]
        first_ms = int(first_ts.timestamp() * 1000)
        tail_resume_ms = int(last_ts.timestamp() * 1000) + TF_MS[tf]
        need_head = first_ms > global_start_ms
        need_tail = tail_resume_ms < global_end_ms
        if not need_head and not need_tail:
            logger.info("  %-14s %-3s  SKIP (%d bars)", symbol, tf, len(existing))
            return existing
        logger.info("  %-14s %-3s  RESUME (%d bars cached)", symbol, tf, len(existing))
    else:
        need_head = True
        need_tail = True
        first_ms = global_end_ms
        tail_resume_ms = global_start_ms
        logger.info("  %-14s %-3s  FULL FETCH", symbol, tf)

    if dry_run:
        return existing if existing is not None else pd.DataFrame()

    head_df = pd.DataFrame()
    if need_head and existing is not None and not existing.empty:
        head_end_ms = first_ms - TF_MS[tf]
        raw_head = _fetch_klines(symbol, tf, global_start_ms, head_end_ms)
        head_df = _klines_to_df(raw_head)

    tail_df = pd.DataFrame()
    if need_tail:
        fetch_start = tail_resume_ms if (existing is not None and not existing.empty) else global_start_ms
        raw_tail = _fetch_klines(symbol, tf, fetch_start, global_end_ms)
        tail_df = _klines_to_df(raw_tail)

    parts = [df for df in [head_df, existing, tail_df] if df is not None and not df.empty]
    if not parts:
        return existing if existing is not None else pd.DataFrame()

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined = combined[
        (combined.index >= pd.Timestamp(GLOBAL_START))
        & (combined.index <= pd.Timestamp(GLOBAL_END))
    ]

    combined.to_parquet(path, engine="pyarrow")
    logger.info("  %-14s %-3s  SAVED %d bars", symbol, tf, len(combined))
    return combined


def fetch_all(assets=None, timeframes=None, dry_run=False):
    assets = assets or ASSETS
    timeframes = timeframes or TIMEFRAMES
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Varanus Neo-Flow Extended — Data Fetcher (33 assets)")
    logger.info("  Assets:     %d", len(assets))
    logger.info("  Timeframes: %s", ", ".join(timeframes))
    logger.info("  Range:      %s → %s", GLOBAL_START.date(), GLOBAL_END.date())
    logger.info("=" * 70)

    results = {}
    for symbol in assets:
        results[symbol] = {}
        logger.info("[%s]", symbol)
        for tf in timeframes:
            results[symbol][tf] = fetch_symbol_tf(symbol, tf, dry_run=dry_run)
        logger.info("")

    logger.info("=" * 70)
    logger.info("SUMMARY")
    for symbol in assets:
        parts = []
        for tf in timeframes:
            df = results[symbol].get(tf)
            n = len(df) if df is not None and not df.empty else 0
            parts.append(f"{tf}={n:>7,}")
        logger.info("  %-14s  %s", symbol, "  ".join(parts))
    logger.info("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varanus Neo-Flow Extended — Data Fetcher")
    parser.add_argument("--asset", type=str, default=None)
    parser.add_argument("--tf", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assets_arg = [args.asset.upper()] if args.asset else None
    tf_arg = [args.tf] if args.tf else None
    fetch_all(assets=assets_arg, timeframes=tf_arg, dry_run=args.dry_run)
