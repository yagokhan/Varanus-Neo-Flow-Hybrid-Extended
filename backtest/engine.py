"""
backtest/engine.py — Core backtest loop with group-aware trade management.

Extended: 33-asset universe, 4 scan TFs, group-specific thresholds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from neo_flow.adaptive_engine import (
    scan_asset,
    calc_log_regression,
    compute_trail_sl,
    compute_hard_sl,
    compute_atr,
    get_leverage,
    HIGH_VOL_ASSETS,
    SCAN_TIMEFRAMES,
)

from backtest.data_loader import (
    AssetData,
    build_scan_dataframes,
    build_htf_dataframe,
    get_1h_timestamps,
    get_sub_bars,
    find_bar_index,
    ts_to_ns,
    BARS_7D,
    ASSETS,
)

from config.groups import get_group, get_thresholds, GroupThresholds

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Transaction Costs (Binance Futures realistic)
# ═══════════════════════════════════════════════════════════════════════════════

TAKER_FEE = 0.0004         # 0.04% per side (Binance Futures VIP0 taker)
SLIPPAGE_DEFAULT = 0.0002  # 0.02% per side (mid/large-cap)
SLIPPAGE_VOLATILE = 0.0005 # 0.05% per side (meme/volatile coins)

# Volatile assets — wider spreads, thinner order books
VOLATILE_ASSETS = {"PEPE", "BONK", "WIF", "FLOKI", "SHIB", "DOGE", "TIA", "SEI"}


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest Parameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestParams:
    """Global backtest parameters + group-specific overrides."""
    # Global defaults (can be overridden per group)
    min_pearson_r: float = 0.80
    min_pvt_r: float = 0.75
    combined_gate: float = 0.80
    hard_sl_mult: float = 2.5
    trail_buffer: float = 0.5
    exhaust_r: float = 0.475
    pos_frac: float = 0.05
    initial_capital: float = 10_000.0
    max_concurrent: int = 8       # 33 assets → allow more positions
    scan_interval_hours: int = 1
    # XGB threshold (per-group override via GroupThresholds)
    xgb_threshold: float = 0.55
    # Group-specific overrides (None = use global defaults)
    group_overrides: dict | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Complete record of one backtest trade."""
    trade_id: int
    asset: str
    direction: int
    entry_ts: pd.Timestamp
    entry_price: float
    best_tf: str
    best_period: int
    confidence: float
    pvt_r: float
    leverage: int
    position_usd: float
    hard_sl: float
    initial_trail_sl: float
    group: str = "B"
    peak_r: float = 0.0
    # Filled on exit
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    bars_held: int = 0
    final_trail_sl: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Active Position
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActivePosition:
    """Mutable state of an open position."""
    trade_id: int
    asset: str
    direction: int
    entry_price: float
    entry_ts: pd.Timestamp
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
    group: str = "B"
    bars_held: int = 0
    peak_r: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest Engine (Group-Aware)
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Event-driven backtest engine for the 33-asset Extended universe."""

    def __init__(
        self,
        all_data: dict[str, dict[str, AssetData]],
        params: BacktestParams | None = None,
    ):
        self.all_data = all_data
        self.params = params or BacktestParams()
        self.trades: list[TradeRecord] = []
        self.positions: dict[str, ActivePosition] = {}
        self.equity_curve: list[float] = []
        self.capital = self.params.initial_capital
        self.realized_pnl = 0.0
        self._trade_counter = 0
        self._scan_count = 0
        self._prev_1h_ns: int | None = None

    def _get_group_thresholds(self, asset: str) -> GroupThresholds:
        """Get group-specific thresholds, with param overrides applied."""
        group = get_group(asset)
        thresholds = get_thresholds(asset, self.params.group_overrides)
        return thresholds

    def _close_position(self, pos, exit_price, exit_ts, exit_reason):
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100.0
        pnl_pct_lev = pnl_pct * pos.leverage
        pnl_usd = pnl_pct_lev / 100.0 * pos.position_usd
        # Transaction costs: fee + slippage, applied per side (entry + exit)
        slip = SLIPPAGE_VOLATILE if pos.asset in VOLATILE_ASSETS else SLIPPAGE_DEFAULT
        cost_usd = pos.position_usd * 2 * (TAKER_FEE + slip)
        pnl_usd -= cost_usd
        self.realized_pnl += pnl_usd
        for tr in reversed(self.trades):
            if tr.trade_id == pos.trade_id:
                tr.exit_ts = exit_ts
                tr.exit_price = exit_price
                tr.exit_reason = exit_reason
                tr.bars_held = pos.bars_held
                tr.peak_r = pos.peak_r
                tr.final_trail_sl = pos.trail_sl
                tr.pnl_pct = pnl_pct_lev
                tr.pnl_usd = pnl_usd
                break
        del self.positions[pos.asset]

    def _update_positions(self, current_1h_ns: int):
        """Update active positions with sub-bar data."""
        prev_ns = self._prev_1h_ns
        if prev_ns is None:
            return
        to_close = []
        for asset, pos in list(self.positions.items()):
            ad = self.all_data.get(asset, {}).get(pos.best_tf)
            if ad is None:
                continue
            gt = self._get_group_thresholds(asset)
            sub_indices = get_sub_bars(ad, prev_ns, current_1h_ns)
            for idx in sub_indices:
                pos.bars_held += 1
                close_arr = ad.close[:idx + 1]
                if len(close_arr) >= pos.best_period:
                    std_dev, pearson_r, slope, intercept = calc_log_regression(
                        close_arr, pos.best_period
                    )
                    midline = np.exp(intercept)
                    pos.midline = midline
                    pos.std_dev = std_dev
                    pos.peak_r = max(pos.peak_r, abs(pearson_r))
                    pos.trail_sl = compute_trail_sl(
                        pos.direction, midline, std_dev,
                        pos.trail_sl, gt.trail_buffer,
                    )
                    current_r = pearson_r
                else:
                    current_r = 0.0

                bar_high = float(ad.high[idx])
                bar_low = float(ad.low[idx])
                bar_close = float(ad.close[idx])
                bar_ts = pd.Timestamp(ad.timestamps[idx], tz="UTC")

                if pos.direction == 1 and bar_low <= pos.hard_sl:
                    to_close.append((pos, pos.hard_sl, bar_ts, "HARD_SL_HIT"))
                    break
                if pos.direction == -1 and bar_high >= pos.hard_sl:
                    to_close.append((pos, pos.hard_sl, bar_ts, "HARD_SL_HIT"))
                    break
                if pos.direction == 1 and bar_low <= pos.trail_sl:
                    to_close.append((pos, pos.trail_sl, bar_ts, "ADAPTIVE_TRAIL_HIT"))
                    break
                if pos.direction == -1 and bar_high >= pos.trail_sl:
                    to_close.append((pos, pos.trail_sl, bar_ts, "ADAPTIVE_TRAIL_HIT"))
                    break
                if abs(current_r) < gt.exhaust_r:
                    to_close.append((pos, bar_close, bar_ts, "TREND_EXHAUSTION"))
                    break
                if pos.bars_held >= 200:
                    to_close.append((pos, bar_close, bar_ts, "TIME_BARRIER"))
                    break

        for pos, exit_price, exit_ts, reason in to_close:
            if pos.asset in self.positions:
                self._close_position(pos, exit_price, exit_ts, reason)

    def _scan_and_enter(self, current_1h_ns: int):
        """Scan all 33 assets and open positions for qualifying signals."""
        if len(self.positions) >= self.params.max_concurrent:
            return
        current_ts = pd.Timestamp(current_1h_ns, tz="UTC")

        for asset in ASSETS:
            if asset in self.positions:
                continue
            if len(self.positions) >= self.params.max_concurrent:
                break

            asset_data = self.all_data.get(asset)
            if asset_data is None:
                continue

            gt = self._get_group_thresholds(asset)
            group = get_group(asset)

            # Build scan DataFrames for ALL 4 TFs
            scan_dfs = build_scan_dataframes(asset_data, current_1h_ns - 1)
            df_4h = build_htf_dataframe(asset_data, current_1h_ns - 4 * 3600 * 10**9)

            if not scan_dfs or df_4h is None:
                continue

            signal = scan_asset(
                asset, scan_dfs, df_4h,
                min_pearson_r=gt.min_confidence,
                min_pvt_r=gt.min_pvt_r,
                combined_gate_threshold=gt.combined_gate,
                group=group,
            )
            if signal is None:
                continue

            # Entry at next bar open on signal's TF
            ad = asset_data.get(signal.best_tf)
            if ad is None:
                continue
            entry_idx = np.searchsorted(ad.timestamps, current_1h_ns, side="right")
            if entry_idx >= len(ad.timestamps):
                continue
            entry_price = float(ad.open_[entry_idx])

            # Position sizing
            lev = get_leverage(signal.confidence)
            vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
            pos_usd = self.capital * gt.pos_frac * lev * vol_scalar

            if pos_usd <= 0 or lev == 0:
                continue

            # Hard SL
            ad_tf = asset_data.get(signal.best_tf)
            idx = find_bar_index(ad_tf.timestamps, current_1h_ns - 1)
            start_idx = max(0, idx - 20)
            atr_df = pd.DataFrame({
                "high": ad_tf.high[start_idx:idx + 1],
                "low": ad_tf.low[start_idx:idx + 1],
                "close": ad_tf.close[start_idx:idx + 1],
            })
            atr_val = float(compute_atr(atr_df, 14).iloc[-1])
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            hard_sl = compute_hard_sl(entry_price, atr_val, signal.direction, gt.hard_sl_mult)

            # Initial trail
            std_dev_price = signal.midline * signal.std_dev
            if signal.direction == 1:
                initial_trail = signal.midline - gt.trail_buffer * std_dev_price
            else:
                initial_trail = signal.midline + gt.trail_buffer * std_dev_price

            self._trade_counter += 1
            entry_ts = current_ts

            tr = TradeRecord(
                trade_id=self._trade_counter,
                asset=asset,
                direction=signal.direction,
                entry_ts=entry_ts,
                entry_price=entry_price,
                best_tf=signal.best_tf,
                best_period=signal.best_period,
                confidence=signal.confidence,
                pvt_r=signal.pvt_r,
                leverage=lev,
                position_usd=pos_usd,
                hard_sl=hard_sl,
                initial_trail_sl=initial_trail,
                group=group,
                peak_r=signal.confidence,
            )
            self.trades.append(tr)

            pos = ActivePosition(
                trade_id=self._trade_counter,
                asset=asset,
                direction=signal.direction,
                entry_price=entry_price,
                entry_ts=entry_ts,
                hard_sl=hard_sl,
                trail_sl=initial_trail,
                midline=signal.midline,
                std_dev=signal.std_dev,
                best_tf=signal.best_tf,
                best_period=signal.best_period,
                confidence=signal.confidence,
                pvt_r=signal.pvt_r,
                leverage=lev,
                position_usd=pos_usd,
                group=group,
            )
            self.positions[asset] = pos

        self._scan_count += 1

    def _compute_equity(self) -> float:
        unrealized = 0.0
        for pos in self.positions.values():
            pnl_pct = (pos.midline - pos.entry_price) / pos.entry_price * pos.direction
            unrealized += pnl_pct * pos.leverage * pos.position_usd
        return self.capital + self.realized_pnl + unrealized

    def run(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> list[TradeRecord]:
        """Run backtest over [start_ts, end_ts]."""
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.realized_pnl = 0.0
        self._trade_counter = 0
        self._scan_count = 0
        self._prev_1h_ns = None

        bar_timestamps = get_1h_timestamps(self.all_data, start_ts, end_ts)
        if len(bar_timestamps) == 0:
            return []

        total_bars = len(bar_timestamps)
        t0 = time.perf_counter()
        scan_interval = self.params.scan_interval_hours

        logger.info(
            "Backtest: %s → %s (%d 1h bars, %d assets, scan every %dh)",
            start_ts.date(), end_ts.date(), total_bars, len(ASSETS), scan_interval,
        )

        for i, current_ns in enumerate(bar_timestamps):
            self._update_positions(current_ns)
            if i % scan_interval == 0:
                self._scan_and_enter(current_ns)
            self.equity_curve.append(self._compute_equity())
            self._prev_1h_ns = current_ns
            if (i + 1) % 5000 == 0:
                elapsed = time.perf_counter() - t0
                pct = (i + 1) / total_bars * 100
                trades_so_far = len([t for t in self.trades if t.exit_ts is not None])
                logger.info(
                    "  %5.1f%% (%d/%d bars, %.1fs) — %d trades, %d open",
                    pct, i + 1, total_bars, elapsed, trades_so_far, len(self.positions),
                )

        for asset, pos in list(self.positions.items()):
            ad = self.all_data.get(asset, {}).get(pos.best_tf)
            if ad is not None:
                last_idx = find_bar_index(ad.timestamps, bar_timestamps[-1])
                exit_price = float(ad.close[last_idx])
                exit_ts = pd.Timestamp(bar_timestamps[-1], tz="UTC")
                self._close_position(pos, exit_price, exit_ts, "END_OF_PERIOD")

        elapsed = time.perf_counter() - t0
        completed = [t for t in self.trades if t.exit_ts is not None]
        logger.info(
            "Backtest complete: %.1fs, %d scans, %d trades",
            elapsed, self._scan_count, len(completed),
        )
        return self.trades
