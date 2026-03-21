"""
NeoFlowHybridEngine.py — Hybrid backtest engine: Neo-Flow physics + XGBoost meta-labeler.

Extended: 33 assets, 4 scan TFs, group-specific XGB thresholds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent))

from neo_flow.adaptive_engine import (
    calc_log_regression,
    compute_trail_sl,
    compute_hard_sl,
    get_leverage,
    HIGH_VOL_ASSETS,
)

from backtest.data_loader import (
    AssetData,
    get_1h_timestamps,
    get_sub_bars,
    find_bar_index,
    ts_to_ns,
    ASSETS,
)

from backtest.engine import BacktestParams, TradeRecord, ActivePosition
from config.groups import get_group, get_thresholds, GroupThresholds
from ml.train_meta_model import FEATURE_COLS, TF_ENCODE, predict_probability

logger = logging.getLogger(__name__)

TF_FROM_IDX = {0: "5m", 1: "30m", 2: "1h", 3: "4h"}


@dataclass
class HybridParams(BacktestParams):
    """BacktestParams + per-group XGBoost thresholds."""
    xgb_threshold: float = 0.55


class NeoFlowHybridEngine:
    """
    Extended Hybrid Engine: Physics + XGBoost meta-labeler with group-aware thresholds.

    Scan flow:
        1. LUT feature lookup (O(1))
        2. Physics gates (R-gate, PVT-gate, HTF filter, combined gate)
        3. XGBoost meta-labeler: predict probability (group-specific threshold)
        4. Enter if XGB_Probability > group_xgb_threshold
    """

    def __init__(
        self,
        all_data: dict[str, dict[str, AssetData]],
        params: HybridParams,
        features: dict[str, np.ndarray],
        xgb_model: xgb.XGBClassifier,
    ):
        self.all_data = all_data
        self.params = params
        self.features = features
        self.xgb_model = xgb_model
        self.trades: list[TradeRecord] = []
        self.positions: dict[str, ActivePosition] = {}
        self.equity_curve: list[float] = []
        self.capital = params.initial_capital
        self.realized_pnl = 0.0
        self._trade_counter = 0
        self._scan_count = 0
        self._prev_1h_ns: int | None = None
        self._physics_pass = 0
        self._xgb_pass = 0
        self._xgb_reject = 0

        self._feat_idx = {}
        for asset, feat in features.items():
            self._feat_idx[asset] = feat["timestamp_ns"]

    def _lookup_feature(self, asset, current_ns):
        ts_arr = self._feat_idx.get(asset)
        if ts_arr is None:
            return None
        idx = np.searchsorted(ts_arr, current_ns, side="right") - 1
        if idx < 0 or idx >= len(ts_arr):
            return None
        return self.features[asset][idx]

    def _get_entry_price(self, asset, tf, signal_ns):
        ad = self.all_data.get(asset, {}).get(tf)
        if ad is None:
            return None
        idx = np.searchsorted(ad.timestamps, signal_ns, side="right") - 1
        if idx < 0:
            return None
        return float(ad.close[idx])

    def _close_position(self, pos, exit_price, exit_ts, exit_reason):
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100.0
        pnl_pct_lev = pnl_pct * pos.leverage
        pnl_usd = pnl_pct_lev / 100.0 * pos.position_usd
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

    def _update_positions(self, current_1h_ns):
        prev_ns = self._prev_1h_ns
        if prev_ns is None:
            return
        to_close = []
        for asset, pos in list(self.positions.items()):
            ad = self.all_data.get(asset, {}).get(pos.best_tf)
            if ad is None:
                continue
            gt = get_thresholds(asset, self.params.group_overrides)
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

    def _scan_and_enter(self, current_1h_ns):
        if len(self.positions) >= self.params.max_concurrent:
            return
        current_ts = pd.Timestamp(current_1h_ns, tz="UTC")

        for asset in ASSETS:
            if asset in self.positions:
                continue
            if len(self.positions) >= self.params.max_concurrent:
                break

            feat = self._lookup_feature(asset, current_1h_ns)
            if feat is None:
                continue

            group = get_group(asset)
            gt = get_thresholds(asset, self.params.group_overrides)

            best_r = float(feat["best_r"])
            if best_r < gt.min_confidence:
                continue

            direction = int(feat["best_direction"])
            pvt_r = float(feat["pvt_r"])
            pvt_dir = int(feat["pvt_direction"])

            if pvt_dir != direction:
                continue
            if best_r >= 0.85 and pvt_r < 0.50:
                continue
            if pvt_r < gt.min_pvt_r:
                continue

            htf = int(feat["htf_bias"])
            if htf == 0 or htf != direction:
                continue

            max_opp = float(feat["max_opposing_r"])
            if max_opp > gt.combined_gate:
                continue

            self._physics_pass += 1

            best_tf = TF_FROM_IDX[int(feat["best_tf_idx"])]
            best_period = int(feat["best_period"])

            xgb_prob = predict_probability(
                self.xgb_model,
                confidence=best_r,
                pvt_r=pvt_r,
                best_tf=best_tf,
                best_period=best_period,
            )

            if xgb_prob <= gt.min_xgb_score:
                self._xgb_reject += 1
                continue

            self._xgb_pass += 1

            entry_price = self._get_entry_price(asset, best_tf, current_1h_ns)
            if entry_price is None:
                continue

            confidence = best_r
            lev = get_leverage(confidence)
            vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
            pos_usd = self.capital * gt.pos_frac * lev * vol_scalar

            if pos_usd <= 0 or lev == 0:
                continue

            atr_val = float(feat["atr_best"])
            if atr_val <= 0:
                continue

            hard_sl = compute_hard_sl(entry_price, atr_val, direction, gt.hard_sl_mult)

            midline = float(feat["best_midline"])
            std_dev = float(feat["best_std"])
            std_dev_price = midline * std_dev
            if direction == 1:
                initial_trail = midline - gt.trail_buffer * std_dev_price
            else:
                initial_trail = midline + gt.trail_buffer * std_dev_price

            self._trade_counter += 1

            tr = TradeRecord(
                trade_id=self._trade_counter,
                asset=asset,
                direction=direction,
                entry_ts=current_ts,
                entry_price=entry_price,
                best_tf=best_tf,
                best_period=best_period,
                confidence=confidence,
                pvt_r=pvt_r,
                leverage=lev,
                position_usd=pos_usd,
                hard_sl=hard_sl,
                initial_trail_sl=initial_trail,
                group=group,
                peak_r=confidence,
            )
            self.trades.append(tr)

            pos = ActivePosition(
                trade_id=self._trade_counter,
                asset=asset,
                direction=direction,
                entry_price=entry_price,
                entry_ts=current_ts,
                hard_sl=hard_sl,
                trail_sl=initial_trail,
                midline=midline,
                std_dev=std_dev,
                best_tf=best_tf,
                best_period=best_period,
                confidence=confidence,
                pvt_r=pvt_r,
                leverage=lev,
                position_usd=pos_usd,
                group=group,
            )
            self.positions[asset] = pos

        self._scan_count += 1

    def _compute_equity(self):
        unrealized = 0.0
        for pos in self.positions.values():
            pnl_pct = (pos.midline - pos.entry_price) / pos.entry_price * pos.direction
            unrealized += pnl_pct * pos.leverage * pos.position_usd
        return self.capital + self.realized_pnl + unrealized

    def run(self, start_ts, end_ts):
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.realized_pnl = 0.0
        self._trade_counter = 0
        self._scan_count = 0
        self._prev_1h_ns = None
        self._physics_pass = 0
        self._xgb_pass = 0
        self._xgb_reject = 0

        bar_timestamps = get_1h_timestamps(self.all_data, start_ts, end_ts)
        if len(bar_timestamps) == 0:
            return []

        t0 = time.perf_counter()
        scan_interval = self.params.scan_interval_hours

        for i, current_ns in enumerate(bar_timestamps):
            self._update_positions(current_ns)
            if i % scan_interval == 0:
                self._scan_and_enter(current_ns)
            self.equity_curve.append(self._compute_equity())
            self._prev_1h_ns = current_ns

        for asset, pos in list(self.positions.items()):
            ad = self.all_data.get(asset, {}).get(pos.best_tf)
            if ad is not None:
                last_idx = find_bar_index(ad.timestamps, bar_timestamps[-1])
                exit_price = float(ad.close[last_idx])
                exit_ts = pd.Timestamp(bar_timestamps[-1], tz="UTC")
                self._close_position(pos, exit_price, exit_ts, "END_OF_PERIOD")

        elapsed = time.perf_counter() - t0
        completed = [t for t in self.trades if t.exit_ts is not None]
        total_checked = self._physics_pass + self._xgb_reject
        filter_rate = self._xgb_reject / total_checked * 100 if total_checked > 0 else 0

        logger.info(
            "Hybrid backtest: %.1fs, %d scans, %d trades | "
            "Physics pass: %d, XGB pass: %d, XGB reject: %d (filter: %.1f%%)",
            elapsed, self._scan_count, len(completed),
            self._physics_pass, self._xgb_pass, self._xgb_reject, filter_rate,
        )
        return self.trades

    def get_diagnostics(self):
        total = self._physics_pass
        return {
            "total_scans": self._scan_count,
            "physics_pass": self._physics_pass,
            "xgb_pass": self._xgb_pass,
            "xgb_reject": self._xgb_reject,
            "xgb_filter_rate_pct": round(self._xgb_reject / total * 100, 2) if total > 0 else 0.0,
            "trades_executed": len([t for t in self.trades if t.exit_ts is not None]),
        }
