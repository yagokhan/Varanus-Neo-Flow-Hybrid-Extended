#!/usr/bin/env python3
"""
blind_test_v1_extended.py — Full blind test with V1 P-Matrix method.

Key features (matches live_extended_v1.py):
  1. Entry gate:   XGB score >= entry_p percentile (TF Matrix authority)
  2. Exit gate:    XGB score < exit_p percentile (Mathematical Exhaustion)
  3. Risk gate:    Global 1.5% Hard SL (overrides P-Floor)
  4. Variable TF:  Leverage and thresholds derived from TF_MATRIX
  5. Recovery:     Escalating Cooldowns [15, 45, 180] min on losses

Date Range: Nov 1, 2025 → Last Scan Today
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import (
    AssetData,
    load_all_assets,
    build_scan_dataframes,
    build_htf_dataframe,
    find_bar_index,
    ts_to_ns,
    ASSETS,
    BLIND_START,
)
from backtest.engine import (
    BacktestEngine,
    BacktestParams,
)
from backtest.metrics import compute_metrics, print_metrics
from config.groups import get_group
from neo_flow.adaptive_engine import (
    calc_log_regression,
    find_best_regression,
    trim_to_7d,
    compute_pvt_regression,
    check_pvt_alignment,
    HIGH_VOL_ASSETS,
)
from ml.train_meta_model import predict_probability

# Import components from live_extended_v1
import live_extended_v1
from live_extended_v1 import (
    TF_MATRIX,
    _get_matrix_params,
    _load_group_thresholds,
    _load_consensus_params,
    PercentileCalculator,
    GroupThresholds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("blind_test_v1")

BASE_DIR = Path(__file__).parent
HARD_SL_PCT = 0.015  # 1.5%
STRESS_COST = 0.5    # 0.5% (for cooldown logic)
COOLDOWN_STEPS = [15, 45, 180] # minutes

# Re-define necessary dataclasses for V1 logic
from dataclasses import dataclass

@dataclass
class ActivePositionV1:
    trade_id: int
    asset: str
    direction: int
    entry_price: float
    entry_ts: pd.Timestamp
    entry_idx: int
    pos_usd: float
    best_tf: str
    best_period: int
    confidence: float
    pvt_r: float
    xgb_prob: float
    midline: float
    std_dev: float
    leverage: int
    group: str
    peak_r: float = 0.0

@dataclass
class TradeRecordV1:
    trade_id: int
    asset: str
    group: str
    direction: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    best_tf: str
    best_period: int
    confidence: float
    pvt_r: float
    xgb_prob: float
    leverage: int
    position_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    bars_held: int = 0
    peak_r: float = 0.0

class CooldownState:
    def __init__(self):
        self.consecutive_losses = 0
        self.cooldown_until_ns = 0

class BlindTestV1Engine(BacktestEngine):
    """
    Backtest engine implementing V1 P-Matrix logic.
    """
    def __init__(self, all_data, xgb_model, params: BacktestParams):
        super().__init__(all_data, params)
        if xgb_model:
            self.xgb_model = xgb_model
        
        self.percentile_calc = PercentileCalculator(self.xgb_model)
        self.group_thresholds = _load_group_thresholds()
        self.consensus = _load_consensus_params()
        self.cooldowns: dict[str, CooldownState] = {a: CooldownState() for a in ASSETS}
        
        self.positions_v1: dict[str, ActivePositionV1] = {}
        self.trades_v1: list[TradeRecordV1] = []

    def run(self, start_ts, end_ts):
        self.trades_v1 = []
        self.positions_v1 = {}
        self.equity_curve = []
        self.realized_pnl = 0.0
        self.capital = self.params.initial_capital
        self._trade_counter = 0
        
        bar_timestamps = self._get_5m_timestamps(start_ts, end_ts)
        if len(bar_timestamps) == 0:
            logger.error("No 5m timestamps found in range %s → %s", start_ts, end_ts)
            return []

        total_bars = len(bar_timestamps)
        t0 = time.perf_counter()

        logger.info(
            "V1 Blind Test: %s → %s (%d 5m bars, %d assets)",
            start_ts.date(), end_ts.date(), total_bars, len(ASSETS),
        )

        for i, current_ns in enumerate(bar_timestamps):
            current_ns = int(current_ns)
            self._update_positions_v1(current_ns)
            is_hourly = (current_ns // 10**9) % 3600 == 0
            self._scan_and_enter_v1(current_ns, full_scan=is_hourly)
            self.equity_curve.append(self.capital + self.realized_pnl)

            if (i + 1) % 5000 == 0:
                elapsed = time.perf_counter() - t0
                pct = (i + 1) / total_bars * 100
                trades_so_far = len([t for t in self.trades_v1 if t.exit_ts is not None])
                logger.info(
                    "  %5.1f%% (%d/%d bars, %.1fs) — %d trades, %d open",
                    pct, i + 1, total_bars, elapsed, trades_so_far, len(self.positions_v1),
                )

        # Close remaining
        for asset, pos in list(self.positions_v1.items()):
            ad = self.all_data.get(asset, {}).get(pos.best_tf)
            if ad is not None:
                idx = find_bar_index(ad.timestamps, bar_timestamps[-1])
                self._close_position_v1(pos, float(ad.close[idx]), bar_timestamps[-1], "END_OF_PERIOD")

        return self.trades_v1

    def _get_5m_timestamps(self, start_ts, end_ts):
        for asset_data in self.all_data.values():
            ad = asset_data.get("5m")
            if ad is not None:
                start_ns = ts_to_ns(start_ts)
                end_ns = ts_to_ns(end_ts)
                mask = (ad.timestamps >= start_ns) & (ad.timestamps <= end_ns)
                return ad.timestamps[mask]
        return np.array([], dtype="int64")

    def _update_positions_v1(self, current_ns):
        to_close = []
        for asset, pos in self.positions_v1.items():
            ad = self.all_data[asset][pos.best_tf]
            idx = find_bar_index(ad.timestamps, current_ns)
            price = float(ad.close[idx])
            
            scan_dfs = build_scan_dataframes(self.all_data[asset], current_ns)
            df_tf = scan_dfs.get(pos.best_tf)
            if df_tf is not None and len(df_tf) >= pos.best_period:
                std, r, slope, intercept = calc_log_regression(df_tf["close"].values, pos.best_period)
                abs_r = abs(r)
                if abs_r > pos.peak_r:
                    pos.peak_r = abs_r

                # 1. Hard SL (1.5%)
                hit_sl, _ = self._check_hard_sl(pos, price)
                if hit_sl:
                    to_close.append((pos, price, current_ns, "HARD_SL_HIT"))
                    continue

                # 2. P-Floor
                current_xgb = predict_probability(
                    self.xgb_model,
                    confidence=abs_r,
                    pvt_r=pos.pvt_r,
                    best_tf=pos.best_tf,
                    best_period=pos.best_period,
                )
                
                exit_threshold = self.percentile_calc.get_exit_threshold(get_group(asset), pos.best_tf)
                if current_xgb < exit_threshold:
                    to_close.append((pos, price, current_ns, "TREND_EXHAUSTION"))
                    continue

            # 3. Time Barrier
            if idx - pos.entry_idx >= 200:
                to_close.append((pos, price, current_ns, "TIME_BARRIER"))

        for pos, price, ts_ns, reason in to_close:
            self._close_position_v1(pos, price, ts_ns, reason)

    def _check_hard_sl(self, pos, current_price):
        if pos.direction == 1:
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price
        if pnl_pct <= -HARD_SL_PCT:
            return True, f"HARD_SL: {pnl_pct*100:.2f}%"
        return False, ""

    def _scan_and_enter_v1(self, current_ns, full_scan=True):
        if len(self.positions_v1) >= self.params.max_concurrent:
            return
        scan_tfs = ["5m", "30m", "1h", "4h"] if full_scan else ["5m"]
        
        for asset in ASSETS:
            if asset in self.positions_v1: continue
            if len(self.positions_v1) >= self.params.max_concurrent: break
            
            cd = self.cooldowns[asset]
            if current_ns < cd.cooldown_until_ns: continue

            asset_data = self.all_data.get(asset)
            gt = self.group_thresholds.get(get_group(asset))
            if not asset_data or not gt: continue

            scan_dfs = build_scan_dataframes(asset_data, current_ns, scan_tfs=scan_tfs)
            df_4h = build_htf_dataframe(asset_data, current_ns)
            if not scan_dfs or df_4h is None: continue

            best, _ = find_best_regression(scan_dfs)
            if not best: continue
            
            abs_r = abs(best.pearson_r)
            if abs_r < gt.min_confidence: continue
            
            direction = 1 if best.slope < 0 else -1
            best_df_7d = trim_to_7d(scan_dfs[best.timeframe], best.timeframe)
            pvt = compute_pvt_regression(best_df_7d, best.period)
            pvt_passes, _ = check_pvt_alignment(direction, abs_r, pvt, gt.min_pvt_r)
            if not pvt_passes: continue
            
            pvt_r = abs(pvt.pearson_r)
            xgb_prob = predict_probability(
                self.xgb_model,
                confidence=abs_r,
                pvt_r=pvt_r,
                best_tf=best.timeframe,
                best_period=best.period,
            )
            
            entry_threshold = self.percentile_calc.get_entry_threshold(get_group(asset), best.timeframe)
            if xgb_prob < entry_threshold: continue

            ad_tf = asset_data[best.timeframe]
            entry_idx = find_bar_index(ad_tf.timestamps, current_ns) + 1
            if entry_idx >= len(ad_tf.timestamps): continue
            entry_price = float(ad_tf.open_[entry_idx])
            entry_ts_ns = int(ad_tf.timestamps[entry_idx])

            m_params = _get_matrix_params(get_group(asset), best.timeframe)
            lev = m_params["leverage"]
            vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
            pos_usd = self.capital * gt.pos_frac * lev * vol_scalar

            self._trade_counter += 1
            self.positions_v1[asset] = ActivePositionV1(
                trade_id=self._trade_counter,
                asset=asset,
                direction=direction,
                entry_price=entry_price,
                entry_ts=pd.Timestamp(entry_ts_ns, tz="UTC"),
                midline=np.exp(best.intercept),
                std_dev=best.std_dev,
                best_tf=best.timeframe,
                best_period=best.period,
                confidence=abs_r,
                pvt_r=pvt_r,
                leverage=lev,
                pos_usd=pos_usd,
                entry_idx=entry_idx,
                xgb_prob=xgb_prob,
                group=get_group(asset),
                peak_r=abs_r,
            )

    def _close_position_v1(self, pos, exit_price, exit_ts_ns, reason):
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price if pos.direction == 1 else (pos.entry_price - exit_price) / pos.entry_price
        pnl_pct -= 0.0008 
        trade_pnl = pos.pos_usd * pnl_pct
        self.realized_pnl += trade_pnl
        self._record_cooldown(pos.asset, pnl_pct * 100, exit_ts_ns)
        
        ad = self.all_data[pos.asset][pos.best_tf]
        exit_idx = find_bar_index(ad.timestamps, exit_ts_ns)
        
        record = TradeRecordV1(
            trade_id=pos.trade_id,
            asset=pos.asset,
            group=pos.group,
            direction=pos.direction,
            entry_ts=pos.entry_ts,
            exit_ts=pd.Timestamp(exit_ts_ns, tz="UTC"),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            best_tf=pos.best_tf,
            best_period=pos.best_period,
            confidence=pos.confidence,
            pvt_r=pos.pvt_r,
            xgb_prob=pos.xgb_prob,
            leverage=pos.leverage,
            position_usd=pos.pos_usd,
            pnl_usd=trade_pnl,
            pnl_pct=pnl_pct * 100,
            exit_reason=reason,
            bars_held=exit_idx - pos.entry_idx,
            peak_r=pos.peak_r,
        )
        self.trades_v1.append(record)
        if pos.asset in self.positions_v1:
            del self.positions_v1[pos.asset]

    def _record_cooldown(self, asset, pnl_pct, current_ns):
        cd = self.cooldowns[asset]
        net_pnl_pct = pnl_pct - STRESS_COST
        if net_pnl_pct > STRESS_COST:
            cd.consecutive_losses = 0
            cd.cooldown_until_ns = 0
            return
        if net_pnl_pct <= 0:
            cd.consecutive_losses += 1
            step_idx = min(cd.consecutive_losses, len(COOLDOWN_STEPS)) - 1
            pause_min = COOLDOWN_STEPS[step_idx]
            cd.cooldown_until_ns = current_ns + (pause_min * 60 * 10**9)

def trades_to_csv_v1(trades: list[TradeRecordV1], path: str | Path):
    rows = []
    for t in trades:
        dur_hours = 0.0
        if t.exit_ts and t.entry_ts:
            dur_hours = (t.exit_ts - t.entry_ts).total_seconds() / 3600
        rows.append({
            "trade_id": t.trade_id,
            "asset": t.asset,
            "group": t.group,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "best_tf": t.best_tf,
            "best_period": t.best_period,
            "confidence": t.confidence,
            "pvt_r": t.pvt_r,
            "leverage": t.leverage,
            "position_usd": t.position_usd,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
            "duration_hours": round(dur_hours, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "pnl_usd": round(t.pnl_usd, 2),
            "peak_r": round(t.peak_r, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Trade log saved: %s (%d trades)", path, len(rows))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-11-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    all_data = load_all_assets()
    model_path = BASE_DIR / "models" / "meta_xgb.json"
    if not model_path.exists():
        model_path = Path("/home/yagokhan/VaranusNeoFlow/models/meta_xgb.json")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_path))

    params = BacktestParams(max_concurrent=8, use_signal_price=False)
    engine = BlindTestV1Engine(all_data, xgb_model, params)
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
    
    trades = engine.run(start_ts, end_ts)
    
    # Pre-process exit reasons for compute_metrics compatibility
    from backtest.engine import TradeRecord as EngineTradeRecord
    
    engine_trades = []
    for t in trades:
        et = EngineTradeRecord(
            trade_id=t.trade_id,
            asset=t.asset,
            direction=t.direction,
            entry_ts=t.entry_ts,
            entry_price=t.entry_price,
            best_tf=t.best_tf,
            best_period=t.best_period,
            confidence=t.confidence,
            pvt_r=t.pvt_r,
            leverage=t.leverage,
            position_usd=t.position_usd,
            hard_sl=0,
            initial_trail_sl=0,
            group=t.group,
            exit_ts=t.exit_ts,
            exit_price=t.exit_price,
            exit_reason=t.exit_reason,
            bars_held=t.bars_held,
            pnl_pct=t.pnl_pct,
            pnl_usd=t.pnl_usd,
            peak_r=t.peak_r
        )
        engine_trades.append(et)
        
    metrics = compute_metrics(engine_trades, initial_capital=10000)
    print_metrics(metrics)
    
    if args.csv:
        trades_to_csv_v1(trades, "blind_test_v1_extended_trades.csv")

if __name__ == "__main__":
    main()
