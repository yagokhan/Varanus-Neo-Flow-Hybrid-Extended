#!/usr/bin/env python3
"""
blind_test_5m.py — Full blind test with 5m-interval scanning.

Matches the live bot's dual-scan behavior:
  - On hourly boundaries: scan ALL 4 TFs (5m, 30m, 1h, 4h) — matches live hourly scan
  - On intermediate 5m bars: scan ONLY 5m TF — matches live WS-5m scan

Uses current optimized thresholds WITHOUT re-optimization.
Runs over the full blind test period (Nov 1, 2025 → now).

Usage:
    python blind_test_5m.py                         # Full blind test
    python blind_test_5m.py --start 2026-01-01      # Custom start
    python blind_test_5m.py --csv                   # Export trades CSV
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import (
    AssetData,
    build_scan_dataframes,
    build_htf_dataframe,
    get_sub_bars,
    find_bar_index,
    ts_to_ns,
    ASSETS,
    BLIND_START,
)
from backtest.engine import (
    BacktestEngine,
    BacktestParams,
    TradeRecord,
    ActivePosition,
)
from backtest.metrics import compute_metrics, print_metrics, trades_to_csv
from config.groups import (
    get_group, get_thresholds, GroupThresholds,
    GROUP_A_MAJORS, GROUP_B_TECH_AI, GROUP_C_MOMENTUM_MEME,
    GROUP_NAMES,
)
from neo_flow.adaptive_engine import (
    scan_asset,
    compute_hard_sl,
    compute_atr,
    get_leverage,
    HIGH_VOL_ASSETS,
)
from ml.train_meta_model import predict_probability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
ONE_HOUR_NS = 3600 * 10**9


def get_5m_timestamps(all_data, start_ts, end_ts):
    """Get 5m timestamp array within [start, end] range."""
    for asset_data in all_data.values():
        ad = asset_data.get("5m")
        if ad is not None:
            start_ns = ts_to_ns(start_ts)
            end_ns = ts_to_ns(end_ts)
            mask = (ad.timestamps >= start_ns) & (ad.timestamps <= end_ns)
            return ad.timestamps[mask]
    return np.array([], dtype="int64")


def is_hourly_boundary(ns_timestamp):
    """Check if a nanosecond timestamp falls on an exact hour boundary."""
    seconds = ns_timestamp // 10**9
    return seconds % 3600 == 0


class BlindTest5mEngine(BacktestEngine):
    """
    Backtest engine that matches the live bot's dual-scan behavior:
    - Every hour: full 4-TF scan (like live hourly scan)
    - Every 5m in between: 5m-only scan (like live WS-5m scan)
    """

    def run(self, start_ts, end_ts):
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.realized_pnl = 0.0
        self._trade_counter = 0
        self._scan_count = 0
        self._prev_1h_ns = None

        # Use 5m timestamps instead of 1h
        bar_timestamps = get_5m_timestamps(self.all_data, start_ts, end_ts)
        if len(bar_timestamps) == 0:
            logger.error("No 5m timestamps found in range %s → %s", start_ts, end_ts)
            return []

        total_bars = len(bar_timestamps)
        t0 = time.perf_counter()

        # Count hourly vs 5m-only scans for logging
        hourly_scans = 0
        fivem_scans = 0

        logger.info(
            "5m Blind Test: %s → %s (%d 5m bars, %d assets)",
            start_ts.date(), end_ts.date(), total_bars, len(ASSETS),
        )

        for i, current_ns in enumerate(bar_timestamps):
            current_ns = int(current_ns)

            # 1. Update existing positions (SL/Trail checks on every bar)
            self._update_positions(current_ns)

            # 2. Scan for new entries
            if is_hourly_boundary(current_ns):
                # Hourly: full 4-TF scan (matches live hourly scan)
                self._scan_and_enter(current_ns)
                hourly_scans += 1
            else:
                # Intermediate 5m bar: 5m-only scan (matches live WS-5m scan)
                self._scan_5m_only(current_ns)
                fivem_scans += 1

            # 3. Record equity
            self.equity_curve.append(self._compute_equity())
            self._prev_1h_ns = current_ns

            # Progress logging
            if (i + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                pct = (i + 1) / total_bars * 100
                trades_so_far = len([t for t in self.trades if t.exit_ts is not None])
                logger.info(
                    "  %5.1f%% (%d/%d bars, %.1fs) — %d trades, %d open",
                    pct, i + 1, total_bars, elapsed, trades_so_far, len(self.positions),
                )

        # Close remaining positions
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
            "5m Blind Test complete: %.1fs | %d hourly scans + %d 5m scans = %d total | %d trades",
            elapsed, hourly_scans, fivem_scans, hourly_scans + fivem_scans, len(completed),
        )
        return self.trades

    def _scan_5m_only(self, current_ns: int):
        """5m-only scan — matches live bot's scan_5m_asset() behavior.
        Only checks the 5m timeframe for regressions, uses same gates."""
        if len(self.positions) >= self.params.max_concurrent:
            return
        current_ts = pd.Timestamp(current_ns, tz="UTC")

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

            # Build scan DataFrames for 5m ONLY (matches live WS-5m)
            scan_dfs = build_scan_dataframes(asset_data, current_ns - 1, scan_tfs=["5m"])
            df_4h = build_htf_dataframe(asset_data, current_ns - 4 * 3600 * 10**9)

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

            # ML Meta-Labeler Filter
            if self.xgb_model is not None:
                try:
                    xgb_prob = predict_probability(
                        self.xgb_model,
                        confidence=signal.confidence,
                        pvt_r=signal.pvt_r,
                        best_tf=signal.best_tf,
                        best_period=signal.best_period,
                    )
                    threshold = gt.min_xgb_score
                    if xgb_prob < threshold:
                        continue
                except Exception as e:
                    logger.error("XGB prediction failed: %s", e)

            # Entry price — use next bar's open for realistic fill
            ad = asset_data.get(signal.best_tf)
            if ad is None:
                continue
            if self.params.use_signal_price:
                entry_price = signal.entry_price
            else:
                entry_idx = np.searchsorted(ad.timestamps, current_ns, side="right")
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
            idx = find_bar_index(ad_tf.timestamps, current_ns - 1)
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
                initial_trail = min(entry_price, signal.midline - gt.trail_buffer * std_dev_price)
            else:
                initial_trail = max(entry_price, signal.midline + gt.trail_buffer * std_dev_price)

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


def _load_optimized_thresholds():
    config_path = BASE_DIR / "config" / "optimized_thresholds.json"
    if not config_path.exists():
        logger.info("No optimized thresholds found, using defaults")
        return None
    config = json.loads(config_path.read_text())
    overrides = {}
    for group, data in config.get("groups", {}).items():
        t = data.get("thresholds", {})
        overrides[group] = GroupThresholds(
            min_confidence=t.get("min_confidence", 0.80),
            min_xgb_score=t.get("min_xgb_score", 0.55),
            trail_buffer=t.get("trail_buffer", 0.50),
            min_pvt_r=t.get("min_pvt_r", 0.75),
            combined_gate=t.get("combined_gate", 0.80),
            hard_sl_mult=t.get("hard_sl_mult", 2.5),
            exhaust_r=t.get("exhaust_r", 0.475),
            pos_frac=t.get("pos_frac", 0.05),
        )
    logger.info("Loaded optimized thresholds for groups: %s", list(overrides.keys()))
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Varanus 5m Blind Test — Live Fidelity")
    parser.add_argument("--start", type=str, default=None, help="Start date (default: blind test start)")
    parser.add_argument("--end", type=str, default=None, help="End date (default: now)")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--csv", action="store_true", help="Export trade log to CSV")
    parser.add_argument("--group", type=str, default=None, help="Single group (A/B/C)")
    args = parser.parse_args()

    # Date range
    start_ts = pd.Timestamp(args.start, tz="UTC") if args.start else BLIND_START
    end_ts = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")

    logger.info("=" * 70)
    logger.info("VARANUS 5m BLIND TEST — Live Fidelity Mode")
    logger.info("Period: %s → %s", start_ts.date(), end_ts.date())
    logger.info("=" * 70)

    logger.info("Loading 33-asset dataset (5m + 30m + 1h + 4h)...")
    from backtest.data_loader import load_all_assets
    all_data = load_all_assets()

    # Filter to specific group if requested
    if args.group:
        group = args.group.upper()
        group_assets = {
            "A": GROUP_A_MAJORS,
            "B": GROUP_B_TECH_AI,
            "C": GROUP_C_MOMENTUM_MEME,
        }
        assets = group_assets.get(group, [])
        all_data = {a: all_data[a] for a in assets if a in all_data}
        logger.info("Filtered to Group %s [%s]: %d assets",
                     group, GROUP_NAMES.get(group, ""), len(all_data))

    overrides = _load_optimized_thresholds()

    params = BacktestParams(
        initial_capital=args.capital,
        max_concurrent=8,
        use_xgb=True,
        group_overrides=overrides,
    )

    engine = BlindTest5mEngine(all_data, params)
    trades = engine.run(start_ts, end_ts)

    if not trades:
        logger.info("No trades taken in the blind test period.")
        return

    metrics = compute_metrics(trades, engine.equity_curve, args.capital)

    print("\n" + "=" * 70)
    print("5m BLIND TEST RESULTS (Live Fidelity)")
    print("=" * 70)
    print_metrics(metrics, args.capital)

    # Win rate by scan type — check if best_tf == "5m" to identify 5m-originated trades
    trades_5m = [t for t in trades if t.best_tf == "5m" and t.exit_ts is not None]
    trades_other = [t for t in trades if t.best_tf != "5m" and t.exit_ts is not None]
    completed = [t for t in trades if t.exit_ts is not None]

    if trades_5m:
        wins_5m = sum(1 for t in trades_5m if t.pnl_usd > 0)
        wr_5m = wins_5m / len(trades_5m) * 100
        pnl_5m = sum(t.pnl_usd for t in trades_5m)
        print(f"\n  5m-TF Trades:    {len(trades_5m):>5d} | WR: {wr_5m:5.1f}% | PnL: ${pnl_5m:+,.2f}")

    if trades_other:
        wins_other = sum(1 for t in trades_other if t.pnl_usd > 0)
        wr_other = wins_other / len(trades_other) * 100
        pnl_other = sum(t.pnl_usd for t in trades_other)
        print(f"  Other-TF Trades: {len(trades_other):>5d} | WR: {wr_other:5.1f}% | PnL: ${pnl_other:+,.2f}")

    # Group breakdown
    print("\n--- Group Breakdown ---")
    for g in ["A", "B", "C"]:
        gt = [t for t in completed if t.group == g]
        if gt:
            wins = sum(1 for t in gt if t.pnl_usd > 0)
            wr = wins / len(gt) * 100
            pnl = sum(t.pnl_usd for t in gt)
            print(f"  Group {g}: {len(gt):>5d} trades | WR: {wr:5.1f}% | PnL: ${pnl:+,.2f}")

    if args.csv:
        csv_path = str(BASE_DIR / "blind_test_5m_trades.csv")
        trades_to_csv(trades, csv_path)
        logger.info("Trades exported to: %s", csv_path)


if __name__ == "__main__":
    main()
