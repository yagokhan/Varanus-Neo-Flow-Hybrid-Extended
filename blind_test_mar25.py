#!/usr/bin/env python3
"""
blind_test_mar25.py — High-fidelity 5m-interval backtest for March 25, 2026.
Matches the live bot's WebSocket-driven 5m scanning behavior.
"""

import sys
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from backtest.data_loader import load_all_assets, ts_to_ns
from backtest.engine import BacktestEngine, BacktestParams
from backtest.metrics import compute_metrics, print_metrics
from config.groups import ALL_ASSETS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_5m_timestamps(all_data, start_ts, end_ts):
    """Get 5m timestamp array within [start, end] range."""
    # Find any asset with 5m data
    for asset_data in all_data.values():
        ad = asset_data.get("5m")
        if ad is not None:
            start_ns = ts_to_ns(start_ts)
            end_ns = ts_to_ns(end_ts)
            mask = (ad.timestamps >= start_ns) & (ad.timestamps <= end_ns)
            return ad.timestamps[mask]
    return np.array([], dtype="int64")

class LiveFidelityEngine(BacktestEngine):
    """Modified engine that treats every 5m step as a potential entry scan."""
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
            return []

        logger.info(f"Running Live-Fidelity Backtest: {start_ts} -> {end_ts}")
        logger.info(f"Interval: 5 minutes | Total steps: {len(bar_timestamps)}")

        for i, current_ns in enumerate(bar_timestamps):
            # 1. Update existing positions (SL/Trail checks)
            self._update_positions(current_ns)
            
            # 2. Scan for new entries EVERY 5 minutes (Matches Live WS behavior)
            self._scan_and_enter(current_ns)
            
            # 3. Record equity
            self.equity_curve.append(self._compute_equity())
            self._prev_1h_ns = current_ns

        # Close any remaining positions at the end
        for asset, pos in list(self.positions.items()):
            ad = self.all_data.get(asset, {}).get(pos.best_tf)
            if ad is not None:
                mask = ad.timestamps == bar_timestamps[-1]
                if mask.any():
                    exit_price = float(ad.close[mask][0])
                    exit_ts = pd.Timestamp(bar_timestamps[-1], tz="UTC")
                    self._close_position(pos, exit_price, exit_ts, "END_OF_PERIOD")

        return self.trades

def main():
    # March 25, 2026 range
    start_ts = pd.Timestamp("2026-03-25 00:00:00", tz="UTC")
    end_ts = pd.Timestamp("2026-03-25 23:55:00", tz="UTC")

    logger.info("Loading dataset...")
    all_data = load_all_assets()

    # Load optimized thresholds if they exist
    from run_backtest import _load_optimized_thresholds
    overrides = _load_optimized_thresholds()

    params = BacktestParams(
        initial_capital=10000.0,
        max_concurrent=8,
        use_xgb=True,
        group_overrides=overrides,
        scan_interval_hours=1 # This will be ignored by our custom engine but good to keep
    )

    engine = LiveFidelityEngine(all_data, params)
    trades = engine.run(start_ts, end_ts)

    if not trades:
        logger.info("\nNo trades taken on March 25 with 5m scanning.")
        return

    metrics = compute_metrics(trades, engine.equity_curve, 10000.0)
    print_metrics(metrics, 10000.0)

    # Summarize trades
    print("\n--- Trade Summary ---")
    for t in trades:
        dur = (t.exit_ts - t.entry_ts).total_seconds() / 3600
        print(f"{t.entry_ts.strftime('%H:%M')} | {t.asset:5} | {t.direction:2} | PnL: {t.pnl_pct:+.2f}% | {t.exit_reason}")

if __name__ == "__main__":
    main()
