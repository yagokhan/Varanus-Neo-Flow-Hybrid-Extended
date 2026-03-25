#!/usr/bin/env python3
"""
extend_blind.py — Fast incremental blind test extension.

Instead of re-running the full 5-month backtest, this script:
  1. Reads the checkpoint CSV (blind_test_checkpoint.csv)
  2. Finds the last trade's exit timestamp
  3. Runs the backtest ONLY from that point to now (or --end)
  4. Appends new trades to the checkpoint
  5. Saves merged result to extended_trades.csv

Usage:
    python extend_blind.py                    # Extend to now
    python extend_blind.py --end "2026-03-26T14:00"  # Extend to specific time
    python extend_blind.py --end "2026-03-26T11:00" --save  # Extend and update checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import load_all_assets
from backtest.engine import BacktestEngine, BacktestParams
from backtest.metrics import compute_metrics, print_metrics, trades_to_csv
from config.groups import GroupThresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CHECKPOINT = BASE_DIR / "blind_test_checkpoint.csv"
OUTPUT = BASE_DIR / "extended_trades.csv"


def _load_optimized_thresholds() -> dict[str, GroupThresholds] | None:
    config_path = BASE_DIR / "config" / "optimized_thresholds.json"
    if not config_path.exists():
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
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Fast incremental blind test extension")
    parser.add_argument("--end", type=str, default=None, help="End time (default: now)")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--save", action="store_true", help="Update checkpoint with new results")
    args = parser.parse_args()

    # Step 1: Load checkpoint
    if not CHECKPOINT.exists():
        logger.error("No checkpoint found at %s. Run the full backtest first.", CHECKPOINT)
        sys.exit(1)

    checkpoint_df = pd.read_csv(CHECKPOINT)
    n_existing = len(checkpoint_df)
    logger.info("Loaded checkpoint: %d trades", n_existing)

    # Step 2: Find where to resume from — use the last entry_ts hour boundary
    last_entry = pd.Timestamp(checkpoint_df["entry_ts"].max())
    # Start from the hour of the last trade (to catch any trades in that hour)
    resume_from = last_entry.floor("h")
    logger.info("Last trade at %s — resuming from %s", last_entry, resume_from)

    # Step 3: Determine end time
    if args.end:
        end_ts = pd.Timestamp(args.end, tz="UTC")
    else:
        end_ts = pd.Timestamp.now(tz="UTC")
    logger.info("Extending blind test: %s → %s", resume_from, end_ts)

    if end_ts <= resume_from:
        logger.info("Nothing to extend — checkpoint is already up to date.")
        return

    # Step 4: Run incremental backtest
    logger.info("Loading dataset...")
    all_data = load_all_assets()
    overrides = _load_optimized_thresholds()

    params = BacktestParams(
        initial_capital=args.capital,
        max_concurrent=8,
        use_xgb=True,
        group_overrides=overrides,
    )

    engine = BacktestEngine(all_data, params)
    new_trades = engine.run(resume_from, end_ts)

    if not new_trades:
        logger.info("No new trades in the extension period.")
        # Still save the checkpoint as output
        checkpoint_df.to_csv(str(OUTPUT), index=False)
        logger.info("Output saved (unchanged): %s", OUTPUT)
        return

    # Step 5: Convert new trades to DataFrame (match CSV columns from trades_to_csv)
    new_rows = []
    for t in new_trades:
        dur_h = (t.exit_ts - t.entry_ts).total_seconds() / 3600 if t.exit_ts else 0
        new_rows.append({
            "trade_id": t.trade_id,
            "asset": t.asset,
            "group": t.group,
            "direction": t.direction,
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
            "hard_sl": t.hard_sl,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
            "duration_hours": round(dur_h, 2),
            "pnl_pct": t.pnl_pct,
            "pnl_usd": t.pnl_usd,
            "peak_r": t.peak_r,
        })
    new_df = pd.DataFrame(new_rows)

    # Remove duplicates: drop new trades whose entry_ts already exists in checkpoint
    checkpoint_entries = set(checkpoint_df["entry_ts"].astype(str))
    new_df["entry_ts_str"] = new_df["entry_ts"].astype(str)
    truly_new = new_df[~new_df["entry_ts_str"].isin(checkpoint_entries)].drop(columns=["entry_ts_str"])

    logger.info("New trades found: %d (after dedup: %d)", len(new_df), len(truly_new))

    # Step 6: Merge and save
    if len(truly_new) > 0:
        merged = pd.concat([checkpoint_df, truly_new], ignore_index=True)
        # Re-number trade IDs
        merged["trade_id"] = range(1, len(merged) + 1)
    else:
        merged = checkpoint_df

    merged.to_csv(str(OUTPUT), index=False)
    logger.info("Saved extended trades: %d total (%d new) → %s", len(merged), len(truly_new), OUTPUT)

    # Print metrics for new trades only
    if new_trades:
        metrics = compute_metrics(new_trades, engine.equity_curve, args.capital)
        print("\n" + "=" * 60)
        print("NEW TRADES ONLY (extension period)")
        print("=" * 60)
        print_metrics(metrics, args.capital)

    # Step 7: Update checkpoint if requested
    if args.save:
        merged.to_csv(str(CHECKPOINT), index=False)
        logger.info("Checkpoint updated: %s", CHECKPOINT)


if __name__ == "__main__":
    main()
