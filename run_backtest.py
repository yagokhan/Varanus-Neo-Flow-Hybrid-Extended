#!/usr/bin/env python3
"""
run_backtest.py — Run backtest for the 33-asset Extended portfolio.

Usage:
    python run_backtest.py                                    # Full WFV
    python run_backtest.py --start 2024-01-01 --end 2025-01-01
    python run_backtest.py --blind                            # Blind test only
    python run_backtest.py --csv                              # Export trades CSV
    python run_backtest.py --group A                          # Single group
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import (
    load_all_assets, generate_wfv_folds,
    BLIND_START, BLIND_END,
)
from backtest.engine import BacktestEngine, BacktestParams
from backtest.metrics import compute_metrics, print_metrics, trades_to_csv
from config.groups import (
    GROUP_A_MAJORS, GROUP_B_TECH_AI, GROUP_C_MOMENTUM_MEME,
    GroupThresholds, DEFAULT_THRESHOLDS, GROUP_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent


def _load_optimized_thresholds() -> dict[str, GroupThresholds] | None:
    """Load group thresholds from optimized config if available."""
    config_path = OUTPUT_DIR / "config" / "optimized_thresholds.json"
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
    parser = argparse.ArgumentParser(description="Varanus Neo-Flow Extended — Backtest Runner")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--blind", action="store_true", help="Run blind test only")
    parser.add_argument("--wfv", action="store_true", help="Run full WFV")
    parser.add_argument("--csv", action="store_true", help="Export trade log to CSV")
    parser.add_argument("--group", type=str, default=None, help="Single group (A/B/C)")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--no-xgb", action="store_true", help="Disable XGBoost meta-labeler filter")
    args = parser.parse_args()

    logger.info("Loading 33-asset dataset...")
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

    # Load optimized or default thresholds
    overrides = _load_optimized_thresholds()

    params = BacktestParams(
        initial_capital=args.capital,
        max_concurrent=8,
        use_xgb=(not args.no_xgb),
        group_overrides=overrides,
    )

    if args.blind:
        start_ts = BLIND_START
        end_ts = BLIND_END
        logger.info("Running BLIND test: %s → %s", start_ts.date(), end_ts.date())
    elif args.start and args.end:
        start_ts = pd.Timestamp(args.start, tz="UTC")
        end_ts = pd.Timestamp(args.end, tz="UTC")
    else:
        # Default: full training period
        start_ts = pd.Timestamp("2023-06-01", tz="UTC")
        end_ts = pd.Timestamp("2025-10-31", tz="UTC")

    engine = BacktestEngine(all_data, params)
    trades = engine.run(start_ts, end_ts)
    metrics = compute_metrics(trades, engine.equity_curve, args.capital)
    print_metrics(metrics, args.capital)

    if args.csv:
        csv_path = str(OUTPUT_DIR / "extended_trades.csv")
        trades_to_csv(trades, csv_path)

    if args.wfv:
        logger.info("\n=== Running Walk-Forward Validation ===")
        folds = generate_wfv_folds()
        for fold in folds:
            engine = BacktestEngine(all_data, params)
            trades = engine.run(fold.test_start, fold.test_end)
            m = compute_metrics(trades, engine.equity_curve, args.capital)
            logger.info(
                "  Fold %d: %d trades, WR=%.1f%%, Sharpe=%.2f, DD=%.2f%%",
                fold.fold_id, m.total_trades, m.win_rate, m.sharpe_ratio, m.max_drawdown_pct,
            )


if __name__ == "__main__":
    main()
