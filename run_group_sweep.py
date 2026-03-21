#!/usr/bin/env python3
"""
run_group_sweep.py — Data-Driven Parameter Sweep for Group-Specific Thresholds

Optimizes entry gate thresholds (Confidence, XGB Score, Trail Sensitivity)
independently for each of the 3 asset groups using Optuna + pre-computed LUT.

Uses FastBacktestEngine (O(1) scan lookups) instead of live regression.
Expected: ~5-15s per trial (vs ~5.5 min without LUT).

Output: config/optimized_thresholds.json with group-specific parameters.

Usage:
    python run_group_sweep.py --trials 100
    python run_group_sweep.py --trials 200 --group A
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Install: pip install optuna")
    sys.exit(1)

from backtest.data_loader import (
    load_all_assets, generate_wfv_folds,
    BLIND_START, BLIND_END,
)
from backtest.engine import BacktestParams, TradeRecord
from backtest.engine_fast import FastBacktestEngine
from backtest.metrics import compute_metrics, print_metrics
from neo_flow.precompute_features import precompute_all_features, save_features, load_features
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

GROUP_ASSETS = {
    "A": GROUP_A_MAJORS,
    "B": GROUP_B_TECH_AI,
    "C": GROUP_C_MOMENTUM_MEME,
}

OUTPUT_DIR = Path(__file__).parent
CONFIG_DIR = OUTPUT_DIR / "config"
DATA_DIR = OUTPUT_DIR / "data"


def _optimize_group(
    group: str,
    all_data: dict,
    features: dict,
    n_trials: int,
    initial_capital: float,
) -> dict:
    """Run Optuna optimization for one group using WFV median fold + FastBacktestEngine."""
    assets = GROUP_ASSETS[group]
    group_data = {a: all_data[a] for a in assets if a in all_data}
    group_features = {a: features[a] for a in assets if a in features}

    if not group_data:
        logger.warning("No data for group %s", group)
        return asdict(DEFAULT_THRESHOLDS[group])

    folds = generate_wfv_folds()
    mid_fold = folds[len(folds) // 2]

    logger.info(
        "=== Group %s [%s]: %d assets, %d trials ===",
        group, GROUP_NAMES[group], len(assets), n_trials,
    )
    logger.info(
        "  Optimizing on val %s → %s (FastBacktestEngine)",
        mid_fold.val_start.date(), mid_fold.val_end.date(),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10),
    )

    def objective(trial):
        gt = GroupThresholds(
            min_confidence=trial.suggest_float("min_confidence", 0.75, 0.92, step=0.01),
            min_xgb_score=trial.suggest_float("min_xgb_score", 0.40, 0.75, step=0.01),
            trail_buffer=trial.suggest_float("trail_buffer", 0.30, 1.50, step=0.05),
            min_pvt_r=trial.suggest_float("min_pvt_r", 0.50, 0.85, step=0.05),
            combined_gate=trial.suggest_float("combined_gate", 0.50, 0.85, step=0.05),
            hard_sl_mult=trial.suggest_float("hard_sl_mult", 1.0, 3.5, step=0.25),
            exhaust_r=trial.suggest_float("exhaust_r", 0.30, 0.60, step=0.05),
            pos_frac=trial.suggest_float("pos_frac", 0.03, 0.12, step=0.01),
        )

        params = BacktestParams(
            min_pearson_r=gt.min_confidence,
            min_pvt_r=gt.min_pvt_r,
            combined_gate=gt.combined_gate,
            hard_sl_mult=gt.hard_sl_mult,
            trail_buffer=gt.trail_buffer,
            exhaust_r=gt.exhaust_r,
            pos_frac=gt.pos_frac,
            initial_capital=initial_capital,
            max_concurrent=4,
            group_overrides={group: gt},
        )

        engine = FastBacktestEngine(group_data, params, group_features)
        trades = engine.run(mid_fold.val_start, mid_fold.val_end)
        metrics = compute_metrics(trades, engine.equity_curve, initial_capital)

        if metrics.total_trades < 5:
            return -10.0

        score = metrics.sharpe_ratio
        if metrics.win_rate >= 60:
            score += 0.5
        if metrics.trail_exit_pct >= 60:
            score += 0.3
        if metrics.max_drawdown_pct > -15:
            score += 0.2
        if metrics.max_drawdown_pct < -20:
            score -= 2.0

        return score

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    best = study.best_params
    logger.info(
        "  Group %s complete: %.1f min, best score=%.4f",
        group, elapsed / 60, study.best_value,
    )
    logger.info("  Best params: %s", best)

    return best


def run_sweep(
    groups: list[str] | None = None,
    n_trials: int = 100,
    initial_capital: float = 10_000.0,
):
    """Run optimization sweep for each group."""
    groups = groups or ["A", "B", "C"]

    logger.info("Loading data for 33 assets...")
    all_data = load_all_assets()

    # Pre-compute features once (saves ~50-100x per trial)
    feature_dir = str(DATA_DIR)
    features_path = DATA_DIR / "features"
    if features_path.exists() and any(features_path.glob("*_scan_features.npy")):
        logger.info("Loading pre-computed features from disk...")
        features = load_features(feature_dir)
    else:
        logger.info("Pre-computing scan features for all 33 assets (one-time)...")
        features = precompute_all_features(all_data)
        save_features(features, feature_dir)

    results = {}
    for group in groups:
        best_params = _optimize_group(group, all_data, features, n_trials, initial_capital)
        results[group] = best_params

    # Save optimized thresholds
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_DIR / "optimized_thresholds.json"

    config = {
        "description": "Varanus Neo-Flow Extended — Group-Specific Optimized Thresholds",
        "groups": {}
    }

    for group in groups:
        config["groups"][group] = {
            "name": GROUP_NAMES[group],
            "assets": GROUP_ASSETS[group],
            "thresholds": results[group],
        }

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output_path.write_text(json.dumps(config, indent=2, default=convert))
    logger.info("Optimized thresholds saved: %s", output_path)

    # Print summary
    print()
    print("=" * 80)
    print("GROUP SWEEP RESULTS")
    print("=" * 80)
    for group in groups:
        print(f"\n  Group {group} [{GROUP_NAMES[group]}]:")
        for k, v in sorted(results[group].items()):
            print(f"    {k:>20}: {v:.4f}")
    print()
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group-Specific Parameter Sweep")
    parser.add_argument("--trials", "-n", type=int, default=100)
    parser.add_argument("--group", "-g", type=str, default=None,
                        help="Optimize single group (A, B, or C)")
    parser.add_argument("--capital", type=float, default=10_000.0)
    args = parser.parse_args()

    groups = [args.group.upper()] if args.group else None
    run_sweep(groups=groups, n_trials=args.trials, initial_capital=args.capital)
