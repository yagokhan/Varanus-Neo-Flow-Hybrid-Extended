#!/usr/bin/env python3
"""
run_optimize.py — Full WFV Optimization + Blind Test for Extended Portfolio.

Runs 8-fold walk-forward validation with Optuna using FastBacktestEngine
(pre-computed LUT), computes consensus params, then evaluates on blind test.

Usage:
    python run_optimize.py --trials 200                   # Full WFV
    python run_optimize.py --blind-only                   # Blind test only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from backtest.engine import BacktestParams
from backtest.engine_fast import FastBacktestEngine
from backtest.metrics import compute_metrics, print_metrics, trades_to_csv
from neo_flow.precompute_features import precompute_all_features, save_features, load_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"


def _load_or_compute_features(all_data):
    """Load cached features or compute from scratch."""
    feature_dir = str(DATA_DIR)
    features_path = DATA_DIR / "features"
    if features_path.exists() and any(features_path.glob("*_scan_features.npy")):
        logger.info("Loading pre-computed features from disk...")
        return load_features(feature_dir)
    else:
        logger.info("Pre-computing scan features for all assets (one-time)...")
        features = precompute_all_features(all_data)
        save_features(features, feature_dir)
        return features


def _optimize_fold(all_data, features, fold, n_trials, initial_capital):
    """Optimize one fold using FastBacktestEngine."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42 + fold.fold_id),
        pruner=MedianPruner(n_startup_trials=10),
    )

    def objective(trial):
        params = BacktestParams(
            min_pearson_r=trial.suggest_float("min_pearson_r", 0.75, 0.92, step=0.01),
            min_pvt_r=trial.suggest_float("min_pvt_r", 0.50, 0.85, step=0.05),
            combined_gate=trial.suggest_float("combined_gate", 0.50, 0.80, step=0.05),
            hard_sl_mult=trial.suggest_float("hard_sl_mult", 1.0, 3.0, step=0.25),
            trail_buffer=trial.suggest_float("trail_buffer", 0.5, 2.0, step=0.25),
            exhaust_r=trial.suggest_float("exhaust_r", 0.30, 0.65, step=0.05),
            pos_frac=trial.suggest_float("pos_frac", 0.03, 0.12, step=0.01),
            initial_capital=initial_capital,
            max_concurrent=8,
        )
        engine = FastBacktestEngine(all_data, params, features)
        trades = engine.run(fold.val_start, fold.val_end)
        metrics = compute_metrics(trades, engine.equity_curve, initial_capital)

        if metrics.total_trades < 10:
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

    logger.info(
        "  Fold %d: %.1f min, best=%.4f",
        fold.fold_id, elapsed / 60, study.best_value,
    )

    # Evaluate on test
    best = study.best_params
    best_params = BacktestParams(
        min_pearson_r=best["min_pearson_r"],
        min_pvt_r=best["min_pvt_r"],
        combined_gate=best["combined_gate"],
        hard_sl_mult=best["hard_sl_mult"],
        trail_buffer=best["trail_buffer"],
        exhaust_r=best["exhaust_r"],
        pos_frac=best["pos_frac"],
        initial_capital=initial_capital,
        max_concurrent=8,
    )

    engine_test = FastBacktestEngine(all_data, best_params, features)
    trades_test = engine_test.run(fold.test_start, fold.test_end)
    test_metrics = compute_metrics(trades_test, engine_test.equity_curve, initial_capital)

    logger.info(
        "  Fold %d test: %d trades, WR=%.1f%%, Sharpe=%.2f, DD=%.2f%%",
        fold.fold_id, test_metrics.total_trades, test_metrics.win_rate,
        test_metrics.sharpe_ratio, test_metrics.max_drawdown_pct,
    )

    return {
        "fold_id": fold.fold_id,
        "best_params": best,
        "test_trades": test_metrics.total_trades,
        "test_win_rate": test_metrics.win_rate,
        "test_sharpe": test_metrics.sharpe_ratio,
        "test_max_dd": test_metrics.max_drawdown_pct,
        "test_pnl_pct": test_metrics.total_pnl_pct,
    }


def run_wfv(n_trials=100, initial_capital=10_000.0):
    """Full 8-fold WFV optimization + blind test."""
    logger.info("Loading 33-asset dataset...")
    all_data = load_all_assets()
    features = _load_or_compute_features(all_data)

    folds = generate_wfv_folds()
    logger.info("Generated %d WFV folds", len(folds))

    fold_results = []
    for fold in folds:
        logger.info("=== FOLD %d ===", fold.fold_id)
        result = _optimize_fold(all_data, features, fold, n_trials, initial_capital)
        fold_results.append(result)

    # Consensus: median across folds
    param_keys = fold_results[0]["best_params"].keys()
    consensus = {}
    for key in param_keys:
        values = [fr["best_params"][key] for fr in fold_results]
        consensus[key] = round(float(np.median(values)), 4)

    logger.info("Consensus params: %s", consensus)

    # Blind test
    logger.info("Running BLIND test: %s → %s", BLIND_START.date(), BLIND_END.date())
    blind_params = BacktestParams(
        min_pearson_r=consensus["min_pearson_r"],
        min_pvt_r=consensus["min_pvt_r"],
        combined_gate=consensus["combined_gate"],
        hard_sl_mult=consensus["hard_sl_mult"],
        trail_buffer=consensus["trail_buffer"],
        exhaust_r=consensus["exhaust_r"],
        pos_frac=consensus["pos_frac"],
        initial_capital=initial_capital,
        max_concurrent=8,
    )
    engine = FastBacktestEngine(all_data, blind_params, features)
    trades = engine.run(BLIND_START, BLIND_END)
    blind_metrics = compute_metrics(trades, engine.equity_curve, initial_capital)
    print_metrics(blind_metrics, initial_capital)

    # Save trades
    trades_to_csv(trades, str(OUTPUT_DIR / "blind_test_trades.csv"))

    # Save results
    output = {
        "folds": fold_results,
        "consensus_params": consensus,
        "blind_test": {
            "trades": blind_metrics.total_trades,
            "win_rate": blind_metrics.win_rate,
            "sharpe": blind_metrics.sharpe_ratio,
            "max_dd": blind_metrics.max_drawdown_pct,
            "pnl_pct": blind_metrics.total_pnl_pct,
            "profit_factor": blind_metrics.profit_factor,
        },
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    result_path = OUTPUT_DIR / "wfv_results.json"
    result_path.write_text(json.dumps(output, indent=2, default=convert))
    logger.info("Results saved: %s", result_path)

    # Print summary
    print()
    print("=" * 80)
    print("WFV SUMMARY — VARANUS NEO-FLOW EXTENDED (33 ASSETS)")
    print("=" * 80)
    print()
    print(f"{'Fold':>4}  {'Trades':>6}  {'WR':>6}  {'Sharpe':>8}  {'DD':>8}  {'PnL':>8}")
    print("-" * 50)
    for fr in fold_results:
        print(f"{fr['fold_id']:>4}  {fr['test_trades']:>6}  "
              f"{fr['test_win_rate']:>5.1f}%  {fr['test_sharpe']:>8.2f}  "
              f"{fr['test_max_dd']:>7.2f}%  {fr['test_pnl_pct']:>+7.1f}%")
    print("-" * 50)
    print(f"\nConsensus: {consensus}")
    print(f"\nBlind Test: {blind_metrics.total_trades} trades, "
          f"WR={blind_metrics.win_rate:.1f}%, Sharpe={blind_metrics.sharpe_ratio:.2f}, "
          f"DD={blind_metrics.max_drawdown_pct:.2f}%, PnL={blind_metrics.total_pnl_pct:+.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WFV Optimization — Extended")
    parser.add_argument("--trials", "-n", type=int, default=100)
    parser.add_argument("--capital", type=float, default=10_000.0)
    args = parser.parse_args()
    run_wfv(n_trials=args.trials, initial_capital=args.capital)
