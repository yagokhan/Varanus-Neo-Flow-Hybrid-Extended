#!/usr/bin/env python3
"""
check_last_scan.py — Diagnostic tool to run the last available data scan offline.
Used to compare backtest/offline logic with live bot behavior.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup Paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from backtest.data_loader import load_all_assets, build_scan_dataframes, build_htf_dataframe
from neo_flow.adaptive_engine import (
    find_best_regression, compute_pvt_regression, check_pvt_alignment,
    get_htf_bias, check_combined_gate, get_leverage
)
from config.groups import get_group, get_thresholds, ALL_ASSETS
from ml.train_meta_model import predict_probability, load_model

# Config
XGB_MODEL_PATH = BASE_DIR / "models" / "meta_xgb.json"
OPTIMIZED_THRESHOLDS_PATH = BASE_DIR / "config" / "optimized_thresholds.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("offline_scan")

def _load_optimized_thresholds():
    """Helper to load optimized thresholds from json."""
    from config.groups import GroupThresholds
    if not OPTIMIZED_THRESHOLDS_PATH.exists():
        return None
    import json
    data = json.loads(OPTIMIZED_THRESHOLDS_PATH.read_text())
    overrides = {}
    for group, gdata in data.get("groups", {}).items():
        t = gdata.get("thresholds", {})
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

def run_diagnostic_scan():
    logger.info("=== OFFLINE DIAGNOSTIC SCAN ===")
    
    # 1. Load Data
    logger.info("Loading Parquet data...")
    all_data = load_all_assets()
    if not all_data:
        logger.error("No data found in data/*.parquet")
        return

    # 2. Find latest common 1H timestamp
    latest_ts = 0
    for asset in all_data:
        if "1h" in all_data[asset]:
            latest_ts = max(latest_ts, all_data[asset]["1h"].timestamps[-1])
    
    scan_ts = pd.Timestamp(latest_ts, tz="UTC")
    logger.info(f"Target Scan Timestamp: {scan_ts} (Latest in data)")
    logger.info("-" * 40)

    # 3. Load ML Model & Optimized Thresholds
    xgb_model = None
    if XGB_MODEL_PATH.exists():
        xgb_model, _ = load_model(XGB_MODEL_PATH)
        logger.info("XGBoost Meta-Labeler: LOADED")
    
    optimized_overrides = _load_optimized_thresholds()
    if optimized_overrides:
        logger.info(f"Optimized Thresholds: LOADED ({list(optimized_overrides.keys())})")

    results = []

    # 4. Scan each asset
    for asset in ALL_ASSETS:
        asset_data = all_data.get(asset)
        if not asset_data: continue
        
        group = get_group(asset)
        gt = get_thresholds(asset, optimized_overrides)
        
        # Build DataFrames
        scan_dfs = build_scan_dataframes(asset_data, latest_ts - 1)
        df_4h = build_htf_dataframe(asset_data, latest_ts - 14400*10**9)
        
        if not scan_dfs or df_4h is None:
            results.append([asset, group, "NO_DATA", "-", "-", "-", "-"])
            continue

        # Gate 1: Physics (Pearson R)
        best, all_reg = find_best_regression(scan_dfs)
        if not best:
            results.append([asset, group, "NO_REGR", "-", "-", "-", "-"])
            continue
            
        abs_r = abs(best.pearson_r)
        direction = 1 if best.slope < 0 else -1
        dir_str = "LONG" if direction == 1 else "SHORT"
        
        if abs_r < gt.min_confidence:
            results.append([asset, group, "WEAK_R", dir_str, f"{abs_r:.3f}", "-", "-"])
            continue

        # Gate 2: PVT
        best_df_7d = scan_dfs[best.timeframe].iloc[-2016:] # Max 7d
        pvt = compute_pvt_regression(best_df_7d, best.period)
        pvt_passes, pvt_reason = check_pvt_alignment(direction, abs_r, pvt, gt.min_pvt_r)
        if not pvt_passes:
            results.append([asset, group, "PVT_REJ", dir_str, f"{abs_r:.3f}", f"{abs(pvt.pearson_r):.3f}", "-"])
            continue

        # Gate 3: HTF Bias
        htf_bias = get_htf_bias(df_4h)
        if htf_bias == 0 or htf_bias != direction:
            status = "HTF_NEUT" if htf_bias == 0 else "HTF_CONFL"
            results.append([asset, group, status, dir_str, f"{abs_r:.3f}", f"{abs(pvt.pearson_r):.3f}", "-"])
            continue

        # Gate 4: Combined Gate
        if not check_combined_gate(direction, all_reg, gt.combined_gate):
            results.append([asset, group, "GATE_BLOK", dir_str, f"{abs_r:.3f}", f"{abs(pvt.pearson_r):.3f}", "-"])
            continue

        # Gate 5: XGBoost
        xgb_prob = 0.0
        if xgb_model:
            xgb_prob = predict_probability(xgb_model, abs_r, abs(pvt.pearson_r), best.timeframe, best.period)
            if xgb_prob < gt.min_xgb_score:
                results.append([asset, group, "XGB_REJ", dir_str, f"{abs_r:.3f}", f"{abs(pvt.pearson_r):.3f}", f"{xgb_prob:.3f}"])
                continue

        # Passed All Gates
        results.append([asset, group, "PASSED", dir_str, f"{abs_r:.3f}", f"{abs(pvt.pearson_r):.3f}", f"{xgb_prob:.3f}"])

    # 5. Display Results
    df_res = pd.DataFrame(results, columns=["Asset", "Grp", "Status", "Dir", "Price_R", "PVT_R", "XGB_Prob"])
    
    # Sort: PASSED first, then by Status
    df_res['sort_val'] = df_res['Status'].apply(lambda x: 0 if x == "PASSED" else 1)
    df_res = df_res.sort_values(['sort_val', 'Status', 'Asset']).drop(columns=['sort_val'])
    
    print(df_res.to_string(index=False))
    print("-" * 40)
    passed = df_res[df_res['Status'] == "PASSED"]
    if not passed.empty:
        logger.info(f"SIGNALS DETECTED ({len(passed)}):")
        print(passed[['Asset', 'Dir', 'Price_R', 'XGB_Prob']].to_string(index=False))
    else:
        logger.info("NO SIGNALS PASSED GATES.")

if __name__ == "__main__":
    run_diagnostic_scan()
