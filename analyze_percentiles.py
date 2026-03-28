
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import sys

# Add project root to path for imports
BASE_DIR = Path("/home/gokhan/Varanus-Neo-Flow-Hybrid-Extended")
sys.path.insert(0, str(BASE_DIR))

from ml.train_meta_model import load_model, predict_probability, TF_ENCODE

def analyze_validation_percentiles():
    # 1. Load Model
    model_path = BASE_DIR / "models" / "meta_xgb.json"
    model, _ = load_model(model_path)
    
    # 2. Find validation files
    val_dir = Path("/home/gokhan/VaranusNeoFlow")
    files = list(val_dir.glob("wfv_fold_*_trades.csv"))
    
    if not files:
        print("No validation files found in VaranusNeoFlow")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} total validation trades.")

    # 3. Generate XGB Probabilities
    probs = []
    for _, row in df.iterrows():
        prob = predict_probability(
            model,
            confidence=row['confidence'],
            pvt_r=row['pvt_r'],
            best_tf=row['best_tf'],
            best_period=row['best_period'],
            pnl_pct=0.0 # dummy
        )
        probs.append(prob)
    
    df['xgb_prob'] = probs

    # 4. Filter for Winners
    winners = df[df['pnl_usd'] > 0]
    print(f"Analyzing {len(winners)} winning trades out of {len(df)} total.")

    # 5. Calculate Percentiles for Entry (Winners distribution)
    metrics = {
        "confidence": "|R| (Physics)",
        "xgb_prob": "XGB Probability"
    }

    results = {}
    for col, label in metrics.items():
        results[col] = {
            "p10": winners[col].quantile(0.10),
            "p25": winners[col].quantile(0.25),
            "p50": winners[col].quantile(0.50),
            "p75": winners[col].quantile(0.75),
            "mean": winners[col].mean(),
            "min": winners[col].min(),
            "max": winners[col].max()
        }

    # 6. Print Results
    print("\n" + "="*50)
    print("VALIDATION PERCENTILES (WINNING TRADES)")
    print("="*50)
    for col, data in results.items():
        print(f"\nMetric: {metrics[col]}")
        print(f"  10th Percentile: {data['p10']:.4f} (Aggressive Entry / Tight Exit)")
        print(f"  25th Percentile: {data['p25']:.4f}")
        print(f"  50th Percentile: {data['p50']:.4f} (Balanced)")
        print(f"  75th Percentile: {data['p75']:.4f} (Conservative)")
        print(f"  Mean:            {data['mean']:.4f}")
    
    # 7. Save results for later use
    with open("percentile_thresholds.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to percentile_thresholds.json")

if __name__ == "__main__":
    analyze_validation_percentiles()
