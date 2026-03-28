
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

def analyze_tf_specific_percentiles():
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
    
    # 5. Group by Timeframe and Calculate Percentiles
    timeframes = ["5m", "30m", "1h", "4h"]
    tf_results = {}

    for tf in timeframes:
        tf_winners = winners[winners['best_tf'] == tf]
        if tf_winners.empty:
            continue
            
        tf_results[tf] = {
            "count": len(tf_winners),
            "confidence": {
                "p10": tf_winners['confidence'].quantile(0.10),
                "p50": tf_winners['confidence'].quantile(0.50),
                "p75": tf_winners['confidence'].quantile(0.75),
            },
            "xgb_prob": {
                "p10": tf_winners['xgb_prob'].quantile(0.10),
                "p50": tf_winners['xgb_prob'].quantile(0.50),
                "p75": tf_winners['xgb_prob'].quantile(0.75),
            }
        }

    # 6. Print Results
    print("\n" + "="*60)
    print("TIMEFRAME-SPECIFIC VALIDATION PERCENTILES (WINNERS)")
    print("="*60)
    for tf in timeframes:
        if tf not in tf_results: continue
        res = tf_results[tf]
        print(f"\n[TF: {tf}] ({res['count']} winning trades)")
        print(f"  |R| Physics:  10th={res['confidence']['p10']:.4f} | 50th={res['confidence']['p50']:.4f}")
        print(f"  XGB Win Prob: 10th={res['xgb_prob']['p10']:.4f}   | 50th={res['xgb_prob']['p50']:.4f}")
    
    # 7. Save results
    with open("tf_percentile_thresholds.json", "w") as f:
        json.dump(tf_results, f, indent=2)
    print("\nResults saved to tf_percentile_thresholds.json")

if __name__ == "__main__":
    analyze_tf_specific_percentiles()
