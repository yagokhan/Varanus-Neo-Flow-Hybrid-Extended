"""
ml/train_meta_model.py — Train XGBoost Meta-Labeler on Extended trade logs.

Extended: supports 4h timeframe encoding (5m=0, 30m=1, 1h=2, 4h=3).
Trained per-group or unified across all 33 assets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install xgboost scikit-learn")
    sys.exit(1)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "meta_xgb.json"

REQUIRED_COLUMNS = {"confidence", "pvt_r", "best_tf", "best_period", "pnl_pct", "pnl_usd"}

# Feature columns — Extended with 4h TF encoding
FEATURE_COLS_LIVE = ["confidence", "pvt_r", "best_tf_encoded", "best_period"]
FEATURE_COLS = FEATURE_COLS_LIVE

# Extended TF encoding: includes 4h
TF_ENCODE = {"5m": 0, "30m": 1, "1h": 2, "4h": 3}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "verbosity": 0,
}


def load_trade_log(path: Path) -> pd.DataFrame:
    """Load and validate a trade log CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logger.info("Loaded %s: %d trades", path.name, len(df))
    return df


def load_and_merge(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate multiple trade log CSVs."""
    frames = []
    for p in paths:
        try:
            frames.append(load_trade_log(p))
        except ValueError as e:
            logger.warning("Skipping %s: %s", p.name, e)
    if not frames:
        raise FileNotFoundError("No valid trade log CSVs found")
    combined = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in ["asset", "entry_ts", "exit_ts", "entry_price", "exit_price"]
                  if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols)
    logger.info("Combined dataset: %d trades", len(combined))
    return combined


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract features and binary target from trade log."""
    df = df.copy()
    if df["best_tf"].dtype == object:
        df["best_tf_encoded"] = df["best_tf"].map(TF_ENCODE)
        df = df.dropna(subset=["best_tf_encoded"])
        df["best_tf_encoded"] = df["best_tf_encoded"].astype(int)
    else:
        df["best_tf_encoded"] = df["best_tf"].astype(int)

    target = (df["pnl_usd"] > 0).astype(int).values
    X = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    target = target[valid_mask.values]
    return X, target


def train_meta_model(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> tuple[xgb.XGBClassifier, dict]:
    """Train XGBoost with stratified k-fold CV."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    class_ratio = n_neg / max(n_pos, 1)

    logger.info("Training: %d samples, %d positive (%.1f%%)", len(y), n_pos, n_pos / len(y) * 100)

    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = class_ratio

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_aucs, cv_accs = [], []
    X_np = X.values
    feature_importances = np.zeros(len(FEATURE_COLS))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_np, y)):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))
        cv_aucs.append(auc)
        cv_accs.append(acc)
        feature_importances += model.feature_importances_
        logger.info("  Fold %d: AUC=%.4f  Acc=%.4f", fold_idx, auc, acc)

    feature_importances /= n_splits
    cv_metrics = {
        "n_samples": len(y),
        "n_positive": int(n_pos),
        "positive_rate": round(n_pos / len(y), 4),
        "mean_auc": round(float(np.mean(cv_aucs)), 4),
        "std_auc": round(float(np.std(cv_aucs)), 4),
        "mean_accuracy": round(float(np.mean(cv_accs)), 4),
        "feature_importance": {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_COLS, feature_importances)
        },
    }

    logger.info("CV: AUC=%.4f ± %.4f", cv_metrics["mean_auc"], cv_metrics["std_auc"])

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_np, y, verbose=False)
    return final_model, cv_metrics


def save_model(model, cv_metrics, output_path, source_files):
    """Save model and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "model_type": "xgboost_meta_labeler_extended",
        "feature_columns": FEATURE_COLS,
        "tf_encoding": TF_ENCODE,
        "source_files": source_files,
        "cv_metrics": cv_metrics,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Model saved: %s", output_path)


def load_model(model_path: Path) -> tuple[xgb.XGBClassifier, dict]:
    """Load a trained meta-labeler model."""
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    meta_path = model_path.with_suffix(".meta.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, metadata


def predict_probability(
    model: xgb.XGBClassifier,
    confidence: float,
    pvt_r: float,
    best_tf: str | int,
    best_period: int,
    pnl_pct: float = 0.0,
) -> float:
    """Predict XGB probability for a single signal."""
    tf_encoded = TF_ENCODE.get(best_tf, best_tf) if isinstance(best_tf, str) else best_tf
    n_model_features = model.n_features_in_
    if n_model_features == 5:
        features = np.array([[confidence, pvt_r, tf_encoded, best_period, pnl_pct]])
    else:
        features = np.array([[confidence, pvt_r, tf_encoded, best_period]])
    return float(model.predict_proba(features)[0, 1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train XGBoost Meta-Labeler (Extended)")
    parser.add_argument("trade_logs", nargs="*")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--folds", "-k", type=int, default=5)
    args = parser.parse_args()

    if not args.trade_logs:
        logger.error("Provide trade log CSV path(s)")
        sys.exit(1)
    paths = [Path(p) for p in args.trade_logs]
    df = load_and_merge(paths)
    X, y = prepare_features(df)
    model, cv_metrics = train_meta_model(X, y, n_splits=args.folds)
    save_model(model, cv_metrics, Path(args.output), [p.name for p in paths])
