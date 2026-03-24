"""
backtest/data_loader.py — Data loading, numpy extraction, walk-forward folds.

Extended: 33 assets × 4 TFs (5m, 30m, 1h, 4h) into memory-efficient numpy arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Import the 33-asset universe from config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.groups import ALL_ASSETS, ALL_SYMBOLS, ASSET_FROM_SYMBOL

ASSETS = ALL_ASSETS
TIMEFRAMES = ["5m", "30m", "1h", "4h"]

BARS_PER_HOUR = {"5m": 12, "30m": 2, "1h": 1, "4h": 0.25}
TF_MS = {"5m": 300_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000}
TF_HOURS = {"5m": 5 / 60, "30m": 0.5, "1h": 1.0, "4h": 4.0}
BARS_7D = {"5m": 2016, "30m": 336, "1h": 168, "4h": 42}

# WFV parameters
WFV_START = pd.Timestamp("2023-01-01", tz="UTC")
WFV_END = pd.Timestamp("2025-10-31 23:00:00", tz="UTC")
BLIND_START = pd.Timestamp("2025-11-01", tz="UTC")
BLIND_END = pd.Timestamp.now(tz="UTC")
WFV_FOLDS = 8
WFV_TRAIN_FRAC = 0.40
WFV_VAL_FRAC = 0.30
WFV_TEST_FRAC = 0.30
EMBARGO_BARS = 24


@dataclass
class AssetData:
    """Preloaded numpy arrays for one asset, one timeframe."""
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open_: np.ndarray
    volume: np.ndarray
    timestamps: np.ndarray
    pvt: np.ndarray


@dataclass
class WFVFold:
    """Walk-forward validation fold date boundaries."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _compute_pvt(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Compute cumulative PVT from close and volume arrays."""
    pct = np.zeros(len(close))
    pct[1:] = (close[1:] - close[:-1]) / close[:-1]
    return np.cumsum(pct * volume)


def load_asset_tf(asset: str, tf: str, data_dir: Path = DATA_DIR) -> AssetData | None:
    """Load a single asset/TF from parquet into numpy arrays."""
    symbol = f"{asset}USDT"
    path = data_dir / f"{symbol}_{tf}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            return None
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    timestamps = df.index.values.astype("datetime64[ns]").astype("int64")
    pvt = _compute_pvt(close, volume)

    return AssetData(
        close=close, high=high, low=low, open_=open_,
        volume=volume, timestamps=timestamps, pvt=pvt,
    )


def load_all_assets(
    assets: list[str] | None = None,
    timeframes: list[str] | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[str, dict[str, AssetData]]:
    """Load all assets × timeframes into memory. Returns {asset: {tf: AssetData}}."""
    assets = assets or ASSETS
    timeframes = timeframes or TIMEFRAMES

    all_data: dict[str, dict[str, AssetData]] = {}
    total_bars = 0
    loaded_count = 0

    for asset in assets:
        all_data[asset] = {}
        for tf in timeframes:
            ad = load_asset_tf(asset, tf, data_dir)
            if ad is not None:
                all_data[asset][tf] = ad
                total_bars += len(ad.close)
                loaded_count += 1

    logger.info(
        "Loaded %d asset/TF combinations (%d assets), %d total bars (%.1f MB est.)",
        loaded_count, len([a for a in all_data if all_data[a]]),
        total_bars, total_bars * 8 * 6 / 1e6,
    )
    return all_data


def ts_to_ns(ts: pd.Timestamp) -> int:
    """Convert Timestamp to int64 nanoseconds."""
    return int(ts.value)


def find_bar_index(timestamps: np.ndarray, ts_ns: int) -> int:
    """Find index of last bar at or before ts_ns."""
    idx = np.searchsorted(timestamps, ts_ns, side="right") - 1
    return max(0, idx)


def slice_window(ad: AssetData, end_ns: int, n_bars: int) -> dict[str, np.ndarray]:
    """Slice last n_bars ending at or before end_ns."""
    idx = find_bar_index(ad.timestamps, end_ns)
    start = max(0, idx - n_bars + 1)
    end = idx + 1
    return {
        "close": ad.close[start:end],
        "high": ad.high[start:end],
        "low": ad.low[start:end],
        "open": ad.open_[start:end],
        "volume": ad.volume[start:end],
        "pvt": ad.pvt[start:end],
        "timestamps": ad.timestamps[start:end],
    }


def build_scan_dataframes(
    asset_data: dict[str, AssetData],
    current_ns: int,
    scan_tfs: list[str] = None,
) -> dict[str, pd.DataFrame]:
    """Build 7-day DataFrames for ALL 4 scan timeframes."""
    scan_tfs = scan_tfs or ["5m", "30m", "1h", "4h"]
    result = {}
    for tf in scan_tfs:
        ad = asset_data.get(tf)
        if ad is None:
            continue
        n = BARS_7D[tf]
        win = slice_window(ad, current_ns, n)
        if len(win["close"]) < 20:
            continue
        df = pd.DataFrame({
            "open": win["open"],
            "high": win["high"],
            "low": win["low"],
            "close": win["close"],
            "volume": win["volume"],
        })
        df.index = pd.to_datetime(win["timestamps"], utc=True)
        df.index.name = "timestamp"
        result[tf] = df
    return result


def build_htf_dataframe(
    asset_data: dict[str, AssetData],
    current_ns: int,
    n_bars: int = 200,
) -> pd.DataFrame | None:
    """Build 4H DataFrame for HTF filter."""
    ad = asset_data.get("4h")
    if ad is None:
        return None
    win = slice_window(ad, current_ns, n_bars)
    if len(win["close"]) < 56:
        return None
    df = pd.DataFrame({
        "open": win["open"],
        "high": win["high"],
        "low": win["low"],
        "close": win["close"],
        "volume": win["volume"],
    })
    df.index = pd.to_datetime(win["timestamps"], utc=True)
    df.index.name = "timestamp"
    return df


def get_1h_timestamps(
    all_data: dict[str, dict[str, AssetData]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> np.ndarray:
    """Get 1h timestamp array within [start, end] range."""
    for asset_data in all_data.values():
        ad = asset_data.get("1h")
        if ad is not None:
            start_ns = ts_to_ns(start_ts)
            end_ns = ts_to_ns(end_ts)
            mask = (ad.timestamps >= start_ns) & (ad.timestamps <= end_ns)
            return ad.timestamps[mask]
    return np.array([], dtype="int64")


def get_sub_bars(
    ad: AssetData,
    prev_1h_ns: int,
    current_1h_ns: int,
) -> list[int]:
    """Get indices of all bars in (prev_1h_ns, current_1h_ns]."""
    mask = (ad.timestamps > prev_1h_ns) & (ad.timestamps <= current_1h_ns)
    return np.where(mask)[0].tolist()


def generate_wfv_folds(
    start: pd.Timestamp = WFV_START,
    end: pd.Timestamp = WFV_END,
    n_folds: int = WFV_FOLDS,
    embargo: int = EMBARGO_BARS,
) -> list[WFVFold]:
    """Generate walk-forward validation folds (8 folds, 40/30/30 split)."""
    total_hours = int((end - start).total_seconds() / 3600)
    fold_bars = int(total_hours * 0.60)
    train_bars = int(fold_bars * WFV_TRAIN_FRAC)
    val_bars = int(fold_bars * WFV_VAL_FRAC)
    test_bars = int(fold_bars * WFV_TEST_FRAC)
    stride = (total_hours - (train_bars + 2 * embargo + val_bars + test_bars)) // (n_folds - 1)

    folds = []
    for k in range(n_folds):
        offset_hours = k * stride
        t_start = start + pd.Timedelta(hours=offset_hours)
        t_end = t_start + pd.Timedelta(hours=train_bars)
        v_start = t_end + pd.Timedelta(hours=embargo)
        v_end = v_start + pd.Timedelta(hours=val_bars)
        te_start = v_end + pd.Timedelta(hours=embargo)
        te_end = te_start + pd.Timedelta(hours=test_bars)
        te_end = min(te_end, end)
        folds.append(WFVFold(
            fold_id=k,
            train_start=t_start, train_end=t_end,
            val_start=v_start, val_end=v_end,
            test_start=te_start, test_end=te_end,
        ))
    return folds
