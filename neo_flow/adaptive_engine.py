"""
neo_flow/adaptive_engine.py — Adaptive Multi-TF Trend Engine (Extended)

Extended from Varanus Neo-Flow Hybrid methodology:
  1. Logarithmic Linear Regression (exact Pine Script calcDev translation)
  2. Linear Regression on PVT for volume confirmation
  3. Multi-TF scanner across 4 timeframes: 5m, 30m, 1h, 4h
  4. PVT alignment gate + volume-price divergence detection
  5. 4H bias filter (MSS + EMA21/55 alignment) — macro trend gate
  6. Adaptive trailing stop (regression midline based)
  7. Combined gate (suppress opposing signals)
  8. Group-aware thresholds (A/B/C asset segmentation)

Key difference from Hybrid:
  - 4h is now BOTH a scan timeframe AND the bias filter
  - Scan searches 4 TFs (5m, 30m, 1h, 4h) for best Pearson R
  - 4h bias filter still operates independently using MSS + EMA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — Extended: 4 scan timeframes
# ═══════════════════════════════════════════════════════════════════════════════

SCAN_TIMEFRAMES = ["5m", "30m", "1h", "4h"]   # 4h now included in scan
HTF_TIMEFRAME   = "4h"                        # Still used for bias filter
PERIOD_MIN      = 20
PERIOD_MAX      = 200
PERIOD_RANGE    = range(PERIOD_MIN, PERIOD_MAX + 1)

# ── Standardized 7-Day Rolling Window ────────────────────────────────────────
ROLLING_WINDOW_DAYS = 7
BARS_7D = {
    "5m":  2016,    # 7 * 24 * 12
    "30m": 336,     # 7 * 24 * 2
    "1h":  168,     # 7 * 24
    "4h":  42,      # 7 * 6
}

# ── Default Thresholds (overridden per-group by GroupThresholds) ─────────────
MIN_PEARSON_R           = 0.80
MIN_PVT_PEARSON_R       = 0.75
PVT_DIVERGENCE_PRICE_R  = 0.85
PVT_DIVERGENCE_WEAK_R   = 0.50
COMBINED_GATE_THRESHOLD = 0.80
TREND_EXHAUST_R         = 0.475
HARD_SL_ATR_MULT        = 2.5
TRAIL_BUFFER_STD        = 0.5
MAX_CONCURRENT          = 8      # Increased for 33-asset universe
MAX_LEVERAGE            = 2.5

# ── EMA periods for HTF filter ──────────────────────────────────────────────
EMA_FAST = 21
EMA_SLOW = 55
MSS_LOOKBACK = 30

# ── High-volatility assets (position size scalar 0.75x) ─────────────────────
HIGH_VOL_ASSETS = {"ICP", "PEPE", "BONK", "WIF", "FLOKI", "SHIB"}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegressionResult:
    """Output of a single log-regression computation."""
    std_dev:   float
    pearson_r: float
    slope:     float
    intercept: float
    midline:   float
    period:    int
    timeframe: str


@dataclass
class PVTResult:
    """Output of PVT linear regression."""
    pearson_r: float
    slope:     float
    direction: int


@dataclass
class ScanSignal:
    """Candidate trade signal from the scanner."""
    asset:        str
    direction:    int
    confidence:   float
    pvt_r:        float
    best_tf:      str
    best_period:  int
    entry_price:  float
    sl_price:     float
    midline:      float
    std_dev:      float
    atr:          float
    group:        str          # "A", "B", or "C"
    regression:   RegressionResult
    pvt:          PVTResult


@dataclass
class ActiveTrade:
    """State of an open position managed by the adaptive trailing stop."""
    asset:           str
    direction:       int
    entry_price:     float
    hard_sl:         float
    trail_sl:        float
    best_trail:      float
    midline:         float
    std_dev:         float
    best_tf:         str
    best_period:     int
    entry_ts:        pd.Timestamp
    group:           str = "B"
    bars_held:       int = 0
    peak_r:          float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Logarithmic Linear Regression — Pine Script calcDev() Translation
# ═══════════════════════════════════════════════════════════════════════════════

def calc_log_regression(
    prices: np.ndarray,
    length: int,
) -> tuple[float, float, float, float]:
    """
    Exact translation of Pine Script calcDev(source, length).

    Slope sign convention (x=1 → newest, x=length → oldest):
      - slope < 0 → prices RISING  → uptrend  → LONG
      - slope > 0 → prices FALLING → downtrend → SHORT
    """
    log_src = np.log(prices[-length:].astype(np.float64)[::-1])

    x = np.arange(1, length + 1, dtype=np.float64)

    sum_x  = x.sum()
    sum_xx = (x * x).sum()
    sum_yx = (x * log_src).sum()
    sum_y  = log_src.sum()

    denom = length * sum_xx - sum_x * sum_x
    slope = (length * sum_yx - sum_x * sum_y) / denom

    average   = sum_y / length
    intercept = average - slope * sum_x / length + slope

    period_1 = length - 1
    regres   = intercept + slope * period_1 * 0.5

    i_vals   = np.arange(length, dtype=np.float64)
    reg_line = intercept + i_vals * slope

    dxt = log_src - average
    dyt = reg_line - regres
    residuals = log_src - reg_line

    sum_dxx = (dxt * dxt).sum()
    sum_dyy = (dyt * dyt).sum()
    sum_dyx = (dxt * dyt).sum()
    sum_dev = (residuals * residuals).sum()

    std_dev = np.sqrt(sum_dev / period_1) if period_1 > 0 else 0.0

    d = np.sqrt(sum_dxx * sum_dyy)
    pearson_r = (sum_dyx / d) if d > 1e-15 else 0.0

    return std_dev, pearson_r, slope, intercept


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Linear Regression (for PVT — no log transform)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_linear_regression(
    values: np.ndarray,
    length: int,
) -> tuple[float, float, float, float]:
    """Linear regression on raw values (not log-transformed)."""
    src = values[-length:].astype(np.float64)[::-1]

    x = np.arange(1, length + 1, dtype=np.float64)

    sum_x  = x.sum()
    sum_xx = (x * x).sum()
    sum_yx = (x * src).sum()
    sum_y  = src.sum()

    denom = length * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0, 0.0, 0.0, 0.0

    slope = (length * sum_yx - sum_x * sum_y) / denom

    average   = sum_y / length
    intercept = average - slope * sum_x / length + slope

    period_1 = length - 1
    regres   = intercept + slope * period_1 * 0.5

    i_vals   = np.arange(length, dtype=np.float64)
    reg_line = intercept + i_vals * slope

    dxt = src - average
    dyt = reg_line - regres
    residuals = src - reg_line

    sum_dxx = (dxt * dxt).sum()
    sum_dyy = (dyt * dyt).sum()
    sum_dyx = (dxt * dyt).sum()
    sum_dev = (residuals * residuals).sum()

    std_dev = np.sqrt(sum_dev / period_1) if period_1 > 0 else 0.0

    d = np.sqrt(sum_dxx * sum_dyy)
    pearson_r = (sum_dyx / d) if d > 1e-15 else 0.0

    return std_dev, pearson_r, slope, intercept


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Price Volume Trend (PVT)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pvt(df: pd.DataFrame) -> np.ndarray:
    """Compute cumulative PVT. PVT[0] = 0."""
    close  = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    pct_change = np.zeros(len(close))
    pct_change[1:] = (close[1:] - close[:-1]) / close[:-1]
    pvt_increments = pct_change * volume
    return np.cumsum(pvt_increments)


def compute_pvt_regression(
    df: pd.DataFrame,
    period: int,
) -> PVTResult:
    """Compute PVT then run linear regression on the last period bars."""
    pvt = compute_pvt(df)
    if len(pvt) < period:
        return PVTResult(pearson_r=0.0, slope=0.0, direction=0)

    _, pearson_r, slope, _ = calc_linear_regression(pvt, period)

    if abs(slope) < 1e-15:
        direction = 0
    elif slope < 0:
        direction = 1
    else:
        direction = -1

    return PVTResult(pearson_r=pearson_r, slope=slope, direction=direction)


def check_pvt_alignment(
    price_direction: int,
    price_abs_r: float,
    pvt: PVTResult,
    min_pvt_r: float = MIN_PVT_PEARSON_R,
) -> tuple[bool, str]:
    """PVT alignment gate + volume-price divergence check."""
    pvt_abs_r = abs(pvt.pearson_r)

    if pvt.direction != price_direction:
        if price_abs_r >= PVT_DIVERGENCE_PRICE_R:
            return False, (
                f"VOLUME-PRICE DIVERGENCE: price |R|={price_abs_r:.4f} "
                f"but PVT direction opposing (pvt_R={pvt.pearson_r:.4f})"
            )
        return False, (
            f"PVT direction mismatch: price={'LONG' if price_direction == 1 else 'SHORT'} "
            f"but PVT={'RISING' if pvt.direction == 1 else 'FALLING' if pvt.direction == -1 else 'FLAT'}"
        )

    if price_abs_r >= PVT_DIVERGENCE_PRICE_R and pvt_abs_r < PVT_DIVERGENCE_WEAK_R:
        return False, (
            f"WEAK VOLUME: price |R|={price_abs_r:.4f} but PVT |R|={pvt_abs_r:.4f} < {PVT_DIVERGENCE_WEAK_R}"
        )

    if pvt_abs_r < min_pvt_r:
        return False, f"PVT too weak: |pvt_R|={pvt_abs_r:.4f} < {min_pvt_r}"

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Multi-TF Period Scanner (now includes 4h)
# ═══════════════════════════════════════════════════════════════════════════════

def trim_to_7d(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Trim to standardized 7-day rolling window."""
    n = BARS_7D.get(tf)
    if n is None or df is None or df.empty:
        return df
    if len(df) <= n:
        return df
    return df.iloc[-n:]


def scan_all_periods(
    close: np.ndarray,
    timeframe: str,
    periods: range = PERIOD_RANGE,
) -> list[RegressionResult]:
    """Compute log-regression for every period on one close array."""
    results: list[RegressionResult] = []
    n = len(close)
    for p in periods:
        if n < p:
            break
        std_dev, pearson_r, slope, intercept = calc_log_regression(close, p)
        results.append(RegressionResult(
            std_dev=std_dev,
            pearson_r=pearson_r,
            slope=slope,
            intercept=intercept,
            midline=np.exp(intercept),
            period=p,
            timeframe=timeframe,
        ))
    return results


def find_best_regression(
    data: dict[str, pd.DataFrame],
) -> tuple[Optional[RegressionResult], list[RegressionResult]]:
    """
    Scan all 4 timeframes × all periods and return the single best result.
    Dynamically searches 5m, 30m, 1h, 4h for highest |Pearson R|.
    """
    all_results: list[RegressionResult] = []

    for tf in SCAN_TIMEFRAMES:
        df = data.get(tf)
        if df is None or df.empty:
            continue
        df_7d = trim_to_7d(df, tf)
        close = df_7d["close"].values.astype(np.float64)
        if len(close) < PERIOD_MIN:
            continue
        # For 4h with only 42 bars, cap the max period
        max_period = min(PERIOD_MAX, len(close))
        period_range = range(PERIOD_MIN, max_period + 1)
        results = scan_all_periods(close, tf, period_range)
        all_results.extend(results)

    if not all_results:
        return None, []

    best = max(all_results, key=lambda r: abs(r.pearson_r))
    return best, all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 4H Trend Filter — MSS + EMA Alignment (Bias Filter)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_mss(df: pd.DataFrame, lookback: int = MSS_LOOKBACK) -> int:
    """Detect Market Structure Shift. Returns +1, -1, or 0."""
    if len(df) < lookback + 1:
        return 0
    window = df.iloc[-(lookback + 1):-1]
    current = df.iloc[-1]
    swing_high = window["high"].max()
    swing_low  = window["low"].min()
    if current["close"] > swing_high:
        return 1
    if current["close"] < swing_low:
        return -1
    return 0


def get_htf_bias(df_4h: pd.DataFrame) -> int:
    """
    Combined 4H higher-timeframe bias: MSS direction + EMA alignment.
    Both MSS and EMA must agree for a non-zero bias.
    No counter-trend trades allowed.
    """
    if df_4h is None or len(df_4h) < max(EMA_SLOW, MSS_LOOKBACK) + 1:
        return 0
    mss = detect_mss(df_4h, MSS_LOOKBACK)
    ema_fast = compute_ema(df_4h["close"], EMA_FAST)
    ema_slow = compute_ema(df_4h["close"], EMA_SLOW)
    ema_bullish = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
    ema_bearish = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])
    if mss == 1 and ema_bullish:
        return 1
    if mss == -1 and ema_bearish:
        return -1
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Combined Gate — Suppress Conflicting Signals
# ═══════════════════════════════════════════════════════════════════════════════

def check_combined_gate(
    direction: int,
    all_results: list[RegressionResult],
    threshold: float = COMBINED_GATE_THRESHOLD,
) -> bool:
    """Returns True if the gate PASSES (no strong opposing signal)."""
    for r in all_results:
        r_direction = -1 if r.slope > 0 else (1 if r.slope < 0 else 0)
        if r_direction != 0 and r_direction != direction:
            if abs(r.pearson_r) > threshold:
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Leverage & Position Sizing
# ═══════════════════════════════════════════════════════════════════════════════

def get_leverage(confidence: float) -> int:
    """Map |pearson_r| confidence to leverage tier."""
    if confidence >= 0.95:
        return 5
    if confidence >= 0.90:
        return 3
    if confidence >= 0.85:
        return 2
    if confidence >= 0.80:
        return 1
    return 0


def compute_position_size(
    capital: float,
    confidence: float,
    asset: str,
    pos_frac: float = 0.05,
) -> tuple[float, int]:
    """pos_usd = capital * pos_frac * leverage * vol_scalar"""
    lev = get_leverage(confidence)
    vol_scalar = 0.75 if asset in HIGH_VOL_ASSETS else 1.0
    pos_usd = capital * pos_frac * lev * vol_scalar
    return pos_usd, lev


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Adaptive Trailing Stop
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trail_sl(
    direction: int,
    midline: float,
    std_dev: float,
    current_trail: float,
    buffer_mult: float = TRAIL_BUFFER_STD,
) -> float:
    """Adaptive trailing stop based on regression midline. Ratchets only."""
    std_dev_price = midline * std_dev
    if direction == 1:
        new_trail = midline - buffer_mult * std_dev_price
        return max(current_trail, new_trail)
    else:
        new_trail = midline + buffer_mult * std_dev_price
        return min(current_trail, new_trail)


def check_trail_hit(
    direction: int,
    trail_sl: float,
    bar: pd.Series,
) -> bool:
    """Check if trailing stop hit during this bar."""
    if direction == 1:
        return float(bar["low"]) <= trail_sl
    else:
        return float(bar["high"]) >= trail_sl


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Hard Stop Loss
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hard_sl(
    entry_price: float,
    atr: float,
    direction: int,
    mult: float = HARD_SL_ATR_MULT,
) -> float:
    """Server-side STOP_MARKET placement (touch-based)."""
    if direction == 1:
        return entry_price - mult * atr
    return entry_price + mult * atr


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Full Asset Scanner (Group-Aware)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_asset(
    asset: str,
    scan_data: dict[str, pd.DataFrame],
    df_4h: pd.DataFrame,
    min_pearson_r: float = MIN_PEARSON_R,
    min_pvt_r: float = MIN_PVT_PEARSON_R,
    combined_gate_threshold: float = COMBINED_GATE_THRESHOLD,
    group: str = "B",
) -> Optional[ScanSignal]:
    """
    Run the full Neo-Flow Extended pipeline for a single asset.
    Searches across 5m, 30m, 1h, 4h for best Pearson R.
    4H bias filter enforced separately (no counter-trend trades).
    """
    # ── Step 1: Find best regression across ALL 4 TFs and periods ────
    best, all_results = find_best_regression(scan_data)

    if best is None:
        return None

    abs_r = abs(best.pearson_r)

    # ── Step 2: Minimum price trend strength (group-specific) ────────
    if abs_r < min_pearson_r:
        return None

    # ── Step 3: Direction from price slope ───────────────────────────
    direction = 1 if best.slope < 0 else -1

    # ── Step 4: PVT alignment gate ───────────────────────────────────
    best_df = scan_data.get(best.timeframe)
    if best_df is None:
        return None
    best_df_7d = trim_to_7d(best_df, best.timeframe)
    pvt = compute_pvt_regression(best_df_7d, best.period)
    pvt_passes, pvt_reason = check_pvt_alignment(direction, abs_r, pvt, min_pvt_r)
    if not pvt_passes:
        return None

    # ── Step 5: 4H trend filter (bias — no counter-trend) ───────────
    htf_bias = get_htf_bias(df_4h)
    if htf_bias == 0:
        return None
    if htf_bias != direction:
        return None

    # ── Step 6: Combined gate ────────────────────────────────────────
    if not check_combined_gate(direction, all_results, combined_gate_threshold):
        return None

    # ── Step 7: Build signal ─────────────────────────────────────────
    entry_price = float(best_df_7d.iloc[-1]["close"])
    atr = float(compute_atr(best_df_7d, 14).iloc[-1])
    if np.isnan(atr) or atr <= 0:
        return None

    sl_price = compute_hard_sl(entry_price, atr, direction)

    signal = ScanSignal(
        asset=asset,
        direction=direction,
        confidence=abs_r,
        pvt_r=abs(pvt.pearson_r),
        best_tf=best.timeframe,
        best_period=best.period,
        entry_price=entry_price,
        sl_price=sl_price,
        midline=best.midline,
        std_dev=best.std_dev,
        atr=atr,
        group=group,
        regression=best,
        pvt=pvt,
    )

    dir_str = "LONG" if direction == 1 else "SHORT"
    logger.info(
        "[%s|%s] SIGNAL  %s  |R|=%.4f  pvt|R|=%.4f  TF=%s  P=%d  "
        "entry=%.6f  SL=%.6f",
        asset, group, dir_str, abs_r, abs(pvt.pearson_r), best.timeframe,
        best.period, entry_price, sl_price,
    )

    return signal


def scan_universe(
    market_data: dict[str, dict[str, pd.DataFrame]],
    open_positions: dict[str, ActiveTrade],
    assets: list[str] | None = None,
    group_thresholds: dict | None = None,
) -> list[ScanSignal]:
    """
    Scan all 33 assets and return qualifying signals.
    Uses group-specific thresholds for entry gates.
    """
    from config.groups import ALL_ASSETS, get_group, get_thresholds

    assets = assets or ALL_ASSETS
    signals: list[ScanSignal] = []

    for asset in assets:
        if asset in open_positions:
            continue

        asset_data = market_data.get(asset)
        if asset_data is None:
            continue

        group = get_group(asset)
        thresholds = get_thresholds(asset, group_thresholds)

        scan_tfs = {tf: asset_data[tf] for tf in SCAN_TIMEFRAMES if tf in asset_data}
        df_4h = asset_data.get("4h")

        if not scan_tfs or df_4h is None:
            continue

        signal = scan_asset(
            asset, scan_tfs, df_4h,
            min_pearson_r=thresholds.min_confidence,
            min_pvt_r=thresholds.min_pvt_r,
            combined_gate_threshold=thresholds.combined_gate,
            group=group,
        )
        if signal is not None:
            signals.append(signal)

    signals.sort(key=lambda s: s.confidence, reverse=True)

    if signals:
        logger.info(
            "Scan complete: %d signal(s) — best: %s [%s] %s |R|=%.4f TF=%s P=%d",
            len(signals), signals[0].asset, signals[0].group,
            "LONG" if signals[0].direction == 1 else "SHORT",
            signals[0].confidence, signals[0].best_tf, signals[0].best_period,
        )
    else:
        logger.info("Scan complete: 0 signals across %d assets", len(assets))

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Trade Management — Update Trailing Stop
# ═══════════════════════════════════════════════════════════════════════════════

def update_active_trade(
    trade: ActiveTrade,
    scan_data: dict[str, pd.DataFrame],
    trail_buffer: float = TRAIL_BUFFER_STD,
) -> ActiveTrade:
    """Recalculate regression on active (TF, period) and update trail."""
    df = scan_data.get(trade.best_tf)
    if df is None:
        return trade
    df_7d = trim_to_7d(df, trade.best_tf)
    if len(df_7d) < trade.best_period:
        return trade
    close = df_7d["close"].values.astype(np.float64)
    std_dev, pearson_r, slope, intercept = calc_log_regression(close, trade.best_period)
    midline = np.exp(intercept)
    trade.midline = midline
    trade.std_dev = std_dev
    trade.peak_r = max(trade.peak_r, abs(pearson_r))
    trade.bars_held += 1
    trade.trail_sl = compute_trail_sl(
        trade.direction, midline, std_dev, trade.trail_sl, trail_buffer,
    )
    return trade


def check_exit_conditions(
    trade: ActiveTrade,
    current_bar: pd.Series,
    current_r: float,
    exhaust_r: float = TREND_EXHAUST_R,
) -> Optional[str]:
    """Check all exit conditions. Returns exit reason or None."""
    if trade.direction == 1 and float(current_bar["low"]) <= trade.hard_sl:
        return "HARD_SL_HIT"
    if trade.direction == -1 and float(current_bar["high"]) >= trade.hard_sl:
        return "HARD_SL_HIT"
    if check_trail_hit(trade.direction, trade.trail_sl, current_bar):
        return "ADAPTIVE_TRAIL_HIT"
    if abs(current_r) < exhaust_r:
        return "TREND_EXHAUSTION"
    if trade.bars_held >= 200:
        return "TIME_BARRIER"
    return None
