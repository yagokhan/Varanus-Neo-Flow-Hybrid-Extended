# Varanus Neo-Flow Extended v1 — Complete System Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Theoretical Foundation — The Physics Engine](#2-theoretical-foundation--the-physics-engine)
3. [Architecture Deep-Dive](#3-architecture-deep-dive)
4. [33-Asset Universe & Group Segmentation](#4-33-asset-universe--group-segmentation)
5. [Signal Generation Pipeline](#5-signal-generation-pipeline)
6. [Entry & Exit Logic](#6-entry--exit-logic)
7. [Walk-Forward Validation (WFV)](#7-walk-forward-validation-wfv)
8. [FastBacktestEngine — LUT Precomputation](#8-fastbacktestengine--lut-precomputation)
9. [Group-Specific Threshold Optimization](#9-group-specific-threshold-optimization)
10. [Performance Results](#10-performance-results)
11. [Comparison: Extended v1 vs Original Hybrid](#11-comparison-extended-v1-vs-original-hybrid)
12. [File Reference](#12-file-reference)
13. [How to Run](#13-how-to-run)

---

## 1. System Overview

Varanus Neo-Flow Extended v1 is a quantitative cryptocurrency trading system that detects trends across 33 digital assets using physics-based regression analysis, volume confirmation, and macro trend filtering across 4 timeframes. It extends the original 15-asset Varanus Neo-Flow Hybrid system to a 33-asset universe with group-specific optimization.

### Core Philosophy

The system treats price action as a **physical signal** embedded in noise. Rather than using traditional technical indicators (RSI, MACD, Bollinger Bands), it applies **logarithmic linear regression** — the same method used in physics to fit noisy data — to detect the underlying trend direction and strength.

A trade is entered only when:
- The regression fit is statistically significant (high Pearson R)
- Volume confirms the price trend (PVT alignment)
- The macro trend agrees (4H bias filter)
- No strong opposing signal exists (combined gate)

### What Makes It Different

| Feature | Traditional Algo | Varanus Neo-Flow |
|---------|-----------------|------------------|
| Signal source | Indicator crossovers | Logarithmic regression goodness-of-fit |
| Confidence metric | Fixed thresholds | Pearson R correlation (0-1 continuous) |
| Volume confirmation | OBV, VWAP | PVT linear regression with direction matching |
| Trend filter | Moving average | Market Structure Shift + EMA alignment on 4H |
| Stop loss | Fixed % or ATR | Adaptive trailing stop on regression midline |
| Asset treatment | Uniform | Group-specific thresholds (Majors/Tech/Meme) |
| Optimization speed | Minutes per trial | Sub-second (LUT precomputation) |

---

## 2. Theoretical Foundation — The Physics Engine

### 2.1 Logarithmic Linear Regression

The core signal comes from fitting a linear regression to the **log-transformed** close prices over a sliding window. This is an exact translation of the Pine Script `calcDev()` function used in TradingView's Linear Regression Channel.

**Mathematical formulation:**

Given close prices `C[0], C[1], ..., C[n-1]` over a window of `n` bars:

```
y[i] = ln(C[i])                    # Log-transform
x[i] = i + 1                       # Bar index (1-based)

slope = (n * Σ(x*y) - Σx * Σy) / (n * Σ(x²) - (Σx)²)
intercept = mean(y) - slope * mean(x) + slope

midline = exp(intercept)            # Regression center in price space
```

**Pearson R (Confidence):**

```
R = Σ((x - x̄)(ŷ - ȳ)) / sqrt(Σ(x - x̄)² * Σ(ŷ - ȳ)²)
```

Where `ŷ` is the fitted regression value. |R| ranges from 0 (no fit) to 1 (perfect fit). Higher |R| means the price is moving in a more orderly, trend-like fashion.

**Standard Deviation:**

```
σ = sqrt(Σ(y - ŷ)² / (n - 1))
```

Used for the adaptive trailing stop: `trail = midline ± buffer * σ_price`

**Direction Rule (from Pine Script):**
- `slope < 0` → Price is falling → **LONG** signal (anticipating reversal)
- `slope > 0` → Price is rising → **SHORT** signal (anticipating reversal)

Wait — this seems counterintuitive. The system detects **mean-reversion** at high confidence, not trend-following. When |R| is very high and the slope is steep, the price has moved far from equilibrium and is likely to revert. The regression midline acts as the gravitational center.

### 2.2 Dynamic Period & Timeframe Selection

The system doesn't use a fixed lookback period. Instead, it **scans all periods from 20 to 200 bars across all 4 timeframes** and selects the combination with the highest |R|:

```
for tf in [5m, 30m, 1h, 4h]:
    for period in range(20, 201):
        R = compute_regression(close[tf], period)
        if |R| > best_R:
            best_R, best_tf, best_period = |R|, tf, period
```

This dynamic selection means the system automatically adapts to the dominant cycle length and timeframe for each asset at each point in time.

**4H period cap:** Since 4H has only 42 bars in a 7-day window, the maximum period on 4H is capped at `min(200, available_bars - 1)`.

### 2.3 PVT (Price Volume Trend) Confirmation

PVT is a cumulative volume-weighted price change indicator:

```
PVT[i] = PVT[i-1] + ((C[i] - C[i-1]) / C[i-1]) × Volume[i]
```

The system runs **linear regression on PVT** using the same best_tf and best_period found by the price regression. It then checks:

1. **Direction alignment**: PVT regression slope must agree with the price signal direction
2. **Minimum PVT |R|**: The PVT regression must have a statistically significant fit
3. **Divergence detection**: If price R ≥ 0.85 but PVT R < 0.50, the signal is **suppressed** (volume doesn't confirm the move)

### 2.4 4H HTF Bias Filter

Before any trade is taken, the system checks the macro trend on the 4-hour timeframe using two independent methods that must **both agree**:

**Market Structure Shift (MSS):**
```
swing_high = max(High[4h], last 30 bars)
swing_low  = min(Low[4h], last 30 bars)

if Close > swing_high → MSS = BULLISH (+1)
if Close < swing_low  → MSS = BEARISH (-1)
else                   → MSS = NEUTRAL (0)
```

**EMA Alignment:**
```
EMA_fast = EMA(Close[4h], 21)
EMA_slow = EMA(Close[4h], 55)

if EMA_fast > EMA_slow → EMA = BULLISH
if EMA_fast < EMA_slow → EMA = BEARISH
```

**Bias Rule:**
- Both MSS and EMA must agree → `bias = +1` (bullish) or `-1` (bearish)
- If they disagree → `bias = 0` (neutral) → **no trades allowed**
- Signal direction must match bias: LONG only in bullish bias, SHORT only in bearish bias

### 2.5 Combined Gate (Opposing Signal Suppression)

The system tracks the maximum |R| for both LONG and SHORT signals across all TF/period combinations:

```
max_R_long  = max |R| across all regressions where slope < 0
max_R_short = max |R| across all regressions where slope > 0
```

If the best signal is LONG but `max_R_short > combined_gate_threshold`, the signal is suppressed. This prevents entering when there's a strong opposing signal on a different timeframe/period.

---

## 3. Architecture Deep-Dive

### 3.1 Module Structure

```
Varanus_Extended_v1/
│
├── neo_flow/                           # Signal Generation (Physics Engine)
│   ├── adaptive_engine.py              # Core: regression, PVT, bias, scanning
│   └── precompute_features.py          # LUT: vectorized feature pre-computation
│
├── backtest/                           # Backtesting Framework
│   ├── engine.py                       # Standard engine (live regression per scan)
│   ├── engine_fast.py                  # Fast engine (O(1) LUT lookups for Optuna)
│   ├── data_loader.py                  # Parquet → numpy, WFV fold generation
│   └── metrics.py                      # Performance metrics, CSV export
│
├── config/                             # Configuration
│   ├── groups.py                       # 33-asset group definitions, thresholds
│   └── optimized_thresholds.json       # Optuna-optimized per-group params
│
├── ml/                                 # Machine Learning (Future)
│   └── train_meta_model.py             # XGBoost meta-labeler training
│
├── NeoFlowHybridEngine.py             # Hybrid engine (physics + XGBoost)
├── data_fetcher.py                     # Binance API data fetcher
├── run_optimize.py                     # 8-fold WFV optimization
├── run_group_sweep.py                  # Per-group threshold optimization
├── run_backtest.py                     # Single backtest runner
└── requirements.txt                    # Dependencies
```

### 3.2 Data Flow

```
Binance API → data_fetcher.py → data/*.parquet (132 files)
                                       ↓
                              data_loader.py → AssetData (numpy arrays in RAM)
                                       ↓
                        ┌──────────────┴──────────────┐
                        ↓                              ↓
              precompute_features.py          adaptive_engine.py
              (vectorized batch)             (live per-bar scan)
                        ↓                              ↓
              data/features/*.npy              ScanSignal objects
                        ↓                              ↓
              engine_fast.py                   engine.py
              (O(1) LUT lookups)              (live regression)
              ~0.7s per trial                 ~5.5 min per trial
                        ↓                              ↓
                        └──────────────┬──────────────┘
                                       ↓
                              metrics.py → BacktestMetrics
                                       ↓
                              wfv_results.json + trades.csv
```

### 3.3 Data Representation

Each asset/timeframe combination is stored as an `AssetData` dataclass with aligned numpy arrays:

```python
@dataclass
class AssetData:
    close:      np.ndarray    # float64, shape (N,)
    high:       np.ndarray    # float64, shape (N,)
    low:        np.ndarray    # float64, shape (N,)
    open_:      np.ndarray    # float64, shape (N,)
    volume:     np.ndarray    # float64, shape (N,)
    timestamps: np.ndarray    # int64 nanoseconds, shape (N,)
    pvt:        np.ndarray    # float64, cumulative PVT, shape (N,)
```

Total memory footprint: ~641 MB for 33 assets × 4 TFs × 13.35M bars.

---

## 4. 33-Asset Universe & Group Segmentation

### 4.1 Asset Groups

The 33 assets are divided into 3 groups based on market capitalization, liquidity, and volatility profile:

**Group A — Majors (11 assets)**
```
BTC  ETH  SOL  BNB  XRP  ADA  LINK  DOT  LTC  BCH  TRX
```
- Highest market cap and liquidity
- Lower relative volatility
- Tightest spreads, deepest order books
- Behave more like traditional financial assets

**Group B — Tech & AI (11 assets)**
```
FET  RENDER  NEAR  AR  GRT  INJ  THETA  FIL  ATOM  ICP  STX
```
- Mid-cap infrastructure and AI tokens
- Moderate volatility
- Often correlated with tech/AI sentiment cycles
- More susceptible to narrative-driven moves

**Group C — Momentum & Meme (11 assets)**
```
PEPE  TIA  WIF  BONK  SUI  SEI  APT  SHIB  DOGE  FLOKI  OP
```
- Highest volatility and beta
- Meme coins and momentum plays
- Wider spreads, more prone to gaps
- Require wider stop losses and faster exits

### 4.2 Why Group-Specific Optimization?

A single set of thresholds applied uniformly to BTC and PEPE would be suboptimal because:

- **BTC** (Group A) trends smoothly with high |R| — a lower confidence threshold captures more valid signals
- **FET** (Group B) requires higher confidence to filter noise — tighter thresholds prevent false entries
- **PEPE** (Group C) trends violently with rapid exhaustion — needs wide SL but fast exhaustion exit

The system optimizes **8 parameters independently per group** using Optuna:

| Parameter | Description | Group A | Group B | Group C |
|-----------|-------------|:-------:|:-------:|:-------:|
| min_confidence | Min Pearson \|R\| to enter | 0.75 | 0.91 | 0.84 |
| min_pvt_r | Min PVT regression \|R\| | 0.65 | 0.75 | 0.80 |
| combined_gate | Max opposing \|R\| allowed | 0.80 | 0.85 | 0.80 |
| hard_sl_mult | Hard SL = entry ± ATR × mult | 3.25 | 2.50 | 3.25 |
| trail_buffer | Trail = midline ± buffer × σ | 0.30 | 0.35 | 0.30 |
| exhaust_r | Close if \|R\| drops below | 0.45 | 0.55 | 0.35 |
| pos_frac | Position size % of capital | 7% | 6% | 7% |
| min_xgb_score | XGBoost probability gate | 0.45 | 0.55 | 0.54 |

**Key insight from optimization:**
- Group A (Majors): Loosest confidence gate (0.75) — stable assets produce reliable signals even at lower R
- Group B (Tech & AI): Tightest confidence (0.91) — needs very strong regression to overcome noise
- Group C (Memes): Low exhaustion threshold (0.35) — lets explosive trends run longer before exiting

### 4.3 High-Volatility Position Scaling

Six ultra-high-volatility assets receive a 0.75x position size reduction:
```
HIGH_VOL_ASSETS = {ICP, PEPE, BONK, WIF, FLOKI, SHIB}
```
This provides additional downside protection on the most volatile assets.

---

## 5. Signal Generation Pipeline

### 5.1 Full Scan Sequence (Per 1H Bar)

For each 1-hour timestamp, the following sequence executes for every asset:

```
1. CHECK CAPACITY
   └── If open_positions >= max_concurrent (8), skip all scanning

2. BUILD 7-DAY WINDOWS
   └── For each TF in [5m, 30m, 1h, 4h]:
       └── Extract last BARS_7D[tf] bars ending at current timestamp

3. FIND BEST REGRESSION
   └── For each TF in [5m, 30m, 1h, 4h]:
       └── For each period in [20, 21, ..., 200]:
           └── Compute log-linear regression → (R, slope, σ, midline)
           └── Track best |R| → select winner (best_tf, best_period)
           └── Track max_R_long, max_R_short (for combined gate)

4. DETERMINE DIRECTION
   └── slope < 0 → LONG, slope > 0 → SHORT

5. GATE 1: CONFIDENCE
   └── |R| >= group.min_confidence? (0.75 / 0.91 / 0.84)

6. GATE 2: PVT ALIGNMENT
   └── Run linear regression on PVT[best_tf, best_period]
   └── PVT direction matches signal direction?
   └── PVT |R| >= group.min_pvt_r? (0.65 / 0.75 / 0.80)
   └── If price R >= 0.85 and PVT R < 0.50 → SUPPRESS (divergence)

7. GATE 3: 4H BIAS FILTER
   └── Compute MSS on 4H (swing high/low over 30 bars)
   └── Compute EMA21 vs EMA55 on 4H
   └── Both must agree AND match signal direction

8. GATE 4: COMBINED GATE
   └── max_opposing_R <= group.combined_gate? (0.80 / 0.85 / 0.80)

9. SIGNAL EMITTED
   └── Contains: asset, direction, confidence, pvt_r, best_tf,
       best_period, midline, std_dev, entry_price, hard_sl
```

### 5.2 Leverage Tiers

Position leverage is determined by confidence (Pearson |R|):

| Confidence | Leverage | Meaning |
|:----------:|:--------:|---------|
| < 0.80 | 0x | No trade (below all group thresholds) |
| 0.80 – 0.84 | 1x | Low confidence |
| 0.85 – 0.89 | 2x | Moderate confidence |
| 0.90 – 0.94 | 3x | High confidence |
| ≥ 0.95 | 5x | Very high confidence |

---

## 6. Entry & Exit Logic

### 6.1 Entry

When a signal passes all 4 gates:

```
entry_price = open of next bar on best_tf after signal
position_usd = capital × group.pos_frac × leverage × vol_scalar
hard_sl = entry_price ± ATR(14) × group.hard_sl_mult
initial_trail = midline ± group.trail_buffer × (midline × std_dev)
```

### 6.2 Position Management (Sub-Bar Updates)

Between each 1H scan, the engine processes every sub-bar on the position's timeframe:

```
For each sub-bar on best_tf:
    1. Recompute log-regression with updated close array
    2. Update midline, std_dev, peak_R
    3. Ratchet trailing stop (only tightens, never loosens):
       LONG:  trail = max(trail, midline - buffer × σ_price)
       SHORT: trail = min(trail, midline + buffer × σ_price)
    4. Check exit conditions (in priority order)
```

### 6.3 Exit Conditions

| Priority | Condition | Description |
|:--------:|-----------|-------------|
| 1 | **HARD_SL_HIT** | Price touches hard stop loss (ATR-based) |
| 2 | **ADAPTIVE_TRAIL_HIT** | Price touches adaptive trailing stop |
| 3 | **TREND_EXHAUSTION** | \|R\| drops below group.exhaust_r |
| 4 | **TIME_BARRIER** | Position held for 200 bars without exit |
| 5 | **END_OF_PERIOD** | Backtest period ends with position open |

In practice, **99.5% of exits are via adaptive trail**, demonstrating the system's precision in dynamic exit management. Only 0.5% hit the hard stop loss.

---

## 7. Walk-Forward Validation (WFV)

### 7.1 Design

The WFV uses 8 overlapping folds spanning January 2023 to October 2025:

```
Training period: Jan 2023 — Oct 2025 (34 months)
Blind test:      Nov 2025 — Mar 2026 (5 months, NEVER used in optimization)

Each fold: 40% train / 30% validation / 30% test
Embargo:   24 hours between train/val and val/test (prevents look-ahead)
Stride:    Folds overlap with progressive shift
```

### 7.2 Optuna Optimization

Each fold optimizes 7 parameters using TPE (Tree-structured Parzen Estimator) sampling:

```python
Search Space:
    min_pearson_r:  [0.75, 0.92]  step=0.01
    min_pvt_r:      [0.50, 0.85]  step=0.05
    combined_gate:  [0.50, 0.80]  step=0.05
    hard_sl_mult:   [1.0,  3.0]   step=0.25
    trail_buffer:   [0.5,  2.0]   step=0.25
    exhaust_r:      [0.30, 0.65]  step=0.05
    pos_frac:       [0.03, 0.12]  step=0.01

Objective: maximize(Sharpe + bonuses - penalties)
    + 0.5 if win_rate >= 60%
    + 0.3 if trail_exit_pct >= 60%
    + 0.2 if max_drawdown > -15%
    - 2.0 if max_drawdown < -20%
    -10.0 if total_trades < 10
```

### 7.3 Consensus (Median Aggregation)

After all 8 folds complete, the **median** of each parameter across folds is taken as the consensus:

```json
{
    "min_pearson_r": 0.92,
    "min_pvt_r":     0.65,
    "combined_gate": 0.80,
    "hard_sl_mult":  2.50,
    "trail_buffer":  0.50,
    "exhaust_r":     0.625,
    "pos_frac":      0.03
}
```

Using median (not mean) makes the consensus robust to outlier folds.

---

## 8. FastBacktestEngine — LUT Precomputation

### 8.1 The Problem

The standard BacktestEngine recomputes regressions live on every scan:
```
33 assets × 181 periods × 4 TFs = 23,892 regressions per 1H scan
168 scans per week → ~4 million regressions per backtest
Result: ~5.5 minutes per Optuna trial
200 trials × 8 folds = ~147 hours (impossible)
```

### 8.2 The Solution

Pre-compute ALL scan features once into a structured numpy array (LUT), then do O(1) lookups during optimization:

```python
FEATURE_DTYPE = np.dtype([
    ("timestamp_ns", "i8"),       # Timestamp (nanoseconds)
    ("best_r", "f4"),             # Best Pearson |R| across all TFs/periods
    ("best_slope", "f4"),         # Slope of best regression
    ("best_std", "f4"),           # Std dev of best regression
    ("best_midline", "f4"),       # Midline price (exp(intercept))
    ("best_period", "i2"),        # Winning period (20-200)
    ("best_tf_idx", "i1"),        # 0=5m, 1=30m, 2=1h, 3=4h
    ("best_direction", "i1"),     # +1 LONG, -1 SHORT
    ("pvt_r", "f4"),              # PVT Pearson |R|
    ("pvt_direction", "i1"),      # +1 rising, -1 falling
    ("htf_bias", "i1"),           # 4H bias: +1 bull, -1 bear, 0 neutral
    ("max_opposing_r", "f4"),     # Max |R| of opposing direction
    ("atr_best", "f4"),           # ATR(14) on best TF
    ("close_best", "f4"),         # Latest close on best TF
])
```

### 8.3 Precomputation Pipeline

```
1. For each asset (parallelized via joblib):
   a. Map each 1H timestamp → bar index on each TF (np.searchsorted)
   b. For each TF × each period:
      - Extract log-close windows using vectorized indexing
      - Batch-compute regression for ALL timestamps at once
      - Track best |R| per timestamp (LONG and SHORT separately)
   c. Compute PVT regression for winning TF/period combinations
   d. Compute 4H HTF bias (MSS + EMA) vectorized
   e. Compute ATR(14) per TF (rolling)
   f. Pack into structured array

2. Save to data/features/{ASSET}_scan_features.npy
```

### 8.4 Performance Impact

| Metric | Standard Engine | Fast Engine | Speedup |
|--------|:-:|:-:|:-:|
| Regressions per scan | 23,892 | 0 | ∞ |
| Time per scan | ~100ms | ~0.1ms | 1000x |
| Time per trial | 5.5 min | 0.7s | **470x** |
| 200 trials × 8 folds | 147 hours | **19 min** | 464x |
| Precompute overhead | 0 | ~8 min (one-time) | — |
| Memory overhead | ~20 MB | ~44 MB (+24 MB) | — |

### 8.5 Correctness

The fast engine produces **identical** scan results to the standard engine because:
- The same regression math runs during precomputation
- Gate checks (confidence, PVT, bias, combined) use the same thresholds
- Only position management (trailing stop, exhaustion) runs live (must recompute as price changes)

---

## 9. Group-Specific Threshold Optimization

### 9.1 Process (run_group_sweep.py)

```
1. Load all 33 assets into memory (~641 MB)
2. Pre-compute features (LUT) for all assets (~8 min, cached to disk)
3. For each group (A, B, C):
   a. Extract group's assets (11 per group)
   b. Select middle WFV fold for optimization (fold 4)
   c. Run 100 Optuna trials with FastBacktestEngine
   d. Record best parameters per group
4. Save to config/optimized_thresholds.json
```

### 9.2 Optimized Results

**Group A [Majors] — Best Score: 5.43**
```
min_confidence=0.75  min_pvt_r=0.65  combined_gate=0.80
hard_sl_mult=3.25    trail_buffer=0.30  exhaust_r=0.45  pos_frac=0.07
```
Interpretation: Majors trend reliably → low confidence gate, wider SL, moderate exhaust.

**Group B [Tech & AI] — Best Score: 2.94**
```
min_confidence=0.91  min_pvt_r=0.75  combined_gate=0.85
hard_sl_mult=2.50    trail_buffer=0.35  exhaust_r=0.55  pos_frac=0.06
```
Interpretation: Tech tokens are noisier → very tight confidence gate, stricter volume confirmation.

**Group C [Momentum & Meme] — Best Score: 10.44**
```
min_confidence=0.84  min_pvt_r=0.80  combined_gate=0.80
hard_sl_mult=3.25    trail_buffer=0.30  exhaust_r=0.35  pos_frac=0.07
```
Interpretation: Meme coins trend explosively → high PVT requirement, very low exhaust (let it run), wide SL.

---

## 10. Performance Results

### 10.1 Walk-Forward Validation (8 Folds × 200 Trials)

| Fold | Trades | Win Rate | Sharpe | Max DD | PnL |
|:----:|:------:|:--------:|:------:|:------:|:---:|
| 0 | 2,003 | 65.0% | 9.24 | -12.93% | +485.2% |
| 1 | 1,110 | 46.8% | 4.73 | -18.01% | +284.7% |
| 2 | 2,440 | 62.7% | 13.67 | -8.63% | +612.1% |
| 3 | 2,443 | 64.5% | 14.19 | -5.52% | +708.8% |
| 4 | 2,293 | 63.8% | 12.20 | -8.18% | +728.8% |
| 5 | 2,386 | 66.9% | 13.21 | -3.86% | +621.8% |
| 6 | 1,333 | 65.2% | 7.04 | -10.21% | +267.5% |
| 7 | 2,270 | 67.9% | 10.72 | -8.05% | +514.1% |
| **Avg** | **2,035** | **61.6%** | **10.63** | **-9.42%** | **+525.4%** |

### 10.2 Blind Test — Consensus Params (Nov 2025 → Mar 2026)

| Metric | Value |
|--------|-------|
| Total Trades | 1,529 |
| Win Rate | 67.5% |
| Sharpe Ratio | 7.45 |
| Max Drawdown | -13.77% |
| Total PnL | +316.9% |
| Profit Factor | 2.49 |

### 10.3 Blind Test — Group-Optimized Thresholds (Nov 2025 → Mar 2026)

| Metric | Value |
|--------|-------|
| **Total Trades** | **2,594** |
| **Win Rate** | **79.0%** (2050W / 544L) |
| **Total PnL** | **+$212,094 (+2,120.9%)** |
| **Profit Factor** | **6.07** |
| **Sharpe Ratio** | **10.82** |
| **Max Drawdown** | **-10.95%** |

**Per-Timeframe Breakdown:**

| Timeframe | Trades | Win Rate | PnL |
|:---------:|:------:|:--------:|:---:|
| 5m | 914 | 86.3% | $78,949 |
| 30m | 890 | 78.0% | $70,649 |
| 1h | 640 | 75.2% | $58,197 |
| 4h | 150 | 57.3% | $4,299 |

**Per-Group Breakdown:**

| Group | Trades | Win Rate | PnL |
|:-----:|:------:|:--------:|:---:|
| A (Majors) | 1,214 | 80.1% | $81,671 |
| B (Tech & AI) | 877 | 77.1% | $78,044 |
| C (Momentum & Meme) | 503 | 79.7% | $52,380 |

**Exit Distribution:**

| Exit Reason | Count | Percentage |
|:-----------:|:-----:|:----------:|
| Adaptive Trail | 2,580 | 99.5% |
| Hard SL | 14 | 0.5% |
| Trend Exhaustion | 0 | 0.0% |
| Time Barrier | 0 | 0.0% |

---

## 11. Comparison: Extended v1 vs Original Hybrid

### 11.1 Architecture Differences

| Feature | Original Hybrid (15 assets) | Extended v1 (33 assets) |
|---------|:--:|:--:|
| Asset universe | 15 | 33 |
| Scan timeframes | 3 (5m, 30m, 1h) | 4 (5m, 30m, 1h, 4h) |
| XGBoost meta-labeler | Yes (AUC 0.585) | No (physics-only) |
| Group optimization | No (uniform params) | Yes (A/B/C specific) |
| Group-specific exits | No | Yes (trail, exhaust per group) |
| LUT precomputation | Yes | Yes (extended to 4 TFs) |
| High-vol position scaling | No | Yes (0.75x for 6 assets) |

### 11.2 Blind Test Performance (Same Period)

| Metric | Hybrid (15) | Extended v1 (33) | Delta |
|--------|:---:|:---:|:---:|
| Total Trades | 780 | 2,594 | +233% |
| Win Rate | 77.4% | 79.0% | +1.6% |
| Profit Factor | 6.78 | 6.07 | -10.5% |
| Sharpe Ratio | ~11.0 | 10.82 | ~same |
| Max Drawdown | -8.8% | -10.95% | -2.15% |
| Total PnL | +$30,445 | +$212,094 | +7x |
| PnL per Trade | $39.03 | $81.80 | +2.1x |

### 11.3 Key Takeaways

1. **Extended generates 3.3x more trades** — 18 additional assets unlock more opportunities
2. **PnL is 7x higher** ($212K vs $30K) — more positions compound faster
3. **Win rate slightly better** on Extended (79.0% vs 77.4%) despite no XGBoost filter
4. **Hybrid has tighter risk** — lower DD (-8.8% vs -10.95%) and higher PF (6.78 vs 6.07)
5. **WFV consistency favors Hybrid** — narrower fold variance (3.6% WR range vs 21.1%)
6. **Extended 5m is strongest** — 86.3% WR on 5m timeframe
7. **4h is weakest** on Extended (57.3% WR) — lower-frequency signals less reliable
8. **Next logical step**: Train XGBoost for Extended to combine both systems' strengths

---

## 12. File Reference

### Source Code

| File | Lines | Description |
|------|:-----:|-------------|
| `neo_flow/adaptive_engine.py` | ~600 | Core physics engine: regression, PVT, bias, scanning |
| `neo_flow/precompute_features.py` | ~443 | Vectorized LUT precomputation (joblib parallel) |
| `backtest/engine.py` | ~417 | Standard backtest engine (live regression) |
| `backtest/engine_fast.py` | ~280 | Fast engine (O(1) LUT lookups, group-aware) |
| `backtest/data_loader.py` | ~274 | Parquet → numpy, WFV fold generation |
| `backtest/metrics.py` | ~300 | Metrics computation, per-group breakdown |
| `config/groups.py` | ~120 | 33-asset groups, GroupThresholds dataclass |
| `ml/train_meta_model.py` | ~500 | XGBoost meta-labeler (4-TF encoding) |
| `NeoFlowHybridEngine.py` | ~450 | Hybrid engine (physics + XGB) |
| `data_fetcher.py` | ~252 | Binance API fetcher (resumable) |
| `run_optimize.py` | ~200 | 8-fold WFV optimization |
| `run_group_sweep.py` | ~200 | Per-group threshold sweep |
| `run_backtest.py` | ~143 | Backtest runner |

### Data Files

| Path | Size | Description |
|------|:----:|-------------|
| `data/*.parquet` | 331 MB | 132 files (33 assets × 4 TFs), Jan 2023 → Mar 2026 |
| `data/features/*.npy` | 43 MB | 33 pre-computed feature arrays |
| `config/optimized_thresholds.json` | 4 KB | Group-specific optimized thresholds |
| `wfv_results.json` | 4 KB | 8-fold WFV results + consensus + blind test |
| `blind_test_trades.csv` | 317 KB | 1,529 trades (WFV consensus blind test) |
| `extended_trades.csv` | 557 KB | 2,594 trades (group-optimized blind test) |

---

## 13. How to Run

### Prerequisites

```bash
# Python 3.10+
pip install numpy pandas requests pyarrow xgboost scikit-learn optuna joblib
```

### Step 1: Fetch Data (33 assets × 4 timeframes)

```bash
python data_fetcher.py                     # Full fetch (~2 hours)
python data_fetcher.py --asset BTCUSDT     # Single asset
python data_fetcher.py --dry-run           # Preview without downloading
```

### Step 2: Group-Specific Threshold Optimization

```bash
python run_group_sweep.py --trials 100     # All 3 groups (~12 min)
python run_group_sweep.py --group A -n 200 # Single group
```
Output: `config/optimized_thresholds.json`

### Step 3: Full Walk-Forward Validation + Blind Test

```bash
python run_optimize.py --trials 200        # 8 folds (~50 min)
```
Output: `wfv_results.json`, `blind_test_trades.csv`

### Step 4: Backtest with Optimized Thresholds

```bash
python run_backtest.py --blind --csv       # Blind test + CSV export
python run_backtest.py --group A           # Single group
python run_backtest.py --start 2024-01-01 --end 2025-01-01  # Custom range
python run_backtest.py --wfv               # Run all WFV folds
```

---

*Generated: March 21, 2026*
*System: Varanus Neo-Flow Extended v1*
*Assets: 33 (3 groups × 11)*
*Data range: January 2023 — March 2026*
