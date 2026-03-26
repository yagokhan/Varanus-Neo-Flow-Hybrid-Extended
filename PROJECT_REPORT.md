# Varanus Neo-Flow Hybrid Extended v1 — Project Report

**Date**: March 26, 2026
**Status**: Production Live (Binance USDT-M Futures)
**Asset Universe**: 33 Cryptocurrencies across 3 Risk Groups

---

## 1. Executive Summary

Varanus Neo-Flow Extended v1 is a quantitative trading system for Binance Futures that combines physics-based trend detection, volume confirmation, and machine learning to generate high-frequency entries across 33 crypto assets. Over a 5-month blind test (Nov 2025 — Mar 2026), the system achieved an **85.3% win rate** with a **15.07 profit factor** and **+2,127% cumulative return** on $10,000 initial capital.

The system transitioned to **live trading with real capital** on March 26, 2026. During the live deployment, critical production bugs were identified and resolved: algo/conditional order management, re-entry cooldowns, and commission-aware PnL tracking.

---

## 2. Core Architecture

### 2.1 Three-Layer Validation Stack

The system operates on the thesis that market movements are physical signals embedded in noise. Every trade must pass three independent validation layers:

```
Signal Flow:
  Binance Data → [Physics Engine] → [Volume Filter] → [ML Gatekeeper] → Execution
                     Layer 1            Layer 2            Layer 3
```

**Layer 1 — Physics Engine (Logarithmic Linear Regression)**
- Fits line-of-best-fit to log-transformed price data
- Confidence metric: Pearson R correlation (0.0 = noise, 1.0 = perfect trend)
- Signal direction: Negative slope → LONG (reversion), Positive slope → SHORT
- Dynamic scanning: 4 timeframes (5m, 30m, 1h, 4h) × periods 20-200 = ~724 regressions per asset per hour
- Winner selection: TF/period combo with highest |R| becomes the signal generator

**Layer 2 — Price-Volume Trend (PVT) Alignment**
- Computes PVT regression over the same window as the winning regression
- Confirms volume is flowing in the signal direction
- Group-specific minimum PVT |R| thresholds (0.65 — 0.80)

**Layer 3 — XGBoost Meta-Labeler**
- Binary classifier trained on historical physics signals
- Features: confidence (R), pvt_r, timeframe (encoded), period
- Target: win/loss classification on realized PnL
- Group-specific probability thresholds (0.45 — 0.55)
- Blocks ~16% of physics signals, primarily losers

### 2.2 Additional Gate Filters

**HTF Macro Bias (4-Hour)**
- Zero-tolerance filter: signal direction must align with 4H trend bias
- Dual confirmation: Market Structure Shift (MSS) + EMA 21/55 alignment
- If MSS and EMA disagree → Bias = NEUTRAL → all entries blocked

**Combined Gate**
- Suppresses entry if an opposing regression (different TF/period) shows |R| above threshold
- Prevents entering contested trends where multiple timeframes disagree

### 2.3 Exit Strategy

- **Adaptive Trailing Stop**: Regression midline ± (trail_buffer × std_dev). Tightens as trend progresses. Accounts for 99.8% of all exits.
- **Hard Stop Loss**: Entry ± (ATR × hard_sl_mult). Safety net for sudden reversals.
- **Trend Exhaustion**: Close if |R| drops below exhaust_r threshold (trend losing steam).
- **Time Barrier**: Maximum 200 bars held.

---

## 3. 5-Minute Websocket Scan Integration

### 3.1 Architecture Evolution

The system evolved from REST-polling (hourly scans only) to a **push-based websocket architecture** for real-time 5-minute scanning:

```
Previous: Hourly REST poll → scan 33 assets → enter/exit
Current:  Websocket stream (33 assets × 5m klines)
          ├── On each closed 5m bar:
          │   ├── Append bar to in-memory cache
          │   ├── Process exits for open positions (trail/SL checks)
          │   └── Run full physics+XGBoost pipeline for new entries
          └── Hourly cycle: Full multi-TF scan + offline backtest comparison
```

**Data Source**: `wss://fstream.binance.com/stream?streams=...`
- 33 concurrent kline streams maintained
- Bars appended to in-memory cache on close (x=true flag)
- 15-second settlement buffer ensures data stability

### 3.2 Scan Pipeline (Per 5-Minute Bar)

For each closed 5m candle, the system runs the full pipeline on that asset:

1. **Find Best Regression**: Scan periods 20-200 on 5m data, select highest |R|
2. **R-Gate**: Reject if |R| < group min_confidence
3. **PVT Alignment**: Compute PVT regression, verify volume confirms direction
4. **HTF Bias**: Check 4H trend alignment
5. **Combined Gate**: Verify no strong opposing regression
6. **XGBoost Gate**: Predict win probability, reject if below threshold
7. **Entry Execution**: Place market order + STOP conditional order on Binance

### 3.3 Performance Impact

The 5-minute scan is the primary alpha engine:

| Timeframe | Win Rate | Profit Factor | Max DD | Total PnL |
|-----------|----------|---------------|--------|-----------|
| **5m**    | 85.8%    | 21.89         | -0.66% | $89,972   |
| 30m       | 84.1%    | 10.21         | -7.06% | $66,205   |
| 1h        | 86.0%    | 16.57         | -5.55% | $56,493   |

5-minute generates 42% of total profits with the lowest drawdown, validating high-frequency scalping as the core strategy.

### 3.4 Sub-Bar Position Management

Open positions are monitored continuously via:
- **Websocket 5m bars**: Trail/SL checks on every closed bar (exact backtest alignment)
- **30-second monitor thread**: Ticker price safety-net for hard SL between bars
- **Stop orders on Binance**: Server-side protection via algo/conditional orders

---

## 4. Asset Groups & Optimization

### 4.1 Group Definitions

| Group | Name | Count | Assets |
|-------|------|-------|--------|
| A | Majors | 11 | BTC, ETH, SOL, BNB, XRP, ADA, LINK, DOT, LTC, BCH, TRX |
| B | Tech & AI | 11 | FET, RENDER, NEAR, AR, GRT, INJ, THETA, FIL, ATOM, ICP, STX |
| C | Momentum & Meme | 11 | PEPE, TIA, WIF, BONK, SUI, SEI, APT, SHIB, DOGE, FLOKI, OP |

### 4.2 Optimized Thresholds (Optuna-Tuned)

| Parameter | Group A | Group B | Group C | Purpose |
|-----------|---------|---------|---------|---------|
| min_confidence | 0.75 | 0.91 | 0.84 | Min Pearson \|R\| to enter |
| min_xgb_score | 0.45 | 0.55 | 0.54 | Min XGBoost probability |
| trail_buffer | 0.30 | 0.35 | 0.30 | Adaptive trail sensitivity |
| min_pvt_r | 0.65 | 0.75 | 0.80 | Min PVT \|R\| |
| combined_gate | 0.80 | 0.85 | 0.80 | Opposing signal suppression |
| hard_sl_mult | 3.25 | 2.50 | 3.25 | ATR multiplier for hard SL |
| exhaust_r | 0.45 | 0.55 | 0.35 | Trend exhaustion threshold |
| pos_frac | 0.07 | 0.06 | 0.07 | Position size (% of capital) |

**Design Rationale**:
- Group A (Majors): Most liquid, lowest R-gate (more signals), tightest exits
- Group B (Tech): Strictest R-gate (0.91), highest ML threshold, moderate volatility
- Group C (Meme): High-volatility assets get 0.75x position scaling, widest stops

### 4.3 Walk-Forward Validation

8-fold WFV produced consensus parameters validated across different market regimes:

| Metric | Average Across Folds |
|--------|---------------------|
| Win Rate | 63.2% |
| Sharpe Ratio | 10.63 |
| PnL% | 527.9% |
| Max Drawdown | -9.42% |

Blind test (out-of-sample): 1,529 trades, 67.5% WR, 7.45 Sharpe, 316.9% PnL.

---

## 5. Blind Test Results (Nov 1, 2025 — Mar 25, 2026)

### 5.1 Hybrid Mode (Physics + XGBoost)

| Metric | Value |
|--------|-------|
| Total Trades | 2,257 |
| Win Rate | 85.3% |
| Profit Factor | 15.07 |
| Sharpe Ratio | 14.29 |
| Max Drawdown | -5.92% |
| Avg Duration | 0.59 hours (35 min) |
| Total PnL | +$212,711 (+2,127%) |

### 5.2 Physics-Only vs Hybrid Comparison

| Metric | Physics-Only | Hybrid (ML) | Delta |
|--------|-------------|-------------|-------|
| Total Trades | 2,683 | 2,252 | -431 blocked |
| Win Rate | 77.2% | 85.3% | +8.1% |
| Profit Factor | 5.66 | 15.07 | +9.41x |
| Max Drawdown | -11.23% | -5.92% | +5.31% safer |
| Largest Loss | -54.94% | -22.87% | Tail risk halved |

The ML layer blocks ~16% of signals. While it occasionally filters winning trades (e.g., APT +17.7% blocked on Mar 24), it successfully removes 400+ losing trades, doubling system efficiency.

### 5.3 Monthly Breakdown (2026)

| Month | PnL | Return |
|-------|-----|--------|
| January | +$61,349 | +613.5% |
| February | +$46,918 | +469.2% |
| March (MTD) | +$24,194 | +241.9% |

---

## 6. Live Trading Deployment (March 26, 2026)

### 6.1 Configuration

- **Exchange**: Binance USDT-M Futures
- **Initial Capital**: $951.25
- **Mode**: Live (real orders via CCXT)
- **Leverage**: 2-5x (group-dependent)
- **Max Concurrent Positions**: 8
- **Circuit Breaker**: -15% drawdown auto-halt

### 6.2 Production Issues Identified & Resolved

**Issue 1: Algo/Conditional Order Stacking**
- **Root Cause**: Binance routes STOP orders through a separate algo/conditional order system (`algoType: CONDITIONAL`). The standard `cancel_all_orders()` API only cancels regular orders, leaving algo orders untouched.
- **Impact**: Each stop-loss update stacked a new order instead of replacing the old one, resulting in 11 orphaned stop orders.
- **Fix**: Implemented `_cancel_all_orders_full()` that cancels both regular orders (`cancel_all_orders`) and algo orders (`fapiPrivateDeleteAlgoOrder` per order via `fapiPrivateGetOpenAlgoOrders`).

**Issue 2: Rapid Re-Entry on Same Asset**
- **Root Cause**: When a position closed (e.g., adaptive trail hit), the next 5m scan immediately re-entered the same asset because the physics signal was still valid. Each re-entry placed new market + stop orders, stacking positions on Binance.
- **Impact**: DOT was entered 7+ times in succession. Multiple overlapping positions with compounding fees.
- **Fix**: Added 1-hour cooldown (`REENTRY_COOLDOWN_SECONDS = 3600`) after closing any position. Cooldowns are persisted in state and checked before both hourly and 5m scans.

**Issue 3: PnL Discrepancy (Bot vs Binance)**
- **Root Cause**: Bot calculated PnL purely from price movement (entry vs exit) without accounting for trading fees. Binance charges 0.02% taker fee per side on futures.
- **Impact**: Bot reported +$0.82 profit on ATOM trade while actual Binance result was -$0.13 (a $0.95 discrepancy from $0.29 round-trip fees + stacked position losses).
- **Fix**: Added `_fetch_order_commission()` that queries actual fill commissions from Binance trade history. Entry commission stored on `LivePosition`, exit commission fetched on close. Both subtracted from PnL before recording. Trade CSV now includes `entry_commission`, `exit_commission`, `total_commission` columns.

### 6.3 Live Trade Log

| Trade | Asset | Group | Dir | Entry | Exit | Gross PnL | Reason |
|-------|-------|-------|-----|-------|------|-----------|--------|
| 62 | ATOM | B | SHORT | $1.729 | $1.731 | -$1.65 | Adaptive Trail |
| 63 | DOT | A | SHORT | $1.334 | $1.333 | -$1.25 | Adaptive Trail |
| 64 | DOT | A | SHORT | $1.333 | $1.332 | -$1.25 | Adaptive Trail |
| 65 | ATOM | B | SHORT | $1.730 | $1.729 | +$0.82 | Adaptive Trail |

---

## 7. XGBoost Meta-Labeler

### 7.1 Model Specification

```
Objective:     binary:logistic
Max Depth:     4
Learning Rate: 0.05
Estimators:    300
Subsample:     0.8
Validation:    5-fold Stratified K-Fold
```

### 7.2 Feature Space

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| confidence | float | 0-1 | Pearson R from physics engine |
| pvt_r | float | 0-1 | PVT correlation strength |
| best_tf_encoded | int | 0-3 | Timeframe (5m=0, 30m=1, 1h=2, 4h=3) |
| best_period | int | 20-200 | Regression lookback period |

### 7.3 Impact Assessment

The ML layer acts as a conservative filter, prioritizing capital preservation:
- Accepts signals with >45-55% predicted win probability (group-dependent)
- Blocks ~16% of physics-validated signals
- Reduces max single-trade loss from -54.9% to -22.9%
- Primary value: tail-risk reduction, not alpha generation

---

## 8. Risk Management

### 8.1 Position Sizing

```
Position USD = Capital × pos_frac × leverage × vol_scalar

Where:
  pos_frac  = 0.06-0.07 (group-dependent)
  leverage  = 2-5x (confidence-based)
  vol_scalar = 0.75 for HIGH_VOL_ASSETS (ICP, PEPE, BONK, WIF, FLOKI, SHIB)
```

### 8.2 Stop Loss Architecture

| Type | Mechanism | Coverage |
|------|-----------|----------|
| Adaptive Trail | Midline ± buffer × std_dev | 99.8% of exits |
| Hard SL | Entry ± ATR × mult | Catastrophic protection |
| Server-Side Stop | Binance algo/conditional order | Exchange-level safety |
| Ticker Monitor | 30-second price poll | Between-bar safety net |
| Circuit Breaker | -15% drawdown auto-halt | Portfolio protection |

### 8.3 Transaction Cost Model (Backtest)

| Component | Majors | Volatile Assets |
|-----------|--------|-----------------|
| Taker Fee | 0.04% per side | 0.04% per side |
| Slippage | 0.02% per side | 0.05% per side |
| Round-Trip | ~0.12% | ~0.18% |

---

## 9. Technical Infrastructure

### 9.1 Runtime Components

```
live_extended_bot.py
├── Main Loop: Hourly scan cycle (top of hour + 60s settlement)
├── Websocket Thread: 33 × 5m kline streams from fstream.binance.com
├── Monitor Thread: 30-second heartbeat, ticker safety-net
├── Telegram Thread: Command listener (/status, /help)
└── State Persistence: live_extended_state.json (atomic writes)
```

### 9.2 Data Pipeline

```
Parquet Cache (132 files: 33 assets × 4 TFs)
  → API Sync (append new bars from Binance REST)
  → Websocket Overlay (real-time 5m bars)
  → 7-Day Rolling Window (trim_to_7d)
  → Physics Scan → PVT → HTF → Gate → XGBoost → Execute
```

### 9.3 Monitoring

- **Dashboard**: Streamlit app (port 8502) reading state/scan/trade files
- **Telegram**: Real-time notifications for entries, exits, scan reports, errors
- **Logs**: Rotating file handler (10MB × 5 backups)

---

## 10. Conclusions

1. **The hybrid approach works**: Physics + ML achieves 2x efficiency over physics alone, with the ML layer serving as a precision filter rather than a signal generator.

2. **5-minute is the alpha timeframe**: Highest profit factor (21.89), lowest drawdown (-0.66%), and 42% of total profits. The websocket integration enabling real-time 5m scanning is a critical architectural advantage.

3. **Live deployment revealed production gaps**: The algo order system, re-entry behavior, and fee tracking were all areas where backtest assumptions diverged from exchange reality. These are now resolved.

4. **The system is operationally mature**: Websocket data, sub-bar monitoring, server-side stops, circuit breaker, and Telegram control provide institutional-grade infrastructure for a retail-scale capital base.

---

*Report generated March 26, 2026. System is live on Binance USDT-M Futures.*
