# Varanus Neo-Flow Extended — Latest System Status
**Last Updated:** March 25, 2026

## 1. Extended Blind Test Results
The blind test has been extended to include data up to the latest scan today.

| Metric | Value |
|--------|-------|
| **Period** | Nov 1, 2025 — March 25, 2026 |
| **Total Trades** | 2,252 |
| **Win Rate** | 85.3% |
| **Total PnL** | +$212,711.48 (+2,127.1%) |
| **Sharpe Ratio** | 14.29 |
| **Max Drawdown** | -5.92% |
| **Profit Factor** | 15.07 |

### Monthly Performance (2026)
*   **January:** +$61,349.09 (+613.5%)
*   **February:** +$46,918.14 (+469.2%)
*   **March (MTD):** +$24,193.61 (+241.9%)

## 2. Technical State
*   **Asset Universe:** 33 assets (Groups A, B, C)
*   **Optimization:** Using optimized thresholds from `config/optimized_thresholds.json`.
*   **XGBoost:** Meta-labeler active (`models/meta_xgb.json`).
*   **Data:** All 132 Parquet files synced up to March 24/25, 2026.
*   **Active Log:** Latest trades saved in `extended_trades.csv`.

## 3. Quick-Start Commands
To skip full re-analysis and perform targeted tasks:

```bash
# Update data only
python3 data_fetcher.py

# Run backtest for the last 7 days only (Fast)
python3 run_backtest.py --start 2026-03-18 --end 2026-03-25 --csv

# Check live bot status
python3 live_extended_bot.py --status
```
