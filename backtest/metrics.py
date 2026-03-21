"""
backtest/metrics.py — Performance metrics and reporting.

Extended: includes per-group breakdown for 33-asset portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.engine import TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Complete performance summary with group breakdowns."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    avg_winner_pnl_pct: float = 0.0
    avg_loser_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    largest_win_pct: float = 0.0
    largest_loss_pct: float = 0.0

    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    avg_duration_hours: float = 0.0
    avg_bars_held: float = 0.0

    trail_exit_count: int = 0
    hard_sl_exit_count: int = 0
    exhaustion_exit_count: int = 0
    time_barrier_exit_count: int = 0
    end_of_period_exit_count: int = 0
    trail_exit_pct: float = 0.0
    hard_sl_exit_pct: float = 0.0
    exhaustion_exit_pct: float = 0.0
    time_barrier_exit_pct: float = 0.0

    trades_by_tf: dict = field(default_factory=dict)
    winrate_by_tf: dict = field(default_factory=dict)
    pnl_by_tf: dict = field(default_factory=dict)

    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    monthly_returns: dict = field(default_factory=dict)

    # Extended: per-group stats
    trades_by_group: dict = field(default_factory=dict)
    winrate_by_group: dict = field(default_factory=dict)
    pnl_by_group: dict = field(default_factory=dict)


def compute_metrics(
    trades: list[TradeRecord],
    equity_curve: list[float] | None = None,
    initial_capital: float = 10_000.0,
) -> BacktestMetrics:
    """Compute full metrics from trade list with group breakdowns."""
    m = BacktestMetrics()
    completed = [t for t in trades if t.exit_ts is not None]
    m.total_trades = len(completed)
    if m.total_trades == 0:
        return m

    winners = [t for t in completed if t.pnl_usd > 0]
    losers = [t for t in completed if t.pnl_usd <= 0]
    m.winning_trades = len(winners)
    m.losing_trades = len(losers)
    m.win_rate = m.winning_trades / m.total_trades * 100

    m.total_pnl_usd = sum(t.pnl_usd for t in completed)
    m.total_pnl_pct = m.total_pnl_usd / initial_capital * 100

    if winners:
        m.avg_winner_pnl_pct = np.mean([t.pnl_pct for t in winners])
        m.largest_win_pct = max(t.pnl_pct for t in winners)
    if losers:
        m.avg_loser_pnl_pct = np.mean([t.pnl_pct for t in losers])
        m.largest_loss_pct = min(t.pnl_pct for t in losers)

    gross_profit = sum(t.pnl_usd for t in winners)
    gross_loss = abs(sum(t.pnl_usd for t in losers))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    if equity_curve and len(equity_curve) > 1:
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak * 100
        m.max_drawdown_pct = float(drawdown.min())
        returns = np.diff(eq) / eq[:-1]
        if len(returns) > 1 and returns.std() > 0:
            m.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(8760))

    durations = []
    for t in completed:
        if t.entry_ts and t.exit_ts:
            durations.append((t.exit_ts - t.entry_ts).total_seconds() / 3600)
    m.avg_duration_hours = np.mean(durations) if durations else 0.0
    m.avg_bars_held = np.mean([t.bars_held for t in completed])

    for t in completed:
        if t.exit_reason == "ADAPTIVE_TRAIL_HIT":
            m.trail_exit_count += 1
        elif t.exit_reason == "HARD_SL_HIT":
            m.hard_sl_exit_count += 1
        elif t.exit_reason == "TREND_EXHAUSTION":
            m.exhaustion_exit_count += 1
        elif t.exit_reason == "TIME_BARRIER":
            m.time_barrier_exit_count += 1
        elif t.exit_reason == "END_OF_PERIOD":
            m.end_of_period_exit_count += 1

    m.trail_exit_pct = m.trail_exit_count / m.total_trades * 100
    m.hard_sl_exit_pct = m.hard_sl_exit_count / m.total_trades * 100
    m.exhaustion_exit_pct = m.exhaustion_exit_count / m.total_trades * 100
    m.time_barrier_exit_pct = m.time_barrier_exit_count / m.total_trades * 100

    # Per-TF breakdown (now includes 4h)
    for tf in ["5m", "30m", "1h", "4h"]:
        tf_trades = [t for t in completed if t.best_tf == tf]
        m.trades_by_tf[tf] = len(tf_trades)
        tf_winners = [t for t in tf_trades if t.pnl_usd > 0]
        m.winrate_by_tf[tf] = len(tf_winners) / len(tf_trades) * 100 if tf_trades else 0.0
        m.pnl_by_tf[tf] = sum(t.pnl_usd for t in tf_trades)

    # Direction breakdown
    longs = [t for t in completed if t.direction == 1]
    shorts = [t for t in completed if t.direction == -1]
    m.long_trades = len(longs)
    m.short_trades = len(shorts)
    m.long_win_rate = len([t for t in longs if t.pnl_usd > 0]) / len(longs) * 100 if longs else 0.0
    m.short_win_rate = len([t for t in shorts if t.pnl_usd > 0]) / len(shorts) * 100 if shorts else 0.0

    # Monthly returns
    for t in completed:
        if t.exit_ts:
            month_key = t.exit_ts.strftime("%Y-%m")
            m.monthly_returns[month_key] = m.monthly_returns.get(month_key, 0.0) + t.pnl_usd

    # ── Extended: Per-Group Breakdown ─────────────────────────────────────
    for group in ["A", "B", "C"]:
        g_trades = [t for t in completed if t.group == group]
        m.trades_by_group[group] = len(g_trades)
        g_winners = [t for t in g_trades if t.pnl_usd > 0]
        m.winrate_by_group[group] = len(g_winners) / len(g_trades) * 100 if g_trades else 0.0
        m.pnl_by_group[group] = sum(t.pnl_usd for t in g_trades)

    return m


def print_metrics(m: BacktestMetrics, initial_capital: float = 10_000.0):
    """Print a formatted metrics report with group breakdowns."""
    from config.groups import GROUP_NAMES

    print()
    print("=" * 80)
    print("VARANUS NEO-FLOW EXTENDED — BACKTEST RESULTS")
    print("=" * 80)
    print()

    print(f"  Total Trades:        {m.total_trades}")
    print(f"  Win Rate:            {m.win_rate:.1f}%  ({m.winning_trades}W / {m.losing_trades}L)")
    print(f"  Long / Short:        {m.long_trades}L / {m.short_trades}S  "
          f"(WR: {m.long_win_rate:.1f}% / {m.short_win_rate:.1f}%)")
    print()

    print(f"  Total PnL:           ${m.total_pnl_usd:,.2f}  ({m.total_pnl_pct:+.1f}%)")
    print(f"  Avg Winner:          {m.avg_winner_pnl_pct:+.2f}%")
    print(f"  Avg Loser:           {m.avg_loser_pnl_pct:+.2f}%")
    print(f"  Largest Win:         {m.largest_win_pct:+.2f}%")
    print(f"  Largest Loss:        {m.largest_loss_pct:+.2f}%")
    print(f"  Profit Factor:       {m.profit_factor:.2f}")
    print()

    print(f"  Max Drawdown:        {m.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:        {m.sharpe_ratio:.2f}")
    print()

    print(f"  Avg Duration:        {m.avg_duration_hours:.1f}h  ({m.avg_bars_held:.0f} bars)")
    print()

    print("  Exit Distribution:")
    print(f"    Adaptive Trail:    {m.trail_exit_count:>4}  ({m.trail_exit_pct:.1f}%)")
    print(f"    Hard SL:           {m.hard_sl_exit_count:>4}  ({m.hard_sl_exit_pct:.1f}%)")
    print(f"    Trend Exhaustion:  {m.exhaustion_exit_count:>4}  ({m.exhaustion_exit_pct:.1f}%)")
    print(f"    Time Barrier:      {m.time_barrier_exit_count:>4}  ({m.time_barrier_exit_pct:.1f}%)")
    print(f"    End of Period:     {m.end_of_period_exit_count:>4}")
    print()

    print("  Per-TF Breakdown:")
    for tf in ["5m", "30m", "1h", "4h"]:
        n = m.trades_by_tf.get(tf, 0)
        wr = m.winrate_by_tf.get(tf, 0.0)
        pnl = m.pnl_by_tf.get(tf, 0.0)
        print(f"    {tf:>3}:  {n:>4} trades  WR={wr:.1f}%  PnL=${pnl:,.2f}")
    print()

    print("  Per-Group Breakdown:")
    for group in ["A", "B", "C"]:
        n = m.trades_by_group.get(group, 0)
        wr = m.winrate_by_group.get(group, 0.0)
        pnl = m.pnl_by_group.get(group, 0.0)
        label = GROUP_NAMES.get(group, group)
        print(f"    Group {group} [{label:>15}]:  {n:>4} trades  WR={wr:.1f}%  PnL=${pnl:,.2f}")
    print()

    if m.monthly_returns:
        print("  Monthly Returns:")
        for month in sorted(m.monthly_returns.keys()):
            pnl = m.monthly_returns[month]
            pct = pnl / initial_capital * 100
            print(f"    {month}:  ${pnl:>+9,.2f}  ({pct:>+6.1f}%)")
    print()
    print("=" * 80)


def trades_to_csv(trades: list[TradeRecord], path: str):
    """Export trade log to CSV with group column."""
    rows = []
    for t in trades:
        if t.exit_ts is None:
            continue
        dur_hours = (t.exit_ts - t.entry_ts).total_seconds() / 3600 if t.exit_ts and t.entry_ts else 0
        rows.append({
            "trade_id": t.trade_id,
            "asset": t.asset,
            "group": t.group,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "best_tf": t.best_tf,
            "best_period": t.best_period,
            "confidence": t.confidence,
            "pvt_r": t.pvt_r,
            "leverage": t.leverage,
            "position_usd": t.position_usd,
            "hard_sl": t.hard_sl,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
            "duration_hours": round(dur_hours, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "pnl_usd": round(t.pnl_usd, 2),
            "peak_r": round(t.peak_r, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Trade log saved: %s (%d trades)", path, len(rows))
