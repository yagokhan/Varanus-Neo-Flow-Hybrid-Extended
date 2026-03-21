"""
config/groups.py — Asset group definitions and group-specific configuration.

33 assets divided into 3 groups for differentiated optimization:
  Group A [Majors]:       High-cap, high-liquidity, lower relative volatility
  Group B [Tech & AI]:    Mid-cap tech/infrastructure, moderate volatility
  Group C [Momentum & Meme]: High-beta, meme/momentum coins, highest volatility
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# Asset Groups
# ═══════════════════════════════════════════════════════════════════════════════

GROUP_A_MAJORS = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "ADA", "LINK", "DOT", "LTC", "BCH", "TRX",
]

GROUP_B_TECH_AI = [
    "FET", "RENDER", "NEAR", "AR", "GRT",
    "INJ", "THETA", "FIL", "ATOM", "ICP", "STX",
]

GROUP_C_MOMENTUM_MEME = [
    "PEPE", "TIA", "WIF", "BONK", "SUI",
    "SEI", "APT", "SHIB", "DOGE", "FLOKI", "OP",
]

ALL_ASSETS = GROUP_A_MAJORS + GROUP_B_TECH_AI + GROUP_C_MOMENTUM_MEME

# Mapping asset → group name
ASSET_TO_GROUP: dict[str, str] = {}
for _a in GROUP_A_MAJORS:
    ASSET_TO_GROUP[_a] = "A"
for _a in GROUP_B_TECH_AI:
    ASSET_TO_GROUP[_a] = "B"
for _a in GROUP_C_MOMENTUM_MEME:
    ASSET_TO_GROUP[_a] = "C"

GROUP_NAMES = {"A": "Majors", "B": "Tech & AI", "C": "Momentum & Meme"}

# Binance symbols (append USDT)
ALL_SYMBOLS = [f"{a}USDT" for a in ALL_ASSETS]
ASSET_FROM_SYMBOL = {f"{a}USDT": a for a in ALL_ASSETS}


# ═══════════════════════════════════════════════════════════════════════════════
# Group-Specific Thresholds (defaults — overridden by optimization)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GroupThresholds:
    """Optimizable entry gate thresholds for one asset group."""
    # Entry gates
    min_confidence: float = 0.80      # Min Pearson |R| to enter
    min_xgb_score: float = 0.55       # Min XGBoost probability to confirm
    # Trail sensitivity
    trail_buffer: float = 0.50        # Adaptive trail = midline ± buffer * std_dev
    # PVT gate
    min_pvt_r: float = 0.75           # Min PVT |R|
    # Combined gate
    combined_gate: float = 0.80       # Suppress opposing |R| above this
    # Risk
    hard_sl_mult: float = 2.5         # Hard SL = entry ± ATR × mult
    exhaust_r: float = 0.475          # Close if |R| drops below this
    pos_frac: float = 0.05            # Position size fraction of capital


# Default thresholds by group (calibrated to volatility profile)
DEFAULT_THRESHOLDS: dict[str, GroupThresholds] = {
    "A": GroupThresholds(
        min_confidence=0.82,
        min_xgb_score=0.55,
        trail_buffer=0.60,       # Tighter trail for lower-vol majors
        min_pvt_r=0.75,
        combined_gate=0.80,
        hard_sl_mult=2.25,       # Tighter SL for majors
        exhaust_r=0.475,
        pos_frac=0.06,           # Larger position (lower risk)
    ),
    "B": GroupThresholds(
        min_confidence=0.80,
        min_xgb_score=0.53,
        trail_buffer=0.50,       # Standard trail for mid-vol
        min_pvt_r=0.70,
        combined_gate=0.75,
        hard_sl_mult=2.50,
        exhaust_r=0.45,
        pos_frac=0.05,
    ),
    "C": GroupThresholds(
        min_confidence=0.78,
        min_xgb_score=0.50,
        trail_buffer=0.40,       # Wider trail for high-vol memes
        min_pvt_r=0.65,
        combined_gate=0.70,
        hard_sl_mult=3.00,       # Wider SL for volatile assets
        exhaust_r=0.425,
        pos_frac=0.04,           # Smaller position (higher risk)
    ),
}


def get_group(asset: str) -> str:
    """Return group letter for an asset."""
    return ASSET_TO_GROUP.get(asset, "B")  # default to B if unknown


def get_thresholds(asset: str, overrides: dict[str, GroupThresholds] | None = None) -> GroupThresholds:
    """Return thresholds for an asset's group."""
    group = get_group(asset)
    if overrides and group in overrides:
        return overrides[group]
    return DEFAULT_THRESHOLDS[group]
