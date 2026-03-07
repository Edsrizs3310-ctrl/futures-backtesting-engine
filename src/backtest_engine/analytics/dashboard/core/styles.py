"""
src/backtest_engine/analytics/dashboard/core/styles.py

Shared Plotly styles and color palettes.
"""

from typing import List

# Shared color palette
PALETTE: dict = {
    "combined":  "#2980B9",
    "long":      "#27AE60",
    "short":     "#E74C3C",
    "bench":     "#BDC3C7",
    "dd_fill":   "#FADBD8",
    "dd_line":   "#E74C3C",
    "winner":    "#27AE60",
    "loser":     "#E74C3C",
    "text":      "#2C3E50",
    "sharpe":    "#8E44AD",
    "neutral":   "#95A5A6",
    "var_95":    "#E67E22",
    "var_99":    "#C0392B",
}

# Distinct palette for per-strategy lines
STRATEGY_COLORS: List[str] = [
    "#9B59B6", "#F1C40F", "#E67E22", "#1ABC9C", "#D35400", "#8E44AD",
]
