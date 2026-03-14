"""
src/backtest_engine/analytics/dashboard/pnl_analysis/distribution_chart.py

PnL distribution chart builders.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.backtest_engine.analytics.dashboard.core.styles import PALETTE


def build_pnl_hist_figure(trades: Optional[pd.DataFrame]) -> go.Figure:
    """
    Builds a trade-level P&L distribution histogram (winner vs loser overlay).

    Args:
        trades: Trades DataFrame with 'pnl' column.
    """
    fig = go.Figure()
    if trades is None or trades.empty or "pnl" not in trades.columns:
        fig.add_annotation(
            text="No Trades", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False, font_size=14,
        )
        fig.update_layout(height=240, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        return fig

    pnls    = trades["pnl"].dropna()
    winners = pnls[pnls > 0]
    losers  = pnls[pnls <= 0]
    bin_sz  = (pnls.max() - pnls.min()) / 50
    bins    = dict(start=pnls.min(), end=pnls.max(), size=bin_sz)

    fig.add_trace(go.Histogram(x=losers, xbins=bins, name="Losers",
                               marker_color=PALETTE["loser"], opacity=0.7))
    fig.add_trace(go.Histogram(x=winners, xbins=bins, name="Winners",
                               marker_color=PALETTE["winner"], opacity=0.7))
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["text"], line_width=0.8)
    fig.update_layout(
        title=dict(text="P&L Distribution (Trades)", font_size=12, x=0),
        barmode="overlay",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        margin=dict(l=0, r=0, t=30, b=0),
        height=240,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
    )
    return fig


def build_pnl_distribution_figure(
    daily_pnl: pd.Series,
    dist_stats: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """
    Builds an enhanced daily P&L distribution with VaR vertical lines and
    statistical stats block (skew, kurtosis, VaR 95%, CVaR 95%).

    Methodology:
        - Histogram split into negative (red) and positive (green) bars.
        - VaR 95% and VaR 99% are stored as positive loss magnitudes and drawn
          on the negative PnL axis as vertical dashed lines
          so the tail thresholds are spatially visible in the distribution.
        - Stats (skew, kurt, VaR 95%, CVaR 95%) annotated inside the chart.

    Args:
        daily_pnl: Daily net PnL series (not cumulative).
        dist_stats: Output of compute_pnl_dist_stats(). Annotations skipped
                    if None.
    """
    fig = go.Figure()

    if daily_pnl is None or daily_pnl.dropna().empty:
        fig.add_annotation(
            text="No daily PnL data", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False, font_size=14,
        )
        fig.update_layout(height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        return fig

    clean   = daily_pnl.dropna()
    winners = clean[clean > 0]
    losers  = clean[clean <= 0]

    pnl_range = float(clean.max()) - float(clean.min())
    bin_size  = pnl_range / 40 if pnl_range > 0 else 1.0
    bins      = dict(start=float(clean.min()), end=float(clean.max()), size=bin_size)

    fig.add_trace(go.Histogram(x=losers, xbins=bins, name="Negative days",
                               marker_color=PALETTE["loser"], opacity=0.75))
    fig.add_trace(go.Histogram(x=winners, xbins=bins, name="Positive days",
                               marker_color=PALETTE["winner"], opacity=0.75))
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["text"], line_width=0.8)

    # VaR vertical lines drawn ON the chart for spatial context
    if dist_stats:
        var95 = dist_stats.get("var_95", float("nan"))
        var99 = dist_stats.get("var_99", float("nan"))
        skew  = dist_stats.get("skew",   float("nan"))
        kurt  = dist_stats.get("kurtosis", float("nan"))
        cvar  = dist_stats.get("cvar_95", float("nan"))

        if not np.isnan(var95):
            fig.add_vline(
                x=-var95, line_dash="dot", line_color=PALETTE["var_95"],
                line_width=1.8,
                annotation_text=f"VaR 95%<br>${var95:,.0f}",
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=PALETTE["var_95"],
            )
        if not np.isnan(var99):
            fig.add_vline(
                x=-var99, line_dash="dot", line_color=PALETTE["var_99"],
                line_width=1.8,
                annotation_text=f"VaR 99%<br>${var99:,.0f}",
                annotation_position="bottom left",
                annotation_font_size=9,
                annotation_font_color=PALETTE["var_99"],
            )

        def _s(v: float, prefix: str = "") -> str:
            return "N/A" if np.isnan(v) else f"{prefix}{v:,.0f}"

        stats_text = (
            f"<b>Skew:</b> {skew:.3f}  "
            f"<b>Kurt:</b> {kurt:.3f}<br>"
            f"<b>CVaR 95%:</b> {_s(cvar, '$')}"
        )
        fig.add_annotation(
            text=stats_text, align="left",
            xref="paper", yref="paper", x=0.99, y=0.97,
            showarrow=False,
            font=dict(size=10, color=PALETTE["text"]),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor=PALETTE["neutral"], borderwidth=1,
        )

    fig.update_layout(
        title=dict(text="Daily PnL Distribution", font_size=12, x=0),
        barmode="overlay",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        margin=dict(l=0, r=0, t=30, b=40),
        height=300,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
    )
    return fig
