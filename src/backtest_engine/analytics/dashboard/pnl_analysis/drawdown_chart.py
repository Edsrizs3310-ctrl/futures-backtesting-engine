"""
src/backtest_engine/analytics/dashboard/pnl_analysis/drawdown_chart.py

Drawdown percentage chart builder.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.backtest_engine.analytics.dashboard.core.styles import PALETTE


def build_drawdown_figure(history: pd.DataFrame) -> go.Figure:
    """
    Builds a filled area drawdown chart.

    Methodology:
        Drawdown at each bar = (equity - running_peak) / running_peak * 100
        Displayed as a percentage to make severity immediately legible.

    Args:
        history: Portfolio history with 'total_value' column.
    """
    running_max = history["total_value"].cummax()
    dd = (history["total_value"] - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.25)",
        line=dict(color=PALETTE["dd_line"], width=1),
        name="Drawdown",
    ))
    fig.update_layout(
        title=dict(text="Drawdown %", font_size=12, x=0),
        yaxis=dict(ticksuffix="%"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=200,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
    )
    return fig
