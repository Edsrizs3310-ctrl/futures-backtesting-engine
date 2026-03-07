"""
src/backtest_engine/analytics/dashboard/pnl_analysis/correlation_heatmap.py

Correlation heatmap builder.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.backtest_engine.analytics.dashboard.core.styles import PALETTE


def build_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation",
) -> go.Figure:
    """
    Builds a Plotly annotated correlation heatmap.

    Methodology:
        Diverging RdBu_r colorscale, fixed [-1,1] range, midpoint 0.
        Cell text annotated so values are readable without colorbar lookup.

    Args:
        corr_matrix: Square correlation DataFrame.
        title: Chart title string.
    """
    fig = go.Figure()

    if corr_matrix is None or corr_matrix.empty:
        fig.add_annotation(
            text="Insufficient data for correlation\n(need ≥ 2 instruments)",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False, font_size=12,
        )
        fig.update_layout(height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        return fig

    labels = corr_matrix.columns.tolist()
    z_vals = corr_matrix.values
    z_text = [[f"{v:.2f}" for v in row] for row in z_vals]

    fig.add_trace(go.Heatmap(
        z=z_vals, x=labels, y=labels,
        text=z_text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(thickness=12, len=0.75,
                      tickvals=[-1, -0.5, 0, 0.5, 1],
                      ticktext=["-1", "-0.5", "0", "0.5", "1"]),
        hovertemplate="%{y} × %{x}: <b>%{text}</b><extra></extra>",
    ))
    n = len(labels)
    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        xaxis=dict(side="bottom", tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=38, b=0),
        height=max(280, n * 60 + 100),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
    )
    return fig
