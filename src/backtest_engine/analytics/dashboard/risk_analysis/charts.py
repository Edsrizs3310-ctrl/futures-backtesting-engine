"""
Plotly chart builders for the dashboard Risk Analysis tab.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.backtest_engine.analytics.dashboard.core.styles import PALETTE, STRATEGY_COLORS
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressScenarioResult


def _empty_figure(message: str, height: int = 280) -> go.Figure:
    """Returns a lightweight annotated figure for missing-data states."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font_size=14,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
    )
    return fig


def build_equity_curve_figure(equity: pd.Series, title: str) -> go.Figure:
    """Builds a clean equity curve for drawdown context."""
    if equity is None or equity.dropna().empty:
        return _empty_figure("No equity data", height=260)

    clean = equity.dropna().astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            name="Equity",
            line=dict(color=PALETTE["combined"], width=2),
        )
    )
    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        xaxis_title="Date",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=260,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_var_es_figure(
    rolling_var: pd.DataFrame,
    primary_confidence: float,
    tail_confidence: float,
    title: str,
) -> go.Figure:
    """Builds a rolling VaR / ES chart with breach markers."""
    if rolling_var is None or rolling_var.empty:
        return _empty_figure("No daily PnL data", height=320)

    frame = rolling_var.dropna(subset=["pnl"], how="all")
    if frame.empty:
        return _empty_figure("No daily PnL data", height=320)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["pnl"],
            mode="lines",
            name="Daily PnL",
            line=dict(color=PALETTE["combined"], width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=-frame["var_primary"],
            mode="lines",
            name=f"VaR {int(primary_confidence * 100)}",
            line=dict(color=PALETTE["var_95"], width=1.8, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=-frame["es_primary"],
            mode="lines",
            name=f"ES {int(primary_confidence * 100)}",
            line=dict(color=PALETTE["var_95"], width=1.3, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=-frame["var_tail"],
            mode="lines",
            name=f"VaR {int(tail_confidence * 100)}",
            line=dict(color=PALETTE["var_99"], width=1.8, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=-frame["es_tail"],
            mode="lines",
            name=f"ES {int(tail_confidence * 100)}",
            line=dict(color=PALETTE["var_99"], width=1.3, dash="dot"),
        )
    )

    primary_breaches = frame[frame["breach_primary"].fillna(False)]
    if not primary_breaches.empty:
        fig.add_trace(
            go.Scatter(
                x=primary_breaches.index,
                y=primary_breaches["pnl"],
                mode="markers",
                name=f"Breach {int(primary_confidence * 100)}",
                marker=dict(color=PALETTE["var_95"], size=6, symbol="circle"),
            )
        )

    tail_breaches = frame[frame["breach_tail"].fillna(False)]
    if not tail_breaches.empty:
        fig.add_trace(
            go.Scatter(
                x=tail_breaches.index,
                y=tail_breaches["pnl"],
                mode="markers",
                name=f"Breach {int(tail_confidence * 100)}",
                marker=dict(color=PALETTE["var_99"], size=7, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        xaxis_title="Date",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=30, b=80),
        height=320,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.3, x=0),
    )
    return fig


def build_risk_distribution_figure(
    daily_pnl: pd.Series,
    summary: Dict[str, float],
    primary_confidence: float,
    tail_confidence: float,
    title: str,
) -> go.Figure:
    """Builds a histogram with VaR / ES markers for the tail-risk view."""
    if daily_pnl is None or daily_pnl.dropna().empty:
        return _empty_figure("No daily PnL data", height=320)

    clean = daily_pnl.dropna().astype(float)
    winners = clean[clean > 0.0]
    losers = clean[clean <= 0.0]
    pnl_range = float(clean.max()) - float(clean.min())
    bin_size = pnl_range / 40.0 if pnl_range > 0 else 1.0
    bins = dict(start=float(clean.min()), end=float(clean.max()), size=bin_size)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=losers,
            xbins=bins,
            name="Negative days",
            marker_color=PALETTE["loser"],
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=winners,
            xbins=bins,
            name="Positive days",
            marker_color=PALETTE["winner"],
            opacity=0.75,
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color=PALETTE["text"], line_width=0.8)

    marker_specs = [
        ("var_primary", f"VaR {int(primary_confidence * 100)}", PALETTE["var_95"], "dash", "top right"),
        ("es_primary", f"ES {int(primary_confidence * 100)}", PALETTE["var_95"], "dot", "top left"),
        ("var_tail", f"VaR {int(tail_confidence * 100)}", PALETTE["var_99"], "dash", "bottom right"),
        ("es_tail", f"ES {int(tail_confidence * 100)}", PALETTE["var_99"], "dot", "bottom left"),
    ]
    for key, label, color, dash, pos in marker_specs:
        value = float(summary.get(key, float("nan")))
        if np.isnan(value):
            continue
        fig.add_vline(
            x=-value,
            line_dash=dash,
            line_color=color,
            line_width=1.7,
            annotation_text=f"{label}<br>${value:,.0f}",
            annotation_font_size=9,
            annotation_font_color=color,
            annotation_position=pos,
        )

    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        barmode="overlay",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=30, b=80),
        height=320,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.3, x=0),
    )
    return fig


def build_drawdown_curve_figure(drawdown_pct: pd.Series, title: str) -> go.Figure:
    """Builds a filled drawdown chart from a generic drawdown series."""
    if drawdown_pct is None or drawdown_pct.dropna().empty:
        return _empty_figure("No drawdown data", height=240)

    clean = drawdown_pct.dropna().astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.22)",
            line=dict(color=PALETTE["dd_line"], width=1.5),
            name="Drawdown",
        )
    )
    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        yaxis=dict(ticksuffix="%"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=240,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_drawdown_distribution_figure(drawdown_episodes: pd.DataFrame, title: str) -> go.Figure:
    """Builds a histogram of drawdown episode depths."""
    if drawdown_episodes is None or drawdown_episodes.empty or "depth_abs_pct" not in drawdown_episodes.columns:
        return _empty_figure("No drawdown episodes", height=240)

    depths = drawdown_episodes["depth_abs_pct"].dropna().astype(float)
    if depths.empty:
        return _empty_figure("No drawdown episodes", height=240)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=depths,
            nbinsx=min(25, max(len(depths), 5)),
            marker_color=PALETTE["dd_line"],
            opacity=0.8,
            name="Episode depth",
        )
    )
    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        xaxis=dict(ticksuffix="%"),
        yaxis_title="Count",
        margin=dict(l=0, r=0, t=30, b=0),
        height=240,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_rolling_volatility_figure(rolling_vol: pd.DataFrame, title: str) -> go.Figure:
    """Builds a multi-window rolling volatility chart."""
    if rolling_vol is None or rolling_vol.empty:
        return _empty_figure("No return series for rolling volatility", height=280)

    fig = go.Figure()
    for idx, column in enumerate(rolling_vol.columns):
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[column],
                mode="lines",
                name=column,
                line=dict(color=STRATEGY_COLORS[idx % len(STRATEGY_COLORS)], width=1.8),
            )
        )

    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        xaxis_title="Date",
        yaxis=dict(ticksuffix="%"),
        margin=dict(l=0, r=0, t=30, b=80),
        height=280,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.3, x=0),
    )
    return fig


def build_stress_test_figure(stress_results: List[StressScenarioResult], title: str) -> go.Figure:
    """Builds a multi-scenario daily equity comparison for stress previews."""
    if not stress_results:
        return _empty_figure("No stress preview scenarios", height=320)

    color_map = {
        "baseline": PALETTE["combined"],
        "volatility": STRATEGY_COLORS[0],
        "slippage": STRATEGY_COLORS[1],
        "commission": STRATEGY_COLORS[2],
        "combined": PALETTE["var_99"],
    }

    fig = go.Figure()
    for scenario in stress_results:
        fig.add_trace(
            go.Scatter(
                x=scenario.equity.index,
                y=scenario.equity,
                mode="lines",
                name=scenario.label,
                line=dict(
                    color=color_map.get(scenario.name, PALETTE["neutral"]),
                    width=2.6 if scenario.name in {"baseline", "combined"} else 1.6,
                    dash="solid" if scenario.name in {"baseline", "combined"} else "dot",
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        xaxis_title="Date",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=30, b=80),
        height=320,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.3, x=0),
    )
    return fig


def build_scenario_comparison_figure(
    baseline_equity: pd.Series,
    scenario_equity: pd.Series,
    title: str,
) -> go.Figure:
    """Builds a baseline-vs-scenario equity comparison for real rerun artifacts."""
    if baseline_equity is None or baseline_equity.dropna().empty:
        return _empty_figure("No baseline equity data", height=320)
    if scenario_equity is None or scenario_equity.dropna().empty:
        return _empty_figure("No scenario equity data", height=320)

    baseline = baseline_equity.dropna().astype(float)
    scenario = scenario_equity.dropna().astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=baseline.index,
            y=baseline,
            mode="lines",
            name="Baseline",
            line=dict(color=PALETTE["combined"], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenario.index,
            y=scenario,
            mode="lines",
            name="Scenario",
            line=dict(color=PALETTE["var_99"], width=2.0, dash="dash"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font_size=12, x=0),
        xaxis_title="Date",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=30, b=80),
        height=320,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.3, x=0),
    )
    return fig
