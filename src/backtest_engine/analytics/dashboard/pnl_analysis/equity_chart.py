"""
src/backtest_engine/analytics/dashboard/pnl_analysis/equity_chart.py

Combined equity chart builders for both single-asset and portfolio modes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest_engine.analytics.dashboard.core.styles import PALETTE, STRATEGY_COLORS


def build_single_equity_figure(
    history: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    benchmark: Optional[pd.DataFrame],
) -> go.Figure:
    """
    Builds an interactive equity curve for single-asset mode.
    """
    fig = go.Figure()
    initial_cap = float(history["total_value"].iloc[0])
    idx = history.index
    portfolio_pnl = history["total_value"] - initial_cap

    # Benchmark
    if benchmark is not None and not benchmark.empty:
        b = benchmark["close"]
        common = idx.intersection(b.index)
        if len(common) > 1:
            b_pnl = (b.loc[common] / float(b.loc[common].iloc[0]) - 1.0) * initial_cap
            fig.add_trace(go.Scatter(
                x=common, y=b_pnl, mode="lines",
                name="B&H (Buy-and-Hold)",
                line=dict(color=PALETTE["bench"], width=1.5, dash="dash"),
            ))

    # Long / short decomposition
    if trades is not None and not trades.empty and "exit_time" in trades.columns:
        for direction, color, label in [
            ("LONG",  PALETTE["long"],  "Long"),
            ("SHORT", PALETTE["short"], "Short"),
        ]:
            sub = trades[trades["direction"] == direction].copy()
            if sub.empty:
                continue
            pnl_s = sub.set_index("exit_time")["pnl"].sort_index()
            pnl_s = pnl_s.groupby(level=0).sum()
            full_idx = idx.union(pnl_s.index)
            cum = (
                pnl_s.reindex(full_idx, fill_value=0)
                .reindex(idx, fill_value=0)
                .cumsum()
            )
            fig.add_trace(go.Scatter(
                x=idx, y=cum, mode="lines",
                name=label,
                line=dict(color=color, width=1.2),
                opacity=0.85,
            ))

    # Strategy total (bold, on top)
    fig.add_trace(go.Scatter(
        x=idx, y=portfolio_pnl, mode="lines",
        name="Strategy",
        line=dict(color=PALETTE["combined"], width=2.2),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["bench"],
                  line_width=0.8, opacity=0.6)
    fig.update_layout(
        title=dict(text="Equity Curve — Cumulative PnL ($)", font_size=13, x=0),
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Cumulative PnL ($)"),
        legend=dict(orientation="h", y=-0.12, x=0,
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=34, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
    )
    return fig


def build_portfolio_equity_figure(
    history: pd.DataFrame,
    benchmark: Optional[pd.DataFrame],
    slots: Optional[Dict[str, str]],
    rolling_sharpe: Optional[pd.Series] = None,
    strategy_summaries: Optional[Dict[str, Dict[str, object]]] = None,
) -> go.Figure:
    """
    Builds an interactive portfolio equity curve figure.
    """
    show_sharpe = (
        rolling_sharpe is not None
        and not rolling_sharpe.dropna().empty
    )

    if show_sharpe:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.76, 0.24],
            vertical_spacing=0.05,
            subplot_titles=["Equity Curve — Cumulative PnL ($)", "Rolling Sharpe (90d)"],
        )
        eq_row, sh_row = 1, 2
    else:
        fig = go.Figure()
        eq_row = sh_row = None

    def _add(trace: go.BaseTraceType) -> None:
        if show_sharpe:
            fig.add_trace(trace, row=eq_row, col=1)
        else:
            fig.add_trace(trace)

    initial_cap: float = float(history["total_value"].iloc[0])
    idx = history.index
    portfolio_pnl = history["total_value"] - initial_cap

    # Benchmark
    if benchmark is not None and not benchmark.empty:
        b = benchmark["close"]
        common = idx.intersection(b.index)
        if len(common) > 1:
            b_pnl = (b.loc[common] / float(b.loc[common].iloc[0]) - 1.0) * initial_cap
            _add(go.Scatter(
                x=common, y=b_pnl, mode="lines",
                name="B&H (Buy-and-Hold)",
                line=dict(color=PALETTE["bench"], width=1.5, dash="dash"),
            ))

    # Per-strategy lines with hover
    if slots:
        for i, (str_slot_id, strat_name) in enumerate(slots.items()):
            col_name = f"slot_{str_slot_id}_pnl"
            if col_name not in history.columns:
                continue
            y_vals = history[col_name].tolist()
            color  = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
            summ   = (strategy_summaries or {}).get(strat_name, {})
            def _fmt(val: object, fmt_str: str) -> str:
                if val == "N/A" or val is None:
                    return "N/A"
                try:
                    return format(val, fmt_str)
                except Exception:
                    return str(val)

            hover = (
                f"<b>{strat_name}</b><br>"
                f"PnL at cursor: <b>$%{{y:,.0f}}</b><br>"
                f"Total PnL: ${_fmt(summ.get('total_pnl', 'N/A'), ',.0f')}<br>"
                f"Trades: {summ.get('trade_count', 'N/A')}<br>"
                f"Win Rate: {_fmt(summ.get('win_rate', 'N/A'), '.1f')}%<br>"
                f"Avg Trade: ${_fmt(summ.get('avg_trade', 'N/A'), ',.0f')}<br>"
                f"Max Loss: ${_fmt(summ.get('max_loss', 'N/A'), ',.0f')}<br>"
                f"<br>--- Stat tests ---<br>"
                f"T-stat: {_fmt(summ.get('tstat', 'N/A'), '.2f')}<br>"
                f"p-value: {_fmt(summ.get('pvalue', 'N/A'), '.3f')}<br>"
                f"Alpha (Ann): {_fmt(summ.get('alpha', 'N/A'), '.1f')}%<br>"
                f"Alpha p-value: {_fmt(summ.get('alpha_p', 'N/A'), '.3f')}<br>"
                f"Beta: {_fmt(summ.get('beta', 'N/A'), '.2f')}<br>"
                f"Beta p-value: {_fmt(summ.get('beta_p', 'N/A'), '.3f')}"
                "<extra></extra>"
            )
            _add(go.Scatter(
                x=idx, y=y_vals,
                hovertemplate=hover,
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#CCCCCC",
                    font=dict(color="#000000", size=11),
                ),
                mode="lines",
                name=f"{strat_name} (S{str_slot_id})",
                line=dict(color=color, width=1.2),
                opacity=0.80,
            ))

    # Portfolio total (bold, on top)
    _add(go.Scatter(
        x=idx, y=portfolio_pnl, mode="lines",
        name="Portfolio Total",
        line=dict(color=PALETTE["combined"], width=2.2),
    ))

    # Rolling Sharpe subplot
    if show_sharpe:
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode="lines",
            name="Rolling Sharpe (90d)",
            line=dict(color=PALETTE["sharpe"], width=1.3),
            fill="tozeroy",
            fillcolor="rgba(142,68,173,0.15)",
        ), row=sh_row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["neutral"],
                      line_width=0.8, opacity=0.5, row=sh_row, col=1)
        fig.update_yaxes(title_text="Sharpe", row=sh_row, col=1)

    # Zero baseline
    hline_kw = dict(line_dash="dash", line_color=PALETTE["bench"],
                    line_width=0.8, opacity=0.6)
    if show_sharpe:
        fig.add_hline(y=0, row=eq_row, col=1, **hline_kw)
    else:
        fig.add_hline(y=0, **hline_kw)

    layout = dict(
        legend=dict(orientation="h", y=-0.12, x=0,
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=34, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
    )
    if show_sharpe:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f",
                         title_text="Cumulative PnL ($)", row=1, col=1)
        fig.update_layout(**layout)
    else:
        fig.update_layout(
            title=dict(text="Equity Curve — Cumulative PnL ($)", font_size=13, x=0),
            yaxis=dict(tickprefix="$", tickformat=",.0f", title="Cumulative PnL ($)"),
            **layout,
        )
    return fig


def build_decomp_chart(decomp_df: pd.DataFrame) -> go.Figure:
    """
    Builds a horizontal grouped bar chart for strategy PnL decomposition.
    """
    fig = go.Figure()

    if decomp_df is None or decomp_df.empty:
        fig.add_annotation(
            text="No decomposition data", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False, font_size=13,
        )
        fig.update_layout(height=200, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        return fig

    strategies   = decomp_df["Strategy"].tolist()
    pnl_contribs = decomp_df["PnL Contrib (%)"].tolist()
    pnl_colors   = [PALETTE["winner"] if v >= 0 else PALETTE["loser"] for v in pnl_contribs]

    fig.add_trace(go.Bar(
        y=strategies, x=pnl_contribs, orientation="h",
        name="PnL Contribution %",
        marker_color=pnl_colors, opacity=0.85,
        text=[f"{v:.1f}%" for v in pnl_contribs],
        textposition="auto",
    ))

    if "Risk Contrib (%)" in decomp_df.columns:
        risk = decomp_df["Risk Contrib (%)"].tolist()
        fig.add_trace(go.Bar(
            y=strategies, x=risk, orientation="h",
            name="Risk Contribution % (β)",
            marker_color=PALETTE["sharpe"], opacity=0.65,
            text=[f"{v:.1f}%" for v in risk],
            textposition="auto",
        ))

    fig.update_layout(
        title=dict(text="PnL & Volatility Contribution by Strategy", font_size=12, x=0),
        barmode="group",
        xaxis=dict(ticksuffix="%", zeroline=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10, orientation="h", y=-0.30),
        margin=dict(l=0, r=0, t=34, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        height=max(200, len(strategies) * 55 + 100),
    )
    return fig
