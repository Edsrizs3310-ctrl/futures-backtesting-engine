"""
src/backtest_engine/analytics/dashboard/charts.py

Public re-export façade — keeps backward compatibility while the actual
chart building logic is split across three focused modules:

    base_charts.py      — shared/mode-agnostic builders
    portfolio_charts.py — portfolio-specific builders
    single_charts.py    — single-asset-specific builders

Import from this module as before:
    from src.backtest_engine.analytics.dashboard.charts import build_equity_figure
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.backtest_engine.analytics.dashboard.pnl_analysis.drawdown_chart import build_drawdown_figure  # noqa: F401
from src.backtest_engine.analytics.dashboard.pnl_analysis.distribution_chart import (
    build_pnl_hist_figure,          # noqa: F401
    build_pnl_distribution_figure,  # noqa: F401
)
from src.backtest_engine.analytics.dashboard.pnl_analysis.correlation_heatmap import build_correlation_heatmap  # noqa: F401
from src.backtest_engine.analytics.dashboard.pnl_analysis.equity_chart import (
    build_portfolio_equity_figure,  # noqa: F401
    build_single_equity_figure,     # noqa: F401
    build_decomp_chart,             # noqa: F401
)


def build_equity_figure(
    history: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    benchmark: Optional[pd.DataFrame],
    run_type: str = "single",
    slots: Optional[Dict[str, str]] = None,
    rolling_sharpe: Optional[pd.Series] = None,
    strategy_summaries: Optional[Dict] = None,
):
    """
    Mode-dispatching wrapper kept for backward compatibility.

    Routes to build_portfolio_equity_figure() or build_single_equity_figure()
    depending on run_type.

    Args:
        history: Bar-level portfolio history.
        trades: Trades DataFrame.
        benchmark: Optional benchmark close price DataFrame.
        run_type: 'portfolio' or 'single'.
        slots: {slot_id: strategy_name} — portfolio mode only.
        rolling_sharpe: Daily rolling Sharpe series — portfolio mode only.
        strategy_summaries: Per-strategy hover tooltip data — portfolio mode only.
    """
    if run_type == "portfolio":
        return build_portfolio_equity_figure(
            history=history,
            benchmark=benchmark,
            slots=slots,
            rolling_sharpe=rolling_sharpe,
            strategy_summaries=strategy_summaries,
        )
    return build_single_equity_figure(
        history=history,
        trades=trades,
        benchmark=benchmark,
    )
