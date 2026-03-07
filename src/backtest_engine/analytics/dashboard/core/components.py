"""
src/backtest_engine/analytics/dashboard/components.py

Data loading and Streamlit component rendering helpers.

Responsibility: File I/O (load Parquet / JSON / text from results/) and
reusable Streamlit widgets (e.g. exit breakdown table, decomp table,
correlation horizon selector).
No chart building happens here — that lives in charts.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def get_results_dir() -> Path:
    """
    Resolves the results/ directory relative to the project root.

    Returns:
        Absolute path to results/.
    """
    # core/components.py → core/ → dashboard/ → analytics/ → backtest_engine/ → src/ → project root
    return Path(__file__).parent.parent.parent.parent.parent.parent / "results"


def load_parquet(filename: str) -> Optional[pd.DataFrame]:
    """
    Loads a Parquet file from the results directory.

    Args:
        filename: File name relative to results/ (e.g. 'history.parquet').

    Returns:
        DataFrame or None if the file does not exist.
    """
    path = get_results_dir() / filename
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_text(filename: str) -> Optional[str]:
    """
    Loads a plain-text file from the results directory.

    Args:
        filename: File name relative to results/ (e.g. 'report.txt').

    Returns:
        String contents or None if the file does not exist.
    """
    path = get_results_dir() / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_json(filename: str) -> Optional[dict]:
    """
    Loads a JSON file from the results directory.

    Args:
        filename: File name relative to results/ (e.g. 'metrics.json').

    Returns:
        Parsed dict or None if the file does not exist.
    """
    path = get_results_dir() / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))




def render_decomp_table(decomp_df: Optional[pd.DataFrame]) -> None:
    """
    Renders the Strategy PnL Decomposition table with conditional colouring.

    Applies green/red to 'Total PnL ($)' and Sharpe columns.
    All other columns use 2-decimal precision where needed.

    Args:
        decomp_df: Output of compute_strategy_decomp(). None -> caption shown.
    """
    if decomp_df is None or decomp_df.empty:
        st.caption("No decomposition data (requires portfolio run with strategy labels).")
        return

    def _color_pnl(val: float) -> str:
        if pd.isna(val):
            return ""
        return "background-color: #d4efdf" if val >= 0 else "background-color: #fadbd8"

    def _color_sharpe(val: float) -> str:
        if pd.isna(val):
            return ""
        if val >= 1.0:
            return "background-color: #d4efdf"
        if val >= 0:
            return "background-color: #fef9e7"
        return "background-color: #fadbd8"

    format_map = {
        "Sharpe":                 "{:.2f}",
        "MTM PnL ($)":            "{:,.0f}",
        "Closed PnL ($)":         "{:,.0f}",
        "PnL Contrib (%)":        "{:.1f}%",
        "Risk Contrib (%)":       "{:.1f}%",
        "Max DD PnL ($)":         "{:,.0f}",
        "Tail PnL (CVaR)":        "{:,.0f}",
        "Signal PnL ($)":         "{:,.0f}",
        "Exec Cost ($)":          "{:,.0f}",
    }
    active_fmt = {k: v for k, v in format_map.items() if k in decomp_df.columns}

    style = decomp_df.style.format(active_fmt, na_rep="N/A")

    if "MTM PnL ($)" in decomp_df.columns:
        style = style.map(_color_pnl, subset=["MTM PnL ($)"])
    if "Sharpe" in decomp_df.columns:
        style = style.map(_color_sharpe, subset=["Sharpe"])

    st.dataframe(style, width="stretch", hide_index=True)



def render_correlation_horizon_selector(key: str = "corr_horizon") -> str:
    """
    Renders a radio selector for the correlation resampling horizon.

    Returns one of: '1d', '1w', '1m'.

    Args:
        key: Streamlit widget key (allows two independent selectors on same page).

    Returns:
        Selected horizon string.
    """
    options = {"1 Day": "1d", "1 Week": "1w", "1 Month": "1m"}
    label = st.radio(
        "Horizon",
        list(options.keys()),
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )
    return options[label]
