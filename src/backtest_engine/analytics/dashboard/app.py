"""
src/backtest_engine/analytics/dashboard/app.py

Streamlit entry point for the backtest research dashboard.

Responsibility: Only page layout and orchestration.
No chart building (charts.py / *_charts.py), no file I/O (components.py),
no math (transforms.py).

Layout — PnL Analysis tab:
    Row 1 : Equity Curve [left 70%]  | Terminal Report Log [right 30%]
    Row 2 : Drawdown % [full width]
    Row 3 : PnL Distribution [left 50%] | Exit Breakdown table [right 50%]
    -- Portfolio mode only --
    Row 4 : Strategy PnL Decomposition table (full width)
    Row 5 : Decomposition bar chart (full width)
    Row 6 : Strategy Correlation [left 50%] | Exposure Correlation [right 50%]

Usage:
    streamlit run src/backtest_engine/analytics/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle, load_result_bundle
from src.backtest_engine.analytics.dashboard.pnl_analysis.drawdown_chart import build_drawdown_figure
from src.backtest_engine.analytics.dashboard.pnl_analysis.distribution_chart import build_pnl_distribution_figure
from src.backtest_engine.analytics.dashboard.pnl_analysis.correlation_heatmap import build_correlation_heatmap
from src.backtest_engine.analytics.dashboard.pnl_analysis.equity_chart import (
    build_portfolio_equity_figure,
    build_decomp_chart,
    build_single_equity_figure,
)
from src.backtest_engine.analytics.dashboard.pnl_analysis.exit_decomposition import (
    build_holding_time_chart,
    build_pnl_decay_chart,
    build_mfe_mae_scatter,
    build_exit_reason_chart,
    build_vol_regime_chart,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.risk_placeholder import render_risk_tab
from src.backtest_engine.analytics.dashboard.simulation_analysis.sim_placeholder import render_simulation_tab
from src.backtest_engine.analytics.dashboard.core.components import (
    render_decomp_table,
    render_correlation_horizon_selector,
)
from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_bar_pnl_matrix,
    compute_strategy_decomp,
    compute_strategy_correlation,
    compute_exposure_correlation,
    compute_rolling_sharpe,
    compute_pnl_dist_stats,
    compute_per_strategy_summary,
    compute_exit_summary,
)


# ── Cached computation wrappers ────────────────────────────────────────────────
# @st.cache_data lives here (not in transforms.py) to keep transforms.py
# importable without a live Streamlit session — critical for unit tests.

@st.cache_data(show_spinner=False)
def _cached_bar_pnl_matrix(history_hash: str, slots_hash: str,
                            history: pd.DataFrame,
                            slots: dict) -> pd.DataFrame:
    """Cache the bar PnL matrix by DataFrame hash + slots hash."""
    return build_bar_pnl_matrix(history, slots)


@st.cache_data(show_spinner=False)
def _cached_rolling_sharpe(history_hash: str, history: pd.DataFrame,
                            window_days: int) -> pd.Series:
    """Cache rolling Sharpe by DataFrame hash + window config."""
    return compute_rolling_sharpe(history, window_days=window_days)


def _derive_daily_pnl(history: pd.DataFrame) -> pd.Series:
    """
    Derives a daily PnL series from the portfolio history total_value column.

    Methodology:
        bar_pnl   = total_value.diff()  (incremental, not cumulative)
        daily_pnl = bar_pnl.resample('1D').sum()
    """
    bar_pnl = history["total_value"].diff().fillna(0.0)
    return bar_pnl.resample("1D").sum()


def _render_pnl_tab(bundle: ResultBundle, window_days: int) -> None:
    """
    Renders the complete PnL Analysis tab for both modes.

    Args:
        bundle: Fully loaded ResultBundle from data_layer.
        window_days: Rolling Sharpe window from settings.
    """
    is_portfolio: bool = bundle.run_type == "portfolio"

    # ── Pre-compute ────────────────────────────────────────────────────────────
    daily_pnl: pd.Series = _derive_daily_pnl(bundle.history)
    dist_stats: dict     = compute_pnl_dist_stats(daily_pnl)

    rolling_sharpe    = None
    strategy_summaries = None
    bar_pnl_matrix    = None

    if is_portfolio and bundle.slots:
        bar_pnl_matrix   = build_bar_pnl_matrix(bundle.history, bundle.slots)
        strategy_summaries = compute_per_strategy_summary(
            bundle.trades, bundle.slots, bundle.history, bundle.instrument_closes, getattr(bundle, "slot_weights", None)
        )
        rolling_sharpe   = compute_rolling_sharpe(
            bundle.history, window_days=window_days
        )

    # ── Row 1: Equity Curve & Drawdown | Terminal Report ───────────────────────
    col_eq, col_log = st.columns([7, 3])

    with col_eq:
        if is_portfolio:
            fig_eq = build_portfolio_equity_figure(
                history=bundle.history,
                benchmark=bundle.benchmark,
                slots=bundle.slots,
                rolling_sharpe=rolling_sharpe,
                strategy_summaries=strategy_summaries,
            )
        else:
            fig_eq = build_single_equity_figure(
                history=bundle.history,
                trades=bundle.trades,
                benchmark=bundle.benchmark,
            )
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = build_drawdown_figure(bundle.history)
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_log:
        st.code(bundle.report or "No report available.", language="")

    st.divider()

    # ── Row 3: Exit Analysis Summary (Screener Table) ──────────────────────────
    st.markdown("#### Exit Analysis Summary")
    st.caption("Select a strategy row below to open detailed interactive exit charts (MFE/MAE, Decay, etc).")

    st_summ = pd.DataFrame()
    if is_portfolio and bundle.slots:
        st_summ = compute_exit_summary(bundle.trades, bundle.slots)
    else:
        st_summ = compute_exit_summary(bundle.trades, {"single": "Single Asset"})

    if not st_summ.empty:
        event = st.dataframe(
            st_summ,
            width="stretch",
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        if event.selection.rows:
            selected_idx = event.selection.rows[0]
            strat_name = st_summ.iloc[selected_idx]["Strategy"]
            if is_portfolio:
                strat_trades = bundle.trades[bundle.trades["strategy"] == strat_name] if not bundle.trades.empty else pd.DataFrame()
            else:
                strat_trades = bundle.trades
            
            _show_exit_analysis_dialog(strat_name, strat_trades)
    else:
        st.info("No detailed exit data available.")

    st.divider()

    # ── Row 4: PnL Distribution (full width since legacy table is removed) ─────
    st.markdown("#### Daily PnL Distribution")
    fig_dist = build_pnl_distribution_figure(daily_pnl, dist_stats)
    st.plotly_chart(fig_dist, use_container_width=True)

    if not is_portfolio:
        return

    st.divider()

    # ── Row 5: Strategy PnL Decomposition table (full width) ──────────────────
    st.markdown("#### Strategy PnL Decomposition")
    decomp_df = compute_strategy_decomp(
        trades_df=bundle.trades,
        history=bundle.history,
        slots=bundle.slots or {},
    )
    render_decomp_table(decomp_df)

    # ── Row 5: Decomposition bar chart (full width, below table) ──────────────
    fig_decomp = build_decomp_chart(decomp_df)
    st.plotly_chart(fig_decomp, use_container_width=True)

    st.divider()

    # ── Row 6: Correlations ────────────────────────────────────────────────────
    st.markdown("#### Correlations")
    st.caption(
        "Strategy: incremental bar PnL — not cumulative equity.  "
        "Exposure: avg instrument exposure ratio per period."
    )
    # Shared horizon selector — same window applies to BOTH heatmaps so they
    # are directly comparable (no confusion from different time scales).
    horizon = render_correlation_horizon_selector(key="corr_horizon")

    col_strat_corr, col_exp_corr = st.columns(2)

    with col_strat_corr:
        st.markdown("**Strategy PnL Correlation**")
        strat_corr_matrix = compute_strategy_correlation(
            bar_pnl_matrix if bar_pnl_matrix is not None else pd.DataFrame(),
            horizon=horizon,
        )
        fig_strat_corr = build_correlation_heatmap(
            strat_corr_matrix,
            title=f"Strategy PnL Correlation ({horizon})",
        )
        st.plotly_chart(fig_strat_corr, use_container_width=True)

    with col_exp_corr:
        st.markdown("**Exposure Correlation (by Instrument)**")
        if bundle.exposure is not None and not bundle.exposure.empty:
            exp_corr_matrix = compute_exposure_correlation(
                bundle.exposure, horizon=horizon
            )
            if exp_corr_matrix.empty:
                st.info(
                    f"Too few samples at **{horizon}** horizon to compute exposure "
                    "correlation meaningfully. Try a shorter horizon (1d) or run "
                    "a longer backtest.",
                    icon="ℹ️",
                )
            else:
                fig_exp_corr = build_correlation_heatmap(
                    exp_corr_matrix,
                    title=f"Exposure Correlation ({horizon})",
                )
                st.plotly_chart(fig_exp_corr, use_container_width=True)
        else:
            st.info("No exposure data (`exposure.parquet` missing from results/portfolio/).")



def main() -> None:
    """
    Streamlit page entry point.

    Pure read-only viewer — the engine does NOT run inside Streamlit.
    Run the backtest separately first, then open the dashboard.
    """
    st.set_page_config(
        page_title="Backtest Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1rem; padding-bottom: 0rem; }
            pre { font-size: 0.55rem; line-height: 1.08; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Settings (read-only, no engine import paths needed in Streamlit) ───────
    try:
        from src.backtest_engine.settings import get_settings
        _settings = get_settings()
        window_days: int  = _settings.rolling_sharpe_window_days
    except Exception:
        window_days = 90   # safe default

    # ── Load results ───────────────────────────────────────────────────────────
    bundle = load_result_bundle()

    if bundle is None:
        st.title("Backtest Dashboard")
        st.error(
            "No backtest results found in `results/`. "
            "Run a backtest first:\n\n"
            "```\npython run.py --backtest --strategy <name>\n"
            "python run.py --portfolio-backtest\n```\n\n"
            "Then open the dashboard separately:\n\n"
            "```\nstreamlit run src/backtest_engine/analytics/dashboard/app.py\n```"
        )
        return

    mode_label = "Portfolio" if bundle.run_type == "portfolio" else "Single-Asset"
    st.title(f"Backtest Dashboard — {mode_label} Mode")

    tab_pnl, tab_risk, tab_sim = st.tabs(["PnL Analysis", "Risk Analysis", "Simulation Analysis"])

    with tab_pnl:
        _render_pnl_tab(bundle, window_days=window_days)

    with tab_risk:
        render_risk_tab()
        
    with tab_sim:
        render_simulation_tab()

@st.dialog("Detailed Exit Analysis", width="large")
def _show_exit_analysis_dialog(title: str, trades: pd.DataFrame):
    st.markdown(f"**{title} Exits**")
    if trades.empty:
        st.write("No trades available.")
        return
        
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_mfe_mae_scatter(trades), use_container_width=True)
    with c2:
        st.plotly_chart(build_pnl_decay_chart(trades), use_container_width=True)
        
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(build_holding_time_chart(trades), use_container_width=True)
    with c4:
        st.plotly_chart(build_vol_regime_chart(trades), use_container_width=True)
        
    st.plotly_chart(build_exit_reason_chart(trades), use_container_width=True)



if __name__ == "__main__":
    main()
