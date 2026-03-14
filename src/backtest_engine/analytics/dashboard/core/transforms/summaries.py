from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
from .pnl import (
    build_bar_pnl_matrix,
    build_strategy_equity_curve,
    resample_pnl_to_horizon,
    _HORIZON_RULE,
    compute_pnl_dist_stats as _compute_pnl_dist_stats,
)
from .correlations import _MIN_CORR_SAMPLES

def compute_strategy_decomp(
    trades_df: pd.DataFrame,
    history: pd.DataFrame,
    slots: Dict[str, str],
    tail_confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Builds the Strategy PnL Decomposition table.

    Columns and methodology:

    Closed PnL ($)
        sum(net_pnl) from closed trades per strategy.

    PnL Contrib (%)
        Closed PnL / sum(|Closed_PnL_all_strategies|) * 100.
        This keeps the comparison anchored to realised trades rather than
        mark-to-market noise embedded in the cumulative slot equity curve.
        Falls back to MTM only when trade-level PnL is unavailable.

    Risk Contrib (%)
        (cov(slot_bar_pnl, portfolio_bar_pnl) / var(portfolio_bar_pnl)) * 100
        This is the beta-decomposition of portfolio variance.
        Values sum to exactly 100%.

    Max DD PnL ($)
        Cumulative PnL of the strategy during the portfolio's maximum
        drawdown period (from peak to trough). Negative means the
        strategy contributed to the portfolio's worst period.

    Tail PnL (CVaR) ($)
        Average daily PnL of the strategy on the portfolio tail days implied by
        `tail_confidence`. This is a marginal CVaR-style measure.

    Signal PnL ($)
        sum(closed pnl + commission + slippage) = PnL before execution friction.
        This is the raw directional signal value net of market impact.

    Execution Cost ($)
        sum(-(commission + slippage))  (negative = we paid fees/slippage)

    Daily PnL Sharpe-like
        Annualised mean/std signal-to-noise ratio computed from the strategy's
        daily dollar PnL series. This is scale-dependent and is intentionally
        not labeled as a return-based Sharpe ratio.

    Args:
        trades_df: Trades DataFrame with strategy, pnl, commission columns.
        history: Portfolio history used for bar-level risk computation.
        slots: {slot_id: strategy_name} — maps history columns to names.
        tail_confidence: Confidence level used to define the lower-tail mask.

    Returns:
        DataFrame with one row per strategy, ready for rendering.
    """
    if history.empty or not slots:
        return pd.DataFrame()
        
    # ── 1. Bar-level PnL from history for MTM and risk ────────────────────────
    bar_pnl = build_bar_pnl_matrix(history, slots)
    if bar_pnl.empty:
        return pd.DataFrame()

    portfolio_bar_pnl: pd.Series = bar_pnl.sum(axis=1)
    daily_portfolio: pd.Series = portfolio_bar_pnl.resample("1D").sum()
    portfolio_var: float = float(portfolio_bar_pnl.var())

    # Strategy MTM PnL is still used for risk attribution and drawdown context,
    # but is intentionally not rendered as a separate decomposition column.
    strat_mtm_pnl: pd.Series = bar_pnl.sum(axis=0)

    # ── 2. Net PnL and execution cost from trade records ──────────────────────
    if trades_df is not None and not trades_df.empty and "strategy" in trades_df.columns:
        strat_closed_pnl: pd.Series = trades_df.groupby("strategy")["pnl"].sum()
        
        # Calculate fees (negative numbers in raw data usually, so we add them)
        exec_cost_s: pd.Series = pd.Series(0.0, index=strat_closed_pnl.index)
        if "commission" in trades_df.columns:
            exec_cost_s += trades_df.groupby("strategy")["commission"].sum().fillna(0.0)
        if "slippage" in trades_df.columns:
            exec_cost_s += trades_df.groupby("strategy")["slippage"].sum().fillna(0.0)
            
        strat_exec_cost = exec_cost_s
    else:
        strat_closed_pnl = pd.Series(0.0, index=strat_mtm_pnl.index)
        strat_exec_cost = pd.Series(0.0, index=strat_mtm_pnl.index)

    abs_total_closed_pnl: float = float(strat_closed_pnl.abs().sum())
    abs_total_mtm_pnl: float = float(strat_mtm_pnl.abs().sum())

    # ── 3. Drawdown and VaR masks ─────────────────────────────────────────────
    # Maximum drawdown window: from overall peak to subsequent trough
    port_cum: pd.Series   = portfolio_bar_pnl.cumsum()
    running_peak: pd.Series = port_cum.cummax()
    trough_after_peak = port_cum.iloc[running_peak.argmax():]
    
    if not trough_after_peak.empty:
        dd_start = running_peak.index[running_peak.argmax()]
        dd_end   = trough_after_peak.index[trough_after_peak.argmin()]
        dd_mask  = (portfolio_bar_pnl.index >= dd_start) & (portfolio_bar_pnl.index <= dd_end)
    else:
        dd_mask = pd.Series(False, index=portfolio_bar_pnl.index)

    # Tail mask parameterized from the requested confidence level.
    if len(daily_portfolio) >= 20:
        var_threshold = float(daily_portfolio.quantile(1.0 - float(tail_confidence)))
        tail_dates    = daily_portfolio[daily_portfolio <= var_threshold].index
    else:
        tail_dates = daily_portfolio.index[:0]  # empty

    ann_factor = np.sqrt(252.0)

    # ── 4. Assemble rows ──────────────────────────────────────────────────────
    rows: List[dict] = []
    for name in strat_mtm_pnl.index:
        mtm_pnl_val: float = float(strat_mtm_pnl[name])
        closed_pnl_val: float = float(strat_closed_pnl.get(name, 0.0))
        exec_cost_val: float = float(strat_exec_cost.get(name, 0.0))

        # Signal PnL restores pre-fee value: Closed PnL - (Execution Cost)
        # assuming Exec Cost is negative (e.g. -10 for fees). Signal = Closed PnL - (-10) = Closed + 10.
        # Wait: if Execution Cost in DB is POSITIVE (stored as absolute value of fees)?
        # For IB fetcher usually cost is positive. Let's make Exec Cost uniformly negative output.
        actual_exec_cost = -abs(exec_cost_val)
        signal_pnl: float  = round(closed_pnl_val - actual_exec_cost, 0)

        # PnL contribution is anchored to realised trade PnL when available.
        contrib_base = abs_total_closed_pnl if abs_total_closed_pnl > 0 else abs_total_mtm_pnl
        contrib_value = closed_pnl_val if abs_total_closed_pnl > 0 else mtm_pnl_val
        pnl_contrib: float = (
            contrib_value / contrib_base * 100.0 if contrib_base > 0 else float("nan")
        )

        # Risk contribution (beta to portfolio) * 100
        if portfolio_var > 0:
            risk_contrib = float(bar_pnl[name].cov(portfolio_bar_pnl) / portfolio_var) * 100.0
        else:
            risk_contrib = float("nan")

        # DD PnL: cumulative return during max portfolio drawdown window
        dd_pnl = float(bar_pnl[name][dd_mask].sum()) if dd_mask.any() else float("nan")

        # Tail PnL: avg strategy daily PnL on worst portfolio days
        if len(tail_dates) > 0:
            slot_daily = bar_pnl[name].resample("1D").sum()
            common_tail = slot_daily.index.intersection(tail_dates)
            tail_pnl = float(slot_daily.loc[common_tail].mean()) if len(common_tail) > 0 else float("nan")
        else:
            tail_pnl = float("nan")

        # Scale-dependent daily PnL signal-to-noise ratio.
        daily_strat_pnl = bar_pnl[name].resample("1D").sum()
        roll_std = float(daily_strat_pnl.std())
        roll_mean = float(daily_strat_pnl.mean())
        sharpe_val = (roll_mean / roll_std * ann_factor) if roll_std > 1e-8 else float("nan")

        rows.append({
            "Strategy":             name,
            "Daily PnL Sharpe-like": round(sharpe_val, 2) if not np.isnan(sharpe_val) else float("nan"),
            "Closed PnL ($)":       round(closed_pnl_val, 0),
            "PnL Contrib (%)":      round(pnl_contrib, 1) if not np.isnan(pnl_contrib) else float("nan"),
            "Risk Contrib (%)":     round(risk_contrib, 1) if not np.isnan(risk_contrib) else float("nan"),
            "Max DD PnL ($)":       round(dd_pnl, 0) if not np.isnan(dd_pnl) else float("nan"),
            "Tail PnL (CVaR)":      round(tail_pnl, 0) if not np.isnan(tail_pnl) else float("nan"),
            "Signal PnL ($)":       signal_pnl,
            "Exec Cost ($)":        actual_exec_cost,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()



# ── Strategy Correlation ───────────────────────────────────────────────────────

def compute_strategy_correlation(
    bar_pnl_matrix: pd.DataFrame,
    horizon: str = "1d",
) -> pd.DataFrame:
    """
    Computes a Pearson correlation matrix of per-strategy incremental PnL.

    Methodology:
        Correlation is computed on *bar/daily PnL series*, not cumulative equity.
        Cumulative equity correlations are artificially inflated by shared drift
        and are meaningless for risk decomposition.

        Horizon resampling lets the user detect correlations that appear only
        at longer time scales (e.g. independent intrabar but correlated daily).

    Args:
        bar_pnl_matrix: Output of build_bar_pnl_matrix() — incremental PnL.
        horizon: Resampling horizon ('1d', '1w', '1m').

    Returns:
        Correlation DataFrame (strategies x strategies). Empty if < 2 strategies
        or too few observations after resampling.
    """
    if bar_pnl_matrix.empty or bar_pnl_matrix.shape[1] < 2:
        return pd.DataFrame()

    resampled = resample_pnl_to_horizon(bar_pnl_matrix, horizon)
    if len(resampled) < _MIN_CORR_SAMPLES:
        return pd.DataFrame()
    return resampled.corr(method="pearson")


# ── Exposure Correlation ───────────────────────────────────────────────────────

def compute_exposure_correlation(
    exposure_df: pd.DataFrame,
    horizon: str = "1d",
) -> pd.DataFrame:
    """
    Computes correlation between per-instrument absolute exposures.

    Methodology:
        Step 1: Keep only `*_notional` columns.
        Step 2: Aggregate by symbol — sum slot_N_SYM_notional across all slots.
        Step 3: Resample using `mean()` (average active exposure per period). 
                We use the raw exposure, NOT the ratio to total gross_exposure. 
                Using the ratio crushes the variance and leads to 0.00 correlation
                for larger horizons when one strategy's exposure dominates.
        Step 4: Compute Pearson correlation on the resampled absolute levels.

    Args:
        exposure_df: Raw bar-level exposure DataFrame with columns like
                     `slot_0_NQ_notional`.
        horizon: Resampling horizon '1d', '1w', '1m' (default: '1d').

    Returns:
        Correlation matrix (symbols x symbols). Empty if < 2 distinct symbols
        or too few samples.
    """
    if exposure_df is None or exposure_df.empty:
        return pd.DataFrame()

    # Step 1: keep only notional columns
    notional_cols = [c for c in exposure_df.columns if c.endswith("_notional")]
    if not notional_cols:
        notional_cols = exposure_df.columns.tolist()

    df = exposure_df[notional_cols].copy()

    # Step 2: extract symbol name (slot_N_SYM_notional -> SYM)
    symbol_map: dict = {}
    for col in notional_cols:
        parts = col.split("_")
        if len(parts) >= 4 and parts[0] == "slot" and parts[-1] == "notional":
            sym = "_".join(parts[2:-1])
        else:
            sym = col
        symbol_map[col] = sym

    # Step 3: sum per-symbol notional across all slots
    symbol_notional: dict = {}
    for col, sym in symbol_map.items():
        symbol_notional[sym] = symbol_notional.get(sym, 0.0) + df[col].fillna(0.0)

    sym_df = pd.DataFrame(symbol_notional, index=df.index)

    if sym_df.shape[1] < 2:
        return pd.DataFrame()

    # Step 4: resample using MEAN (average absolute exposure level)
    # This provides a more robust correlation measure for longer intervals.
    rule = _HORIZON_RULE.get(horizon, "1D")
    resampled = sym_df.abs().resample(rule).mean()

    if len(resampled) < _MIN_CORR_SAMPLES:
        return pd.DataFrame()

    # Step 5: drop columns with near-zero variance (always flat strategy)
    active_cols = [c for c in resampled.columns if resampled[c].std() > 1e-8]
    if len(active_cols) < 2:
        return pd.DataFrame()

    return resampled[active_cols].corr(method="pearson")


# ── Rolling Sharpe ─────────────────────────────────────────────────────────────

def compute_rolling_sharpe(
    history: pd.DataFrame,
    window_days: int = 90,
    bars_per_day: float = 13.0,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Computes rolling Sharpe ratio on *daily* equity returns.

    Methodology:
        Step 1: Resample bar-level equity to end-of-day snapshots.
        Step 2: Compute daily percentage returns:
                  r_t = equity_t / equity_{t-1} - 1
        Step 3: Rolling window = window_days calendar days.
        Step 4: Annualise by sqrt(252).

        Using *daily* returns instead of bar-level returns avoids the
        +/-10-15 Sharpe artefact caused by near-zero intraday return std.

        The std guard (std < 1e-8) prevents division by near-zero when the
        strategy is flat for extended periods.

    Args:
        history: Portfolio history with 'total_value'.
        window_days: Rolling window in calendar days (default: 90 from settings).
        bars_per_day: Not used for computation — stored for reference only.
        risk_free_rate: Annualised risk-free rate (default 0).

    Returns:
        pd.Series of daily rolling Sharpe values (indexed by date).
    """
    if history.empty or "total_value" not in history.columns:
        return pd.Series(dtype=float)

    # Step 1: resample to end-of-day equity level
    daily_equity: pd.Series = (
        history["total_value"]
        .resample("1D")
        .last()
        .dropna()
    )

    if len(daily_equity) < 3:
        return pd.Series(dtype=float)

    # Step 2: daily returns (not PnL — must be return = equity_t/equity_{t-1} - 1)
    daily_ret: pd.Series = daily_equity.pct_change(fill_method=None).dropna()

    ann_factor: float = np.sqrt(252.0)
    rf_daily:   float = risk_free_rate / 252.0

    # Step 3: rolling window in days
    rolling_mean: pd.Series = (daily_ret - rf_daily).rolling(
        window=window_days, min_periods=max(window_days // 2, 5)
    ).mean()
    rolling_std: pd.Series = daily_ret.rolling(
        window=window_days, min_periods=max(window_days // 2, 5)
    ).std()

    # Step 4: Sharpe — clip std at 1e-8 to prevent inf on flat periods
    safe_std: pd.Series = rolling_std.clip(lower=1e-8)
    return (rolling_mean / safe_std) * ann_factor


# ── PnL Distribution Stats ─────────────────────────────────────────────────────

def compute_pnl_dist_stats(
    daily_pnl: pd.Series,
    var_confidence: float = 0.95,
) -> Dict[str, float]:
    """Backward-compatible wrapper around the shared PnL distribution helper."""
    return _compute_pnl_dist_stats(daily_pnl, var_confidence=var_confidence)


# ── Per-strategy summary for equity hover tooltip ──────────────────────────────


def compute_per_strategy_summary(
    trades_df: pd.DataFrame,
    slots: Dict[str, str],
    history: Optional[pd.DataFrame] = None,
    instrument_closes: Optional[pd.DataFrame] = None,
    slot_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Builds a lightweight per-strategy stat dict for equity hover tooltips.

    Methodology:
        For each strategy, derives from the closed-trade record:
        - total_pnl, trade_count, win_rate, avg_trade, max_loss.
        - tstat, pvalue using 1-sample T-test on trade PnLs.
        - alpha, beta vs the strategy's traded instrument.
        These are attached as customdata to the Plotly scatter trace so they
        appear in the hovertemplate without any server round-trip.

    Args:
        trades_df: Trades DataFrame with 'strategy' and 'pnl' columns.
        slots: {slot_id: strategy_name} mapping.
        history: Portfolio history DataFrame for initial capital reference.
        instrument_closes: DataFrame of daily instrument closes.
        slot_weights: Optional {slot_id: weight} to correctly scale returns.

    Returns:
        Dict {strategy_name: {metric_name: value}}.
    """
    result: Dict[str, Dict[str, object]] = {}

    if trades_df is None or trades_df.empty or "strategy" not in trades_df.columns:
        return result

    # Portfolio initial capital to convert slot $ PnL to % return for regression
    try:
        from src.backtest_engine.settings import BacktestSettings
        initial_cap = float(BacktestSettings().initial_capital)
    except Exception:
        initial_cap = 1_000_000.0

    if history is not None and not history.empty and "total_value" in history.columns:
        initial_cap = float(history["total_value"].iloc[0])

    for str_id, strat_name in slots.items():
        sub: pd.DataFrame = trades_df[trades_df["strategy"] == strat_name]
        
        # Default stats structure
        stats_dict = {
            "total_pnl":   0.0,
            "trade_count": 0,
            "win_rate":    0.0,
            "avg_trade":   0.0,
            "max_loss":    0.0,
            "tstat":       0.0,
            "pvalue":      1.0,
            "alpha":       0.0,
            "alpha_p":     1.0,
            "beta":        0.0,
            "beta_p":      1.0,
        }
        
        if not sub.empty:
            pnls: pd.Series    = sub["pnl"]
            winners: pd.Series = pnls[pnls > 0]
            
            stats_dict["total_pnl"]   = round(float(pnls.sum()), 0)
            stats_dict["trade_count"] = len(sub)
            stats_dict["win_rate"]    = round(float(len(winners) / len(pnls) * 100), 1)
            stats_dict["avg_trade"]   = round(float(pnls.mean()), 0)
            stats_dict["max_loss"]    = round(float(pnls.min()), 0)

            # T-Stat and P-Value
            if len(pnls) > 1 and pnls.std() > 0:
                t_stat, p_val = stats.ttest_1samp(pnls, 0.0)
                stats_dict["tstat"] = float(t_stat)
                stats_dict["pvalue"] = float(p_val)

        # Alpha & Beta Calculation
        if history is not None and instrument_closes is not None and not sub.empty:
            # Determine the main symbol traded by this strategy
            symbols = sub["symbol"].value_counts().index.tolist()
            if symbols:
                target_sym = symbols[0]
                strat_pnl_col = f"slot_{str_id}_pnl"

                if strat_pnl_col in history.columns and target_sym in instrument_closes.columns:
                    slot_weight = None
                    if slot_weights and str_id in slot_weights:
                        slot_weight = float(slot_weights[str_id])

                    strategy_equity = build_strategy_equity_curve(
                        history=history,
                        slot_id=str(str_id),
                        slot_weight=slot_weight,
                        slot_count=len(slots),
                    )
                    strat_rets = (
                        strategy_equity
                        .resample("1D")
                        .last()
                        .dropna()
                        .pct_change(fill_method=None)
                        .dropna()
                    )

                    # Instrument Returns
                    inst_close = instrument_closes[target_sym].resample("1D").last()
                    inst_rets  = inst_close.pct_change(fill_method=None).fillna(0.0)

                    # Align timestamps and drop any remaining NaNs
                    aligned = pd.concat([strat_rets, inst_rets], axis=1, join="inner").dropna()

                    if len(aligned) > 2:
                        y = aligned.iloc[:, 0].values
                        x = aligned.iloc[:, 1].values

                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                        stats_dict["beta"]   = float(slope)
                        stats_dict["beta_p"] = float(p_value)

                        # Annualised alpha: intercept is daily excess return on portfolio
                        # capital, so × 252 gives annual, × 100 converts to percentage.
                        ann_alpha = intercept * 252
                        stats_dict["alpha"] = float(ann_alpha * 100)

                        # Alpha p-value from the intercept t-test of the same regression
                        n      = len(x)
                        mean_x = np.mean(x)
                        ss_x   = np.sum((x - mean_x) ** 2)

                        if ss_x > 0 and n > 2:
                            residuals    = y - (intercept + slope * x)
                            residual_var = np.sum(residuals ** 2) / (n - 2)
                            se_intercept = np.sqrt(
                                residual_var * ((1.0 / n) + (mean_x ** 2 / ss_x))
                            )
                            if np.isfinite(se_intercept) and se_intercept > 0:
                                t_alpha   = intercept / se_intercept
                                alpha_pval = stats.t.sf(np.abs(t_alpha), n - 2) * 2
                                stats_dict["alpha_p"] = float(alpha_pval)

        result[strat_name] = stats_dict

    return result

def compute_exit_summary(trades_df: pd.DataFrame, slots: dict) -> pd.DataFrame:
    """
    Computes summary metrics for the Exit Analysis screener table.
    """
    if trades_df is None or trades_df.empty or not slots:
        return pd.DataFrame()
        
    rows = []
    for slot_id, strat_name in slots.items():
        if "strategy" in trades_df.columns:
            st_t = trades_df[trades_df["strategy"] == strat_name]
        else:
            st_t = trades_df

        if st_t.empty:
            continue
            
        mfe_val = float(st_t["mfe"].mean()) if "mfe" in st_t.columns else 0.0
        mae_val = float(st_t["mae"].mean()) if "mae" in st_t.columns else 0.0
        pnl_val = float(st_t["pnl_decay_60m"].mean()) if "pnl_decay_60m" in st_t.columns else 0.0
        
        # Avg holding time string
        ht_str = "N/A"
        if "holding_time" in st_t.columns:
            ht = st_t["holding_time"].dropna()
            if not ht.empty:
                avg_ht = ht.mean()
                if pd.notna(avg_ht):
                    total_min = int(avg_ht.total_seconds() / 60)
                    hours, mins = divmod(total_min, 60)
                    if hours > 0:
                        ht_str = f"{hours}h {mins}m"
                    else:
                        ht_str = f"{mins}m"

        # Win rate, total trades and avg trade
        trade_count = len(st_t)
        if trade_count > 0:
            pnls = st_t["pnl"]
            winners = pnls[pnls > 0]
            win_rate = float(len(winners) / trade_count * 100)
            avg_trade = float(pnls.mean())
        else:
            win_rate = 0.0
            avg_trade = 0.0

        rows.append({
            "Strategy": strat_name,
            "Total Trades": trade_count,
            "Win Rate %": round(win_rate, 1),
            "Avg Trade ($)": round(avg_trade, 0),
            "PnL Decay (1h)": round(pnl_val, 0) if not np.isnan(pnl_val) else 0.0,
            "Highest PnL (MFE)": round(mfe_val, 0) if not np.isnan(mfe_val) else 0.0,
            "Avg MAE": round(mae_val, 0) if not np.isnan(mae_val) else 0.0,
            "Avg Hold Time": ht_str,
        })
        
    return pd.DataFrame(rows)

