"""
Unit tests for dashboard/transforms.py.

Invariants tested:
    1. bar_pnl_matrix contains incremental (not cumulative) PnL.
    2. PnL contribution percentages sum to ~100% when total PnL is positive.
    3. Strategy correlation diagonal = 1, off-diagonal < 1 for decorrelated series.
    4. VaR 95% is <= 0 for a mixed PnL series with losses.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_bar_pnl_matrix,
    compute_strategy_decomp,
    compute_strategy_correlation,
    compute_exposure_correlation,
    compute_pnl_dist_stats,
    compute_rolling_sharpe,
    compute_per_strategy_summary,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_history(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic portfolio history DataFrame with two slot PnL columns.

    The cumulative slot_0_pnl and slot_1_pnl columns are deliberately
    constructed from different random walks so they are decorrelated at the
    bar level.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="30min")

    s0_bar = rng.normal(10, 80, n)
    s1_bar = rng.normal(-5, 60, n)

    df = pd.DataFrame({
        "total_value": 100_000 + np.cumsum(s0_bar + s1_bar),
        "slot_0_pnl":  np.cumsum(s0_bar),
        "slot_1_pnl":  np.cumsum(s1_bar),
    }, index=idx)
    return df


def _make_slots() -> dict:
    return {"0": "StrategyA", "1": "StrategyB"}


def _make_trades(slots: dict) -> pd.DataFrame:
    """Generates minimal trades DataFrame with strategy, pnl, gross_pnl columns."""
    rng = np.random.default_rng(0)
    n = 60
    strategies = [list(slots.values())[i % 2] for i in range(n)]
    pnls = rng.normal(50, 200, n)
    return pd.DataFrame({
        "strategy":  strategies,
        "pnl":       pnls,
        "gross_pnl": pnls + rng.uniform(1, 10, n),  # gross > net (fees)
        "direction": ["LONG"] * n,
    })


# ── Test 1: bar PnL matrix is incremental (not cumulative) ────────────────────

def test_bar_pnl_matrix_not_cumulative() -> None:
    """
    Ensures build_bar_pnl_matrix() returns *incremental* bar PnL, not the
    raw cumulative equity columns.

    A cumulative series is strictly monotonically increasing for a profitable
    strategy — the incremental series must have both positive and negative bars.
    """
    history = _make_history()
    slots   = _make_slots()
    matrix  = build_bar_pnl_matrix(history, slots)

    assert not matrix.empty, "Bar PnL matrix must not be empty."
    assert "StrategyA" in matrix.columns
    assert "StrategyB" in matrix.columns

    for col in matrix.columns:
        series = matrix[col].dropna()
        # A cumulative series has std(diff) ≈ 0; incremental has both signs
        assert (series > 0).any(), f"{col}: no positive bars found — may be cumulative?"
        assert (series < 0).any(), f"{col}: no negative bars found — may be cumulative?"
        # Additional invariant: NOT monotonically non-decreasing
        is_monotone = (series.diff().dropna() >= 0).all()
        assert not is_monotone, f"{col}: series is monotonically non-decreasing (cumulative bug)."


# ── Test 2: PnL contribution absolute sum is ~100% ────────────────────────────

def test_decomp_contribution_abs_sum_is_100_pct() -> None:
    """
    Ensures that the sum of absolute PnL Contribution %
    values across all strategies equals ~100%.
    """
    history = _make_history(seed=1)
    slots   = _make_slots()
    trades  = _make_trades(slots)

    decomp = compute_strategy_decomp(trades, history, slots)
    assert not decomp.empty, "Decomp table must not be empty."

    # Using the new robust denominator sum(|pnl|), the absolute sum of contributions
    # should be exactly 100.0.
    total_abs_contribution = decomp["PnL Contrib (%)"].abs().sum()
    assert math.isclose(total_abs_contribution, 100.0, abs_tol=0.11), (
        f"Absolute PnL contributions sum to {total_abs_contribution:.3f}%, expected ~100%."
    )


# ── Test 3: Correlation diagonal = 1, off-diagonal < 1 ───────────────────────

def test_strategy_correlation_diagonal_is_one() -> None:
    """
    Ensures that the strategy correlation matrix has diagonal = 1.0 and that
    decorrelated random series produce off-diagonal < 1.
    """
    history = _make_history(seed=7)
    slots   = _make_slots()
    bar_pnl = build_bar_pnl_matrix(history, slots)

    corr = compute_strategy_correlation(bar_pnl, horizon="1d")

    assert not corr.empty, "Correlation matrix must not be empty."
    # Diagonal must be exactly 1 (self-correlation)
    for col in corr.columns:
        assert math.isclose(corr.loc[col, col], 1.0, abs_tol=1e-9), (
            f"Diagonal element [{col},{col}] = {corr.loc[col, col]:.6f}, expected 1.0"
        )

    # Off-diagonal must be < 1 for decorrelated normal random walks
    for r in corr.index:
        for c in corr.columns:
            if r != c:
                assert abs(corr.loc[r, c]) < 1.0, (
                    f"Off-diagonal [{r},{c}] = {corr.loc[r, c]:.4f} is not < 1."
                )


# ── Test 4: VaR 95% is <= 0 for mixed PnL ────────────────────────────────────

def test_pnl_dist_stats_var_is_non_positive() -> None:
    """
    Ensures that the historical VaR 95% is <= 0 when the PnL series contains
    losses (which it must for any realistic strategy).

    VaR is the 5th percentile of the daily PnL distribution.
    If there are any losing days, the 5th percentile must be negative.
    """
    rng = np.random.default_rng(42)
    daily = pd.Series(rng.normal(0, 100, 250))  # plenty of negative values

    dist_stats = compute_pnl_dist_stats(daily)

    assert "var_95" in dist_stats
    assert dist_stats["var_95"] <= 0, (
        f"VaR 95% = {dist_stats['var_95']:.2f} — expected <= 0 for mixed PnL series."
    )
    # CVaR (Expected Shortfall) must be even more negative than VaR
    if not np.isnan(dist_stats.get("cvar_95", float("nan"))):
        assert dist_stats["cvar_95"] <= dist_stats["var_95"], (
            "CVaR must be <= VaR 95% (Expected Shortfall >= VaR by definition)."
        )


# ── Test 5: Rolling Sharpe has finite values after warm-up ────────────────────

def test_rolling_sharpe_finite_after_warmup() -> None:
    """
    Ensures rolling Sharpe series has finite (non-NaN) values after the
    initial warm-up window, confirming the calculation is not degenerate.

    The new implementation resamples to daily frequency internally, so the
    history fixture must span well more than `window_days` calendar days.
    _make_history(n=500) creates 500 30-min bars ≈ 38 trading days — not
    enough for a 90-day window. Use n=2000 bars ≈ 153 trading days.
    """
    history = _make_history(n=2000)     # ~153 trading days @ 13 bars/day
    sharpe  = compute_rolling_sharpe(history, window_days=45)  # 45-day window

    assert isinstance(sharpe, pd.Series)
    finite_values = sharpe.dropna()
    assert len(finite_values) > 0, "No finite Sharpe values after warm-up window."


# ── Test 6: Per-strategy summary returns trade count ─────────────────────────

def test_per_strategy_summary_trade_counts() -> None:
    """Ensures per-strategy summary correctly counts trades per strategy."""
    slots  = _make_slots()
    trades = _make_trades(slots)

    summary = compute_per_strategy_summary(trades, slots)

    for strat_name in slots.values():
        assert strat_name in summary, f"Missing strategy {strat_name} in summary."
        expected_count = int((trades["strategy"] == strat_name).sum())
        assert summary[strat_name]["trade_count"] == expected_count, (
            f"Trade count mismatch for {strat_name}: "
            f"got {summary[strat_name]['trade_count']}, expected {expected_count}."
        )


# ── Test 7: Empty inputs return empty outputs gracefully ──────────────────────

def test_transforms_handle_empty_inputs() -> None:
    """All transform functions must return empty, not raise, on empty inputs."""
    empty_df   = pd.DataFrame()
    empty_hist = pd.DataFrame(columns=["total_value"])

    assert build_bar_pnl_matrix(empty_hist, {}).empty
    assert compute_strategy_decomp(empty_df, empty_hist, {}).empty
    assert compute_strategy_correlation(empty_df).empty
    assert compute_exposure_correlation(empty_df).empty

    stats = compute_pnl_dist_stats(pd.Series([], dtype=float))
    assert all(np.isnan(v) for v in stats.values())

    sharpe = compute_rolling_sharpe(empty_hist)
    assert isinstance(sharpe, pd.Series)


# ── Test 8: Per-strategy summary includes Alpha, Beta, T-stat, etc. ────────────

def test_per_strategy_summary_statistics() -> None:
    """Ensures per-strategy summary computes the new statistics correctly."""
    slots = {"0": "StrategyA"}
    rng = np.random.default_rng(42)
    n_days = 60
    
    idx = pd.date_range("2024-01-01", periods=n_days, freq="1D")
    
    # Mock history
    history = pd.DataFrame({
        "total_value": np.full(n_days, 100000.0),
        "slot_0_pnl": np.cumsum(rng.normal(10, 50, n_days))
    }, index=idx)
    
    # Mock instrument closes
    instrument_closes = pd.DataFrame({
        "TEST_SYM": np.cumsum(rng.normal(0, 2, n_days)) + 100
    }, index=idx)
    
    # Mock trades
    trades = pd.DataFrame({
        "strategy": ["StrategyA"] * 10,
        "symbol": ["TEST_SYM"] * 10,
        "pnl": rng.normal(50, 20, 10),
        "direction": ["LONG"] * 10
    })
    
    summary = compute_per_strategy_summary(trades, slots, history=history, instrument_closes=instrument_closes)
    
    strat_stats = summary["StrategyA"]
    
    # Verifying presence and validity of new keys
    for key in ["tstat", "pvalue", "alpha", "alpha_p", "beta", "beta_p"]:
        assert key in strat_stats, f"Missing key {key} in strategy summary"
        assert not np.isnan(strat_stats[key]), f"Key {key} should not be NaN"
        
    # Check that t-stat computation matches scipy's output directly
    from scipy import stats
    expected_t, _ = stats.ttest_1samp(trades["pnl"], 0.0)
    assert math.isclose(strat_stats["tstat"], float(expected_t), rel_tol=1e-5)
