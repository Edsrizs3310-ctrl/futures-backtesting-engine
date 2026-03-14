"""
Unit tests for dashboard risk transforms.
"""

from __future__ import annotations

import math

import pandas as pd

from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_risk_profile,
    build_strategy_equity_curve,
    compute_drawdown_episodes,
    compute_stress_scenarios,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers


def test_build_strategy_equity_curve_uses_slot_weight_and_fills_nans() -> None:
    """Strategy standalone equity must use allocated capital plus cumulative slot PnL."""
    idx = pd.date_range("2024-01-01", periods=4, freq="1D")
    history = pd.DataFrame(
        {
            "total_value": [100_000.0, 100_000.0, 100_000.0, 100_000.0],
            "slot_0_pnl": [float("nan"), 100.0, 50.0, 150.0],
        },
        index=idx,
    )

    equity = build_strategy_equity_curve(history, slot_id="0", slot_weight=0.25, slot_count=4)
    expected = pd.Series([25_000.0, 25_100.0, 25_050.0, 25_150.0], index=idx)

    pd.testing.assert_series_equal(equity, expected)


def test_compute_drawdown_episodes_tracks_depth_and_duration() -> None:
    """Drawdown episodes must preserve peak-to-trough depth and recovery duration."""
    idx = pd.date_range("2024-01-01", periods=8, freq="1D")
    drawdown = pd.Series([0.0, -5.0, -2.0, 0.0, 0.0, -3.0, -7.0, -1.0], index=idx)

    episodes = compute_drawdown_episodes(drawdown)

    assert len(episodes) == 2
    assert math.isclose(float(episodes.iloc[0]["depth_pct"]), -5.0, abs_tol=1e-9)
    assert math.isclose(float(episodes.iloc[0]["duration_days"]), 2.0, abs_tol=1e-9)
    assert math.isclose(float(episodes.iloc[1]["depth_pct"]), -7.0, abs_tol=1e-9)
    assert math.isclose(float(episodes.iloc[1]["duration_days"]), 2.0, abs_tol=1e-9)


def test_build_risk_profile_returns_tail_metrics_and_stress_results() -> None:
    """Risk profile should compute VaR / ES, rolling diagnostics and stress scenarios."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1D")
    equity = pd.Series(
        [100_000.0, 100_400.0, 99_800.0, 100_100.0, 99_300.0, 99_900.0, 99_100.0, 99_500.0, 98_700.0, 99_000.0],
        index=idx,
    )
    trades = pd.DataFrame(
        {
            "symbol": ["ES", "ES", "ES", "ES"],
            "quantity": [1.0, 1.0, 1.0, 1.0],
            "commission": [5.0, 5.0, 5.0, 5.0],
            "slippage": [0.25, 0.25, 0.25, 0.25],
            "exit_time": idx[[2, 4, 6, 8]],
        }
    )

    profile = build_risk_profile(
        label="Test Strategy",
        equity=equity,
        trades_df=trades,
        instrument_specs={"ES": {"multiplier": 50.0}},
        primary_confidence=0.95,
        tail_confidence=0.99,
        rolling_var_window_days=5,
        rolling_vol_windows=(3, 5),
        stress_multipliers=StressMultipliers(volatility=2.0, slippage=3.0, commission=2.0),
        risk_free_rate=0.0,
    )

    assert profile.label == "Test Strategy"
    assert profile.summary["var_primary"] >= 0.0
    assert profile.summary["es_primary"] >= profile.summary["var_primary"]
    assert not profile.rolling_var.empty
    assert not profile.rolling_vol.empty
    assert len(profile.stress_results) == 5

    results_by_name = {scenario.name: scenario for scenario in profile.stress_results}
    assert set(results_by_name) == {"baseline", "volatility", "slippage", "commission", "combined"}
    assert results_by_name["baseline"].equity.iloc[0] == equity.iloc[0]
    assert results_by_name["combined"].equity.iloc[0] == equity.iloc[0]
    assert results_by_name["combined"].metrics["final_pnl"] <= results_by_name["baseline"].metrics["final_pnl"]
    assert results_by_name["combined"].metrics["max_drawdown_pct"] >= results_by_name["baseline"].metrics["max_drawdown_pct"]


def test_stress_scenarios_allow_sub_one_cost_multipliers_and_keep_anchor() -> None:
    """Stress preview must keep the initial equity anchor and support lower cost levels coherently."""
    idx = pd.date_range("2024-01-01", periods=4, freq="1D")
    daily_equity = pd.Series([100_000.0, 100_100.0, 100_050.0, 100_200.0], index=idx)
    daily_pnl = daily_equity.diff().fillna(0.0)
    trades = pd.DataFrame(
        {
            "symbol": ["ES", "ES"],
            "commission": [10.0, 10.0],
            "slippage": [25.0, 25.0],
            "exit_time": idx[[1, 3]],
        }
    )

    results = compute_stress_scenarios(
        daily_equity=daily_equity,
        daily_pnl=daily_pnl,
        trades_df=trades,
        instrument_specs={"ES": {"multiplier": 50.0}},
        stress_multipliers=StressMultipliers(volatility=1.0, slippage=0.5, commission=0.5),
        primary_confidence=0.95,
        tail_confidence=0.99,
        risk_free_rate=0.0,
    )

    results_by_name = {scenario.name: scenario for scenario in results}
    assert results_by_name["slippage"].equity.iloc[0] == daily_equity.iloc[0]
    assert results_by_name["commission"].equity.iloc[0] == daily_equity.iloc[0]
    assert results_by_name["combined"].equity.iloc[0] == daily_equity.iloc[0]
    assert results_by_name["combined"].metrics["final_pnl"] > results_by_name["baseline"].metrics["final_pnl"]
