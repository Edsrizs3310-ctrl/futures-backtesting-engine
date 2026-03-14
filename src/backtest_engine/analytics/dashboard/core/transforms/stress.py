from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers, StressScenarioResult
from .risk import compute_var_es_metrics, compute_drawdown_series, compute_annualised_sharpe

def _build_trade_cost_series(
    trades_df: Optional[pd.DataFrame],
    instrument_specs: Dict[str, Dict[str, float]],
) -> tuple[pd.Series, pd.Series]:
    """
    Aggregates daily commission and slippage costs from trade records.

    Methodology:
        Exported trade artifacts store `commission` and `slippage` as positive
        dollar cost magnitudes at the completed-trade level. The daily stress
        preview therefore sums those stored dollar costs directly without
        guessing per-contract units in the dashboard layer.
    """
    if trades_df is None or trades_df.empty or "exit_time" not in trades_df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df = trades_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df = df.dropna(subset=["exit_time"])
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    daily_index = df["exit_time"].dt.normalize()
    commission_daily = (
        df.groupby(daily_index)["commission"].sum().astype(float)
        if "commission" in df.columns
        else pd.Series(dtype=float)
    )

    if "slippage" in df.columns:
        slippage_daily = df.groupby(daily_index)["slippage"].sum().abs().astype(float)
    else:
        slippage_daily = pd.Series(dtype=float)

    return commission_daily, slippage_daily

def _align_series_to_index(series: pd.Series, index: pd.Index) -> pd.Series:
    """Reindexes a numeric series to a target index with zero fill."""
    if series is None or series.empty:
        return pd.Series(0.0, index=index)
    return series.reindex(index, fill_value=0.0).astype(float)

def _build_equity_from_daily_pnl(initial_equity: float, daily_pnl: pd.Series) -> pd.Series:
    """Reconstructs a daily equity curve from initial equity and daily PnL."""
    if daily_pnl is None or daily_pnl.empty:
        return pd.Series(dtype=float)
    return pd.Series(initial_equity + daily_pnl.astype(float).cumsum(), index=daily_pnl.index)


def _preserve_anchor(daily_pnl: pd.Series) -> pd.Series:
    """
    Preserves the baseline first-period equity anchor across all scenarios.

    Methodology:
        The first daily PnL sample in the risk pipeline is a synthetic 0.0 anchor
        created by `diff().fillna(0.0)`, not a real realized increment. Stress
        transformations should never modify that first point, otherwise the
        stressed curve starts from a different effective equity than baseline.
    """
    anchored = daily_pnl.astype(float).copy()
    if not anchored.empty:
        anchored.iloc[0] = 0.0
    return anchored


def _apply_volatility_preview(clean_pnl: pd.Series, multiplier: float) -> pd.Series:
    """
    Applies the volatility preview while keeping the equity anchor unchanged.

    Methodology:
        Only true realized increments after the first anchor point are demeaned
        and rescaled. The preview contract is: multiplier 1.0 = baseline
        dispersion, >1.0 amplifies dispersion, and <1.0 compresses dispersion.
    """
    anchored = _preserve_anchor(clean_pnl)
    if len(anchored) <= 1:
        return anchored

    stressed = anchored.copy()
    realized = anchored.iloc[1:]
    mean_pnl = float(realized.mean())
    centered_pnl = realized - mean_pnl
    stressed.iloc[1:] = mean_pnl + centered_pnl * float(multiplier)
    return stressed

def _build_scenario_metrics(
    equity: pd.Series,
    daily_pnl: pd.Series,
    primary_confidence: float,
    tail_confidence: float,
    risk_free_rate: float,
) -> Dict[str, float]:
    """Computes comparable summary metrics for a baseline or stressed path."""
    var_metrics = compute_var_es_metrics(daily_pnl, primary_confidence, tail_confidence)
    drawdown = compute_drawdown_series(equity)
    daily_returns = equity.pct_change(fill_method=None).dropna() if not equity.empty else pd.Series(dtype=float)
    sharpe = compute_annualised_sharpe(daily_returns, risk_free_rate=risk_free_rate)

    return {
        "final_pnl": float(daily_pnl.sum()),
        "end_equity": float(equity.iloc[-1]) if not equity.empty else float("nan"),
        "var_primary": var_metrics["var_primary"],
        "es_primary": var_metrics["es_primary"],
        "var_tail": var_metrics["var_tail"],
        "es_tail": var_metrics["es_tail"],
        "max_drawdown_pct": abs(float(drawdown.min())) if not drawdown.empty else float("nan"),
        "sharpe": sharpe,
    }

def compute_stress_scenarios(
    daily_equity: pd.Series,
    daily_pnl: pd.Series,
    trades_df: Optional[pd.DataFrame],
    instrument_specs: Dict[str, Dict[str, float]],
    stress_multipliers: StressMultipliers,
    primary_confidence: float,
    tail_confidence: float,
    risk_free_rate: float = 0.0,
) -> List[StressScenarioResult]:
    """
    Builds baseline and stressed risk scenarios from the realised daily PnL path.

    Methodology:
        Volatility preview scales the demeaned realized daily PnL path while
        preserving the baseline first-period equity anchor. Commission and
        slippage multipliers represent total cost levels relative to the
        realized baseline:
            - 1.0 keeps realized costs unchanged,
            - 2.0 doubles realized costs,
            - 0.5 halves realized costs.
        The preview never reruns fills, signals, or sizing; it transforms the
        realized daily PnL path only.
    """
    if daily_equity is None or daily_equity.dropna().empty or daily_pnl is None or daily_pnl.empty:
        return []

    clean_equity = daily_equity.dropna().astype(float)
    clean_pnl = daily_pnl.reindex(clean_equity.index, fill_value=0.0).astype(float)
    initial_equity = float(clean_equity.iloc[0])

    commission_daily, slippage_daily = _build_trade_cost_series(trades_df, instrument_specs)
    commission_daily = _align_series_to_index(commission_daily, clean_pnl.index)
    slippage_daily = _align_series_to_index(slippage_daily, clean_pnl.index)

    baseline_pnl = _preserve_anchor(clean_pnl)
    volatility_pnl = _apply_volatility_preview(clean_pnl, stress_multipliers.volatility)
    commission_adjustment = commission_daily * (float(stress_multipliers.commission) - 1.0)
    slippage_adjustment = slippage_daily * (float(stress_multipliers.slippage) - 1.0)

    scenario_map = {
        "baseline": baseline_pnl,
        "volatility": volatility_pnl,
        "slippage": baseline_pnl - slippage_adjustment,
        "commission": baseline_pnl - commission_adjustment,
        "combined": volatility_pnl - slippage_adjustment - commission_adjustment,
    }
    scenario_labels = {
        "baseline": "Baseline",
        "volatility": f"Volatility Preview x{stress_multipliers.volatility:.1f}",
        "slippage": f"Slippage Preview x{stress_multipliers.slippage:.1f}",
        "commission": f"Commission Preview x{stress_multipliers.commission:.1f}",
        "combined": "Combined Preview Shock",
    }

    baseline_final_pnl = float(baseline_pnl.sum())
    results: List[StressScenarioResult] = []
    for name, scenario_pnl in scenario_map.items():
        scenario_equity = _build_equity_from_daily_pnl(initial_equity, scenario_pnl)
        scenario_metrics = _build_scenario_metrics(
            scenario_equity,
            scenario_pnl,
            primary_confidence=primary_confidence,
            tail_confidence=tail_confidence,
            risk_free_rate=risk_free_rate,
        )
        results.append(
            StressScenarioResult(
                name=name,
                label=scenario_labels[name],
                equity=scenario_equity,
                daily_pnl=scenario_pnl,
                metrics=scenario_metrics,
                pnl_delta=float(scenario_metrics["final_pnl"] - baseline_final_pnl),
            )
        )

    return results

