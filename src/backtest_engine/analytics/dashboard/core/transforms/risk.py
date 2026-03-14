from __future__ import annotations
from typing import Dict, List, Sequence, Optional
import numpy as np
import pandas as pd
from src.backtest_engine.analytics.dashboard.risk_analysis.models import RiskProfile, StressMultipliers, StressScenarioResult

def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Computes point-in-time drawdown percentages for an equity curve.

    Methodology:
        Drawdown is measured relative to the running peak so that both
        portfolio-level and strategy-level risk can be compared on the same
        peak-to-trough basis.
    """
    if equity is None or equity.dropna().empty:
        return pd.Series(dtype=float)

    clean_equity = equity.dropna().astype(float)
    running_peak = clean_equity.cummax()
    return (clean_equity - running_peak) / running_peak * 100.0

def compute_drawdown_episodes(drawdown_pct: pd.Series) -> pd.DataFrame:
    """
    Extracts discrete drawdown episodes from a point-in-time drawdown series.

    Methodology:
        Consecutive timestamps with drawdown < 0 belong to the same episode.
        For each episode we keep the peak-to-trough depth and total recovery
        duration, which is the relevant distribution for path-dependent risk.
    """
    if drawdown_pct is None or drawdown_pct.dropna().empty:
        return pd.DataFrame(
            columns=["start", "trough", "end", "depth_pct", "depth_abs_pct", "duration_days"]
        )

    clean_drawdown = drawdown_pct.dropna().astype(float)
    rows: List[dict] = []
    episode_start = None
    trough_time = None
    trough_value = 0.0

    for timestamp, value in clean_drawdown.items():
        if value < 0.0 and episode_start is None:
            episode_start = timestamp
            trough_time = timestamp
            trough_value = float(value)
            continue

        if episode_start is None:
            continue

        if value < trough_value:
            trough_value = float(value)
            trough_time = timestamp

        if value >= 0.0:
            duration_days = (timestamp - episode_start).total_seconds() / 86_400.0
            rows.append(
                {
                    "start": episode_start,
                    "trough": trough_time,
                    "end": timestamp,
                    "depth_pct": trough_value,
                    "depth_abs_pct": abs(trough_value),
                    "duration_days": duration_days,
                }
            )
            episode_start = None
            trough_time = None
            trough_value = 0.0

    if episode_start is not None:
        end_time = clean_drawdown.index[-1]
        duration_days = (end_time - episode_start).total_seconds() / 86_400.0
        rows.append(
            {
                "start": episode_start,
                "trough": trough_time,
                "end": end_time,
                "depth_pct": trough_value,
                "depth_abs_pct": abs(trough_value),
                "duration_days": duration_days,
            }
        )

    return pd.DataFrame(rows)

def compute_var_es_metrics(
    daily_pnl: pd.Series,
    primary_confidence: float,
    tail_confidence: float,
) -> Dict[str, float]:
    """
    Computes historical VaR / Expected Shortfall for two confidence levels.

    Methodology:
        Uses historical simulation on daily PnL so the risk numbers preserve
        the empirical asymmetry and fat tails of the realised strategy path.
        Dashboard-facing values are returned as positive loss magnitudes, not as
        negative tail quantiles, so table and card comparisons use one explicit
        convention.
    """
    if daily_pnl is None or daily_pnl.dropna().empty:
        return {
            "var_primary": float("nan"),
            "es_primary": float("nan"),
            "var_tail": float("nan"),
            "es_tail": float("nan"),
        }

    clean = daily_pnl.dropna().astype(float)
    var_primary_threshold = float(clean.quantile(1.0 - primary_confidence))
    var_tail_threshold = float(clean.quantile(1.0 - tail_confidence))

    es_primary_tail = clean[clean <= var_primary_threshold]
    es_tail_tail = clean[clean <= var_tail_threshold]

    var_primary = max(0.0, -var_primary_threshold)
    var_tail = max(0.0, -var_tail_threshold)
    es_primary = max(0.0, -float(es_primary_tail.mean())) if not es_primary_tail.empty else float("nan")
    es_tail = max(0.0, -float(es_tail_tail.mean())) if not es_tail_tail.empty else float("nan")

    return {
        "var_primary": var_primary,
        "es_primary": es_primary,
        "var_tail": var_tail,
        "es_tail": es_tail,
    }

def _expected_shortfall_from_array(values: np.ndarray, confidence: float) -> float:
    """Returns the historical expected shortfall of a numeric array."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    threshold = float(np.quantile(arr, 1.0 - confidence))
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return float("nan")
    return max(0.0, -float(tail.mean()))

def compute_rolling_var_es(
    daily_pnl: pd.Series,
    window_days: int,
    primary_confidence: float,
    tail_confidence: float,
) -> pd.DataFrame:
    """
    Computes rolling historical VaR / ES and breach flags on daily PnL.

    Methodology:
        The rolling window allows the dashboard to show regime drift in tail
        risk instead of only a full-sample scalar. Breach flags compare the
        realised daily PnL with the contemporaneous rolling VaR threshold.
    """
    if daily_pnl is None or daily_pnl.dropna().empty:
        return pd.DataFrame()

    clean = daily_pnl.dropna().astype(float)
    min_periods = min(window_days, max(window_days // 2, 5))
    rolling = clean.rolling(window=window_days, min_periods=min_periods)

    frame = pd.DataFrame(index=clean.index)
    frame["pnl"] = clean
    frame["var_primary"] = -rolling.quantile(1.0 - primary_confidence)
    frame["var_tail"] = -rolling.quantile(1.0 - tail_confidence)
    frame["var_primary"] = frame["var_primary"].clip(lower=0.0)
    frame["var_tail"] = frame["var_tail"].clip(lower=0.0)
    frame["es_primary"] = rolling.apply(
        lambda values: _expected_shortfall_from_array(values, primary_confidence),
        raw=True,
    )
    frame["es_tail"] = rolling.apply(
        lambda values: _expected_shortfall_from_array(values, tail_confidence),
        raw=True,
    )
    frame["breach_primary"] = frame["pnl"] <= -frame["var_primary"]
    frame["breach_tail"] = frame["pnl"] <= -frame["var_tail"]
    return frame

def compute_rolling_volatility(
    daily_returns: pd.Series,
    windows: Sequence[int],
) -> pd.DataFrame:
    """
    Computes annualised rolling volatility curves from daily returns.

    Methodology:
        Risk is annualised from daily returns with sqrt(252) so strategy-level
        and portfolio-level volatility remain directly comparable despite
        different capital bases.
    """
    if daily_returns is None or daily_returns.dropna().empty:
        return pd.DataFrame()

    clean_returns = daily_returns.dropna().astype(float)
    frame = pd.DataFrame(index=clean_returns.index)

    for window in windows:
        if window <= 0:
            continue
        min_periods = min(window, max(window // 2, 5))
        frame[f"{window}D"] = (
            clean_returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252.0) * 100.0
        )

    return frame

def compute_annualised_sharpe(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Computes annualised Sharpe ratio from daily returns.

    Methodology:
        This keeps stress-test comparisons on a consistent return basis even
        when the shocked PnL path changes the absolute dollar scale.
    """
    if daily_returns is None or daily_returns.dropna().empty:
        return float("nan")

    clean_returns = daily_returns.dropna().astype(float)
    if len(clean_returns) < 2:
        return float("nan")

    daily_rf = risk_free_rate / 252.0
    excess_returns = clean_returns - daily_rf
    volatility = float(clean_returns.std())
    if volatility <= 1e-8:
        return float("nan")

    return float(excess_returns.mean() / volatility * np.sqrt(252.0))

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

def build_risk_profile(
    label: str,
    equity: pd.Series,
    trades_df: Optional[pd.DataFrame],
    instrument_specs: Dict[str, Dict[str, float]],
    primary_confidence: float,
    tail_confidence: float,
    rolling_var_window_days: int,
    rolling_vol_windows: Sequence[int],
    stress_multipliers: StressMultipliers,
    risk_free_rate: float = 0.0,
) -> RiskProfile:
    """
    Builds the complete risk payload for a single analyzable stream.

    Methodology:
        The renderer supplies an equity curve that already respects the scope
        boundary:
            - single mode uses the standalone backtest equity,
            - portfolio mode uses either total portfolio equity or one isolated
              strategy-equity reconstruction.

        All downstream metrics then operate on that single stream only, which
        avoids leaking portfolio-only diversification effects into a strategy
        drilldown.
    """
    from .stress import compute_stress_scenarios
    clean_equity = equity.dropna().astype(float) if equity is not None else pd.Series(dtype=float)
    if clean_equity.empty:
        return RiskProfile(
            label=label,
            equity=pd.Series(dtype=float),
            daily_pnl=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            drawdown=pd.Series(dtype=float),
            drawdown_episodes=pd.DataFrame(),
            rolling_var=pd.DataFrame(),
            rolling_vol=pd.DataFrame(),
            summary={},
            stress_results=[],
        )

    daily_equity = clean_equity.resample("1D").last().dropna()
    daily_pnl = daily_equity.diff().fillna(0.0)
    daily_returns = daily_equity.pct_change(fill_method=None).dropna()
    drawdown = compute_drawdown_series(clean_equity)
    drawdown_episodes = compute_drawdown_episodes(drawdown)
    rolling_var = compute_rolling_var_es(
        daily_pnl=daily_pnl,
        window_days=rolling_var_window_days,
        primary_confidence=primary_confidence,
        tail_confidence=tail_confidence,
    )
    rolling_vol = compute_rolling_volatility(daily_returns, rolling_vol_windows)
    var_metrics = compute_var_es_metrics(daily_pnl, primary_confidence, tail_confidence)
    stress_results = compute_stress_scenarios(
        daily_equity=daily_equity,
        daily_pnl=daily_pnl,
        trades_df=trades_df,
        instrument_specs=instrument_specs,
        stress_multipliers=stress_multipliers,
        primary_confidence=primary_confidence,
        tail_confidence=tail_confidence,
        risk_free_rate=risk_free_rate,
    )

    episode_depths = (
        drawdown_episodes["depth_abs_pct"].astype(float)
        if not drawdown_episodes.empty and "depth_abs_pct" in drawdown_episodes.columns
        else pd.Series(dtype=float)
    )
    latest_vol = (
        float(rolling_vol.dropna(how="all").iloc[-1].dropna().iloc[0])
        if not rolling_vol.empty and not rolling_vol.dropna(how="all").empty
        else float("nan")
    )
    sharpe = compute_annualised_sharpe(daily_returns, risk_free_rate=risk_free_rate)

    summary = {
        "var_primary": var_metrics["var_primary"],
        "es_primary": var_metrics["es_primary"],
        "var_tail": var_metrics["var_tail"],
        "es_tail": var_metrics["es_tail"],
        "max_drawdown_pct": abs(float(drawdown.min())) if not drawdown.empty else float("nan"),
        "avg_drawdown_pct": float(episode_depths.mean()) if not episode_depths.empty else float("nan"),
        "median_drawdown_pct": float(episode_depths.median()) if not episode_depths.empty else float("nan"),
        "drawdown_95_pct": float(episode_depths.quantile(0.95)) if not episode_depths.empty else float("nan"),
        "max_drawdown_duration_days": (
            float(drawdown_episodes["duration_days"].max())
            if not drawdown_episodes.empty and "duration_days" in drawdown_episodes.columns
            else float("nan")
        ),
        "latest_vol_pct": latest_vol,
        "total_pnl": float(daily_pnl.sum()),
        "sharpe": sharpe,
    }

    return RiskProfile(
        label=label,
        equity=clean_equity,
        daily_pnl=daily_pnl,
        daily_returns=daily_returns,
        drawdown=drawdown,
        drawdown_episodes=drawdown_episodes,
        rolling_var=rolling_var,
        rolling_vol=rolling_vol,
        summary=summary,
        stress_results=stress_results,
    )

