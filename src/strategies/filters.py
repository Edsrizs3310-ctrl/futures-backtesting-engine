"""
Reusable signal filters for single-asset strategies.

All filters are stateless and operate on pre-computed pandas Series so they
fit neatly into the vectorised pre-computation pattern inside BaseStrategy
__init__.  Each filter returns a boolean Series (True = trade allowed).

Available filters
─────────────────
VolatilityRegimeFilter  — blocks entries in low/high volatility regimes.
TrendFilter             — blocks mean-reversion entries during strong trends.
ADFFilter               — verifies stationarity of any price / spread series.
KalmanBeta              — Numba-compiled dynamic hedge-ratio estimator.
HalfLifeFilter          — estimates OU mean-reversion speed parameter.

Shared helpers (no separate class)
─────────────────────────────────
apply_wfo_dataclass_overrides — merge WFO trial keys from engine.settings into a config dataclass.
wilder_atr                     — true range + Wilder EWM average (same as most strategies’ ATR).
hour_of_day_mask               — boolean mask for a simple [start_hour, end_hour) hour window.
gate_trade_direction           — apply "both" / "long" / "short" to long/short booleans.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import adfuller


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def _kalman_beta_loop(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    Q: float,
    R: float,
) -> np.ndarray:
    """
    Numba-compiled 2-D Kalman Filter for rolling dynamic Beta estimation.

    State vector: [alpha, beta].  Runs in O(n) with ~50-100x speedup vs
    a pure Python loop of equivalent complexity.

    Args:
        x: Independent variable array (hedge / regressor).
        y: Dependent variable array (lead / regressand).
        n: Number of observations.
        Q: Process noise variance  (higher → faster adaptation).
        R: Measurement noise variance (higher → smoother estimate).

    Returns:
        beta_arr: Rolling beta estimates of length n.
    """
    beta_arr = np.zeros(n)
    a, b = 0.0, 1.0
    P00, P01, P10, P11 = 1.0, 0.0, 0.0, 1.0

    for t in range(n):
        xt, yt = x[t], y[t]

        # Predict
        P00 += Q
        P11 += Q

        # Innovation covariance
        S = P00 + xt * (P10 + P01) + xt * xt * P11 + R

        # Kalman gains
        K0 = (P00 + P01 * xt) / S
        K1 = (P10 + P11 * xt) / S

        error = yt - (a + b * xt)
        a += K0 * error
        b += K1 * error

        # Update covariance  (P = P - K*H*P)
        t00 = P00 - (K0 * P00 + K0 * xt * P10)
        t01 = P01 - (K0 * P01 + K0 * xt * P11)
        t10 = P10 - (K1 * P00 + K1 * xt * P10)
        t11 = P11 - (K1 * P01 + K1 * xt * P11)
        P00, P01, P10, P11 = t00, t01, t10, t11

        beta_arr[t] = b

    return beta_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers (vectorised indicators & config)
# ═══════════════════════════════════════════════════════════════════════════════


def apply_wfo_dataclass_overrides(engine: Any, cfg: Any, prefix: str) -> None:
    """
    Merge optional Walk-Forward Optimisation parameters from ``engine.settings``.

    WFO injects keys like ``{prefix}_{field_name}``; when present they override
    the dataclass defaults. Mutates ``cfg`` in place.
    """
    for field in dataclasses.fields(cfg):
        wfo_key = f"{prefix}_{field.name}"
        if hasattr(engine.settings, wfo_key):
            setattr(cfg, field.name, getattr(engine.settings, wfo_key))


def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    span: int,
) -> pd.Series:
    """
    Average True Range using Wilder-style smoothing (``ewm(span=..., adjust=False)``).

    True range is the bar-wise max of high-low, |high-prev_close|, |low-prev_close|.
    """
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=span, adjust=False).mean()


def hour_of_day_mask(
    index: pd.Index,
    start_h: int,
    end_h: int,
    enabled: bool,
) -> np.ndarray:
    """
    Boolean mask for bars whose clock hour lies in ``[start_h, end_h)``, or
    wraps midnight when ``start_h > end_h``.
    """
    if not enabled:
        return np.ones(len(index), dtype=bool)
    dt = pd.DatetimeIndex(pd.to_datetime(index, utc=False))
    h = dt.hour.to_numpy()
    if start_h <= end_h:
        return (h >= start_h) & (h < end_h)
    return (h >= start_h) | (h < end_h)


def gate_trade_direction(
    trade_direction: str,
    long_ok: bool,
    short_ok: bool,
) -> Tuple[bool, bool]:
    """
    Restrict long/short entry flags from a ``trade_direction`` setting.

    Args:
        trade_direction: "both" | "long" | "short" (case-insensitive).
        long_ok: Unchecked long entry availability.
        short_ok: Unchecked short entry availability.

    Returns:
        (long_ok_gated, short_ok_gated).
    """
    d = (trade_direction or "both").strip().lower()
    if d == "long":
        return long_ok, False
    if d == "short":
        return False, short_ok
    return long_ok, short_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Public filter classes
# ═══════════════════════════════════════════════════════════════════════════════


class VolatilityRegimeFilter:
    """
    Blocks trading when the asset is in a Compression or Expansion regime.

    Methodology:
        1. Compute a short-term rolling volatility (std of returns or prices).
        2. Rank it within a longer historical window to get a percentile score.
        3. Block entries below `min_pct` (compression) or above `max_pct`
           (expansion / panic), because mean-reversion edges vanish in both.

    The returned `allowed` Series is shifted back by one bar so that the
    strategy only sees past information — zero look-ahead bias.
    """

    def __init__(
        self,
        price: pd.Series,
        regime_window: int,
        history_window: int,
        min_pct: float = 0.20,
        max_pct: float = 0.80,
    ) -> None:
        """
        Args:
            price: Close prices (or any value series) indexed by bar timestamp.
            regime_window: Bars used for short-term volatility estimation.
            history_window: Bars used for percentile ranking of that volatility.
            min_pct: Lower percentile bound; entries blocked below this level.
            max_pct: Upper percentile bound; entries blocked above this level.
        """
        self.min_pct = min_pct
        self.max_pct = max_pct
        rolling_vol = price.rolling(
            window=regime_window, min_periods=regime_window // 2
        ).std()
        vol_pct = rolling_vol.rolling(
            window=history_window, min_periods=history_window // 2
        ).rank(pct=True)
        # Shift prevents look-ahead: strategy sees yesterday's regime label.
        self._pct: pd.Series = vol_pct.shift(1)

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the current bar's volatility regime permits trading.

        Args:
            timestamp: Current bar's index value.

        Returns:
            True if regime percentile is within [min_pct, max_pct].
        """
        try:
            pct = self._pct.at[timestamp]
        except KeyError:
            return True  # No data yet → do not restrict
            
        if np.isnan(pct):
            return True
        return self.min_pct <= pct <= self.max_pct

    def as_series(self) -> pd.Series:
        """Returns raw percentile Series for inspection / overlay plotting."""
        return self._pct


class TrendFilter:
    """
    Blocks mean-reversion entries when a strong directional trend is detected.

    Methodology:
        Fits a rolling OLS regression: price ~ trend_index.
        Computes the T-statistic of the slope.  A high |T-stat| indicates
        that the price series is trending, making mean-reversion bets risky.

    Returns `is_allowed = True` only when |T-stat| < `max_t_stat`.
    """

    def __init__(
        self,
        price: pd.Series,
        window: int,
        max_t_stat: float = 2.0,
    ) -> None:
        """
        Args:
            price: Close price Series indexed by bar timestamp.
            window: Rolling regression window in bars.
            max_t_stat: Maximum absolute T-statistic to allow entry.
        """
        self.max_t_stat = max_t_stat
        t_arr = np.arange(len(price))
        t_series = pd.Series(t_arr, index=price.index)

        cov_st = price.rolling(window=window, min_periods=window // 2).cov(t_series)
        var_t  = t_series.rolling(window=window, min_periods=window // 2).var()
        slope  = cov_st / var_t
        intercept = (
            price.rolling(window=window, min_periods=window // 2).mean()
            - slope * t_series.rolling(window=window, min_periods=window // 2).mean()
        )
        residual = price - (intercept + slope * t_series)
        res_var  = residual.rolling(window=window, min_periods=window // 2).var()
        se_slope = np.sqrt(np.maximum(res_var / ((window - 1) * var_t), 0.0))
        t_stat   = slope / se_slope

        self._t_stat: pd.Series = t_stat.shift(1)

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the trend T-statistic is below the threshold.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if |T-stat| < max_t_stat (weak trend → mean-reversion viable).
        """
        try:
            t = self._t_stat.at[timestamp]
        except KeyError:
            return True
            
        if np.isnan(t):
            return True
        return abs(t) < self.max_t_stat

    def as_series(self) -> pd.Series:
        """Returns raw T-statistic Series for inspection."""
        return self._t_stat


class ShockFilter:
    """
    Blocks entries after abnormally violent bars and opening gaps.

    Methodology:
        Uses prior-bar ATR as a stable volatility yardstick and rejects signal
        bars whose gap, total range, or close-to-close move is far larger than
        normal. This is intentionally a soft "do not touch panic bars" filter,
        not a full volatility-regime model.
    """

    def __init__(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        atr_window: int = 14,
        max_gap_atr: float = 1.0,
        max_range_atr: float = 2.5,
        max_close_change_atr: float = 1.75,
    ) -> None:
        """
        Args:
            open_: Open price Series indexed by bar timestamp.
            high: High price Series indexed by bar timestamp.
            low: Low price Series indexed by bar timestamp.
            close: Close price Series indexed by bar timestamp.
            atr_window: ATR lookback used as the trailing volatility baseline.
            max_gap_atr: Maximum allowed |open - prev_close| in ATR units.
            max_range_atr: Maximum allowed intrabar range in ATR units.
            max_close_change_atr: Maximum allowed |close - prev_close| in ATR units.
        """
        atr_ref = wilder_atr(high, low, close, atr_window).shift(1)
        prev_close = close.shift(1)

        gap_atr = (open_ - prev_close).abs() / atr_ref
        range_atr = (high - low).abs() / atr_ref
        close_change_atr = (close - prev_close).abs() / atr_ref

        allowed = (
            (gap_atr <= max_gap_atr)
            & (range_atr <= max_range_atr)
            & (close_change_atr <= max_close_change_atr)
        )
        allowed = allowed.where(atr_ref.notna(), True)

        self._allowed: pd.Series = allowed.fillna(True)
        self._gap_atr: pd.Series = gap_atr
        self._range_atr: pd.Series = range_atr
        self._close_change_atr: pd.Series = close_change_atr

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the current bar is not an obvious shock bar.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if gap, range, and close-change all remain within ATR bounds.
        """
        try:
            return bool(self._allowed.at[timestamp])
        except KeyError:
            return True

    def as_series(self) -> pd.Series:
        """Returns the boolean allow/block mask."""
        return self._allowed

    def diagnostics(self) -> pd.DataFrame:
        """Returns ATR-normalised shock diagnostics for plotting or debugging."""
        return pd.DataFrame(
            {
                "gap_atr": self._gap_atr,
                "range_atr": self._range_atr,
                "close_change_atr": self._close_change_atr,
                "allowed": self._allowed,
            }
        )


class AtrStretchFilter:
    """
    Blocks entries when price is already too far from its local baseline.

    Methodology:
        Computes an EMA baseline and scales the signed distance to that baseline
        by prior ATR. This catches late, exhausted entries where price is
        already several ATRs away from its short-term equilibrium.

    Long-only strategies typically use `is_long_allowed()`. Short-only
    strategies typically use `is_short_allowed()`.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        baseline_window: int = 20,
        atr_window: int = 14,
        max_long_stretch_atr: float = 1.75,
        max_short_stretch_atr: float = 1.75,
    ) -> None:
        """
        Args:
            high: High price Series indexed by bar timestamp.
            low: Low price Series indexed by bar timestamp.
            close: Close price Series indexed by bar timestamp.
            baseline_window: EMA span used as the local equilibrium estimate.
            atr_window: ATR lookback used to normalise distance from baseline.
            max_long_stretch_atr: Maximum allowed positive stretch for long entries.
            max_short_stretch_atr: Maximum allowed negative stretch for short entries.
        """
        baseline = close.ewm(span=baseline_window, adjust=False).mean()
        atr_ref = wilder_atr(high, low, close, atr_window).shift(1)
        signed_stretch = (close - baseline) / atr_ref

        self.max_long_stretch_atr = max_long_stretch_atr
        self.max_short_stretch_atr = max_short_stretch_atr
        self._signed_stretch: pd.Series = signed_stretch
        self._long_allowed: pd.Series = (
            (signed_stretch <= max_long_stretch_atr).where(atr_ref.notna(), True).fillna(True)
        )
        self._short_allowed: pd.Series = (
            (signed_stretch >= -max_short_stretch_atr).where(atr_ref.notna(), True).fillna(True)
        )

    def is_long_allowed(self, timestamp) -> bool:
        """
        Returns True when a long entry is not too extended above baseline.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if signed stretch is below the configured long threshold.
        """
        try:
            return bool(self._long_allowed.at[timestamp])
        except KeyError:
            return True

    def is_short_allowed(self, timestamp) -> bool:
        """
        Returns True when a short entry is not too extended below baseline.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if signed stretch is above the configured short threshold.
        """
        try:
            return bool(self._short_allowed.at[timestamp])
        except KeyError:
            return True

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when neither side is in an extreme stretched state.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if the absolute stretch remains within both configured bounds.
        """
        return self.is_long_allowed(timestamp) and self.is_short_allowed(timestamp)

    def get(self, timestamp, default: float = np.nan) -> float:
        """
        Returns the signed ATR-normalised distance from the EMA baseline.

        Args:
            timestamp: Current bar index value.
            default: Fallback value when the timestamp is not available.

        Returns:
            Positive values mean price is stretched upward; negative downward.
        """
        try:
            val = self._signed_stretch.at[timestamp]
        except KeyError:
            return default
        return float(val) if not np.isnan(val) else default

    def as_series(self) -> pd.Series:
        """Returns the signed stretch Series in ATR units."""
        return self._signed_stretch


class ADFFilter:
    """
    Macro stationarity filter using the Augmented Dickey-Fuller (ADF) test.

    Methodology:
        Runs a rolling ADF test over a resampled lower-resolution Series.
        Blocks trading when the p-value exceeds `max_pvalue`, which indicates
        that the series has a unit root (non-stationary / trending) and that
        mean-reversion assumptions break down.

    Useful for Z-score / spread strategies on any single time-series.
    """

    def __init__(
        self,
        series: pd.Series,
        adf_window: int = 72,
        timeframe: str = "1h",
        max_pvalue: float = 0.05,
    ) -> None:
        """
        Args:
            series: Price / spread Series at the base bar resolution.
            adf_window: Number of resampled bars per ADF rolling window.
            timeframe: Resample frequency (e.g. '1h', '4h', '1D').
            max_pvalue: Block entries when ADF p-value exceeds this level.
        """
        self.max_pvalue = max_pvalue
        resampled = series.resample(timeframe, label="right", closed="right").last().dropna()
        pvalues = pd.Series(index=resampled.index, dtype=float, name="adf_pvalue")

        failures = 0
        for i in range(adf_window, len(resampled)):
            window_slice = resampled.iloc[i - adf_window : i]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = adfuller(window_slice.values, maxlag=1, autolag=None)
                    pvalues.iloc[i] = result[1]
            except Exception:
                failures += 1
                pvalues.iloc[i] = np.nan

        if failures > 0:
            print(f"[ADFFilter] ADF fit failed on {failures} windows (NaN assigned).")

        self._pvalue: pd.Series = (
            pvalues.reindex(series.index, method="ffill").shift(1).reindex(series.index, method="ffill")
        )

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the ADF test indicates a stationary regime.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if ADF p-value < max_pvalue (reject unit root → stationary).
        """
        try:
            pval = self._pvalue.at[timestamp]
        except KeyError:
            return False  # No result yet → block by default (conservative)
            
        if np.isnan(pval):
            return False
        return pval < self.max_pvalue

    def as_series(self) -> pd.Series:
        """Returns raw p-value Series for inspection."""
        return self._pvalue


class KalmanBeta:
    """
    Dynamic hedge-ratio estimator using a Numba-accelerated Kalman Filter.

    Suitable for any two co-moving price series where the relationship may
    drift over time (e.g. intraday beta to SPY, currency pair correlation).
    Wraps the compiled `_kalman_beta_loop` with scale-normalisation to keep
    Q and R numerically stable regardless of price magnitude.
    """

    def __init__(
        self,
        x: pd.Series,
        y: pd.Series,
        Q: float = 1e-5,
        R: float = 1e-1,
    ) -> None:
        """
        Args:
            x: Independent variable price Series (regressor / hedge leg).
            y: Dependent variable price Series (regressand / lead leg).
            Q: Process noise variance.  Larger → beta adapts faster.
            R: Measurement noise variance.  Larger → beta smoother.
        """
        common_idx = x.index.intersection(y.index)
        x_a = x.loc[common_idx].values.astype(float)
        y_a = y.loc[common_idx].values.astype(float)

        # Scale to unit magnitude to ensure Q/R remain numerically meaningful.
        scale = float(x_a[0]) if x_a[0] != 0 else 1.0
        beta_arr = _kalman_beta_loop(x_a / scale, y_a / scale, len(x_a), Q, R)

        self._beta: pd.Series = pd.Series(beta_arr, index=common_idx).shift(1)

    def get(self, timestamp, default: float = 1.0) -> float:
        """
        Returns the beta estimate for a given bar timestamp.

        Args:
            timestamp: Current bar index value.
            default: Fallback value when no estimate is available.

        Returns:
            Float beta (hedge ratio).
        """
        try:
            val = self._beta.at[timestamp]
        except KeyError:
            return default
            
        return default if np.isnan(val) else float(val)

    def as_series(self) -> pd.Series:
        """Returns the full beta Series for inspection or plotting."""
        return self._beta


class HalfLifeFilter:
    """
    Estimates the Half-Life of mean reversion for a price or spread series.

    Methodology:
        Fits a rolling OLS regression of the series differences (Δy) against
        its lagged values (y_{t-1}) to estimate the Ornstein-Uhlenbeck mean-reversion
        speed parameter (λ). 
        The half-life is then calculated as -ln(2) / λ.

    Returns `is_allowed = True` when the estimated Half-Life is positive and 
    less than or equal to `max_half_life`, indicating a reasonably fast mean-reverting regime.
    """

    def __init__(
        self,
        series: pd.Series,
        window: int = 100,
        max_half_life: float = 50.0,
        lambda_min: Optional[float] = 1e-4,
        max_cap: Optional[float] = 500.0,
    ) -> None:
        """
        Args:
            series: Price or spread Series indexed by bar timestamp.
            window: Rolling regression window in bars.
            max_half_life: Maximum accepted half-life to allow trading.
            lambda_min: Minimum mean-reverting speed to consider the series stationary.
            max_cap: Hard cap limit for calculated Half-Life to prevent explosion.
        """
        self.max_half_life = max_half_life
        self.lambda_min = lambda_min
        self.max_cap = max_cap
        
        y_lag = series.shift(1)
        dy = series - y_lag
        
        # Calculate trailing variance of X and covariance of X and Y
        var_x = y_lag.rolling(window=window, min_periods=window // 2).var()
        cov_xy = y_lag.rolling(window=window, min_periods=window // 2).cov(dy)
        
        # Calculate slope directly. Replace inf values from division-by-zero with NaN.
        slope = cov_xy / var_x
        slope = slope.replace([np.inf, -np.inf], np.nan)
        
        # Calculate Half-Life
        half_life = pd.Series(np.nan, index=series.index)
        
        # Only valid where slope is strictly negative and below lambda bounds
        # (e.g., if lambda_min = 1e-4, we need slope < -1e-4)
        if self.lambda_min is not None:
            valid_idx = slope < -abs(self.lambda_min)
        else:
            valid_idx = slope < -1e-8
        
        # Calculate and hard-cap the value to prevent extreme numbers crashing downstream logic
        raw_hl = -np.log(2) / slope[valid_idx]
        if self.max_cap is not None:
            half_life[valid_idx] = np.minimum(raw_hl, self.max_cap)
        else:
            half_life[valid_idx] = raw_hl
        
        # Shift prevents look-ahead bias
        self._half_life: pd.Series = half_life.shift(1)

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the Half-Life is defined and reasonably short.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if 0 < Half-Life <= max_half_life.
        """
        try:
            hl = self._half_life.at[timestamp]
        except KeyError:
            return False  # No valid half-life -> block entry
            
        if np.isnan(hl):
            return False
            
        return 0 < hl <= self.max_half_life

    def get(self, timestamp, default: float = np.nan) -> float:
        """
        Returns the raw half-life estimate.
        
        Args:
            timestamp: Current bar index value.
            default: Fallback value when no estimate is available.
            
        Returns:
            Float half-life or default.
        """
        try:
            hl = self._half_life.at[timestamp]
        except KeyError:
            return default
            
        return float(hl) if not np.isnan(hl) else default

    def as_series(self) -> pd.Series:
        """Returns the raw Half-Life Series for inspection."""
        return self._half_life
