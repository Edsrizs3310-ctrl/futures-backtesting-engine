"""
ICT-style Daily FVG Reversal strategy (M30 execution).

Why this exists:
    The strategy formalises a discretionary idea into a deterministic contract
    compatible with BacktestEngine:
      1) Build higher-timeframe context from DAILY Fair Value Gaps (FVG).
      2) Wait for an M30 sweep/reclaim of the Daily FVG equilibrium (EQ).
      3) Require an internal M30 structure confirmation.
      4) Enter on an M30 retrace into a fresh inversion/FVG zone.
      5) Use 1:1 RR exits because the setup assumes choppy conditions.

Execution model:
    - Signals are produced on bar[t].
    - Orders are filled by engine at open[t+1].
    - No manual shifting is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy


@dataclass
class IctDailyFvgReversalConfig:
    """Configuration for Daily-FVG-to-M30 reversal logic.

    Attributes:
        atr_window: ATR lookback for SL buffer calculation.
        sl_buffer_atr_mult: Extra ATR-based buffer added beyond confirmation low/high.
        rr_target: Take-profit target in R multiples. Default is 1.0 (1:1 RR).
        confirmation_lookback_bars: Bars used for internal structure shift check.
        daily_zone_max_age_days: Maximum age of a daily FVG zone to stay tradable.
        setup_max_age_bars: Maximum base bars to keep a pending setup active.
        require_internal_shift: When True, requires additional base-TF shift.
        trade_direction: Allowed side: "both", "long", or "short".
    """

    atr_window: int = 14
    sl_buffer_atr_mult: float = 0.10
    rr_target: float = 1.0
    confirmation_lookback_bars: int = 2
    choch_max_age_bars: int = 12
    daily_zone_max_age_days: int = 14
    setup_max_age_bars: int = 24
    require_internal_shift: bool = False
    trade_direction: str = "both"


class _DailyZone(NamedTuple):
    """Daily FVG zone used as higher-timeframe context."""

    direction: str
    zone_low: float
    zone_high: float
    eq: float
    formed_day: pd.Timestamp


class _PendingSetup(NamedTuple):
    """Pending M30 setup waiting for retrace entry."""

    direction: str
    fvg_low: float
    fvg_high: float
    sl_anchor: float
    created_bar_idx: int


class IctDailyFvgReversalStrategy(BaseStrategy):
    """Daily FVG + M30 confirmation strategy.

    Methodology:
        This is an ICT-inspired deterministic approximation for automated
        backtesting. It intentionally avoids subjective pattern matching and
        uses explicit numeric rules to keep results reproducible.
    """

    def __init__(
        self,
        engine,
        config: Optional[IctDailyFvgReversalConfig] = None,
    ) -> None:
        super().__init__(engine)

        cfg = config or IctDailyFvgReversalConfig()
        self.config = cfg

        self._open: pd.Series = engine.data["open"]
        self._high: pd.Series = engine.data["high"]
        self._low: pd.Series = engine.data["low"]
        self._close: pd.Series = engine.data["close"]

        self._bar_index: pd.Series = pd.Series(
            np.arange(len(self._close), dtype=int), index=self._close.index
        )

        self._atr: pd.Series = self._compute_atr(cfg.atr_window)

        # Map each base timestamp to latest eligible Daily FVG context zone.
        self._zone_by_ts: pd.Series = self._build_zone_context_series(
            max_age_days=cfg.daily_zone_max_age_days
        )

        # Precompute M30 ChoCH events and project them to base index.
        self._m30_bull_choch_by_ts, self._m30_bear_choch_by_ts = self._build_m30_choch_context()

        self._setup: Optional[_PendingSetup] = None
        self._invested: bool = False
        self._position_side: Optional[str] = None
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0

        print(
            "[ICT_DAILY_FVG] Ready | "
            f"rr={cfg.rr_target:.2f} | "
            f"confirm_lb={cfg.confirmation_lookback_bars} | "
            f"choch_max_age={cfg.choch_max_age_bars} | "
            f"zone_max_age_days={cfg.daily_zone_max_age_days}"
        )

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        """Returns WFO search bounds for optional optimisation."""

        return {
            "ict_dfvg_atr_window": (10, 30, 2),
            "ict_dfvg_sl_buffer_atr_mult": (0.0, 0.3, 0.05),
            "ict_dfvg_confirmation_lookback_bars": (2, 6, 1),
            "ict_dfvg_choch_max_age_bars": (2, 10, 1),
            "ict_dfvg_setup_max_age_bars": (3, 12, 1),
            "ict_dfvg_require_internal_shift": [True, False],
        }

    def on_bar(self, bar: pd.Series) -> List[Order]:
        """Evaluates exits first, then setup lifecycle and potential entries."""

        timestamp = bar.name
        bar_idx = int(self._bar_index.get(timestamp, -1))
        if bar_idx < 3:
            return []

        atr_val = float(self._atr.get(timestamp, np.nan))
        if np.isnan(atr_val):
            return []

        c_open = float(bar["open"])
        c_high = float(bar["high"])
        c_low = float(bar["low"])
        c_close = float(bar["close"])

        orders: List[Order] = []

        if self._invested:
            exit_order = self._maybe_exit(c_high=c_high, c_low=c_low)
            if exit_order is not None:
                orders.append(exit_order)
                return orders

        if self._setup is not None:
            if (bar_idx - self._setup.created_bar_idx) > self.config.setup_max_age_bars:
                self._setup = None

        if self._setup is None and not self._invested:
            zone = self._zone_by_ts.get(timestamp)
            if zone is not None:
                self._setup = self._try_build_setup(
                    bar_idx=bar_idx,
                    zone=zone,
                    c_open=c_open,
                    c_high=c_high,
                    c_low=c_low,
                    c_close=c_close,
                )

        if self._setup is not None and not self._invested:
            entry_order = self._try_enter_from_setup(
                setup=self._setup,
                c_open=c_open,
                c_high=c_high,
                c_low=c_low,
                c_close=c_close,
                atr_val=atr_val,
            )
            if entry_order is not None:
                self._setup = None
                orders.append(entry_order)

        return orders

    def _compute_atr(self, window: int) -> pd.Series:
        """Computes ATR via Wilder-style EWM for robust stop placement."""

        tr = pd.concat(
            [
                self._high - self._low,
                (self._high - self._close.shift(1)).abs(),
                (self._low - self._close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=window, adjust=False).mean()

    def _build_zone_context_series(self, max_age_days: int) -> pd.Series:
        """Builds timestamp->zone mapping using only completed daily bars."""

        if not isinstance(self._close.index, pd.DatetimeIndex):
            return pd.Series([None] * len(self._close), index=self._close.index, dtype=object)

        daily = (
            pd.DataFrame(
                {
                    "open": self._open,
                    "high": self._high,
                    "low": self._low,
                    "close": self._close,
                }
            )
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )

        zones: Dict[pd.Timestamp, _DailyZone] = {}
        for i in range(2, len(daily)):
            day = daily.index[i]
            high_2 = float(daily.iloc[i - 2]["high"])
            low_2 = float(daily.iloc[i - 2]["low"])
            d_high = float(daily.iloc[i]["high"])
            d_low = float(daily.iloc[i]["low"])

            if d_low > high_2:
                zone_low = high_2
                zone_high = d_low
                eq = 0.5 * (zone_low + zone_high)
                zones[day.normalize()] = _DailyZone(
                    direction="bull",
                    zone_low=zone_low,
                    zone_high=zone_high,
                    eq=eq,
                    formed_day=day.normalize(),
                )
            elif d_high < low_2:
                zone_low = d_high
                zone_high = low_2
                eq = 0.5 * (zone_low + zone_high)
                zones[day.normalize()] = _DailyZone(
                    direction="bear",
                    zone_low=zone_low,
                    zone_high=zone_high,
                    eq=eq,
                    formed_day=day.normalize(),
                )

        context: List[Optional[_DailyZone]] = []
        latest_zone: Optional[_DailyZone] = None

        daily_days = daily.index.normalize()
        for ts in self._close.index:
            day_key = pd.Timestamp(ts).normalize()

            # Previous trading day (not previous calendar day).
            prev_pos = int(daily_days.searchsorted(day_key, side="left")) - 1
            if prev_pos >= 0:
                prev_trading_day = daily_days[prev_pos]
                if prev_trading_day in zones:
                    latest_zone = zones[prev_trading_day]

            if latest_zone is not None:
                age_days = (day_key - latest_zone.formed_day).days
                if age_days > max_age_days:
                    latest_zone = None

            context.append(latest_zone)

        return pd.Series(context, index=self._close.index, dtype=object)

    def _build_m30_choch_context(self) -> tuple[pd.Series, pd.Series]:
        """
        Builds M30 ChoCH event context and projects it to base timestamps.

        Bull ChoCH:
            A new lower swing-low followed by a close above the latest swing-high.
        Bear ChoCH:
            A new higher swing-high followed by a close below the latest swing-low.
        """

        if not isinstance(self._close.index, pd.DatetimeIndex):
            false_mask = pd.Series(False, index=self._close.index, dtype=bool)
            return false_mask, false_mask

        m30 = (
            pd.DataFrame(
                {
                    "open": self._open,
                    "high": self._high,
                    "low": self._low,
                    "close": self._close,
                }
            )
            .resample("30min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )

        bull_event = pd.Series(False, index=m30.index, dtype=bool)
        bear_event = pd.Series(False, index=m30.index, dtype=bool)

        last_swing_high: Optional[tuple[int, float]] = None
        prev_swing_high: Optional[tuple[int, float]] = None
        last_swing_low: Optional[tuple[int, float]] = None
        prev_swing_low: Optional[tuple[int, float]] = None

        highs = m30["high"].to_numpy()
        lows = m30["low"].to_numpy()
        closes = m30["close"].to_numpy()

        for i in range(2, len(m30)):
            pivot_i = i - 1
            if highs[pivot_i] > highs[pivot_i - 1] and highs[pivot_i] > highs[pivot_i + 1]:
                prev_swing_high = last_swing_high
                last_swing_high = (pivot_i, highs[pivot_i])

            if lows[pivot_i] < lows[pivot_i - 1] and lows[pivot_i] < lows[pivot_i + 1]:
                prev_swing_low = last_swing_low
                last_swing_low = (pivot_i, lows[pivot_i])

            if (
                last_swing_high is not None
                and last_swing_low is not None
                and prev_swing_low is not None
                and last_swing_low[1] < prev_swing_low[1]
                and closes[i] > last_swing_high[1]
            ):
                bull_event.iloc[i] = True

            if (
                last_swing_high is not None
                and last_swing_low is not None
                and prev_swing_high is not None
                and last_swing_high[1] > prev_swing_high[1]
                and closes[i] < last_swing_low[1]
            ):
                bear_event.iloc[i] = True

        bull_event_ts = pd.Series(pd.NaT, index=m30.index, dtype="datetime64[ns]")
        bull_event_ts.loc[bull_event] = bull_event.index[bull_event]
        bear_event_ts = pd.Series(pd.NaT, index=m30.index, dtype="datetime64[ns]")
        bear_event_ts.loc[bear_event] = bear_event.index[bear_event]

        bull_latest = bull_event_ts.ffill().reindex(self._close.index, method="ffill")
        bear_latest = bear_event_ts.ffill().reindex(self._close.index, method="ffill")

        max_age = pd.Timedelta(minutes=30 * self.config.choch_max_age_bars)
        base_ts = pd.Series(self._close.index, index=self._close.index)

        bull_recent = (base_ts - bull_latest <= max_age) & bull_latest.notna()
        bear_recent = (base_ts - bear_latest <= max_age) & bear_latest.notna()
        return bull_recent.fillna(False), bear_recent.fillna(False)

    def _try_build_setup(
        self,
        bar_idx: int,
        zone: _DailyZone,
        c_open: float,
        c_high: float,
        c_low: float,
        c_close: float,
    ) -> Optional[_PendingSetup]:
        """Creates setup after EQ sweep/reclaim + internal M30 structure shift."""

        if zone.direction == "bull" and self.config.trade_direction == "short":
            return None
        if zone.direction == "bear" and self.config.trade_direction == "long":
            return None

        bar_ts = self._close.index[bar_idx]
        has_bull_choch = bool(self._m30_bull_choch_by_ts.get(bar_ts, False))
        has_bear_choch = bool(self._m30_bear_choch_by_ts.get(bar_ts, False))

        lb = self.config.confirmation_lookback_bars
        prev_high = float(self._high.iloc[bar_idx - lb : bar_idx].max())
        prev_low = float(self._low.iloc[bar_idx - lb : bar_idx].min())
        in_daily_zone = c_low <= zone.zone_high and c_high >= zone.zone_low

        # Bull setup: sweep below EQ then reclaim above EQ + bullish structure shift.
        if zone.direction == "bull":
            swept_eq = c_low <= zone.eq
            reclaimed = c_close > zone.eq and c_close > c_open
            structure_shift = c_close > prev_high
            fvg_ok = c_low > float(self._high.iloc[bar_idx - 2])
            shift_ok = structure_shift if self.config.require_internal_shift else True
            if in_daily_zone and swept_eq and reclaimed and has_bull_choch and shift_ok and fvg_ok:
                return _PendingSetup(
                    direction="LONG",
                    fvg_low=float(self._high.iloc[bar_idx - 2]),
                    fvg_high=c_low,
                    sl_anchor=min(c_low, zone.zone_low),
                    created_bar_idx=bar_idx,
                )

        # Bear setup: sweep above EQ then reclaim below EQ + bearish structure shift.
        if zone.direction == "bear":
            swept_eq = c_high >= zone.eq
            reclaimed = c_close < zone.eq and c_close < c_open
            structure_shift = c_close < prev_low
            fvg_ok = c_high < float(self._low.iloc[bar_idx - 2])
            shift_ok = structure_shift if self.config.require_internal_shift else True
            if in_daily_zone and swept_eq and reclaimed and has_bear_choch and shift_ok and fvg_ok:
                return _PendingSetup(
                    direction="SHORT",
                    fvg_low=c_high,
                    fvg_high=float(self._low.iloc[bar_idx - 2]),
                    sl_anchor=max(c_high, zone.zone_high),
                    created_bar_idx=bar_idx,
                )

        return None

    def _try_enter_from_setup(
        self,
        setup: _PendingSetup,
        c_open: float,
        c_high: float,
        c_low: float,
        c_close: float,
        atr_val: float,
    ) -> Optional[Order]:
        """Enters on retrace into M30 FVG with directional close confirmation."""

        tick_size = self.settings.get_instrument_spec(self.settings.default_symbol)["tick_size"]

        if setup.direction == "LONG":
            touched_fvg = c_low <= setup.fvg_high and c_high >= setup.fvg_low
            confirmed = c_close > c_open
            if touched_fvg and confirmed:
                entry = c_close
                sl = setup.sl_anchor - max(tick_size, atr_val * self.config.sl_buffer_atr_mult)
                risk = max(tick_size, entry - sl)
                tp = entry + (risk * self.config.rr_target)
                self._activate_position("LONG", sl, tp)
                return self.market_order("BUY", self.settings.fixed_qty, reason="SIGNAL")

        if setup.direction == "SHORT":
            touched_fvg = c_high >= setup.fvg_low and c_low <= setup.fvg_high
            confirmed = c_close < c_open
            if touched_fvg and confirmed:
                entry = c_close
                sl = setup.sl_anchor + max(tick_size, atr_val * self.config.sl_buffer_atr_mult)
                risk = max(tick_size, sl - entry)
                tp = entry - (risk * self.config.rr_target)
                self._activate_position("SHORT", sl, tp)
                return self.market_order("SELL", self.settings.fixed_qty, reason="SIGNAL")

        return None

    def _maybe_exit(self, c_high: float, c_low: float) -> Optional[Order]:
        """Checks SL/TP hit and returns exit order when necessary."""

        if self._position_side == "LONG":
            sl_hit = c_low <= self._sl_price
            tp_hit = c_high >= self._tp_price
            if sl_hit or tp_hit:
                reason = "STOP_LOSS" if sl_hit else "TAKE_PROFIT"
                self._reset_position()
                return self.market_order("SELL", self.settings.fixed_qty, reason=reason)

        if self._position_side == "SHORT":
            sl_hit = c_high >= self._sl_price
            tp_hit = c_low <= self._tp_price
            if sl_hit or tp_hit:
                reason = "STOP_LOSS" if sl_hit else "TAKE_PROFIT"
                self._reset_position()
                return self.market_order("BUY", self.settings.fixed_qty, reason=reason)

        return None

    def _activate_position(self, side: str, sl: float, tp: float) -> None:
        """Stores active position metadata for deterministic exits."""

        self._invested = True
        self._position_side = side
        self._sl_price = sl
        self._tp_price = tp

    def _reset_position(self) -> None:
        """Resets active position state after exit."""

        self._invested = False
        self._position_side = None
        self._sl_price = 0.0
        self._tp_price = 0.0
