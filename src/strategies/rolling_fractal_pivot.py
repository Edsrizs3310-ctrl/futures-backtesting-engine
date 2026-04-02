"""
Rolling lookback pivot (3-bar fractal) strategy.

PineScript-aligned logic:
    • Scan offsets i = 1 .. lookback_bars for confirmed pivot highs/lows
      (center bar strictly higher/lower than its immediate neighbors).
    • Track the max pivot high and min pivot low price seen in that window.
    • Long: reclaim above the rolling min pivot low with strong IBS, after a
      wick pierces below; short is symmetric vs max pivot high.
    • Stops/targets from entry close ± ATR multiples (Wilder-style ATR).

Execution matches the rest of the platform: signals use bar[t] OHLC; orders
fill at open[t+1]. No extra manual shift on indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import (
    AtrStretchFilter,
    ShockFilter,
    apply_wfo_dataclass_overrides,
    gate_trade_direction,
    hour_of_day_mask,
    wilder_atr,
)

_IBS_DENOM_EPS = 1e-5


@dataclass
class RollingFractalPivotConfig:
    """
    Parameters for the rolling fractal pivot strategy.

    penetration_ticks:
        Minimum penetration beyond the fractal level in exchange ticks.
        This keeps the filter contract-aligned for CME futures regardless of
        the symbol's quoted price scale.
    atr_length / atr_mult_tp / atr_mult_sl:
        Wilder EWM ATR on TR; stop and limit distances from entry close.
    ibs_buy_threshold / ibs_sell_threshold:
        Internal bar strength = (close - low) / (high - low); long needs IBS
        above buy threshold, short below sell threshold.
    lookback_bars:
        How many past bars (offsets 1..N) contribute pivot scans.
    start_hour / end_hour / enable_time_filter:
        Exchange-time hour window; if start > end, the window wraps midnight.
    trade_direction:
        "both" | "long" | "short"
    """

    penetration_ticks: float = 2.0
    atr_length: int = 14
    atr_mult_tp: float = 2.0
    atr_mult_sl: float = 2.0
    ibs_buy_threshold: float = 0.7
    ibs_sell_threshold: float = 0.3
    lookback_bars: int = 10
    # Engine session(settings.py) runs first; 
    # edit those for HH:MM precision—this window is secondary.
    start_hour: int = 2     # (whole hours only)
    end_hour: int = 15
    enable_time_filter: bool = True
    trade_direction: str = "both"
    use_shock_filter: bool = True
    shock_atr_window: int = 14
    shock_max_gap_atr: float = 1.0
    shock_max_range_atr: float = 2.75
    shock_max_close_change_atr: float = 1.8
    use_stretch_filter: bool = True
    stretch_baseline_window: int = 20
    stretch_atr_window: int = 14
    stretch_max_long_atr: float = 2.0
    stretch_max_short_atr: float = 2.0


def _rolling_fractal_levels(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each bar index pos, compute max pivot-high and min pivot-low prices
    over offsets i = 1..lookback where a 3-bar fractal is confirmed at pos-i.
    """
    n = high.shape[0]
    out_high = np.full(n, np.nan, dtype=np.float64)
    out_low = np.full(n, np.nan, dtype=np.float64)

    for pos in range(n):
        best_hi = np.nan
        best_lo = np.nan
        for i in range(1, lookback + 1):
            if pos < i + 1:
                continue
            idx = pos - i
            h0, hm, hp = high[idx], high[idx - 1], high[idx + 1]
            if h0 > hm and h0 > hp:
                best_hi = h0 if np.isnan(best_hi) else max(best_hi, h0)
            l0, lm, lp = low[idx], low[idx - 1], low[idx + 1]
            if l0 < lm and l0 < lp:
                best_lo = l0 if np.isnan(best_lo) else min(best_lo, l0)
        out_high[pos] = best_hi
        out_low[pos] = best_lo

    return out_high, out_low


class RollingFractalPivotStrategy(BaseStrategy):
    """
    Fractal-based breakout/reclaim entries with ATR bracket exits.

    Methodology:
        Rolling extrema of confirmed pivot prices define reference levels; entries
        require a pierce past the level and a close back through it with IBS
        confirmation, optionally gated by a session hour window.
    """

    def __init__(
        self,
        engine,
        config: Optional[RollingFractalPivotConfig] = None,
    ) -> None:
        super().__init__(engine)
        cfg = config or RollingFractalPivotConfig()

        apply_wfo_dataclass_overrides(engine, cfg, "rfp")

        self.config = cfg

        high = engine.data["high"].astype(float)
        low = engine.data["low"].astype(float)
        close = engine.data["close"].astype(float)
        open_ = engine.data["open"].astype(float)
        tick_size = float(
            self.settings.get_instrument_spec(self.settings.default_symbol)["tick_size"]
        )

        lb = max(3, int(cfg.lookback_bars))

        hi_a = high.to_numpy(dtype=np.float64)
        lo_a = low.to_numpy(dtype=np.float64)
        highest_fr, lowest_fr = _rolling_fractal_levels(hi_a, lo_a, lb)

        hl_rng = (high - low).to_numpy(dtype=np.float64)
        denom = np.maximum(hl_rng, _IBS_DENOM_EPS)
        ibs = (close.to_numpy(dtype=np.float64) - lo_a) / denom

        in_win = hour_of_day_mask(
            close.index,
            cfg.start_hour,
            cfg.end_hour,
            cfg.enable_time_filter,
        )

        lf = lowest_fr
        hf = highest_fr
        c = close.to_numpy(dtype=np.float64)
        o = open_.to_numpy(dtype=np.float64)
        lo = lo_a
        hi = hi_a

        penetration_price = tick_size * float(cfg.penetration_ticks)

        buy_raw = (
            in_win
            & np.isfinite(lf)
            & (lo < lf)
            & (c > lf)
            & (o > lf)
            & (ibs > cfg.ibs_buy_threshold)
            & ((lf - lo) >= penetration_price)
        )
        sell_raw = (
            in_win
            & np.isfinite(hf)
            & (hi > hf)
            & (c < hf)
            & (o < hf)
            & (ibs < cfg.ibs_sell_threshold)
            & ((hi - hf) >= penetration_price)
        )
        both = buy_raw & sell_raw
        long_sig = buy_raw & ~both
        short_sig = sell_raw & ~both

        atr = wilder_atr(high, low, close, cfg.atr_length)

        self._long_sig = pd.Series(long_sig, index=close.index)
        self._short_sig = pd.Series(short_sig, index=close.index)
        self._atr = atr
        self._shock_filter: Optional[ShockFilter] = None
        if cfg.use_shock_filter:
            self._shock_filter = ShockFilter(
                open_=open_,
                high=high,
                low=low,
                close=close,
                atr_window=cfg.shock_atr_window,
                max_gap_atr=cfg.shock_max_gap_atr,
                max_range_atr=cfg.shock_max_range_atr,
                max_close_change_atr=cfg.shock_max_close_change_atr,
            )

        self._stretch_filter: Optional[AtrStretchFilter] = None
        if cfg.use_stretch_filter:
            self._stretch_filter = AtrStretchFilter(
                high=high,
                low=low,
                close=close,
                baseline_window=cfg.stretch_baseline_window,
                atr_window=cfg.stretch_atr_window,
                max_long_stretch_atr=cfg.stretch_max_long_atr,
                max_short_stretch_atr=cfg.stretch_max_short_atr,
            )

        self._invested = False
        self._position_side: Optional[str] = None
        self._sl_price = 0.0
        self._tp_price = 0.0

        n_long = int(long_sig.sum())
        n_short = int(short_sig.sum())
        print(
            f"[Rolling Fractal Pivot] Ready | lookback={lb} | "
            f"ATR={cfg.atr_length} | penetration={penetration_price:g} "
            f"({cfg.penetration_ticks:g}t default) | "
            f"long={n_long:,} short={n_short:,} signals"
        )

    def on_bar(self, bar: pd.Series) -> List[Order]:
        ts = bar.name

        try:
            atr_val = float(self._atr.at[ts])
        except KeyError:
            return []

        if np.isnan(atr_val):
            return []

        try:
            long_ok = bool(self._long_sig.at[ts])
            short_ok = bool(self._short_sig.at[ts])
        except KeyError:
            return []

        if self._shock_filter is not None and not self._shock_filter.is_allowed(ts):
            long_ok = False
            short_ok = False
        if self._stretch_filter is not None:
            long_ok = long_ok and self._stretch_filter.is_long_allowed(ts)
            short_ok = short_ok and self._stretch_filter.is_short_allowed(ts)

        c_close = float(bar["close"])
        c_high = float(bar["high"])
        c_low = float(bar["low"])

        orders: List[Order] = []

        if self._invested:
            if self._position_side == "LONG":
                if c_low <= self._sl_price or c_high >= self._tp_price:
                    reason = "STOP_LOSS" if c_low <= self._sl_price else "TAKE_PROFIT"
                    orders.append(self.market_order("SELL", self.settings.fixed_qty, reason=reason))
                    self._reset_state()
                    return orders
            elif self._position_side == "SHORT":
                if c_high >= self._sl_price or c_low <= self._tp_price:
                    reason = "STOP_LOSS" if c_high >= self._sl_price else "TAKE_PROFIT"
                    orders.append(self.market_order("BUY", self.settings.fixed_qty, reason=reason))
                    self._reset_state()
                    return orders
            return orders

        long_ok, short_ok = gate_trade_direction(
            self.config.trade_direction, long_ok, short_ok
        )

        if not long_ok and not short_ok:
            return orders

        sl_m = self.config.atr_mult_sl
        tp_m = self.config.atr_mult_tp
        dist = atr_val * sl_m

        if long_ok:
            self._invested = True
            self._position_side = "LONG"
            self._sl_price = c_close - dist
            self._tp_price = c_close + atr_val * tp_m
            orders.append(
                self.market_order("BUY", self.settings.fixed_qty, reason="RFP_LONG"),
            )
        elif short_ok:
            self._invested = True
            self._position_side = "SHORT"
            self._sl_price = c_close + dist
            self._tp_price = c_close - atr_val * tp_m
            orders.append(
                self.market_order("SELL", self.settings.fixed_qty, reason="RFP_SHORT"),
            )

        return orders

    def _reset_state(self) -> None:
        self._invested = False
        self._position_side = None
        self._sl_price = 0.0
        self._tp_price = 0.0

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        return {
            "rfp_lookback_bars": (5, 18, 1),
            "rfp_penetration_ticks": (1.0, 8.0, 1.0),
            "rfp_atr_mult_tp": (1.5, 4.0, 0.5),
            "rfp_atr_mult_sl": (1.0, 3.0, 0.25),
            "rfp_ibs_buy_threshold": (0.60, 0.85, 0.05),
            "rfp_ibs_sell_threshold": (0.15, 0.40, 0.05),
        }
