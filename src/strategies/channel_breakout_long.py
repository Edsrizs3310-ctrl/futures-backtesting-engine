"""
Donchian-style channel breakout with EMA trend filter.

Channel (length bars):
    up_bound   = MAX(High, length), down_bound = MIN(Low, length)
    Break up:   high >= up_bound[t-1] + tick  (long entry / short cover)
    Break down: low  <= down_bound[t-1] - tick (long exit / short entry)

Trend filter (EMA):
    close > EMA(ema_period)  → only long breakouts (break up)
    close < EMA(ema_period)  → only short breakouts (break down)

``trade_direction`` can narrow to ``long`` / ``short`` / ``both`` (see config).

Execution: signals on bar close; orders fill next bar open (engine contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import (
    ShockFilter,
    apply_wfo_dataclass_overrides,
    gate_trade_direction,
)


@dataclass
class ChannelBreakoutLongConfig:
    """
    Parameters for channel breakout + EMA regime filter.

    length:
        Donchian window (highest high / lowest low).
    ema_period:
        EMA length for trend filter (close vs EMA gates long vs short channel trades).
    trade_direction:
        ``both`` — EMA selects long above / short below; ``long`` / ``short`` — only that leg.
    """

    length: int = 50
    ema_period: int = 200
    trade_direction: str = "long"  # "both" | "long" | "short"
    use_shock_filter: bool = True
    shock_atr_window: int = 14
    shock_max_gap_atr: float = 1.25
    shock_max_range_atr: float = 3.0
    shock_max_close_change_atr: float = 2.0


class ChannelBreakoutLongStrategy(BaseStrategy):
    """
    Donchian channel breakout with EMA trend filter and symmetric short leg.

    Methodology:
        Pre-compute break-up / break-down masks and EMA. Long entries only when
        close is above the EMA; short entries only when close is below the EMA.
    """

    def __init__(
        self,
        engine,
        config: Optional[ChannelBreakoutLongConfig] = None,
    ) -> None:
        super().__init__(engine)
        cfg = config or ChannelBreakoutLongConfig()
        apply_wfo_dataclass_overrides(engine, cfg, "chbrk")

        self.config = cfg
        L = max(1, int(cfg.length))
        ema_n = max(1, int(cfg.ema_period))

        close = engine.data["close"].astype(float)
        high = engine.data["high"].astype(float)
        low = engine.data["low"].astype(float)
        open_ = engine.data["open"].astype(float)

        tick = float(self.settings.get_instrument_spec(self.settings.default_symbol)["tick_size"])

        ema = close.ewm(span=ema_n, adjust=False).mean()
        self._above_ema = close > ema
        self._below_ema = close < ema

        up_bound = high.rolling(L, min_periods=L).max()
        down_bound = low.rolling(L, min_periods=L).min()
        entry_stop = up_bound.shift(1) + tick
        exit_stop = down_bound.shift(1) - tick

        # Break up: long entry + cover short. Break down: short entry + exit long.
        self._break_up = (high >= entry_stop) & entry_stop.notna()
        self._break_down = (low <= exit_stop) & exit_stop.notna()

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

        self._invested = False
        self._position_side: Optional[str] = None

        n_up = int(self._break_up.sum())
        n_dn = int(self._break_down.sum())
        n_bars = len(high)
        print(
            f"[Channel Breakout] Ready | L={L} EMA={ema_n} | "
            f"direction={cfg.trade_direction} | "
            f"break_up={n_up:,} break_down={n_dn:,} / {n_bars:,} bars | tick={tick}"
        )

    def on_bar(self, bar: pd.Series) -> List[Order]:
        ts = bar.name
        orders: List[Order] = []

        try:
            bu = bool(self._break_up.loc[ts])
            bd = bool(self._break_down.loc[ts])
            above = bool(self._above_ema.loc[ts])
            below = bool(self._below_ema.loc[ts])
        except KeyError:
            return []

        shock_ok = True if self._shock_filter is None else self._shock_filter.is_allowed(ts)

        td = self.config.trade_direction.strip().lower()
        long_raw = bu and above and shock_ok
        short_raw = bd and below and shock_ok
        long_ok, short_ok = gate_trade_direction(td, long_raw, short_raw)

        if self._invested:
            if self._position_side == "LONG" and bd:
                orders.append(
                    self.market_order(
                        "SELL",
                        self.settings.fixed_qty,
                        reason="CHBRK_LONG_EXIT",
                    )
                )
                self._invested = False
                self._position_side = None
                return orders
            if self._position_side == "SHORT" and bu:
                orders.append(
                    self.market_order(
                        "BUY",
                        self.settings.fixed_qty,
                        reason="CHBRK_SHORT_COVER",
                    )
                )
                self._invested = False
                self._position_side = None
                return orders
            return orders

        if long_ok:
            self._invested = True
            self._position_side = "LONG"
            orders.append(
                self.market_order(
                    "BUY",
                    self.settings.fixed_qty,
                    reason="CHBRK_LONG_ENTRY",
                )
            )
        elif short_ok:
            self._invested = True
            self._position_side = "SHORT"
            orders.append(
                self.market_order(
                    "SELL",
                    self.settings.fixed_qty,
                    reason="CHBRK_SHORT_ENTRY",
                )
            )

        return orders

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        return {
            "chbrk_length": (5, 150, 5),
            "chbrk_ema_period": (100, 2000, 100),
        }
