# GC 5m works good!

"""
3-Bar mean reversion setup (CME / single-asset backtest).

Execution model:
    Entries are placed as DAY limit orders for the next bar. Long entries use
    close minus a small ATR fraction; short entries use close plus the same.
    If the limit is not reached intraday, the entry expires and the strategy
    resets to flat intent.

Rules (signal evaluated at bar close; orders fill at the following bar open):
    Long:
        • Three consecutive down closes: C[t] < C[t-1] < C[t-2].
        • Current bar's low is the lowest low of the last N bars (including t).
        • Regime: prior completed daily close > rolling daily SMA (default 50).
    Short:
        • Three consecutive up closes: C[t] > C[t-1] > C[t-2].
        • Current bar's high is the highest high of the last N bars.
        • Regime: prior completed daily close <= that SMA (strict bear / not-bull).
    Exit:
        • Flat at the session's last bar of the entry day (same calendar day as
          the bar where the entry fill occurs — engine executes that exit at the
          daily close when applicable).

All masks are pre-computed in __init__; on_bar() is O(1) lookup + state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import (
    ShockFilter,
    apply_wfo_dataclass_overrides,
    gate_trade_direction,
    wilder_atr,
)


@dataclass
class ThreeBarMeanReversionConfig:
    """
    Parameters for the 3-bar mean reversion strategy.

    regime_window:
        Length of the simple moving average on daily last closes for trend filter.
    extreme_lookback:
        Bars in the rolling window for the “lowest low / highest high” clause
        (spec uses 5).
    trade_direction:
        "both" | "long" | "short"
    """

    regime_window: int = 50
    extreme_lookback: int = 5
    trade_direction: str = "long"
    use_shock_filter: bool = True
    shock_atr_window: int = 14
    shock_max_gap_atr: float = 1.0
    shock_max_range_atr: float = 2.5
    shock_max_close_change_atr: float = 1.75
    entry_atr_window: int = 14
    entry_limit_atr_frac: float = 0.10


class ThreeBarMeanReversionStrategy(BaseStrategy):
    """Pre-computed 3-bar mean reversion signals with session EOD flat exit."""

    def __init__(self, engine, config: Optional[ThreeBarMeanReversionConfig] = None) -> None:
        super().__init__(engine)
        cfg = config or ThreeBarMeanReversionConfig()
        apply_wfo_dataclass_overrides(engine, cfg, "tbar")

        self.config = cfg

        close = engine.data["close"].astype(float)
        low = engine.data["low"].astype(float)
        high = engine.data["high"].astype(float)
        open_ = engine.data["open"].astype(float)

        c0, c1, c2 = close, close.shift(1), close.shift(2)
        falling_3 = (c0 < c1) & (c1 < c2)
        rising_3 = (c0 > c1) & (c1 > c2)

        lb = cfg.extreme_lookback
        rolling_low_min = low.rolling(lb, min_periods=lb).min()
        rolling_high_max = high.rolling(lb, min_periods=lb).max()
        long_extreme = low <= rolling_low_min
        short_extreme = high >= rolling_high_max

        daily_last = close.resample("1D").last().dropna()
        sma_d = daily_last.rolling(cfg.regime_window, min_periods=cfg.regime_window).mean()
        # float64 avoids object-dtype ffill warnings when expanding daily mask to intraday
        daily_bull_f = (daily_last > sma_d).astype(np.float64)
        regime_prior = daily_bull_f.shift(1)
        regime_ff = regime_prior.reindex(close.index).ffill()
        valid_regime = regime_ff.notna()
        bull = (regime_ff == 1.0) & valid_regime
        bear = (regime_ff == 0.0) & valid_regime
        long_sig = falling_3 & long_extreme & bull
        short_sig = rising_3 & short_extreme & bear

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

        entry_atr = wilder_atr(high=high, low=low, close=close, span=cfg.entry_atr_window)
        self._long_sig = long_sig
        self._short_sig = short_sig
        self._entry_limit_long = close - entry_atr * float(cfg.entry_limit_atr_frac)
        self._entry_limit_short = close + entry_atr * float(cfg.entry_limit_atr_frac)

        n = len(close)
        ts_list = close.index.to_list()
        self._ts_to_pos = {ts: i for i, ts in enumerate(ts_list)}
        dates_np = np.array([ts.date() if hasattr(ts, "date") else ts for ts in ts_list], dtype=object)
        is_eod = np.zeros(n, dtype=bool)
        if n > 0:
            is_eod[-1] = True
        if n > 1:
            is_eod[:-1] = dates_np[1:] != dates_np[:-1]
        self._is_eod = pd.Series(is_eod, index=close.index)
        self._bar_dates = dates_np

        self._invested = False
        self._awaiting_entry = False
        self._position_side: Optional[str] = None
        self._exit_session_date: Optional[date] = None
        self._entry_signal_date: Optional[date] = None

        n_long = int(long_sig.sum())
        n_short = int(short_sig.sum())
        print(
            f"[3Bar MR] Ready | regime_SMA={cfg.regime_window}d | lookback={lb} | "
            f"long={n_long:,} short={n_short:,} bars"
        )

    def on_bar(self, bar: pd.Series) -> List[Order]:
        ts = bar.name

        try:
            pos = self._ts_to_pos[ts]
        except KeyError:
            return []

        orders: List[Order] = []
        current_qty = float(self.get_position())
        current_date = self._bar_dates[pos]

        if self._awaiting_entry:
            if current_qty != 0.0:
                self._awaiting_entry = False
                self._invested = True
                self._position_side = "LONG" if current_qty > 0 else "SHORT"
                self._exit_session_date = current_date
            elif self._entry_signal_date is not None and current_date != self._entry_signal_date:
                self._reset_entry_state()

        if self._invested:
            eod = bool(self._is_eod.loc[ts])
            if current_date == self._exit_session_date and eod:
                if self._position_side == "LONG":
                    orders.append(
                        self.market_order("SELL", self.settings.fixed_qty, reason="EOD_FLAT")
                    )
                else:
                    orders.append(
                        self.market_order("BUY", self.settings.fixed_qty, reason="EOD_FLAT")
                    )
                self._reset_position_state()
            return orders

        if self._awaiting_entry:
            return orders

        long_ok = bool(self._long_sig.loc[ts])
        short_ok = bool(self._short_sig.loc[ts])
        if self._shock_filter is not None and not self._shock_filter.is_allowed(ts):
            long_ok = False
            short_ok = False

        long_ok, short_ok = gate_trade_direction(
            self.config.trade_direction, long_ok, short_ok
        )

        if not long_ok and not short_ok:
            return orders

        if pos + 1 >= len(self._bar_dates):
            return orders

        if long_ok:
            limit_price = self._entry_limit_long.loc[ts]
            if pd.isna(limit_price):
                return orders
            self._awaiting_entry = True
            self._position_side = "LONG"
            self._entry_signal_date = current_date
            orders.append(
                self.limit_order(
                    "BUY",
                    self.settings.fixed_qty,
                    limit_price=float(limit_price),
                    reason="MR3_LONG",
                    time_in_force="DAY",
                )
            )
        elif short_ok:
            limit_price = self._entry_limit_short.loc[ts]
            if pd.isna(limit_price):
                return orders
            self._awaiting_entry = True
            self._position_side = "SHORT"
            self._entry_signal_date = current_date
            orders.append(
                self.limit_order(
                    "SELL",
                    self.settings.fixed_qty,
                    limit_price=float(limit_price),
                    reason="MR3_SHORT",
                    time_in_force="DAY",
                )
            )

        return orders

    def _reset_entry_state(self) -> None:
        self._awaiting_entry = False
        self._position_side = None
        self._entry_signal_date = None

    def _reset_position_state(self) -> None:
        self._invested = False
        self._awaiting_entry = False
        self._position_side = None
        self._exit_session_date = None
        self._entry_signal_date = None

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        return {
            "tbar_regime_window": (30, 120, 10),
            "tbar_extreme_lookback": (3, 9, 1),
            "tbar_entry_atr_window": (10, 30, 5),
            "tbar_entry_limit_atr_frac": (0.05, 0.30, 0.05),
            "tbar_shock_max_gap_atr": (0.75, 1.75, 0.25),
            "tbar_shock_max_range_atr": (1.75, 3.25, 0.25),
        }
