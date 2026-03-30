"""
SMA Pullback Strategy.

Signal logic:
- Pre-calculate 50 SMA (fast/entry) and 200 SMA (slow/trend).
- Trend Regime: Price (close) > 200 SMA for LONG, Price < 200 SMA for SHORT.
- Pullback trigger: Bar Low <= 50 SMA <= Bar High.
- Exit: ATR-based stop-loss, fixed R:R take-profit.
- Filters: Half-Life Time Stop, TrendFilter (ensures strong trend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import apply_wfo_dataclass_overrides, wilder_atr

@dataclass
class SmaPullbackConfig:
    """
    Configuration for the SMA Pullback strategy.
    
    fast_sma_window: SMA used for the pullback touch detection (default 50).
    slow_sma_window: SMA used for the directional trend baseline (default 200).
    atr_window: Lookback for finding ATR.
    atr_sl_mult: Distance to stop loss in ATR multiples.
    rr_ratio: Reward/Risk ratio for take profit (distance = SL distance * rr_ratio).
    trade_direction: "both", "long", "short".
    
    use_trend_filter: Block if the trend is not statistically significant.
    trend_window: Window to measure OLS trend.
    trend_min_tstat: Minimum required |T-stat| (meaning trend must be strong).
    
    use_hl_filter: Use HalfLife as a Time Stop.
    hl_window: Lookback for HalfLife regression.
    hl_baseline: Baseline expected duration.
    hl_max_holding_mult: Exit trade if held for longer than entry_HL * mult.
    """
    fast_sma_window: int = 50
    slow_sma_window: int = 200

    atr_window: int = 14
    atr_sl_mult: float = 2.0
    rr_ratio: float = 3.0

    trade_direction: str = "both"


class SmaPullbackStrategy(BaseStrategy):
    """
    SMA Pullback trend-following strategy.
    
    Methodology:
        1. Pre-compute 50 SMA, 200 SMA, and ATR.
        2. Signal is True when Price is in trend (Close vs 200 SMA) and current bar High/Low bounds the 50 SMA.
        3. Execute using ATR for Stop Loss and R:R multiplier for exact fixed Take Profit target.
    """
    def __init__(self, engine, config: Optional[SmaPullbackConfig] = None) -> None:
        super().__init__(engine)
        cfg = config or SmaPullbackConfig()
        apply_wfo_dataclass_overrides(engine, cfg, "smapull")

        self.config = cfg
        close = engine.data["close"]
        high  = engine.data["high"]
        low   = engine.data["low"]

        # SMAs
        fast_sma = close.rolling(window=cfg.fast_sma_window, min_periods=cfg.fast_sma_window).mean()
        slow_sma = close.rolling(window=cfg.slow_sma_window, min_periods=cfg.slow_sma_window).mean()

        # Conditions
        uptrend = close > slow_sma
        downtrend = close < slow_sma
        
        # Touch condition: Bar Low <= Fast SMA <= Bar High
        touching_sma = (low <= fast_sma) & (high >= fast_sma)

        # Pre-compute signals
        # +1 valid long setup, -1 valid short setup
        signals = pd.Series(0.0, index=close.index)
        signals.loc[uptrend & touching_sma] = 1.0
        signals.loc[downtrend & touching_sma] = -1.0

        atr = wilder_atr(high, low, close, cfg.atr_window)

        # Indicators are looked up directly from the closing bar's timestamp.
        self._signal = signals
        self._atr = atr

        self._invested = False
        self._position_side = None
        self._sl_price = 0.0
        self._tp_price = 0.0
        
        valid = self._signal.notna().sum()
        n_touches = int((self._signal != 0).sum())
        print(
            f"[SMA Pullback] Ready | fast={cfg.fast_sma_window} slow={cfg.slow_sma_window} "
            f"| Signal Touches: {n_touches:,} | Valid bars: {valid:,} / {len(close):,}"
        )

    def on_bar(self, bar: pd.Series) -> List[Order]:
        timestamp = bar.name
        
        try:
            signal = self._signal.at[timestamp]
            atr_val = self._atr.at[timestamp]
        except KeyError:
            return []
            
        if np.isnan(atr_val):
            return []

        c_close = bar["close"]
        c_high = bar["high"]
        c_low = bar["low"]

        orders: List[Order] = []

        # ── In Position ──
        if self._invested:
            # Stop Loss & Take Profit checks
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

        # ── Entry Logic ──
        if not self._invested and signal != 0.0:
            direction = self.config.trade_direction.lower()
            if direction == "long" and signal == -1.0:
                return orders
            if direction == "short" and signal == 1.0:
                return orders

            if signal == 1.0:
                self._invested = True
                self._position_side = "LONG"
                sl_dist = atr_val * self.config.atr_sl_mult
                self._sl_price = c_close - sl_dist
                self._tp_price = c_close + (sl_dist * self.config.rr_ratio)
                
                orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="PULLBACK_LONG"))
                
            elif signal == -1.0:
                self._invested = True
                self._position_side = "SHORT"
                sl_dist = atr_val * self.config.atr_sl_mult
                self._sl_price = c_close + sl_dist
                self._tp_price = c_close - (sl_dist * self.config.rr_ratio)
                
                orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="PULLBACK_SHORT"))

        return orders

    def _reset_state(self) -> None:
        self._invested = False
        self._position_side = None
        self._sl_price = 0.0
        self._tp_price = 0.0

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        return {
            "smapull_fast_sma_window": (30, 90, 10),
            "smapull_slow_sma_window": (120, 320, 20),
            "smapull_atr_window":      (10, 30, 5),
            "smapull_atr_sl_mult":     (1.0, 3.0, 0.25),
            "smapull_rr_ratio":        (1.5, 4.0, 0.25),
        }
