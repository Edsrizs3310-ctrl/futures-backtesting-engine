"""
tests/unit/test_strategies.py

Parametric contract tests for all registered strategies.
"""
from typing import Dict, Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtest_engine.execution import Order
from src.strategies.channel_breakout_long import ChannelBreakoutLongStrategy
from src.strategies.mean_reversion_three_bar import ThreeBarMeanReversionStrategy
from src.strategies.registry import get_strategy_ids, load_strategy_by_id


class MockSettings:
    """Canonical minimal fake settings."""
    def __init__(self) -> None:
        self.default_symbol = "BTCUSDT"
        self.fixed_qty = 0.1
        self.low_interval = "1m"
        self.max_cache_staleness_days = 7
        
    def get_instrument_spec(self, symbol: str) -> Dict[str, Any]:
        return {
            "tick_size": 0.25,
            "min_price_increment": 0.25,
            "price_precision": 2,
            "qty_step": 1.0,
            "qty_precision": 0,
        }


class MockPortfolio:
    """Canonical minimal fake portfolio."""
    def __init__(self) -> None:
        self.positions: Dict[str, float] = {"BTCUSDT": 0.0}


class MockEngine:
    """Canonical minimal fake engine."""
    def __init__(self) -> None:
        self.settings = MockSettings()
        self.portfolio = MockPortfolio()
        
        # Strategies typically expect engine.data to be a dictionary or DataFrame
        # with at least open, high, low, close, volume pandas Series.
        idx = pd.date_range("2020-01-01", periods=100, freq="1min")
        self.data = {
            "open": pd.Series(100.0, index=idx),
            "high": pd.Series(105.0, index=idx),
            "low": pd.Series(95.0, index=idx),
            "close": pd.Series(102.0, index=idx),
            "volume": pd.Series(1000.0, index=idx),
        }


@pytest.mark.parametrize("strategy_id", get_strategy_ids())
def test_strategy_contract(strategy_id: str) -> None:
    """
    Ensures that every registered strategy satisfies the BaseStrategy contract:
    - Can be instantiated with a canonical minimal fake engine.
    - Exposes a callable on_bar(bar) method.
    - Can return its search space without error.
    """
    strategy_class = load_strategy_by_id(strategy_id)
    engine = MockEngine()
    
    # 1. Instantiation contract
    strategy = strategy_class(engine=engine)
    
    # 2. Callable on_bar method contract
    assert hasattr(strategy, "on_bar"), f"Strategy {strategy_id} missing on_bar"
    assert callable(strategy.on_bar), f"Strategy {strategy_id} on_bar is not callable"
    
    # Execute a loose test on on_bar just to ensure it's not fundamentally broken when given typical data.
    # While some indicators may be uninitialized, the contract dictates a List[Order] return.
    dummy_bar = pd.Series({
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
    })
    dummy_bar.name = pd.Timestamp("2020-01-01 00:00:00")
    
    try:
        result = strategy.on_bar(dummy_bar)
        assert isinstance(result, list), f"Strategy {strategy_id} on_bar must return a list"
    except Exception as e:
        pytest.fail(f"Strategy {strategy_id} failed on dummy on_bar call with minimal canonical engine data: {e}")
        
    # 3. Search space contract
    space = strategy_class.get_search_space()
    assert isinstance(space, dict), f"Strategy {strategy_id} get_search_space() must return a dict"


def test_three_bar_mr_emits_day_limit_entry_on_signal() -> None:
    """Three-bar mean reversion must use a DAY limit order for entries."""
    idx = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-02 00:00:00",
            "2020-01-03 00:00:00",
            "2020-01-04 00:00:00",
            "2020-01-05 00:00:00",
            "2020-01-06 00:00:00",
        ]
    )
    data = pd.DataFrame(
        {
            "open": [80.0, 90.0, 130.0, 120.0, 110.0, 112.0],
            "high": [81.0, 91.0, 131.0, 121.0, 111.0, 113.0],
            "low": [79.0, 89.0, 129.0, 119.0, 100.0, 111.0],
            "close": [80.0, 90.0, 130.0, 120.0, 110.0, 112.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=idx,
    )

    engine = MockEngine()
    engine.data = data
    engine.settings.tbar_regime_window = 3
    engine.settings.tbar_extreme_lookback = 3
    engine.settings.tbar_trade_direction = "long"
    engine.settings.tbar_use_shock_filter = False
    engine.settings.tbar_entry_limit_atr_frac = 0.10
    engine.portfolio.positions["BTCUSDT"] = 0.0
    engine.settings.default_symbol = "BTCUSDT"

    renamed_data = data.copy()
    renamed_data.index = idx
    engine.data = renamed_data
    strategy = ThreeBarMeanReversionStrategy(engine=engine)

    bar = pd.Series(
        {
            "open": 110.0,
            "high": 111.0,
            "low": 100.0,
            "close": 110.0,
            "volume": 1000.0,
        },
        name=idx[-2],
    )
    orders = strategy.on_bar(bar)

    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, Order)
    assert order.order_type == "LIMIT"
    assert order.time_in_force == "DAY"
    assert order.side == "BUY"
    assert order.limit_price is not None
    assert order.limit_price < float(bar["close"])


def test_channel_breakout_emits_ioc_stop_entry() -> None:
    """Channel breakout should stage the next-bar breakout with an IOC stop order."""
    idx = pd.date_range("2020-01-01", periods=6, freq="1h")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [1000.0] * 6,
        },
        index=idx,
    )

    engine = MockEngine()
    engine.data = data
    engine.settings.chbrk_length = 3
    engine.settings.chbrk_ema_period = 2
    engine.settings.chbrk_trade_direction = "long"
    engine.settings.chbrk_use_shock_filter = False
    engine.settings.chbrk_entry_buffer_ticks = 1

    strategy = ChannelBreakoutLongStrategy(engine=engine)
    orders = strategy.on_bar(data.iloc[-1])

    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, Order)
    assert order.order_type == "STOP"
    assert order.time_in_force == "IOC"
    assert order.side == "BUY"
    assert order.stop_price is not None
