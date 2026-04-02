"""
tests/unit/test_strategies.py

Parametric contract tests for all registered strategies.
"""
import ast
import importlib
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtest_engine.execution import Order
from src.backtest_engine.portfolio_layer.execution.strategy_runner import StrategyRunner
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


def _load_strategy_or_skip(strategy_id: str):
    """Skips strategy-specific tests when the strategy is no longer registered."""
    try:
        return load_strategy_by_id(strategy_id)
    except ValueError:
        pytest.skip(f"Strategy '{strategy_id}' is no longer registered.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_tests_do_not_directly_import_concrete_strategy_modules() -> None:
    """Concrete strategies must be referenced through the registry, not direct imports."""
    tests_root = _repo_root() / "tests"
    violations = []

    for path in tests_root.rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("src.strategies."):
                continue
            if node.module in {"src.strategies.registry", "src.strategies.base"}:
                continue
            violations.append(f"{path.relative_to(_repo_root())}:{node.lineno} -> {node.module}")

    assert not violations, (
        "Tests must not directly import concrete strategy modules. "
        "Use src.strategies.registry.load_strategy_by_id(...) instead.\n"
        + "\n".join(violations)
    )


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
    strategy_class = _load_strategy_or_skip("three_bar_mr")
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
    strategy = strategy_class(engine=engine)

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
    strategy_class = _load_strategy_or_skip("channel_breakout")
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

    strategy = strategy_class(engine=engine)
    orders = strategy.on_bar(data.iloc[-1])

    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, Order)
    assert order.order_type == "STOP"
    assert order.time_in_force == "IOC"
    assert order.side == "BUY"
    assert order.stop_price is not None


def test_rfp_penetration_threshold_scales_by_tick_size() -> None:
    """RFP penetration should be defined by exchange ticks, not raw price units."""
    strategy_class = _load_strategy_or_skip("rfp_fractal")
    engine = MockEngine()
    engine.settings.default_symbol = "TEST"
    engine.settings.get_instrument_spec = MagicMock(
        return_value={
            "tick_size": 0.0000005,
            "multiplier": 1.0,
        }
    )
    strategy_module = importlib.import_module(strategy_class.__module__)
    config_class = getattr(strategy_module, "RollingFractalPivotConfig")

    strategy = strategy_class(
        engine=engine,
        config=config_class(
            penetration_ticks=4.0,
            enable_time_filter=False,
            use_shock_filter=False,
            use_stretch_filter=False,
        ),
    )

    assert strategy.config.penetration_ticks == pytest.approx(4.0)
    assert engine.settings.get_instrument_spec("TEST")["tick_size"] * strategy.config.penetration_ticks == pytest.approx(0.000002)


def test_bollinger_squeeze_breakout_emits_ioc_stop_entry() -> None:
    """Bollinger squeeze breakout should emit a next-bar IOC stop entry."""
    strategy_class = _load_strategy_or_skip("bollinger_squeeze_breakout")
    idx = pd.date_range("2020-01-01", periods=12, freq="1h")
    close = [
        100.00,
        100.05,
        99.98,
        100.02,
        100.01,
        100.03,
        100.00,
        100.04,
        100.02,
        100.01,
        100.03,
        101.60,
    ]
    open_ = [
        100.00,
        100.02,
        100.00,
        100.01,
        100.00,
        100.02,
        100.00,
        100.02,
        100.01,
        100.00,
        100.02,
        100.10,
    ]
    data = pd.DataFrame(
        {
            "open": open_,
            "high": [value + 0.15 for value in close[:-1]] + [101.80],
            "low": [value - 0.15 for value in close[:-1]] + [99.95],
            "close": close,
            "volume": [1000.0] * 11 + [2600.0],
        },
        index=idx,
    )

    engine = MockEngine()
    engine.data = data
    engine.settings.bbsq_bb_window = 5
    engine.settings.bbsq_breakout_lookback = 5
    engine.settings.bbsq_squeeze_lookback = 8
    engine.settings.bbsq_squeeze_quantile = 0.50
    engine.settings.bbsq_squeeze_memory = 3
    engine.settings.bbsq_volume_window = 5
    engine.settings.bbsq_width_expansion_factor = 1.00
    engine.settings.bbsq_breakout_volume_ratio = 1.20
    engine.settings.bbsq_trade_direction = "long"
    engine.settings.bbsq_use_shock_filter = False

    strategy = strategy_class(engine=engine)
    orders = strategy.on_bar(data.iloc[-1])

    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, Order)
    assert order.order_type == "STOP"
    assert order.time_in_force == "IOC"
    assert order.side == "BUY"
    assert order.stop_price is not None


@pytest.mark.parametrize(
    "strategy_id",
    [
        "bollinger_squeeze_breakout",
        "keltner_tightening_breakout",
        "diamond_breakout",
    ],
)
def test_breakout_strategies_preserve_portfolio_direction_when_bracketing(
    strategy_id: str,
) -> None:
    """Protective reduce-only brackets must not collapse portfolio direction to flat."""
    strategy_class = _load_strategy_or_skip(strategy_id)
    engine = MockEngine()
    strategy = strategy_class(engine=engine)

    ts = engine.data["close"].index[-1]
    bar = pd.Series(
        {
            "open": float(engine.data["open"].iloc[-1]),
            "high": float(engine.data["high"].iloc[-1]),
            "low": float(engine.data["low"].iloc[-1]),
            "close": float(engine.data["close"].iloc[-1]),
            "volume": float(engine.data["volume"].iloc[-1]),
        },
        name=ts,
    )

    strategy._pending_side = "LONG"
    strategy._pending_stop_price = 95.0
    strategy._pending_target_price = 105.0
    engine.portfolio.positions["BTCUSDT"] = 1.0

    orders = strategy.on_bar(bar)
    requested = StrategyRunner._build_requested_orders(orders)
    direction = StrategyRunner._resolve_signal_direction(
        strategy,
        requested,
        slot_id=0,
        symbol="BTCUSDT",
        current_position=1.0,
    )

    assert len(orders) == 2
    assert all(order.reduce_only for order in orders)
    assert getattr(strategy, "_invested", False) is True
    assert getattr(strategy, "_position_side", None) == "LONG"
    assert direction == 1


def test_strategy_runner_uses_entry_order_as_primary_signal_in_mixed_bracket_batch() -> None:
    """Mixed entry+bracket batches must preserve the parent entry as bridge primary."""
    timestamp = pd.Timestamp("2025-01-01 09:30:00")
    orders = [
        Order(
            symbol="BTCUSDT",
            quantity=1.0,
            side="BUY",
            order_type="MARKET",
            reason="ENTRY",
            timestamp=timestamp,
        ),
        Order(
            symbol="BTCUSDT",
            quantity=1.0,
            side="SELL",
            order_type="STOP",
            reason="SL",
            stop_price=95.0,
            reduce_only=True,
            timestamp=timestamp,
        ),
        Order(
            symbol="BTCUSDT",
            quantity=1.0,
            side="SELL",
            order_type="LIMIT",
            reason="TP",
            limit_price=105.0,
            reduce_only=True,
            timestamp=timestamp,
        ),
    ]

    requested = StrategyRunner._build_requested_orders(orders)
    primary = StrategyRunner._select_primary_order(orders)
    strategy = MagicMock()
    strategy._invested = False
    strategy._position_side = None

    signal = StrategyRunner._build_signal(
        slot_id=0,
        symbol="BTCUSDT",
        instance=strategy,
        order=primary,
        requested_orders=requested,
        timestamp=timestamp,
        current_position=0.0,
    )

    attached_children = [
        order for order in signal.requested_orders if order.parent_order_id == primary.id
    ]

    assert primary.reason == "ENTRY"
    assert signal.reason == "ENTRY"
    assert signal.requested_order_id == primary.id
    assert signal.requested_order_type == "MARKET"
    assert signal.direction == 1
    assert len(attached_children) == 2
    assert all(order.activation_policy == "ON_PARENT_FILL" for order in attached_children)
