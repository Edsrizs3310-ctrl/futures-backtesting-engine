"""Tests for typed pending portfolio orders and delta netting."""

from src.backtest_engine.portfolio_layer.domain.contracts import PortfolioConfig, StrategySlot
from src.backtest_engine.portfolio_layer.domain.orders import PendingPortfolioOrder
from src.backtest_engine.portfolio_layer.domain.signals import StrategySignal, TargetPosition
from src.backtest_engine.portfolio_layer.engine.engine import PortfolioBacktestEngine
from src.backtest_engine.config import BacktestSettings
from src.strategies.registry import load_strategy_by_id


def _engine() -> PortfolioBacktestEngine:
    strategy_class = load_strategy_by_id("sma_pullback")
    config = PortfolioConfig(
        slots=[
            StrategySlot(
                strategy_class=strategy_class,
                symbols=["ES"],
                weight=1.0,
                timeframe="30m",
            )
        ],
        initial_capital=100_000.0,
        rebalance_frequency="intrabar",
    )
    settings = BacktestSettings(commission_rate=0.0, spread_ticks=0)
    return PortfolioBacktestEngine(config=config, settings=settings)


def test_compute_orders_returns_typed_pending_portfolio_order() -> None:
    """Portfolio delta computation must return typed pending orders, not tuples."""
    engine = _engine()
    targets = [TargetPosition(slot_id=0, symbol="ES", target_qty=3.0, reason="SIGNAL")]

    orders = engine._compute_orders(targets, pending_orders=[])

    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, PendingPortfolioOrder)
    assert order.slot_id == 0
    assert order.symbol == "ES"
    assert order.side == "BUY"
    assert order.quantity == 3.0


def test_compute_orders_nets_existing_pending_quantity() -> None:
    """Pending carry-forward quantity must participate in target delta netting."""
    engine = _engine()
    engine.book.positions[(0, "ES")] = 2.0
    targets = [TargetPosition(slot_id=0, symbol="ES", target_qty=5.0, reason="SIGNAL")]
    pending = [
        PendingPortfolioOrder(
            slot_id=0,
            symbol="ES",
            side="SELL",
            quantity=1.0,
            reason="SIGNAL",
        )
    ]

    orders = engine._compute_orders(targets, pending_orders=pending)

    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert orders[0].quantity == 4.0


def test_pending_portfolio_order_priority_reason_matches_engine_contract() -> None:
    """Priority reasons must remain explicit on the typed pending order object."""
    order = PendingPortfolioOrder(
        slot_id=0,
        symbol="ES",
        side="SELL",
        quantity=1.0,
        reason="RISK_LIQ",
    )

    assert order.is_priority is True
    assert order.signed_quantity == -1.0


def test_compute_orders_marks_flattening_delta_as_reduce_only() -> None:
    """Target deltas that only reduce exposure must be tagged reduce_only."""
    engine = _engine()
    engine.book.positions[(0, "ES")] = 3.0
    targets = [TargetPosition(slot_id=0, symbol="ES", target_qty=0.0, reason="SIGNAL")]

    orders = engine._compute_orders(targets, pending_orders=[])

    assert len(orders) == 1
    assert orders[0].side == "SELL"
    assert orders[0].quantity == 3.0
    assert orders[0].reduce_only is True


def test_effective_pending_quantity_caps_reduce_only_order_to_current_position() -> None:
    """Reduce-only orders must never execute beyond the remaining opposing position."""
    engine = _engine()
    engine.book.positions[(0, "ES")] = 2.0
    order = PendingPortfolioOrder(
        slot_id=0,
        symbol="ES",
        side="SELL",
        quantity=5.0,
        reduce_only=True,
    )

    effective_qty = engine._effective_pending_quantity(order)

    assert effective_qty == 2.0


def test_compute_orders_uses_current_signal_execution_template() -> None:
    """Fresh signal metadata must shape the queued portfolio delta order."""
    engine = _engine()
    targets = [TargetPosition(slot_id=0, symbol="ES", target_qty=3.0, reason="TP")]
    templates = {
        (0, "ES"): StrategySignal(
            slot_id=0,
            symbol="ES",
            direction=1,
            reason="TP",
            requested_order_id="abc123",
            requested_side="BUY",
            requested_quantity=1.0,
            requested_order_type="LIMIT",
            requested_limit_price=4999.25,
            requested_time_in_force="DAY",
            requested_reduce_only=True,
        )
    }

    orders = engine._compute_orders(
        targets,
        pending_orders=[],
        signal_templates=templates,
    )

    assert len(orders) == 1
    order = orders[0]
    assert order.order_type == "LIMIT"
    assert order.limit_price == 4999.25
    assert order.stop_price is None
    assert order.time_in_force == "DAY"
    assert order.source == "SIGNAL_TEMPLATE"
    assert order.requested_order_id == "abc123"
    assert order.reduce_only is True


def test_compute_orders_does_not_hedge_against_active_resting_signal_template() -> None:
    """Allocator deltas must not hedge around a live non-market templated order."""
    engine = _engine()
    targets = [TargetPosition(slot_id=0, symbol="ES", target_qty=0.0, reason="EXIT")]
    pending = [
        PendingPortfolioOrder(
            slot_id=0,
            symbol="ES",
            side="BUY",
            quantity=2.0,
            reason="SIGNAL",
            order_type="LIMIT",
            limit_price=95.0,
            time_in_force="GTC",
            source="SIGNAL_TEMPLATE",
        )
    ]

    orders = engine._compute_orders(targets, pending_orders=pending)

    assert orders == []


def test_target_drift_is_deferred_while_resting_template_is_active() -> None:
    """Allocator-only target drift must not replace a live resting template."""
    engine = _engine()
    targets = [TargetPosition(slot_id=0, symbol="ES", target_qty=4.0, reason="SIGNAL")]
    pending = [
        PendingPortfolioOrder(
            slot_id=0,
            symbol="ES",
            side="BUY",
            quantity=2.0,
            reason="SIGNAL",
            order_type="LIMIT",
            limit_price=95.0,
            time_in_force="GTC",
            source="SIGNAL_TEMPLATE",
        )
    ]

    orders = engine._compute_orders(targets, pending_orders=pending)

    assert orders == []


def test_pending_portfolio_order_identifies_resting_execution_owner() -> None:
    """Only non-market signal-template orders should own resting execution state."""
    resting = PendingPortfolioOrder(
        slot_id=0,
        symbol="ES",
        side="SELL",
        quantity=1.0,
        order_type="STOP",
        stop_price=90.0,
        source="SIGNAL_TEMPLATE",
    )
    market = PendingPortfolioOrder(
        slot_id=0,
        symbol="ES",
        side="SELL",
        quantity=1.0,
        order_type="MARKET",
        source="SIGNAL_TEMPLATE",
    )
    target_sync = PendingPortfolioOrder(
        slot_id=0,
        symbol="ES",
        side="SELL",
        quantity=1.0,
        order_type="LIMIT",
        source="TARGET_SYNC",
    )

    assert resting.owns_resting_execution_state is True
    assert market.owns_resting_execution_state is False
    assert target_sync.owns_resting_execution_state is False
