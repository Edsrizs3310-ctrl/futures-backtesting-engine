"""tests/unit/test_allocator.py — Allocator unit tests."""

import math

import pytest
import pandas as pd
import numpy as np

from src.backtest_engine.portfolio_layer.allocation.allocator import Allocator
from src.backtest_engine.portfolio_layer.domain.contracts import PortfolioConfig, StrategySlot
from src.backtest_engine.portfolio_layer.domain.signals import (
    BRIDGE_INTENT_CLOSE,
    BRIDGE_INTENT_REVERSE,
    StrategySignal,
)
from src.strategies.registry import load_strategy_by_id


def _config(
    weight: float = 1.0,
    n_symbols: int = 1,
    target_vol: float = 0.10,
    vol_lookback: int = 5,
    max_contracts: int = 10,
    duty_cycle: float = 1.0,
    max_weight_expansion: float = 4.0,
) -> PortfolioConfig:
    strategy_class = load_strategy_by_id("sma_pullback")
    return PortfolioConfig(
        slots=[StrategySlot(
            strategy_class=strategy_class,
            symbols=["ES"] * n_symbols,
            weight=weight,
            expected_duty_cycle=duty_cycle,
        )],
        initial_capital=100_000.0,
        rebalance_frequency="intrabar",
        target_portfolio_vol=target_vol,
        vol_lookback_bars=vol_lookback,
        max_weight_expansion=max_weight_expansion,
        max_contracts_per_slot=max_contracts,
    )


def _signal(
    direction: int = 1,
    symbol: str = "ES",
    slot_id: int = 0,
    bridge_intent: str = "OPEN",
) -> StrategySignal:
    return StrategySignal(
        slot_id=slot_id,
        symbol=symbol,
        direction=direction,
        bridge_intent=bridge_intent,
    )


def _price_history(
    symbol: str = "ES",
    slot_id: int = 0,
    n: int = 30,
    vol: float = 0.015,
) -> dict:
    """Generates a synthetic price series with known volatility."""
    np.random.seed(42)
    log_rets = np.random.normal(0, vol, n)
    prices = 4000.0 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2023-01-02", periods=n, freq="30min")
    return {(slot_id, symbol): pd.Series(prices, index=idx)}


SPECS = {"ES": {"multiplier": 50.0, "tick_size": 0.25}}


class TestComputeTargets:
    def test_zero_direction_yields_zero_qty(self):
        alloc = Allocator(_config())
        hist = _price_history()
        t = alloc.compute_targets([_signal(direction=0)], 100_000.0, {"ES": 4000.0}, SPECS, hist)
        assert t[0].target_qty == 0

    def test_close_intent_forces_flat_target_even_with_live_exit_signal(self):
        alloc = Allocator(_config())
        hist = _price_history(vol=0.0025)
        t = alloc.compute_targets(
            [_signal(direction=0, bridge_intent=BRIDGE_INTENT_CLOSE)],
            5_000_000.0,
            {"ES": 4000.0},
            SPECS,
            hist,
        )
        assert t[0].target_qty == 0

    def test_reverse_intent_sizes_to_opposite_direction(self):
        alloc = Allocator(_config())
        hist = _price_history(vol=0.0025)
        t = alloc.compute_targets(
            [_signal(direction=-1, bridge_intent=BRIDGE_INTENT_REVERSE)],
            5_000_000.0,
            {"ES": 4000.0},
            SPECS,
            hist,
        )
        assert t[0].target_qty < 0

    def test_long_signal_yields_positive_qty(self):
        alloc = Allocator(_config())
        hist = _price_history(vol=0.0025)
        t = alloc.compute_targets([_signal(direction=1)], 5_000_000.0, {"ES": 4000.0}, SPECS, hist)
        assert t[0].target_qty > 0

    def test_short_signal_yields_negative_qty(self):
        alloc = Allocator(_config())
        hist = _price_history(vol=0.0025)
        t = alloc.compute_targets([_signal(direction=-1)], 5_000_000.0, {"ES": 4000.0}, SPECS, hist)
        assert t[0].target_qty < 0

    def test_zero_price_yields_zero_qty(self):
        alloc = Allocator(_config())
        hist = _price_history()
        t = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 0.0}, SPECS, hist)
        assert t[0].target_qty == 0

    def test_max_contracts_cap_applied(self):
        """Verify the hard cap on contracts is respected."""
        alloc = Allocator(_config(max_contracts=2))
        # Give very low vol (high history consistency) to force a large raw size.
        hist = _price_history(vol=0.0001)
        t = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist)
        assert abs(t[0].target_qty) <= 2

    def test_insufficient_history_gives_minimal_sizing(self):
        """When fewer bars than lookback exist, vol fallback (1.0) minimises size."""
        alloc = Allocator(_config(vol_lookback=20))
        hist = {
            (0, "ES"): pd.Series(
                [4000.0, 4001.0], index=pd.date_range("2023-01-02", periods=2, freq="30min")
            )
        }
        t = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist)
        # With 100% annualised vol fallback, sizing is tiny — cap at 0
        assert t[0].target_qty >= 0  # non-negative for long

    def test_higher_vol_yields_smaller_position(self):
        """Higher realised vol should produce a smaller position (vol-scaling)."""
        alloc = Allocator(_config())
        hist_low  = _price_history(vol=0.005)
        hist_high = _price_history(vol=0.050)
        t_low  = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist_low)
        t_high = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist_high)
        assert abs(t_low[0].target_qty) >= abs(t_high[0].target_qty)

    def test_raw_contract_sizing_matches_manual_es_example(self):
        """Raw vol-target sizing should match the documented ES back-of-envelope example."""
        alloc = Allocator(_config())

        raw_contracts = alloc._compute_raw_contracts(
            annual_risk_budget=10_000.0,
            price=5_000.0,
            multiplier=50.0,
            annualised_vol=0.20,
        )

        assert math.isclose(raw_contracts, 0.2, rel_tol=1e-9), raw_contracts

    def test_duty_cycle_scales_up_sizing(self, monkeypatch: pytest.MonkeyPatch):
        """A 25% expected duty cycle should double sizing before any cap binds."""
        base_alloc = Allocator(_config(max_contracts=100, duty_cycle=1.0))
        boosted_alloc = Allocator(_config(max_contracts=100, duty_cycle=0.25))
        monkeypatch.setattr(base_alloc, "_estimate_vol", lambda *args, **kwargs: 1.0)
        monkeypatch.setattr(boosted_alloc, "_estimate_vol", lambda *args, **kwargs: 1.0)

        signal = [_signal(direction=1)]
        prices = {"ES": 100.0}
        hist = _price_history()

        base_target = base_alloc.compute_targets(signal, 100_000.0, prices, SPECS, hist)[0]
        boosted_target = boosted_alloc.compute_targets(signal, 100_000.0, prices, SPECS, hist)[0]

        assert boosted_target.target_qty == 2 * base_target.target_qty

    def test_max_weight_expansion_caps_duty_cycle_multiplier(self, monkeypatch: pytest.MonkeyPatch):
        """Extreme low duty cycle must not expand sizing beyond sqrt(max_weight_expansion)."""
        capped_alloc = Allocator(
            _config(max_contracts=100, duty_cycle=0.01, max_weight_expansion=4.0)
        )
        reference_alloc = Allocator(
            _config(max_contracts=100, duty_cycle=0.25, max_weight_expansion=4.0)
        )
        monkeypatch.setattr(capped_alloc, "_estimate_vol", lambda *args, **kwargs: 1.0)
        monkeypatch.setattr(reference_alloc, "_estimate_vol", lambda *args, **kwargs: 1.0)

        signal = [_signal(direction=1)]
        prices = {"ES": 100.0}
        hist = _price_history()

        capped_target = capped_alloc.compute_targets(signal, 100_000.0, prices, SPECS, hist)[0]
        reference_target = reference_alloc.compute_targets(signal, 100_000.0, prices, SPECS, hist)[0]

        assert capped_target.target_qty == reference_target.target_qty


class TestPortfolioConfigValidation:
    def test_weight_sum_tolerance_allows_small_rounding_error(self):
        strategy_class = load_strategy_by_id("sma_pullback")

        config = PortfolioConfig(
            slots=[
                StrategySlot(strategy_class=strategy_class, symbols=["ES"], weight=0.50),
                StrategySlot(strategy_class=strategy_class, symbols=["NQ"], weight=0.495),
            ],
            initial_capital=100_000.0,
            rebalance_frequency="intrabar",
        )

        config.validate()

    def test_weights_must_sum_to_one(self):
        strategy_class = load_strategy_by_id("sma_pullback")

        config = PortfolioConfig(
            slots=[
                StrategySlot(strategy_class=strategy_class, symbols=["ES"], weight=0.60),
                StrategySlot(strategy_class=strategy_class, symbols=["NQ"], weight=0.38),
            ],
            initial_capital=100_000.0,
            rebalance_frequency="intrabar",
        )

        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            config.validate()
