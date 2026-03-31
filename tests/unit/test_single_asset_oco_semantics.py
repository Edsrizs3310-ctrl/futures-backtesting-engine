from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.backtest_engine.config import BacktestSettings
from src.backtest_engine.single_asset import BacktestEngine
from src.strategies.base import BaseStrategy


def _settings(**overrides) -> BacktestSettings:
    settings = BacktestSettings(
        commission_rate=0.0,
        spread_ticks=0,
        initial_capital=10_000.0,
        use_trading_hours=False,
        **overrides,
    )
    settings.default_symbol = "ES"
    settings.instrument_specs = {"ES": {"tick_size": 1.0, "multiplier": 1.0}}
    return settings


class BracketTargetStrategy(BaseStrategy):
    """Enters long once, then emits a native reduce-only stop/target bracket."""

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self.calls = 0
        self._bracket_sent = False

    def on_bar(self, bar) -> list:
        self.calls += 1
        current_qty = float(self.get_position())
        if current_qty > 0:
            self._invested = True
            self._position_side = "LONG"
        else:
            self._invested = False
            self._position_side = None

        if self.calls == 1:
            return [self.market_order("BUY", 1, reason="SIGNAL")]

        if current_qty > 0 and not self._bracket_sent:
            self._bracket_sent = True
            return [
                self.stop_order("SELL", 1, stop_price=95.0, reason="SL", reduce_only=True),
                self.limit_order("SELL", 1, limit_price=105.0, reason="TP", reduce_only=True),
            ]

        return []


def _coarse_conflict_data() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 9, 30),
            datetime(2025, 1, 1, 10, 0),
            datetime(2025, 1, 1, 10, 30),
            datetime(2025, 1, 1, 11, 0),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 106.0, 100.0],
            "low": [100.0, 100.0, 94.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )


def _intrabar_target_first() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 10, 5),
            datetime(2025, 1, 1, 10, 10),
            datetime(2025, 1, 1, 10, 15),
            datetime(2025, 1, 1, 10, 20),
            datetime(2025, 1, 1, 10, 25),
            datetime(2025, 1, 1, 10, 30),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 105.0, 104.0, 94.0, 100.0, 100.0],
            "high": [106.0, 105.0, 104.0, 94.0, 100.0, 100.0],
            "low": [100.0, 104.0, 94.0, 94.0, 100.0, 100.0],
            "close": [105.0, 104.0, 94.0, 100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )


def _intrabar_incomplete() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 10, 5),
            datetime(2025, 1, 1, 10, 10),
            datetime(2025, 1, 1, 10, 15),
            datetime(2025, 1, 1, 10, 20),
            datetime(2025, 1, 1, 10, 25),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 105.0, 104.0, 94.0, 100.0],
            "high": [106.0, 105.0, 104.0, 94.0, 100.0],
            "low": [100.0, 104.0, 94.0, 94.0, 100.0],
            "close": [105.0, 104.0, 94.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )


def test_single_engine_target_fill_cancels_stop_sibling() -> None:
    """A single-engine target fill must cancel the protective stop sibling."""
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 9, 30),
            datetime(2025, 1, 1, 10, 0),
            datetime(2025, 1, 1, 10, 30),
            datetime(2025, 1, 1, 11, 0),
        ]
    )
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 94.0],
            "high": [100.0, 100.0, 106.0, 94.0],
            "low": [100.0, 100.0, 99.0, 94.0],
            "close": [100.0, 100.0, 104.0, 94.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    engine = BacktestEngine(settings=_settings(), data=data)
    engine.run(BracketTargetStrategy)

    assert len(engine.execution.trades) == 1
    assert engine.execution.trades[0].exit_reason == "TP"
    assert engine.portfolio.positions.get("ES", 0.0) == 0.0
    assert len(engine.execution.fills) == 2


def test_single_engine_same_bar_bracket_conflict_uses_pessimistic_stop() -> None:
    """If stop and target are both reachable on one bar, the stop must win."""
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 9, 30),
            datetime(2025, 1, 1, 10, 0),
            datetime(2025, 1, 1, 10, 30),
            datetime(2025, 1, 1, 11, 0),
        ]
    )
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 106.0],
            "high": [100.0, 100.0, 106.0, 106.0],
            "low": [100.0, 100.0, 94.0, 106.0],
            "close": [100.0, 100.0, 95.0, 106.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    engine = BacktestEngine(settings=_settings(), data=data)
    engine.run(BracketTargetStrategy)

    assert len(engine.execution.trades) == 1
    assert engine.execution.trades[0].exit_reason == "SL"
    assert engine.execution.trades[0].exit_price == 95.0
    assert engine.portfolio.positions.get("ES", 0.0) == 0.0


def test_single_engine_lower_tf_replay_can_override_pessimistic_stop() -> None:
    """Lower-TF replay should let the target win when it clearly fires first."""
    engine = BacktestEngine(
        settings=_settings(
            intrabar_conflict_resolution="lower_timeframe",
            intrabar_resolution_timeframe="5m",
        ),
        data=_coarse_conflict_data(),
    )
    engine._intrabar_data = _intrabar_target_first()
    engine.run(BracketTargetStrategy)

    assert len(engine.execution.trades) == 1
    assert engine.execution.trades[0].exit_reason == "TP"


def test_single_engine_incomplete_lower_tf_replay_falls_back_to_stop() -> None:
    """Incomplete replay coverage must still fall back to the stop-first policy."""
    engine = BacktestEngine(
        settings=_settings(
            intrabar_conflict_resolution="lower_timeframe",
            intrabar_resolution_timeframe="5m",
        ),
        data=_coarse_conflict_data(),
    )
    engine._intrabar_data = _intrabar_incomplete()
    engine.run(BracketTargetStrategy)

    assert len(engine.execution.trades) == 1
    assert engine.execution.trades[0].exit_reason == "SL"
