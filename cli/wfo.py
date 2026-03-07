"""
cli/wfo.py

Walk-Forward Optimization CLI handler.

Responsibility: Load strategy, run WalkForwardOptimizer.
Called by run.py --wfo.
"""

from __future__ import annotations

import sys
from typing import Any

from src.data.data_lake import DataLake
from src.strategies.registry import load_strategy_by_id


def _load_strategy(name: str) -> Any:
    """Loads strategy from central registry."""
    try:
        return load_strategy_by_id(name)
    except ValueError as e:
        print(f"[Error] {e}")
        sys.exit(1)

def run(strategy_name: str, settings: Any) -> None:
    """
    Runs Walk-Forward Validation for the given strategy.

    Args:
        strategy_name: Short strategy name.
        settings: BacktestSettings instance.
    """
    from src.backtest_engine.optimization.wfv_optimizer import WalkForwardOptimizer

    strategy_class = _load_strategy(strategy_name)

    print("=" * 60)
    print(f"  WFV: {strategy_class.__name__}")
    print(f"  Symbol   : {settings.default_symbol}")
    print(f"  Timeframe: {settings.low_interval}")
    print("=" * 60)

    data_lake = DataLake(settings)
    cache_errors = data_lake.validate_cache_requirements(
        requirements=[(settings.default_symbol, settings.low_interval)],
    )
    if cache_errors:
        print("[Data] Cache freshness check failed:")
        for err in cache_errors:
            print(f"  - {err}")
        print(
            f"[Data] Update cache first. "
            f"Max allowed age: {settings.max_cache_staleness_days} days."
        )
        print(f"[Data] Example: python run.py --download {settings.default_symbol}")
        sys.exit(1)

    wfv = WalkForwardOptimizer(settings=settings)
    wfv.run(strategy_class=strategy_class)
