"""
src/strategies/registry.py

Centralized registry for all available trading strategies.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

# Central registry of strategy metadata
# Keys are the short names (IDs) used in CLI and YAML configs.
# Values are dictionaries with metadata.
STRATEGIES = {
    "ict_ob": {
        "class_path": "src.strategies.ict_order_block:IctOrderBlockStrategy",
        "name": "IctOrderBlockStrategy",
        "description": "Popular Media / ICT",
    },
    "sma_pullback": {
        "class_path": "src.strategies.sma_pullback:SmaPullbackStrategy",
        "name": "SmaPullbackStrategy",
        "description": "Trend Following",
    },
    "three_bar_mr": {
        "class_path": (
            "src.strategies.mean_reversion_three_bar:ThreeBarMeanReversionStrategy"
        ),
        "name": "ThreeBarMeanReversionStrategy",
        "description": "Mean Reversion",
    },
    "rfp_fractal": {
        "class_path": (
            "src.strategies.rolling_fractal_pivot:RollingFractalPivotStrategy"
        ),
        "name": "RollingFractalPivotStrategy",
        "description": "Fractal pivot / IBS reclaim",
    },
    "channel_breakout": {
        "class_path": (
            "src.strategies.channel_breakout_long:ChannelBreakoutLongStrategy"
        ),
        "name": "ChannelBreakoutLongStrategy",
        "description": "Donchian channel breakout (long only)",
    },
}

STRATEGY_ALIASES = {
    "ict_order_block": "ict_ob",
    "rolling_fractal_pivot": "rfp_fractal",
    "channel_breakout_long": "channel_breakout",
}


def resolve_strategy_id(strategy_id: str) -> str:
    """
    Resolves CLI/YAML aliases to the canonical registry identifier.

    Args:
        strategy_id: Raw strategy identifier from the user or config.

    Returns:
        Canonical registry identifier stored in ``STRATEGIES``.
    """
    normalized = str(strategy_id or "").strip()
    if normalized in STRATEGIES:
        return normalized
    return STRATEGY_ALIASES.get(normalized, normalized)


def get_strategy_ids(include_aliases: bool = False) -> List[str]:
    """
    Returns all strategy identifiers exposed by the registry.

    Args:
        include_aliases: Whether to append accepted CLI aliases.

    Returns:
        List of strategy identifiers.
    """
    strategy_ids = list(STRATEGIES.keys())
    if include_aliases:
        strategy_ids.extend(sorted(STRATEGY_ALIASES.keys()))
    return strategy_ids


def get_strategy_metadata(strategy_id: str) -> Dict[str, str]:
    """
    Returns metadata for a given strategy ID or alias.

    Args:
        strategy_id: Canonical strategy ID or accepted alias.

    Returns:
        Metadata dictionary or an empty dict when not found.
    """
    return STRATEGIES.get(resolve_strategy_id(strategy_id), {})


def load_strategy_by_id(strategy_id: str) -> Any:
    """
    Returns the strategy class for the given short name.

    Args:
        strategy_id: Strategy identifier ('ict_ob', 'sma_pullback', 'three_bar_mr', ...).

    Returns:
        Strategy class (subclass of BaseStrategy).
    """
    resolved_strategy_id = resolve_strategy_id(strategy_id)
    if resolved_strategy_id not in STRATEGIES:
        available = ", ".join(get_strategy_ids())
        aliases = ", ".join(sorted(STRATEGY_ALIASES.keys()))
        raise ValueError(
            f"Unknown strategy '{strategy_id}'. Available: {available}. "
            f"Aliases: {aliases}"
        )

    module_path, class_name = STRATEGIES[resolved_strategy_id]["class_path"].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_strategy_class_by_name(class_name: str) -> Any:
    """
    Finds and loads a strategy by its class name (e.g. 'SmaPullbackStrategy').
    Useful for portfolio YAML configs that still use the class name.
    """
    for strategy_id, metadata in STRATEGIES.items():
        if metadata["name"] == class_name:
            return load_strategy_by_id(strategy_id)

    # If not found by name, try to load by ID directly as a fallback
    try:
        return load_strategy_by_id(class_name)
    except ValueError:
        raise ValueError(f"Unknown strategy class/id: '{class_name}'")
