# Strategy Development Guide

This document is the **strict instruction manual** for creating new strategies in this backtesting engine. Both human developers and LLMs must adhere to these rules to ensure new strategies work seamlessly, process data efficiently, and do not introduce lookahead bias.

## ⚠️ Core Engineering Rules

The backtesting engine is designed for **Research-Grade** performance. It runs an event-driven loop over historical data. To make this fast and safe, the architecture strictly separates *vectorised pre-computation* from *event-driven logic*.

### 1. Vectorised `__init__` (Heavy Lifting)
All indicators, signals, and mathematical transformations **must** be pre-computed using Pandas/NumPy array operations inside the strategy's `__init__` method.
- Access the full historical dataset via `self.engine.data`.
- Save the resulting `pd.Series` objects to class attributes (e.g., `self._atr = ...`).
- **DO NOT shift indicators `shift(1)`:** The engine naturally executes generated orders at the **NEXT** bar's Open. Shifting data manually will cause timing mismatches.

### 2. O(1) `on_bar` (Lightweight Event Loop)
The `on_bar(self, bar: pd.Series) -> List[Order]` method is called once for every row in the dataset.
- It **must** execute in O(1) time.
- Do not perform rolling window calculations here. Do not slice `engine.data` here.
- Retrieve pre-computed indicators using a timestamp (index) lookup: `value = self._my_indicator.get(bar.name, np.nan)`.

### 3. Inheritance
All strategies **MUST** inherit from `BaseStrategy` (`src.strategies.base.BaseStrategy`).

---

## 🛠️ Step-by-Step Implementation Guide

### Step 1: Define the Configuration Dataclass
Create a `@dataclass` for your strategy's parameters. This provides type-hinting and a clean parameter space.

```python
from dataclasses import dataclass

@dataclass
class MyStrategyConfig:
    fast_window: int = 10
    slow_window: int = 50
    sl_mult: float = 1.5
```

### Step 2: Implement the Strategy Class
Inherit from `BaseStrategy`. In your `__init__`, merge configurations and pre-compute your data.

```python
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from src.strategies.base import BaseStrategy
from src.backtest_engine.execution import Order

class MyStrategy(BaseStrategy):
    def __init__(self, engine, config: Optional[MyStrategyConfig] = None) -> None:
        super().__init__(engine)
        cfg = config or MyStrategyConfig()
        
        # 1. Optuna/WFO parameter override loop (Mandatory for optimization)
        import dataclasses
        for field in dataclasses.fields(cfg):
            wfo_key = f"mystrat_{field.name}" # Prefix matching your get_search_space
            if hasattr(engine.settings, wfo_key):
                setattr(cfg, field.name, getattr(engine.settings, wfo_key))
        
        self.config = cfg
        close = engine.data["close"]

        # 2. Vectorised Operations (Pre-compute EVERYTHING)
        fast_sma = close.rolling(cfg.fast_window).mean()
        slow_sma = close.rolling(cfg.slow_window).mean()
        
        # 3. Store the series
        self._fast_sma = fast_sma
        self._slow_sma = slow_sma
        
        # 4. State variables
        self._invested = False
```

### Step 3: Implement `on_bar`
Write the logic that fires on each tick. Use `bar.name` (the timestamp) to query your pre-computed `pd.Series`.

```python
    def on_bar(self, bar: pd.Series) -> List[Order]:
        timestamp = bar.name
        
        # O(1) Lookup
        fast = self._fast_sma.get(timestamp, np.nan)
        slow = self._slow_sma.get(timestamp, np.nan)
        close = bar["close"]
        
        # Guard against NaN (burn-in period)
        if np.isnan(fast) or np.isnan(slow):
            return []
            
        orders: List[Order] = []
        
        if not self._invested and fast > slow:
            self._invested = True
            # Use the inherited market_order helper
            orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="CROSS_UP"))
            
        elif self._invested and fast < slow:
            self._invested = False
            orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="CROSS_DOWN"))
            
        return orders
```

### Step 4: Implement `get_search_space` (Optuna WFO)
Expose your parameters to the Walk-Forward Optimization engine.

```python
    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        """
        Optuna search bounds. Keys MUST match the WFO injected keys in __init__
        (e.g., 'mystrat_fast_window').
        
        Formats:
          - (start, stop, step)  -> Int or Float range
          - [a, b, c]            -> Categorical
        """
        return {
            "mystrat_fast_window": (5, 20, 1),
            "mystrat_slow_window": (30, 100, 5),
            "mystrat_sl_mult":     (1.0, 3.0, 0.5),
        }
```

### Step 5: Register the Strategy
To make your new strategy visible to the engine, CLI, and portfolio configuration, you **must** register it in `src/strategies/registry.py`.

Open `src/strategies/registry.py` and add a new entry to the `STRATEGIES` dictionary:

```python
STRATEGIES = {
    # ... existing strategies ...
    "my_strat": {
        "class_path": "src.strategies.my_strategy_file:MyStrategy", # ModulePath:ClassName
        "name": "MyStrategy",
        "description": "A short tag describing the strategy style",
    },
}
```
If you forget this step, the engine will crash with an `Unknown strategy` `ValueError` when trying to run it!

---

## 🛑 Common Pitfalls (Do Not Do These)

1. **`shift(1)` Bias:** Do not shift indicators backwards in `__init__`. The engine runs on bar `T` and executes the order on `T+1` Open. Shifting the data creates a double-delay or lookahead.
2. **DataFrame Slicing in `on_bar`:** Never do `engine.data.loc[:timestamp]`. This turns an O(1) operation into an O(N) operation and will make the backtester gridlock to a halt. Always precompute!
3. **Optimizing Risk Parameters:** The WFO engine `validation.py` expressly forbids optimizing position sizing, leverage, or capital params. Keep optimization restricted to entry/exit structural logic.
