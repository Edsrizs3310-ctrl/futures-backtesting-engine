# Strategy Development Guide

This package contains concrete trading strategies and the registry that exposes them to the rest of the system.

Each production strategy should be self-contained in its own file under
`src/strategies/`. Shared code belongs in `src/strategies/filters/` only when
it is a pure indicator, mask builder, or other stateless helper. Avoid
cross-strategy execution helpers or "family" loaders that hide runtime logic.

## Registry-Driven Workflow

Treat `registry.py` as the only canonical list of available strategies.

Practical workflow:

1. Create or update a strategy module under `src/strategies/`.
2. Register or unregister it in `registry.py`.
3. Run the relevant tests.

This README intentionally does not list concrete strategy IDs. The strategy set
changes frequently, and duplicating that inventory in docs creates stale
maintenance overhead with no operational value.

## Contract

All strategies:

- inherit from `BaseStrategy`
- receive an engine instance in `__init__`
- precompute expensive indicators up front
- keep `on_bar()` lightweight

## Execution Timing

Do not manually shift signals to "fix" timing.

The engine contract is already:

- strategy evaluates `bar[t]`
- returned orders execute at `open[t+1]`

Adding extra `shift(1)` logic usually creates a delayed strategy, not a safer one.

Non-market orders still follow the same no-lookahead rule:

- a `LIMIT` / `STOP` / `STOP_LIMIT` created on `bar[t]` first becomes eligible on `bar[t+1]`
- `DAY` expires on the next calendar day
- `IOC` is attempted on the first eligible bar and then cancelled if unfilled

## Native Order Support

Single-asset strategies can return:

- `MARKET`
- `LIMIT`
- `STOP`
- `STOP_LIMIT`

Use the convenience factories in `BaseStrategy` rather than instantiating `Order` directly when possible.

Important single-engine bracket rule:

- multiple same-bar `reduce_only=True` non-market orders are auto-grouped as one protective OCO bracket
- if both stop and target are reachable on the same coarse bar, the single engine first tries lower-timeframe replay when `intrabar_conflict_resolution=lower_timeframe`
- if replay data is missing, incomplete, or anomalous, the single engine falls back to the pessimistic policy and lets the stop win
- this rule exists because the legacy strategy contract returns only `List[Order]`, not an explicit bracket object

Practical implication:

- native stop/target brackets are now safe in the single engine only when emitted as same-bar `reduce_only` non-market siblings
- if you emit unrelated resting orders on the same bar, do not mark them all `reduce_only=True` unless you actually want OCO behavior

## Portfolio Bridge Notes

The portfolio engine is still target-driven, but it now preserves enough raw intent for resting entries to work.

Current bridge behavior:

- the live portfolio position is the source of truth for bridge state, but raw non-`reduce_only` order intent is still preserved
- if a strategy is flat and emits a non-`reduce_only` `LIMIT` / `STOP` / `STOP_LIMIT` entry, the bridge infers provisional direction from that raw order side
- if a strategy is already invested and emits an opposite-side non-`reduce_only` order, the bridge maps it to explicit `CLOSE` or `REVERSE` intent instead of blindly reusing the live position sign
- `reduce_only=True` orders are excluded from entry-direction fallback so protective exits do not request fresh exposure
- same-bar entry plus protective bracket metadata is preserved and forwarded into the portfolio OMS as parent/child intent

Practical implication:

- flat pending-entry strategies such as `three_bar_mr` and `channel_breakout` can trade in portfolio mode without pre-setting `_invested=True`
- explicit exit/reversal strategies can now close or reverse live portfolio positions without local bridge workarounds

## Adding A Strategy

1. Create a new module in `src/strategies/`.
2. Keep the strategy's config and execution state in that same file.
3. Use `src/strategies/filters/` only for stateless helpers.
4. Precompute indicators in `__init__`.
5. Implement `on_bar()` using O(1) lookups.
6. Implement `get_search_space()` in the same module if WFO support is needed.
7. Register the strategy in `registry.py`.

## Optimization Guidance

`get_search_space()` is for walk-forward optimization and should reflect robust ranges, not ultra-fine parameter mining.

Guidelines:

- prefer coarse steps over tiny increments
- optimize the 4-6 parameters that most affect regime, entry quality, and risk placement
- avoid stuffing the search space with every boolean or cosmetic knob
- keep the number of optimized parameters within the global WFO budget in `BacktestSettings.wfo_max_parameters`
- if a strategy uses `trade_direction`, it can be a categorical search dimension when directionality is materially part of the thesis

Good candidates:

- lookback lengths
- ATR windows / ATR multiples
- regime filters
- entry offsets for `LIMIT` / `STOP` logic

Usually weak candidates:

- logging toggles
- redundant mirrored thresholds
- overly precise decimal steps that will not generalize OOS

## Registry

`registry.py` is the source of truth for:

- CLI strategy IDs
- accepted aliases
- class loading for portfolio YAML configs

If the strategy is not registered, it is not part of the platform.

## Design Guidance

- keep data slicing out of `on_bar()`
- keep comments and docstrings in English
- put reusable execution settings in `src/backtest_engine/settings.py`
- add tests for unusual entry, exit, or stateful behavior
- if a strategy emits native protective exits, add at least one regression test covering bracket cancel behavior
