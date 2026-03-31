## Roadmap for Aug-May

- [ ] Define the future `simulation_analysis` artifact contract separately from risk scenario reruns.
- [ ] Design Monte Carlo path generation inputs, persistence format, and reproducibility metadata.
- [ ] Add bankruptcy / margin-call simulation metrics only after simulation artifacts exist.
- [ ] Finish stress testing end-to-end: backend exists, but the frontend surface is still incomplete.
- [ ] Finish Monte Carlo simulations end-to-end: backend direction exists, but there is no real frontend workflow yet.
- [ ] Plan archival and retention rules for large simulation result sets as a separate storage task.
- [ ] Decide whether future simulation workers need queue partitioning beyond the current RQ terminal queue.
- [ ] Add Forex/CFD/Crypto support.
- [ ] Add metric to WFO: Win Rate IS vs OOS
- [ ] Add Shuffled/Bootstrap test for WFO.
- [ ] Add regime analysis in WFO
- [ ] In terminal UI, vol after entry, change to proper vol drag analysis.
- [ ] Define and document maker/taker-style execution assumptions for `LIMIT` vs `MARKET` / `STOP` / `STOP_LIMIT` orders.
- [ ] Add calibrated default `commission_rate_by_order_type` and `spread_tick_multipliers_by_order_type` profiles instead of relying on one flat friction model.
- [ ] Align optimization/WFO rough cost estimation with per-order-type execution costs so `LIMIT` fee/spread assumptions match real backtest fills.
- [ ] Decide whether passive `LIMIT` orders should model lower spread capture / lower slippage only, or also explicit fee rebates and queue-priority uncertainty. (the last two probably no)
- [ ] Add regression tests and docs for per-order-type friction semantics across single backtest, portfolio backtest, and WFO.