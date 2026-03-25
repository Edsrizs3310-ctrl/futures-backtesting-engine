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
