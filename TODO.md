## Simulation Analysis Backlog

- [ ] Define the future `simulation_analysis` artifact contract separately from risk scenario reruns.
- [ ] Design Monte Carlo path generation inputs, persistence format, and reproducibility metadata.
- [ ] Add bankruptcy / margin-call simulation metrics only after simulation artifacts exist.
- [ ] Plan archival and retention rules for large simulation result sets as a separate storage task.
- [ ] Decide whether future simulation workers need queue partitioning beyond the current RQ terminal queue.
- [ ] trend expansion strategy.
- [ ] Create a multi-optimizer. So you can backtest dozens of strategys on one ticker imidiatley at 10 different pop-up terminals.
- [ ] Create a multi-strategy backtester, so you can test dozens of single strategies on one ticker using a matplotlib popup.
- [ ] delete streamlit old logic, when user approves, and migrate fully to terminal_ui.
