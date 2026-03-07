# Analytics Module

This directory contains the post-execution analytics, reporting, and dashboard visualization logic for the backtesting engine. It calculates key performance indicators (KPIs), formats terminal output, exports results for visual analytics, and hosts the Streamlit dashboard.

## File Breakdown

- **`core.py`**: Central `PerformanceMetrics` orchestrator. Coordinates calculations from `metrics.py` and `trades.py`, and formatting from `report.py`, providing a stable public API for the main engine and optimizers to call without knowing internal math details.
- **`metrics.py`**: Pure, stateless math functions for equity-curve-level performance metrics (CAGR, Sharpe, Sortino, Volatility, Max Drawdown, Calmar).
- **`trades.py`**: Trade-level statistical analysis. Computes closed-trade KPIs (Win Rate, Profit Factor, Averages, and T-Statistics/P-Values/Alpha/Beta).
- **`exit_analysis.py`**: Data enrichment layer. Computes Maximum Favorable Excursion (MFE), Maximum Adverse Excursion (MAE), holding times, entry volatility, and PnL decay. Runs once at the end of the backtest.
- **`report.py`**: Text report formatter. Converts metrics and trade data into a human-readable ASCII table for terminal output. Contains purely presentation logic.
- **`exporter.py`**: Backtest results exporter. Persists artifacts (`history.parquet`, `trades.parquet`, `metrics.json`, `report.txt`) to the `results/` folder so the Streamlit dashboard can load and render them asynchronously.

## Subdirectories

- **`dashboard/`**: Contains the Streamlit visual UI web application that renders the backtest results.
  - **`core/`**: Shared data handlers, Streamlit components, dataframe transforms, and plotting palettes.
  - **`pnl_analysis/`**: Components and charts for the default PnL Analysis view (Equity curves, PnL distributions, Drawdowns, Exit Decompositions, and Correlations).
  - **`risk_analysis/`**: Placeholder for future advanced Risk visualizations.
  - **`simulation_analysis/`**: Placeholder for future Monte Carlo scenarios and simulations.
