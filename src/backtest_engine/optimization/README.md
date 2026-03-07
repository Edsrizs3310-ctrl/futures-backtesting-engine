# Optimization Layer

This directory houses the Research-Grade Strategy Optimization and Walk-Forward Validation (WFV) suite. It strictly separates standard parameter tuning from robust, out-of-sample temporal validation to prevent data leakage and curve-fitting.

## Core Components

- **`optimizer.py`**: The standard Bayesian Optimization Engine (`OptunaOptimizer`). It runs the `BacktestEngine` repeatedly using Optuna's sophisticated TPE (Tree-structured Parzen Estimator) sampler to maximize a risk-adjusted objective score.
- **`wfv_optimizer.py`**: The orchestrator for Walk-Forward Validation (`WalkForwardOptimizer`). Generates rolling or expanding date boundaries, optimizes aggressively In-Sample (IS), and evaluates blindly Out-Of-Sample (OOS). Contains "Skeptic" analytics like Deflated Sharpe Ratio (DSR) and performance degradation analysis to penalize overfit strategies.
- **`fold_generator.py`**: Implements Purged & Embargoed Cross-Validation splits (`PurgedFoldGenerator`). Critical for time-series finance to ensure training and testing periods do not overlap or share correlated data points.
- **`objective.py`**: The composite scoring function optimizer targets (`objective_score`). Blends Sharpe and Sortino ratios with soft penalties for insufficient trade counts (activity) or excessive drawdowns (stability).
- **`cost_model.py`**: Connects optimization to real-world friction (`CostModel`). Retrieves symbol-specific configurations from `settings.py` to calculate accurate round-trip transaction costs (commissions) and slippage estimates during high-speed tuning.
- **`validation.py`**: Pre-flight validation gatekeeper (`Validator`). Enforces strict engineering rules before optimization begins (e.g., maximum parameter limits, blocking optimization of risk/position-sizing inputs).
