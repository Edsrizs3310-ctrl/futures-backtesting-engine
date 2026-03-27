from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest_engine.optimization.fold_generator import PurgedFoldGenerator
from src.backtest_engine.optimization.objective import objective_score
from src.backtest_engine.optimization.optimizer import OptunaOptimizer
from src.backtest_engine.optimization.wfv_optimizer import WalkForwardOptimizer
from src.backtest_engine.optimization.wfv_optimizer import WFVReport
from src.backtest_engine.optimization.wfv_optimizer import FoldResult
from src.backtest_engine.settings import BacktestSettings


def test_purged_fold_generator_keeps_test_window_intact_when_embargo_is_set() -> None:
    """
    Embargo must reduce train history, not consume the left side of OOS samples.
    """
    data = pd.DataFrame({"close": range(100)})
    splitter = PurgedFoldGenerator(
        n_folds=2,
        test_size=0.1,
        purge_bars=2,
        embargo_bars=3,
    )

    folds = list(splitter.split(data))
    _, first_test_idx = folds[0]

    assert len(first_test_idx) == 10
    assert first_test_idx[0] == 80


def test_fold_degradation_marks_non_positive_is_scores_as_failure() -> None:
    """
    Negative/zero IS scores should not be treated as neutral degradation.
    """
    fold = FoldResult(
        fold_id=1,
        train_start="2024-01-01",
        train_end="2024-03-01",
        test_start="2024-03-02",
        test_end="2024-04-01",
        best_params={},
        is_score=-1.0,
        oos_score=0.2,
        n_trials=0,
        trial_std=0.0,
        oos_stats={},
    )

    assert fold.degradation == -1.0


def test_fold_win_rate_drift_and_expected_value_helpers() -> None:
    """
    Fold helper properties should expose WR drift and EV for IS/OOS diagnostics.
    """
    fold = FoldResult(
        fold_id=1,
        train_start="2024-01-01",
        train_end="2024-03-01",
        test_start="2024-03-02",
        test_end="2024-04-01",
        best_params={},
        is_score=1.0,
        oos_score=0.8,
        n_trials=10,
        trial_std=0.2,
        is_stats={"win_rate": 0.60},
        oos_stats={"win_rate": 0.48, "avg_win": 2.0, "avg_loss": -1.0},
    )

    assert round(fold.win_rate_degradation, 4) == -0.2
    assert round(fold.oos_expected_value, 4) == 0.44


def test_objective_score_returns_hard_rejection_for_too_few_trades() -> None:
    """
    A trade-starved run should return an explicit hard rejection score.
    """
    score = objective_score(
        {
            "total_trades": 0,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.0,
            "max_drawdown": -10.0,
        },
        min_trades=10,
    )

    assert score == -1.0


def test_objective_score_handles_zero_target_trades_without_division_error() -> None:
    """
    Activity penalty path should be safe even if target_trades is misconfigured.
    """
    score = objective_score(
        {
            "total_trades": 10,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 1.0,
            "max_drawdown": -10.0,
        },
        min_trades=1,
        target_trades=0,
    )

    assert score >= 0.0


@dataclass
class _DummySettings:
    alpha: int = 1

    def model_copy(self, update):
        merged = {"alpha": self.alpha, **update}
        return _DummySettings(**merged)


def test_optimizer_normalizes_max_drawdown_to_percent(monkeypatch) -> None:
    """
    MaxDD input should end up in percent regardless of source convention.
    """

    class _FakeAnalytics:
        def calculate_metrics(self, history, trades):
            return {
                "Total Trades": 5,
                "Sharpe Ratio": 1.0,
                "Sortino Ratio": 1.1,
                "Calmar Ratio": 0.8,
                "Max Drawdown": -25.0,  # already in percent
            }

    class _FakePortfolio:
        def get_history_df(self):
            return pd.DataFrame({"total_value": [100_000.0, 99_000.0]})

    class _FakeEngine:
        def __init__(self, *args, **kwargs):
            self.analytics = _FakeAnalytics()
            self.portfolio = _FakePortfolio()
            self.execution = type("_Exec", (), {"trades": []})()

        def run(self, strategy_class, step_callback=None):
            return None

    monkeypatch.setattr(
        "src.backtest_engine.optimization.optimizer.BacktestEngine",
        _FakeEngine,
    )

    optimizer = OptunaOptimizer(settings=_DummySettings())
    result = optimizer._run_strategy(strategy_class=object, params={})

    assert result["stats"]["max_drawdown"] == -25.0


def test_purged_fold_generator_rejects_zero_fold_size() -> None:
    """
    Tiny test_size values should fail fast instead of creating range(step=0).
    """
    data = pd.DataFrame({"close": range(10)})
    splitter = PurgedFoldGenerator(n_folds=2, test_size=0.01)

    try:
        list(splitter.split(data))
    except ValueError as exc:
        assert "Fold size became zero" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero fold size")


def test_wfv_skips_oos_evaluation_when_is_optimization_fails(monkeypatch) -> None:
    """
    If IS optimization has no successful trials, OOS must not run on empty params.
    """
    settings = BacktestSettings(wfo_n_folds=1, wfo_test_size_pct=0.2, wfo_n_trials=1)
    wfo = WalkForwardOptimizer(settings=settings)
    data = pd.DataFrame(
        {"close": range(30)},
        index=pd.date_range("2024-01-01", periods=30, freq="D"),
    )

    monkeypatch.setattr(wfo.data_lake, "load", lambda symbol, timeframe: data)
    monkeypatch.setattr(
        wfo.base_optimizer,
        "optimize_on_slice",
        lambda **kwargs: {
            "best_params": {},
            "best_score": -1.0,
            "n_trials": 0,
            "trial_std": 0.0,
        },
    )

    def _must_not_be_called(**kwargs):
        raise AssertionError("evaluate_on_slice must not be called for failed IS optimization")

    monkeypatch.setattr(wfo.base_optimizer, "evaluate_on_slice", _must_not_be_called)

    class _Strategy:
        @staticmethod
        def get_search_space():
            return {}

    report = wfo.run(
        strategy_class=_Strategy,
        verbose=False,
        print_report=False,
        show_progress_bar=False,
    )

    assert report.fold_results
    assert report.fold_results[0].is_score == -1.0
    assert report.fold_results[0].oos_score == -1.0


def test_wfv_report_adds_warning_on_large_win_rate_drift() -> None:
    """
    Report should surface a WR drift warning when IS->OOS drop is materially large.
    """
    folds = [
        FoldResult(
            fold_id=1,
            train_start="2024-01-01",
            train_end="2024-03-01",
            test_start="2024-03-02",
            test_end="2024-04-01",
            best_params={},
            is_score=1.0,
            oos_score=0.5,
            n_trials=10,
            trial_std=0.2,
            is_stats={"win_rate": 0.60},
            oos_stats={"win_rate": 0.45, "avg_win": 1.0, "avg_loss": -1.0},
        )
    ]
    report = WFVReport(symbol="ES", strategy_name="S", n_folds=1, fold_results=folds)
    report.compute()

    assert any("WinRate Drift" in warning for warning in report.warnings)
