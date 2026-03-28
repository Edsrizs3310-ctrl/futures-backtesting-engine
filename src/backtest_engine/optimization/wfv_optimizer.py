"""
Research-Grade Walk-Forward Validation Optimizer.

Adapted for event-driven BacktestEngine strategies.

Includes 'Skeptic' analysis tools:
1. Deflated Sharpe Ratio (DSR) estimation.
2. Performance Degradation Analysis.
3. Regime-Specific Stability checks.
"""
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from collections import Counter
import time
import pandas as pd
import numpy as np
import math
import importlib

from ..settings import BacktestSettings
from .fold_generator import PurgedFoldGenerator
from .optimizer import OptunaOptimizer
from src.data.data_lake import DataLake

# Avoid static import resolution issues in type checkers when optional
# runtime environments differ from IDE analysis environments.
optuna = importlib.import_module("optuna")

# Suppress Optuna logging to warnings only
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════
# HELPER: MATH FOR SKEPTICS
# ═══════════════════════════════════════════════════════════════════

def estimated_dsr(
    sharpe: float,
    n_trials: int,
    trial_std: float,
) -> float:
    """
    Estimate Probabilistic Sharpe Ratio adjusted for Multiple Testing (DSR proxy).

    Based on Bailey, Lopez de Prado (2014).

    Args:
        sharpe: The best IS Sharpe Ratio found.
        n_trials: Number of optimization trials run.
        trial_std: Standard deviation of scores across trials.

    Returns:
        Probability (0.0 - 1.0) that the strategy is NOT a false positive.
    """
    if n_trials < 2 or trial_std <= 1e-6:
        return 0.5

    expected_max_sr = trial_std * math.sqrt(2 * math.log(n_trials))

    if expected_max_sr == 0:
        return 0.0

    z_score = (sharpe - expected_max_sr) / trial_std

    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """Results from a single WFV fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]

    # Scores
    is_score: float
    oos_score: float

    # DSR inputs
    n_trials: int
    trial_std: float

    # Metrics
    oos_stats: Dict[str, Any]
    is_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def degradation(self) -> float:
        """Percentage drop from IS to OOS."""
        if self.is_score <= 0.0:
            return -1.0
        return (self.oos_score - self.is_score) / self.is_score

    @property
    def dsr_probability(self) -> float:
        """Probability this fold's success isn't luck."""
        return estimated_dsr(self.is_score, self.n_trials, self.trial_std)

    @property
    def is_failed(self) -> bool:
        """Treat non-positive IS score as failed optimization quality."""
        return self.is_score <= 0.0

    @property
    def is_win_rate(self) -> float:
        return float(self.is_stats.get("win_rate", 0.0))

    @property
    def oos_win_rate(self) -> float:
        return float(self.oos_stats.get("win_rate", 0.0))

    @property
    def win_rate_degradation(self) -> float:
        """
        Relative Win Rate drift from IS to OOS.
        > -0.10: normal, -0.10..-0.20: warning, < -0.20: red flag.
        """
        if self.is_failed or self.is_win_rate <= 0.0:
            return 0.0
        return (self.oos_win_rate - self.is_win_rate) / self.is_win_rate

    @property
    def oos_expected_value(self) -> float:
        """
        Per-trade OOS expected value proxy:
        EV = WR * AvgWin - (1 - WR) * |AvgLoss|
        """
        if self.is_failed:
            return 0.0
        wr = self.oos_win_rate
        avg_win = float(self.oos_stats.get("avg_win", 0.0))
        avg_loss_abs = abs(float(self.oos_stats.get("avg_loss", 0.0)))
        return wr * avg_win - (1 - wr) * avg_loss_abs


@dataclass
class WFVReport:
    """Aggregated results with Skeptic Analysis."""

    symbol: str
    strategy_name: str
    n_folds: int
    fold_results: List[FoldResult]

    # Aggregates
    median_oos_score: float = 0.0
    median_degradation: float = 0.0
    avg_dsr: float = 0.0

    # Candidate Selection
    candidate_params: Dict[str, Any] = field(default_factory=dict)
    verdict: str = "FAIL"
    warnings: List[str] = field(default_factory=list)
    pass_min_profitable_folds: int = 3
    warn_min_profitable_folds: int = 2
    pass_min_consecutive_profitable_folds: int = 2
    warn_min_consecutive_profitable_folds: int = 1
    min_sharpe_per_fold: float = 0.0

    # Computational Profiling
    total_wfo_time_sec: float = 0.0
    avg_fold_time_sec: float = 0.0
    avg_trial_time_sec: float = 0.0

    def compute(self) -> None:
        if not self.fold_results:
            return

        oos_scores = [f.oos_score for f in self.fold_results]
        degradations = [f.degradation for f in self.fold_results]
        dsrs = [f.dsr_probability for f in self.fold_results]

        self.median_oos_score = float(np.median(oos_scores))
        self.median_degradation = float(np.median(degradations))
        self.avg_dsr = float(np.mean(dsrs))

        self._analyze_robustness()
        self._select_candidate_params()

    def _analyze_robustness(self):
        """Apply Senior Quant logic to determine verdict."""
        n_profitable = self._count_quality_profitable_folds(self.min_sharpe_per_fold)
        max_consecutive_profitable = self._max_consecutive_quality_profitable_folds(
            self.min_sharpe_per_fold
        )

        if self.median_degradation < -0.50:
            self.warnings.append(
                f"Severe Overfitting: Median decay is {self.median_degradation:.0%}"
            )

        if self.avg_dsr < 0.5:
            self.warnings.append(
                f"Low Significance: DSR {self.avg_dsr:.2f} implies results "
                f"indistinguishable from noise."
            )

        wr_degradations = [
            f.win_rate_degradation
            for f in self.fold_results
            if not f.is_failed and f.is_win_rate > 0.0
        ]
        if wr_degradations:
            median_wr_degradation = float(np.median(wr_degradations))
            if median_wr_degradation < -0.20:
                self.warnings.append(
                    f"WinRate Drift RED: median IS→OOS degradation {median_wr_degradation:+.1%}"
                )
            elif median_wr_degradation < -0.10:
                self.warnings.append(
                    f"WinRate Drift YELLOW: median IS→OOS degradation {median_wr_degradation:+.1%}"
                )

            oos_wr_std = float(
                np.std([f.oos_win_rate for f in self.fold_results if not f.is_failed])
            )
            if oos_wr_std > 0.10:
                self.warnings.append(
                    f"Unstable OOS WinRate: fold std is {oos_wr_std:.1%} (possible regime/overfit mix)."
                )

        negative_ev_folds = sum(
            1
            for f in self.fold_results
            if not f.is_failed and f.oos_expected_value < 0.0
        )
        if negative_ev_folds > 0:
            self.warnings.append(
                f"Negative OOS EV in {negative_ev_folds}/{len(self.fold_results)} folds."
            )

        if (
            n_profitable >= self.pass_min_profitable_folds
            and max_consecutive_profitable >= self.pass_min_consecutive_profitable_folds
            and self.median_degradation > -0.40
            and self.avg_dsr > 0.6
        ):
            self.verdict = "PASS"
        elif (
            n_profitable >= self.warn_min_profitable_folds
            and max_consecutive_profitable >= self.warn_min_consecutive_profitable_folds
            and self.median_degradation > -0.60
        ):
            self.verdict = "WARNING"
        else:
            self.verdict = "FAIL"

    def _count_quality_profitable_folds(self, min_sharpe: float) -> int:
        """Count OOS-positive folds that also satisfy minimum Sharpe quality."""
        return sum(
            1
            for fold in self.fold_results
            if fold.oos_score > 0.0
            and float(fold.oos_stats.get("sharpe_ratio", 0.0)) >= min_sharpe
        )

    def _max_consecutive_quality_profitable_folds(self, min_sharpe: float) -> int:
        """Return the longest streak of OOS-positive, quality-qualified folds."""
        max_streak = 0
        current_streak = 0
        for fold in self.fold_results:
            is_quality_profitable = (
                fold.oos_score > 0.0
                and float(fold.oos_stats.get("sharpe_ratio", 0.0)) >= min_sharpe
            )
            if is_quality_profitable:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
                continue
            current_streak = 0
        return max_streak

    def _select_candidate_params(self) -> None:
        """Median/Mode of robust folds."""
        robust_folds = [f for f in self.fold_results if f.oos_score > 0]
        if not robust_folds:
            return

        source = robust_folds if len(robust_folds) >= 2 else self.fold_results

        consensus = {}
        first_params = source[0].best_params

        for key, val in first_params.items():
            all_vals = [f.best_params.get(key) for f in source]
            all_vals = [v for v in all_vals if v is not None]

            if not all_vals:
                continue

            if isinstance(val, (int, float)) and not isinstance(val, bool):
                median_val = np.median(all_vals)
                consensus[key] = (
                    int(round(median_val))
                    if isinstance(val, int)
                    else float(median_val)
                )
            else:
                consensus[key] = Counter(all_vals).most_common(1)[0][0]

        self.candidate_params = consensus


class WalkForwardOptimizer:
    """
    Orchestrator for Research-Grade Walk-Forward Validation.

    Uses date-bounded BacktestEngine runs for full compatibility
    with event-driven strategies (on_bar loop).
    """

    def __init__(
        self,
        settings: Optional[BacktestSettings] = None,
    ) -> None:
        if settings is None:
            raise ValueError("BacktestSettings must be provided to WalkForwardOptimizer.")
        self.settings = settings
        self.base_optimizer = OptunaOptimizer(settings=self.settings)
        self.data_lake = DataLake(self.settings)

    def run(
        self,
        strategy_class: Type,
        n_folds: Optional[int] = None,
        test_size_pct: Optional[float] = None,
        n_trials: Optional[int] = None,
        purge_bars: int = 0,
        embargo_bars: int = 0,
        anchored: bool = False,
        verbose: bool = True,
        print_report: bool = True,
        show_progress_bar: bool = True,
    ) -> WFVReport:
        """
        Run Walk-Forward Validation for a strategy.

        Loads the full dataset, generates date-based folds, then for each
        fold runs IS optimization and OOS evaluation via BacktestEngine.

        Args:
            strategy_class: Strategy class implementing BaseStrategy.
            n_folds: Number of walk-forward folds.
            test_size_pct: Fraction of data for each test fold.
            n_trials: Number of Optuna trials per fold.
            purge_bars: Gap between Train and Test to remove overlap.
            embargo_bars: Additional gap to prevent correlation leakage.
            anchored: True for expanding window, False for rolling window.
            verbose: Whether progress and fold diagnostics are printed.
            print_report: Whether the final human-readable report is printed.
            show_progress_bar: Whether Optuna prints its fold-level progress bar.

        Returns:
            WFVReport with fold results and skeptic analysis.
        """
        n_folds = n_folds or self.settings.wfo_n_folds
        test_size_pct = test_size_pct or self.settings.wfo_test_size_pct
        n_trials = n_trials or self.settings.wfo_n_trials
        
        symbol = self.settings.default_symbol
        timeframe = self.settings.low_interval

        # Load full dataset to determine date boundaries
        if verbose:
            print(f"\n[WFV] Loading {symbol} @ {timeframe} for fold generation...")
        data = self.data_lake.load(symbol, timeframe=timeframe)

        if data.empty:
            if verbose:
                print("[WFV] No data found. Aborting.")
            return WFVReport(symbol, strategy_class.__name__, 0, [])

        if verbose:
            print(
                f"[WFV] Data range: {data.index[0].date()} -> {data.index[-1].date()} "
                f"| {len(data):,} bars"
            )

        # Generate index-based folds
        splitter = PurgedFoldGenerator(
            n_folds=n_folds,
            test_size=test_size_pct,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            anchored=anchored,
        )
        folds = list(splitter.split(data))

        fold_results = []
        total_trials = 0
        wfo_start_time = time.time()

        # Silence Optuna's trial-by-trial logs for a cleaner console
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if verbose:
            print(f"\n[WFV] Starting {n_folds}-Fold Walk-Forward on {symbol}...")

        for i, (train_idx, test_idx) in enumerate(folds):
            # Convert indices to date boundaries AND slice data
            train_start = data.index[train_idx[0]]
            train_end = data.index[train_idx[-1]]
            test_start = data.index[test_idx[0]]
            test_end = data.index[test_idx[-1]]
            
            # Extract DataFrame slices once per fold
            train_slice = data.iloc[train_idx]
            test_slice = data.iloc[test_idx]

            if verbose:
                print(
                    f"\n  Fold {i + 1}/{len(folds)}: "
                    f"IS {train_start.date()} -> {train_end.date()} | "
                    f"OOS {test_start.date()} -> {test_end.date()}"
                )

            # 1. Optimize (IS) — dataset injected
            fold_start_time = time.time()
            opt_result = self.base_optimizer.optimize_on_slice(
                strategy_class=strategy_class,
                start_date=train_start,
                end_date=train_end,
                data=train_slice,
                n_trials=n_trials,
                fold_id=i,
                show_progress_bar=show_progress_bar,
            )
            fold_end_time = time.time()

            n_trials_actual = opt_result.get("n_trials", n_trials)
            total_trials += n_trials_actual
            trial_std = opt_result.get("trial_std", 0.0)

            # 2. Evaluate (OOS) — dataset injected
            if opt_result["best_score"] <= 0.0 or not opt_result["best_params"]:
                eval_result = {"score": -1.0, "stats": {}}
            else:
                eval_result = self.base_optimizer.evaluate_on_slice(
                    strategy_class=strategy_class,
                    params=opt_result["best_params"],
                    start_date=test_start,
                    end_date=test_end,
                    data=test_slice,
                )

            fold_results.append(
                FoldResult(
                    fold_id=i + 1,
                    train_start=str(train_start.date()),
                    train_end=str(train_end.date()),
                    test_start=str(test_start.date()),
                    test_end=str(test_end.date()),
                    best_params=opt_result["best_params"],
                    is_score=opt_result["best_score"],
                    oos_score=eval_result["score"],
                    n_trials=n_trials_actual,
                    trial_std=trial_std,
                    is_stats=opt_result.get("best_stats", {}),
                    oos_stats=eval_result["stats"],
                )
            )

            if verbose:
                print(
                    f"  Fold {i + 1}: IS {opt_result['best_score']:.2f} -> "
                    f"OOS {eval_result['score']:.2f} "
                    f"({fold_end_time - fold_start_time:.1f}s)"
                )

        wfo_end_time = time.time()
        total_time = wfo_end_time - wfo_start_time

        report = WFVReport(
            symbol=symbol,
            strategy_name=strategy_class.__name__,
            n_folds=len(folds),
            fold_results=fold_results,
            pass_min_profitable_folds=self.settings.wfo_pass_min_profitable_folds,
            warn_min_profitable_folds=self.settings.wfo_warn_min_profitable_folds,
            pass_min_consecutive_profitable_folds=(
                self.settings.wfo_pass_min_consecutive_profitable_folds
            ),
            warn_min_consecutive_profitable_folds=(
                self.settings.wfo_warn_min_consecutive_profitable_folds
            ),
            min_sharpe_per_fold=self.settings.wfo_min_sharpe_per_fold,
        )
        report.total_wfo_time_sec = total_time
        report.avg_fold_time_sec = total_time / len(folds) if folds else 0.0
        report.avg_trial_time_sec = total_time / total_trials if total_trials else 0.0

        report.compute()

        if print_report:
            self._print_human_report(report)
        return report

    def format_human_report(self, report: WFVReport) -> str:
        """
        Formats the existing human-readable WFO audit report.

        Methodology:
            Batch WFO needs the exact same analyst-facing text output as the
            regular command, but saved to files instead of printed inline.  The
            formatter keeps that content in one place so standard ``--wfo`` and
            ``wfo-batch`` stay consistent.

        Args:
            report: Aggregated WFO report instance.

        Returns:
            Multi-line terminal report string.
        """

        def _bar(val, max_val=2.0, width=10):
            if val < 0:
                return "💀 " + " " * (width - 2)
            normalized = min(1.0, val / max_val)
            chars = int(normalized * width)
            return "█" * chars + "░" * (width - chars)

        def _col(text, width=10, align="<"):
            return f"{str(text):{align}{width}}"

        lines: List[str] = []
        lines.append(f"\n\n{'=' * 80}")
        lines.append(f" WFV AUDIT REPORT: {report.strategy_name} @ {report.symbol}")
        lines.append(f"{'=' * 80}")

        # 1. High Level Summary
        lines.append(f"\n[EXECUTIVE SUMMARY]")
        lines.append(f"  Verdict:        {report.verdict}")
        lines.append(
            f"  Median OOS:     {report.median_oos_score:.4f} (Composite Score)"
        )
        lines.append(
            f"  Perf Decay:     {report.median_degradation:+.1%} (IS -> OOS)"
        )
        lines.append(
            f"  Skeptic Confidence (DSR): {report.avg_dsr:.0%} "
            f"(Prob. Skill > Noise)"
        )

        lines.append(f"\n[COMPUTATIONAL PROFILE]")
        lines.append(f"  Total WFO Runtime: {report.total_wfo_time_sec / 60:.1f} mins")
        lines.append(f"  Average Fold Time: {report.avg_fold_time_sec:.1f} secs")
        lines.append(f"  Average Trial Time: {report.avg_trial_time_sec:.3f} secs")

        if report.warnings:
            lines.append(f"\n[RISK FLAGS]")
            for w in report.warnings:
                lines.append(f"  ! {w}")

        # 2. Fold Detail
        lines.append(f"\n[FOLD ANALYSIS]")
        header = (
            f"{_col('Fold', 4)} | "
            f"{_col('Period', 22)} | "
            f"{_col('IS Score', 8)} | "
            f"{_col('OOS Score', 9)} | "
            f"{_col('Decay', 7)} | "
            f"{_col('WR IS', 6)} | "
            f"{_col('WR OOS', 7)} | "
            f"{_col('WR Δ', 7)} | "
            f"{_col('Visual', 12)} | "
            f"{_col('DD%', 6)}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for f in report.fold_results:
            if f.degradation < -0.5:
                decay_str = "CRASH"
            else:
                decay_str = f"{f.degradation:+.0%}"

            dd_val = f.oos_stats.get("max_drawdown", 0)
            dd_str = f"{dd_val:.1f}"
            if f.is_failed:
                is_wr, oos_wr, wr_delta = "n/a", "n/a", "n/a"
            else:
                is_wr = f"{f.is_win_rate:.0%}"
                oos_wr = f"{f.oos_win_rate:.0%}"
                wr_delta = f"{f.win_rate_degradation:+.0%}"

            lines.append(
                f"{_col(str(f.fold_id), 4)} | "
                f"{_col(f'{f.test_start}..', 22)} | "
                f"{_col(f'{f.is_score:.2f}', 8)} | "
                f"{_col(f'{f.oos_score:.2f}', 9)} | "
                f"{_col(decay_str, 7)} | "
                f"{_col(is_wr, 6)} | "
                f"{_col(oos_wr, 7)} | "
                f"{_col(wr_delta, 7)} | "
                f"{_col(_bar(f.oos_score, 1.5), 12)} | "
                f"{_col(dd_str, 6)}"
            )

        # 3. Params
        if report.candidate_params:
            lines.append(f"\n[CANDIDATE PARAMETERS]")
            lines.append(f"  {report.candidate_params}")
        else:
            lines.append(
                "\n[NO CANDIDATE PARAMETERS] Strategy failed stability checks."
            )

        lines.append(f"{'=' * 80}\n")
        return "\n".join(lines)

    def _print_human_report(self, report: WFVReport) -> None:
        """Prints the standard Quant-style WFO report."""
        print(self.format_human_report(report))
