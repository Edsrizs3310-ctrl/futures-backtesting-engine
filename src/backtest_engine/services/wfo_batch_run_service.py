"""
Lightweight batch orchestrator for Walk-Forward Optimization runs.

Methodology:
    Each scenario reuses the existing ``WalkForwardOptimizer`` unchanged for
    scoring and verdict logic.  The batch layer only coordinates many isolated
    runs, renders one verdict heatmap, prints compact PASS/WARNING tables, and
    exports candidate parameters to a visible results directory.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from src.backtest_engine.optimization.wfv_optimizer import WalkForwardOptimizer
from src.backtest_engine.services.batch_models import BatchScenario, WfoBatchResult
from src.backtest_engine.services.batch_plot_service import show_wfo_batch_heatmap
from src.backtest_engine.services.run_helpers import validate_cache_requirements_or_exit
from src.backtest_engine.settings import BacktestSettings
from src.strategies.registry import load_strategy_by_id, resolve_strategy_id


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    """Normalizes symbol inputs to uppercase, preserving order."""
    return [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]


def _normalize_timeframes(timeframes: Sequence[str]) -> List[str]:
    """Normalizes timeframe inputs, preserving order."""
    return [str(timeframe).strip() for timeframe in timeframes if str(timeframe).strip()]


def _build_scenarios(
    strategy_names: Sequence[str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> List[BatchScenario]:
    """Creates the Cartesian product of requested WFO batch scenarios."""
    normalized_strategies = [resolve_strategy_id(name) for name in strategy_names]
    normalized_symbols = _normalize_symbols(symbols)
    normalized_timeframes = _normalize_timeframes(timeframes)
    return [
        BatchScenario(strategy_id=strategy_id, symbol=symbol, timeframe=timeframe)
        for strategy_id in normalized_strategies
        for symbol in normalized_symbols
        for timeframe in normalized_timeframes
    ]


def _validate_strategies_or_exit(strategy_names: Sequence[str]) -> None:
    """Fails fast when any requested strategy ID or alias is unknown."""
    try:
        for strategy_name in strategy_names:
            load_strategy_by_id(resolve_strategy_id(strategy_name))
    except ValueError as exc:
        print(f"[Error] {exc}")
        sys.exit(1)


def _resolve_max_workers(
    scenario_count: int,
    settings: BacktestSettings,
    requested_workers: Optional[int],
) -> int:
    """Bounds worker count to a safe, useful range."""
    cpu_count = os.cpu_count() or int(settings.batch_max_workers)
    base_workers = int(requested_workers or settings.batch_max_workers)
    return max(1, min(base_workers, scenario_count, cpu_count))


def _render_progress_bar(
    current: int,
    total: int,
    width: int,
    label: str,
    start_time: float,
) -> None:
    """Prints one in-place progress bar with elapsed time, ETA, and speed."""
    safe_total = max(total, 1)
    progress = current / safe_total
    filled = int(width * progress)
    bar = "#" * filled + "-" * (width - filled)

    elapsed = time.monotonic() - start_time
    avg_sec = elapsed / current if current > 0 else 0.0
    remaining = avg_sec * (safe_total - current)

    def _fmt_t(secs: float) -> str:
        m, s = divmod(int(secs), 60)
        return f"{m:02d}:{s:02d}"

    timing = f"[{_fmt_t(elapsed)}<{_fmt_t(remaining)}, {avg_sec:.2f}s/it]"
    percent = f"{progress:3.0%}"
    message = f"\r[WFO Batch] {percent}|{bar}| {current}/{total} {label} {timing}".rstrip()
    sys.stdout.write(message)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _run_wfo_batch_worker(
    scenario: BatchScenario,
    settings_payload: Dict[str, Any],
) -> WfoBatchResult:
    """
    Executes one WFO batch scenario inside a worker process.

    Args:
        scenario: Scenario definition.
        settings_payload: Serialized base settings from the parent process.

    Returns:
        Serializable WFO batch result.
    """
    settings = BacktestSettings(**settings_payload).model_copy(
        update={
            "default_symbol": scenario.symbol,
            "low_interval": scenario.timeframe,
        }
    )

    try:
        # Suppress Optuna INFO logs (study creation, trial messages) that would
        # leak through Python's logging system even inside redirect_stderr.
        logging.disable(logging.WARNING)
        strategy_class = load_strategy_by_id(scenario.strategy_id)
        optimizer = WalkForwardOptimizer(settings=settings)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report = optimizer.run(
                strategy_class=strategy_class,
                verbose=False,
                print_report=False,
                show_progress_bar=False,
            )
        logging.disable(logging.NOTSET)
        text_report = optimizer.format_human_report(report)
        return WfoBatchResult(
            scenario=scenario,
            status="completed",
            verdict=report.verdict,
            n_folds=report.n_folds,
            median_oos_score=report.median_oos_score,
            median_degradation=report.median_degradation,
            avg_dsr=report.avg_dsr,
            total_wfo_time_sec=report.total_wfo_time_sec,
            avg_fold_time_sec=report.avg_fold_time_sec,
            avg_trial_time_sec=report.avg_trial_time_sec,
            candidate_params=report.candidate_params,
            warnings=list(report.warnings),
            text_report=text_report,
        )
    except Exception as exc:
        return WfoBatchResult(
            scenario=scenario,
            status="failed",
            verdict="FAIL",
            error=str(exc),
        )


def _print_verdict_table(title: str, results: Sequence[WfoBatchResult]) -> None:
    """
    Prints one compact terminal table for a verdict section.

    Args:
        title: Section title.
        results: Already-filtered and ordered result set.
    """
    if not results:
        return

    widths = {
        "strategy": 18,
        "symbol": 8,
        "tf": 8,
        "verdict": 10,
        "oos": 12,
        "decay": 10,
        "dsr": 8,
        "folds": 7,
        "runtime": 12,
    }
    header = (
        f"{'Strategy':<{widths['strategy']}} "
        f"{'Ticker':<{widths['symbol']}} "
        f"{'TF':<{widths['tf']}} "
        f"{'Verdict':<{widths['verdict']}} "
        f"{'Median OOS':>{widths['oos']}} "
        f"{'Decay':>{widths['decay']}} "
        f"{'DSR':>{widths['dsr']}} "
        f"{'Folds':>{widths['folds']}} "
        f"{'Runtime m':>{widths['runtime']}}"
    )
    print(f"\n[{title}]")
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.scenario.strategy_id:<{widths['strategy']}} "
            f"{result.scenario.symbol:<{widths['symbol']}} "
            f"{result.scenario.timeframe:<{widths['tf']}} "
            f"{result.verdict:<{widths['verdict']}} "
            f"{result.median_oos_score:>{widths['oos']}.4f} "
            f"{result.median_degradation:>{widths['decay']}.1%} "
            f"{result.avg_dsr:>{widths['dsr']}.0%} "
            f"{result.n_folds:>{widths['folds']}d} "
            f"{(result.total_wfo_time_sec / 60.0):>{widths['runtime']}.2f}"
        )


def _write_wfo_batch_exports(
    results: Sequence[WfoBatchResult],
    settings: BacktestSettings,
) -> Optional[Path]:
    """
    Writes candidate exports for PASS/WARNING scenarios.

    Methodology:
        The terminal table intentionally omits parameter dictionaries to stay
        readable.  Candidate parameters and the full text reports are exported
        into a visible timestamped directory for copy/paste workflow.

    Args:
        results: Completed WFO batch results.
        settings: Base backtest settings.

    Returns:
        Export root path, or ``None`` when nothing qualified for export.
    """
    exportable_results = [
        result
        for result in results
        if result.status == "completed" and result.verdict in {"PASS", "WARNING"}
    ]
    if not exportable_results:
        return None

    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_root = settings.get_wfo_batch_results_path() / timestamp_label
    configs_dir = export_root / "configs"
    reports_dir = export_root / "reports"
    configs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = configs_dir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "symbol",
                "timeframe",
                "verdict",
                "median_oos_score",
                "median_degradation",
                "avg_dsr",
                "n_folds",
                "total_wfo_time_sec",
                "avg_fold_time_sec",
                "avg_trial_time_sec",
                "warnings_json",
                "candidate_params_json",
                "report_path",
            ],
        )
        writer.writeheader()
        for result in exportable_results:
            report_path = reports_dir / f"{result.scenario.scenario_id}.txt"
            report_path.write_text(result.text_report, encoding="utf-8")
            writer.writerow(
                {
                    "strategy": result.scenario.strategy_id,
                    "symbol": result.scenario.symbol,
                    "timeframe": result.scenario.timeframe,
                    "verdict": result.verdict,
                    "median_oos_score": f"{result.median_oos_score:.6f}",
                    "median_degradation": f"{result.median_degradation:.6f}",
                    "avg_dsr": f"{result.avg_dsr:.6f}",
                    "n_folds": result.n_folds,
                    "total_wfo_time_sec": f"{result.total_wfo_time_sec:.6f}",
                    "avg_fold_time_sec": f"{result.avg_fold_time_sec:.6f}",
                    "avg_trial_time_sec": f"{result.avg_trial_time_sec:.6f}",
                    "warnings_json": json.dumps(result.warnings, ensure_ascii=True),
                    "candidate_params_json": json.dumps(
                        result.candidate_params, ensure_ascii=True, sort_keys=True
                    ),
                    "report_path": str(report_path),
                }
            )

    return export_root


def run_wfo_batch_backtests(
    strategy_names: Sequence[str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
    settings: BacktestSettings,
    max_workers: Optional[int] = None,
) -> None:
    """
    Runs many independent WFO scenarios and renders one verdict heatmap.

    Args:
        strategy_names: Strategy identifiers or accepted aliases.
        symbols: One or many futures symbols.
        timeframes: One or many timeframe strings.
        settings: Base backtest settings.
        max_workers: Optional worker override.
    """
    _validate_strategies_or_exit(strategy_names)
    scenarios = _build_scenarios(strategy_names, symbols, timeframes)
    if not scenarios:
        print("[WFO Batch] No scenarios requested.")
        return

    validate_cache_requirements_or_exit(
        requirements=[(scenario.symbol, scenario.timeframe) for scenario in scenarios],
        settings=settings,
    )

    resolved_workers = _resolve_max_workers(
        scenario_count=len(scenarios),
        settings=settings,
        requested_workers=max_workers,
    )
    settings_payload = settings.model_dump(mode="python")

    print("=" * 72)
    print("  Lightweight WFO Batch")
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Workers   : {resolved_workers}")
    print("=" * 72)

    results: List[WfoBatchResult] = []
    progress_width = int(settings.batch_progress_bar_width)
    start_time = time.monotonic()

    if len(scenarios) == 1:
        result = _run_wfo_batch_worker(scenarios[0], settings_payload)
        results.append(result)
        _render_progress_bar(1, 1, progress_width, scenarios[0].legend_label, start_time)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = {
                executor.submit(_run_wfo_batch_worker, scenario, settings_payload): scenario
                for scenario in scenarios
            }
            completed = 0
            for future in as_completed(futures):
                scenario = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = WfoBatchResult(
                        scenario=scenario,
                        status="failed",
                        verdict="FAIL",
                        error=str(exc),
                    )
                results.append(result)
                completed += 1
                _render_progress_bar(
                    completed,
                    len(scenarios),
                    progress_width,
                    scenario.legend_label,
                    start_time,
                )

    completed_results = [result for result in results if result.status == "completed"]
    pass_results = sorted(
        [result for result in completed_results if result.verdict == "PASS"],
        key=lambda item: item.median_oos_score,
        reverse=True,
    )
    warning_results = sorted(
        [result for result in completed_results if result.verdict == "WARNING"],
        key=lambda item: item.median_oos_score,
        reverse=True,
    )
    failed_results = [result for result in results if result.status != "completed"]

    if results:
        show_wfo_batch_heatmap(
            results=results,
            figure_width=float(settings.batch_plot_figure_width),
            figure_height=float(settings.batch_plot_figure_height),
        )
    else:
        print("[WFO Batch] No successful scenarios to visualize.")

    _print_verdict_table("PASS", pass_results)
    _print_verdict_table("WARNING", warning_results)

    export_root = _write_wfo_batch_exports(completed_results, settings)
    if export_root is not None:
        print(f"\n[WFO Batch] Candidate exports saved -> {export_root}")
    else:
        print("\n[WFO Batch] No PASS/WARNING candidate parameters were exported.")

    if failed_results:
        print("[WFO Batch] Failed scenarios:")
        for result in failed_results:
            print(f"  - {result.scenario.legend_label}: {result.error}")
