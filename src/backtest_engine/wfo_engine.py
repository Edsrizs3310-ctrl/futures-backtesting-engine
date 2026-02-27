"""
Walk-Forward Optimisation (WFO) Engine.

Strategy-agnostic 3-phase optimiser powered by Optuna (TPE + ASHA).

Phase 1 — Coarse Search   : Fast exploration on limited IS history.
Phase 2 — Full Fidelity   : Full-dataset validation of top Phase 1 candidates.
Phase 3 — Rolling Windows : IS/OOS folds with warm-start between folds.

Any strategy implementing BaseStrategy.get_search_space() is compatible.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from src.backtest_engine.engine import BacktestEngine
from src.backtest_engine.settings import BacktestSettings, get_settings


# ── Utilities ──────────────────────────────────────────────────────────────────


class _HiddenPrints:
    """Suppresses stdout during WFO iterations to keep console output clean."""

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_):
        sys.stdout.close()
        sys.stdout = self._orig


# ═══════════════════════════════════════════════════════════════════════════════
# WFO Engine
# ═══════════════════════════════════════════════════════════════════════════════


class WFOEngine:
    """
    Walk-Forward Optimisation engine for any single-asset BaseStrategy.

    Methodology:
        1. Strategy provides search space via get_search_space().
        2. Phase 1 runs a fast coarse Optuna search over recent IS history.
        3. Phase 2 evaluates top Phase 1 candidates on the full dataset.
        4. Phase 3 runs rolling IS/OOS folds with fold-to-fold warm starts
           to surface robust, non-overfitted parameter sets.

    The objective function scores each trial using a composite metric:
        Sharpe + 0.5 * Sortino − 2 * |MaxDD| + 0.2 * WinRate − trade_penalty

    Args:
        strategy_class: Any class implementing BaseStrategy.
        settings: Optional settings override; defaults to singleton.
    """

    def __init__(
        self,
        strategy_class: Type,
        settings: Optional[BacktestSettings] = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.settings = settings or get_settings()

        # Strategy-defined search space (overrides any legacy settings field)
        self.search_space: Dict[str, Any] = strategy_class.get_search_space()
        if not self.search_space:
            print(
                f"[WFO] Warning: {strategy_class.__name__}.get_search_space() returned "
                f"an empty dict. No parameters will be optimised."
            )

        # Internal date pointers shared between run_* methods and objective()
        self._is_start_dt: Optional[pd.Timestamp] = None
        self._is_end_dt: Optional[pd.Timestamp] = None

    # ── Search space application ───────────────────────────────────────────────

    def _apply_params(self, trial: optuna.Trial, s: BacktestSettings) -> None:
        """
        Samples parameter values from the trial and applies them to settings.

        Supports three bound formats:
            - (start, stop, step): int or float range with step.
            - (start, stop): continuous range without step.
            - [v1, v2, ...]: categorical choice.

        Args:
            trial: Active Optuna trial.
            s: Settings object to mutate in-place.
        """
        for param, bounds in self.search_space.items():
            if isinstance(bounds, list):
                val = trial.suggest_categorical(param, bounds)
            elif isinstance(bounds, tuple):
                if len(bounds) == 3:
                    start, stop, step = bounds
                    if all(isinstance(v, int) for v in (start, stop, step)):
                        val = trial.suggest_int(param, start, stop, step=step)
                    else:
                        val = trial.suggest_float(param, float(start), float(stop), step=float(step))
                elif len(bounds) == 2:
                    start, stop = bounds
                    if isinstance(start, int) and isinstance(stop, int):
                        val = trial.suggest_int(param, start, stop)
                    else:
                        val = trial.suggest_float(param, float(start), float(stop))
                else:
                    raise ValueError(f"[WFO] Invalid bounds for param '{param}': {bounds}")
            else:
                continue
            setattr(s, param, val)

    # ── Objective score ────────────────────────────────────────────────────────

    def _score(self, engine: BacktestEngine) -> float:
        """
        Composite robust score for a completed backtest run.

        Formula:
            score = Sharpe + 0.5 * Sortino − 2 * |MaxDD| + 0.2 * WinRate − trade_penalty

        A trade_penalty discourages parameter sets that produce fewer trades than
        `wfo_prune_min_trades * 2` (insufficient statistical significance).

        Args:
            engine: Finished BacktestEngine with populated analytics.

        Returns:
            Float score; higher is better.
        """
        history = engine.portfolio.get_history_df()
        metrics = engine.analytics.calculate_metrics(history, engine.execution.trades)

        if not metrics:
            return -100.0

        sharpe   = metrics.get("Sharpe Ratio", 0.0)
        sortino  = metrics.get("Sortino Ratio", 0.0)
        max_dd   = metrics.get("Max Drawdown", 0.0)
        win_rate = metrics.get("Win Rate", 0.0)
        trades   = metrics.get("Total Trades", 0)

        if trades == 0:
            return -100.0

        min_trades = self.settings.wfo_prune_min_trades
        trade_penalty = max(0.0, (min_trades * 2 - trades) * 0.1)

        return sharpe + 0.5 * sortino - 2.0 * abs(max_dd) + 0.2 * win_rate - trade_penalty

    # ── Optuna objective ───────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function compatible with ASHA pruning.

        Runs a BacktestEngine over the currently configured IS window,
        reports intermediate PnL values for pruning, and returns the
        composite robust score at the end.

        Args:
            trial: Optuna trial delivered by study.optimize().

        Returns:
            Robust score (float); raise TrialPruned to abort early.
        """
        s = get_settings()
        self._apply_params(trial, s)

        end_dt = self._is_end_dt or pd.Timestamp.now()
        start_dt = self._is_start_dt or (
            end_dt - pd.DateOffset(months=self.settings.wfo_coarse_months)
        )

        engine = BacktestEngine(start_date=start_dt, end_date=end_dt, settings=s)

        def _pruning_callback(eng: BacktestEngine, _date, step: int, total: int) -> None:
            interval = max(1, total // 10)
            if step % interval != 0 or step == 0:
                return
            pnl = eng.portfolio.total_value - eng.portfolio.initial_capital
            trial.report(pnl, step)
            history = eng.portfolio.get_history_df()
            max_dd_pct = 0.0
            if not history.empty:
                tv = history["total_value"]
                max_dd_pct = ((tv.cummax() - tv) / eng.settings.initial_capital).max() * 100
            if (
                trial.should_prune()
                or max_dd_pct > self.settings.wfo_prune_max_dd_pct
                or pnl < self.settings.wfo_prune_min_pnl
            ):
                raise optuna.TrialPruned()

        with _HiddenPrints():
            engine.run(self.strategy_class, step_callback=_pruning_callback)

        score = self._score(engine)
        trades = len(engine.execution.trades)
        trial.set_user_attr("trades", trades)

        if trades < self.settings.wfo_prune_min_trades:
            raise optuna.TrialPruned()

        return score

    # ── Phase 1: Coarse Search ─────────────────────────────────────────────────

    def run_coarse_search(self, n_trials: Optional[int] = None) -> None:
        """
        Phase 1: Fast hyperparameter exploration on limited IS history.

        Uses TPE + ASHA.  Top-k candidates are promoted to Phase 2.

        Args:
            n_trials: Number of Optuna trials; defaults to settings value.
        """
        n_trials = n_trials or self.settings.wfo_coarse_trials
        strategy_name = self.strategy_class.__name__

        print("=" * 60)
        print(f"  WFO PHASE 1: COARSE SEARCH — {strategy_name}")
        print(f"  Trials: {n_trials} | IS: last {self.settings.wfo_coarse_months} months")
        print("=" * 60)

        start_time = time.time()

        def _eta_callback(study: optuna.Study, trial: optuna.Trial) -> None:
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            elapsed = time.time() - start_time
            if completed == 0:
                return
            eta_s = int((elapsed / completed) * (n_trials - completed))
            best = study.best_value if completed > 0 else "N/A"
            print(f"  Trial {len(study.trials)}/{n_trials} | Best: {best} | ETA: {timedelta(seconds=eta_s)}")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
            study_name=f"WFO_Coarse_{strategy_name}",
        )

        try:
            study.optimize(
                self._objective, n_trials=n_trials,
                callbacks=[_eta_callback], show_progress_bar=False,
            )
        except KeyboardInterrupt:
            print("\n[WFO] Phase 1 interrupted by user.")

        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        print(f"\n  Complete: {len(complete)} | Pruned: {len(pruned)}")

        if not complete:
            print("[WFO] No trials completed. Adjust search space or pruning thresholds.")
            return

        top_k = self.settings.wfo_top_k_candidates
        top_trials = sorted(
            [t for t in complete if t.value is not None],
            key=lambda t: t.value, reverse=True,
        )[:top_k]

        self._print_trial_table(top_trials)

        if not self._confirm(f"Continue to Phase 2 (Full Test) for top {len(top_trials)}?"):
            return
        self.run_full_fidelity(top_trials)

    # ── Phase 2: Full Fidelity ─────────────────────────────────────────────────

    def run_full_fidelity(self, top_trials: List[optuna.trial.FrozenTrial]) -> None:
        """
        Phase 2: Validates Phase 1 winners on the complete dataset.

        Args:
            top_trials: List of completed FrozenTrial objects from Phase 1.
        """
        print("\n" + "=" * 60)
        print("  WFO PHASE 2: FULL FIDELITY TEST")
        print("=" * 60)

        results: List[Tuple[float, dict, BacktestEngine]] = []

        for i, trial in enumerate(top_trials):
            print(f"\n[Phase 2] Trial {i + 1}/{len(top_trials)} | Phase 1 score: {trial.value:.4f}")
            s = get_settings()
            for k, v in trial.params.items():
                setattr(s, k, v)
            engine = BacktestEngine(settings=s)
            with _HiddenPrints():
                engine.run(self.strategy_class)
            score = self._score(engine)
            results.append((score, trial.params, engine))
            print(f"  → Full score: {score:.4f} | Trades: {len(engine.execution.trades)}")

        results.sort(key=lambda x: x[0], reverse=True)
        final_top = results[: self.settings.wfo_final_top_k]

        print("\n" + "=" * 60)
        print(f"  PHASE 2 TOP {len(final_top)} PARAMS")
        print("=" * 60)
        for i, (score, params, eng) in enumerate(final_top):
            print(f"\n  {i + 1}. Score: {score:.4f} | Trades: {len(eng.execution.trades)}")
            for k, v in params.items():
                print(f"      {k}: {v}")

        if not self._confirm(f"Continue to Phase 3 (Rolling Windows)?"):
            return
        self.run_rolling_wfo(
            [(score, params) for score, params, _ in final_top]
        )

    # ── Phase 3: Rolling Windows ───────────────────────────────────────────────

    def run_rolling_wfo(
        self, top_candidates: List[Tuple[float, dict]]
    ) -> None:
        """
        Phase 3: Rolling IS/OOS walk-forward validation.

        For each candidate, runs multiple IS optimisations (warm-started from
        the previous fold's best params) and evaluates OOS performance to produce
        a final, instability-penalised robust score.

        Args:
            top_candidates: List of (phase2_score, param_dict) tuples.
        """
        is_months  = self.settings.wfo_rolling_is_months
        oos_months = self.settings.wfo_rolling_oos_months
        step       = self.settings.wfo_rolling_step_months
        n_trials   = self.settings.wfo_rolling_trials

        print("\n" + "=" * 60)
        print(f"  WFO PHASE 3: ROLLING WINDOWS (IS: {is_months}m | OOS: {oos_months}m | Step: {step}m)")
        print("=" * 60)

        # Determine full dataset bounds
        with _HiddenPrints():
            dummy = BacktestEngine(settings=self.settings)
            df = dummy.data_lake.load(
                self.settings.default_symbol,
                timeframe=self.settings.low_interval,
            )

        if df.empty:
            print("[Phase 3] Error: No data loaded. Aborting.")
            return

        total_start = df.index.min()
        total_end   = df.index.max()
        folds = self._generate_folds(total_start, total_end, is_months, oos_months, step)

        if not folds:
            print("[Phase 3] Insufficient data for even one IS/OOS fold.")
            return

        print(f"  Data range: {total_start.date()} → {total_end.date()} | Folds: {len(folds)}")

        final_results = []
        for i, (base_score, base_params) in enumerate(top_candidates):
            print(f"\n[Phase 3] Candidate {i + 1}/{len(top_candidates)} (P2 score: {base_score:.4f})")
            oos_scores: List[float] = []
            current_params = base_params.copy()

            for f_idx, fold in enumerate(folds):
                print(
                    f"  Fold {f_idx + 1:2d} | IS: {fold['is_start'].date()} → {fold['is_end'].date()} "
                    f"| OOS: → {fold['oos_end'].date()}"
                )

                self._is_start_dt = fold["is_start"]
                self._is_end_dt   = fold["is_end"]

                study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(seed=42 + f_idx),
                )
                study.enqueue_trial(current_params)
                with _HiddenPrints():
                    study.optimize(self._objective, n_trials=n_trials, show_progress_bar=False)

                comp = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                fold_params = study.best_trial.params if comp else current_params
                is_score    = study.best_value if comp else 0.0

                # OOS evaluation
                s = get_settings()
                for k, v in fold_params.items():
                    setattr(s, k, v)
                oos_engine = BacktestEngine(
                    start_date=fold["oos_start"],
                    end_date=fold["oos_end"],
                    settings=s,
                )
                with _HiddenPrints():
                    oos_engine.run(self.strategy_class)

                oos_score = self._score(oos_engine)
                oos_scores.append(oos_score)
                print(f"         IS score (opt): {is_score:8.4f} | OOS score: {oos_score:8.4f}")
                current_params = fold_params

            self._is_start_dt = None
            self._is_end_dt   = None

            if oos_scores:
                mean_oos = float(np.mean(oos_scores))
                std_oos  = float(np.std(oos_scores))
                penalty  = std_oos * 0.5
                final_score = mean_oos - penalty
            else:
                mean_oos = std_oos = penalty = 0.0
                final_score = -100.0

            print(
                f"  → Final OOS score: {final_score:.4f} "
                f"(mean: {mean_oos:.4f}, std: {std_oos:.4f}, penalty: {penalty:.4f})"
            )
            final_results.append(
                (final_score, base_params, current_params, mean_oos, std_oos)
            )

        final_results.sort(key=lambda x: x[0], reverse=True)

        print("\n" + "=" * 60)
        print("  WFO PHASE 3 COMPLETE — ROLLING OOS RANKING")
        print("=" * 60)
        for i, (f_score, _init, end_params, mean, std) in enumerate(final_results):
            print(f"\n  Rank {i + 1}: Final score = {f_score:.4f} (mean: {mean:.4f}, std: {std:.4f})")
            print("  Recommended params (end of final fold):")
            for k, v in list(end_params.items())[:6]:
                print(f"    {k}: {v}")

        print("\n[WFO] Full pipeline complete.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_folds(
        start: pd.Timestamp,
        end: pd.Timestamp,
        is_months: int,
        oos_months: int,
        step: int,
    ) -> List[Dict[str, pd.Timestamp]]:
        """
        Generates non-overlapping IS/OOS fold windows.

        Args:
            start: First available data date.
            end: Last available data date.
            is_months: In-Sample window length in months.
            oos_months: Out-of-Sample window length in months.
            step: Walk-forward step size in months.

        Returns:
            List of fold dicts with keys is_start, is_end, oos_start, oos_end.
        """
        folds = []
        current = start
        while True:
            is_end  = current + pd.DateOffset(months=is_months)
            oos_end = is_end  + pd.DateOffset(months=oos_months)
            if is_end >= end:
                break
            oos_end = min(oos_end, end)
            folds.append(
                {
                    "is_start":  current,
                    "is_end":    is_end,
                    "oos_start": is_end,
                    "oos_end":   oos_end,
                }
            )
            if oos_end >= end:
                break
            current += pd.DateOffset(months=step)
        return folds

    @staticmethod
    def _print_trial_table(trials: List[optuna.trial.FrozenTrial]) -> None:
        """Prints a formatted summary table of the top Phase 1 trials."""
        print("\n" + "-" * 80)
        print(f"{'Rank':<5} | {'Trial':<6} | {'Score':<10} | {'Trades':<8} | Key Params")
        print("-" * 80)
        for i, t in enumerate(trials):
            trds = t.user_attrs.get("trades", 0)
            params = ", ".join(f"{k}={v}" for k, v in list(t.params.items())[:3]) + "..."
            print(f"{i + 1:<5} | {t.number:<6} | {t.value:<10.4f} | {trds:<8} | {params}")
        print("-" * 80)

    @staticmethod
    def _confirm(message: str) -> bool:
        """Prompts the user for a yes/no confirmation."""
        try:
            return input(f"\n{message} [y/n]: ").strip().lower() == "y"
        except EOFError:
            print("Non-interactive mode — proceeding automatically.")
            return True
