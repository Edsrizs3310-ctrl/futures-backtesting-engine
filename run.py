"""
Entry point for the Backtesting Engine.

Execution modes:
    1. Single-Asset Backtest      (--backtest)
    2. Walk-Forward Optimization  (--wfo)
    3. Multi-Strategy Portfolio   (--portfolio-backtest)
    4. Dashboard only             (--dashboard)

── 1. Single-Asset Backtesting ──────────────────────────────────────────────
    python run.py --backtest --strategy zscore

── 2. Walk-Forward Optimization (WFO) ───────────────────────────────────────
    python run.py --wfo --strategy zscore

── 3. Portfolio Backtesting ─────────────────────────────────────────────────
    python run.py --portfolio-backtest
    python run.py --portfolio-backtest --portfolio-config my_config.yaml

── 4. Open Dashboard (standalone, no new backtest) ──────────────────────────
    python run.py --dashboard

To run a backtest AND open the dashboard immediately after:
    python run.py --backtest --strategy zscore --dashboard
    python run.py --portfolio-backtest --dashboard

── Data Management ──────────────────────────────────────────────────────────
    python run.py --download ES NQ CL GC
"""

import argparse
import subprocess
import sys
from pathlib import Path


_DASHBOARD_PATH = (
    Path(__file__).parent
    / "src" / "backtest_engine" / "analytics" / "dashboard" / "app.py"
)
_PROJECT_ROOT = Path(__file__).parent


def _launch_dashboard() -> None:
    """
    Launches Streamlit dashboard as a child process.

    Runs in the foreground so the terminal shows Streamlit logs.
    The backtest process has already finished writing artifacts before
    this call — no race condition.
    """
    print("\n[Dashboard] Launching Streamlit dashboard...")
    print(f"[Dashboard] URL: http://localhost:8501\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(_DASHBOARD_PATH)],
        cwd=str(_PROJECT_ROOT),
        check=False,
    )


if __name__ == "__main__":
    from src.strategies.registry import STRATEGIES
    strategy_list = ", ".join(STRATEGIES.keys())

    parser = argparse.ArgumentParser(
        description="Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--download", nargs="+",
                        help="Download data for symbols via IB")
    parser.add_argument("--backtest", action="store_true",
                        help="Run single-asset backtest")
    parser.add_argument("--wfo", action="store_true",
                        help="Run Walk-Forward Optimization")
    parser.add_argument("--strategy", type=str, default="sma",
                        help=f"Strategy name ({strategy_list})")
    parser.add_argument("--portfolio-backtest", action="store_true",
                        help="Run multi-strategy portfolio backtest")
    parser.add_argument("--portfolio-config", type=str,
                        default="src/backtest_engine/portfolio_layer/portfolio_config_example.yaml",
                        help="Path to YAML portfolio config")
    parser.add_argument("--dashboard", action="store_true",
                        help=(
                            "Launch Streamlit dashboard. "
                            "When combined with --backtest/--portfolio-backtest, "
                            "opens AFTER the backtest completes. "
                            "Standalone: 'python run.py --dashboard'."
                        ))
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # ── Mode: dashboard only (no backtest) ────────────────────────────────────
    if args.dashboard and not args.backtest and not args.portfolio_backtest and not args.wfo:
        _launch_dashboard()
        sys.exit(0)

    # ── Mode: data download ───────────────────────────────────────────────────
    from src.backtest_engine.settings import get_settings
    settings = get_settings()

    if args.download:
        from src.data import IBFetcher
        print("=" * 60)
        print(f"  Downloading data: {args.download}")
        print("=" * 60)
        fetcher = IBFetcher(settings=settings)
        for sym in args.download:
            fetcher.fetch_all_timeframes(sym)
        print("Download complete.")

    # ── Mode: single backtest ─────────────────────────────────────────────────
    if args.backtest:
        from cli.single import run as run_backtest
        run_backtest(args.strategy, settings, launch_dashboard=False)
        if args.dashboard:
            _launch_dashboard()

    # ── Mode: WFO ─────────────────────────────────────────────────────────────
    if args.wfo:
        from cli.wfo import run as run_wfo
        run_wfo(args.strategy, settings)

    # ── Mode: portfolio backtest ───────────────────────────────────────────────
    if getattr(args, "portfolio_backtest", False):
        from cli.portfolio import run as run_portfolio
        run_portfolio(args.portfolio_config, launch_dashboard=False)
        if args.dashboard:
            _launch_dashboard()
