# Usage Guide

Detailed command examples for the backtesting CLI live here so `run.py` can
stay a small entry point instead of a large in-file manual.

## Quick Start

1. Install dependencies:
   `pip install -r requirements.txt`
2. Download cached data:
   `python run.py --download ES NQ YM RTY CL GC YM SI`
3. Validate cache:
   `python run.py --validate-data`
   `python run.py --validate-data YM --tf 5m`
4. Run a backtest:
   `python run.py --backtest --strategy sma_pullback --symbol ES --tf 1h`
5. Open the dashboard:
   `python run.py --dashboard`

## Available Strategies

Current examples:

- `sma_pullback` for trend-following pullback entries
- `ict_ob` with alias `ict_order_block`
- additional strategies are discoverable through `python run.py --help`

## Known Symbols

Preloaded in `instrument_specs`; unknown symbols fall back to generic defaults.

- `ES`, `NQ`, `YM`, `RTY` for US equity index futures
- `CL`, `NG` for energy
- `GC`, `SI`, `PL` for metals (`PL` data quality may be poor)
- `ZC`, `ZB` for grains and bonds
- `6E` for CME FX futures

## Commands

### Data Download

```bash
python run.py --download 6E 6B 6J 6A 6C 6S
python run.py --validate-data
python run.py --validate-data YM
python run.py --validate-data YM --tf 5m
```

### Single Backtest

```bash
python run.py --backtest --strategy channel_breakout --symbol 6A --tf 30m --dashboard
python run.py --backtest --strategy rfp_fractal --symbol NQ --tf 1h --dashboard
```

### Walk-Forward Optimization

```bash
python run.py --wfo --strategy three_bar_mr --symbol YM --tf 1h
```

### Portfolio Backtest

```bash
python run.py --portfolio-backtest --dashboard
python run.py --portfolio-backtest --portfolio-config path/to/config.yaml
```

### Batch Runs

```bash
python run.py batch --strategies diamond_breakout --symbol ES NQ YM RTY CL NG GC SI 6E 6B 6J 6A 6C 6S --tf 5m 30m 1h
python run.py batch --strategies kc_breakout --symbol ES NQ YM --tf 1m 5m
```

### WFO Batch

```bash
python run.py wfo-batch --strategies sma_pullback ict_ob --symbol ES --tf 1h
python run.py wfo-batch --strategies sma_pullback --symbol ES NQ CL GC YM RTY --tf 1h
```

### Terminal Dashboard

```bash
python run.py --dashboard
python run.py --dashboard --dashboard-port 8080
```

## Notes

- Strategy IDs and aliases are interchangeable on the CLI.
- `--tf` uses the same timeframe labels as cache filenames, such as `30m`, `1h`, `4h`, and `1D`.
- `batch` and `wfo-batch` accept multiple `--symbol` and `--tf` values.
- `--workers N` overrides the process-pool size for batch modes.
- Settings can be overridden with `QUANT_BACKTEST_` environment variables or a local `.env` file.
