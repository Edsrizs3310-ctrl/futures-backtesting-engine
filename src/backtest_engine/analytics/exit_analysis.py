"""
src/backtest_engine/analytics/exit_analysis.py

Data enrichment layer for Exit Analysis.
Computes MFE, MAE, Holding Time, and PnL Decay by slicing OHLCV data.
This runs once at the end of the backtest to offload dashboard computation.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def enrich_trades_with_exit_analytics(
    trades_df: pd.DataFrame, 
    data_map: Dict[Any, pd.DataFrame],
) -> pd.DataFrame:
    """
    Enriches a trades DataFrame with exit analytics:
      - holding_time (Timedelta)
      - mfe (Maximum Favorable Excursion, in $)
      - mae (Maximum Adverse Excursion, in $)
      - pnl_decay_5m, 15m, 30m, 60m (Hypothetical PnL if exited at T+N)
      - entry_volatility (14-period standard deviation of returns at entry)

    Args:
        trades_df: Basic trades dataframe containing entry_time, exit_time, symbol, direction, entry_price.
        data_map: In single mode, dict of {symbol: df}. 
                  In portfolio mode, dict of {(slot_id, symbol): df}.
                  
    Returns:
        Enriched DataFrame.
    """
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()
    multiplier_cache: Dict[str, float] = {}

    try:
        from src.backtest_engine.settings import get_settings

        settings = get_settings()
    except Exception:
        settings = None
    
    # Pre-allocate columns with proper dtypes
    df["holding_time"] = pd.Series(dtype='timedelta64[ns]')
    df["mfe"] = np.nan
    df["mae"] = np.nan
    
    horizons = [5, 15, 30, 60, 120, 240, 480, 720, 1440]
    for h in horizons:
        df[f"pnl_decay_{h}m"] = np.nan
    df["entry_volatility"] = np.nan

    for idx, row in df.iterrows():
        entry = row.get("entry_time")
        exit_ = row.get("exit_time")
        symbol = row.get("symbol")
        direction = row.get("direction", "LONG")
        sign = 1.0 if direction == "LONG" else -1.0
        entry_price = row.get("entry_price")
        qty_raw = row.get("quantity", 1.0)
        try:
            qty = abs(float(qty_raw))
        except Exception:
            qty = 1.0
        if pd.isna(qty) or qty <= 0:
            qty = 1.0

        multiplier = multiplier_cache.get(symbol, None)
        if multiplier is None:
            multiplier = 1.0
            if settings is not None:
                try:
                    spec = settings.get_instrument_spec(symbol)
                    multiplier = float(spec.get("multiplier", 1.0))
                except Exception:
                    multiplier = 1.0
            multiplier_cache[symbol] = multiplier
        
        # Determine data map key. Portfolio uses (slot_id, symbol).
        slot_id = row.get("slot_id", None)
        
        df_sym = data_map.get((slot_id, symbol)) if slot_id is not None else None
        if df_sym is None or df_sym.empty:
            df_sym = data_map.get(symbol)
                
        if df_sym is None or df_sym.empty or pd.isna(entry) or pd.isna(exit_) or pd.isna(entry_price):
            continue
            
        df.at[idx, "holding_time"] = exit_ - entry
        
        # Round trip costs to deduct from hypothetical PnLs
        comm = row.get("commission", 0.0)
        slip = row.get("slippage", 0.0)
        costs = (0.0 if pd.isna(comm) else float(comm)) + (0.0 if pd.isna(slip) else float(slip))
        
        # MFE / MAE
        try:
            trade_bars = df_sym.loc[entry:exit_]
            if not trade_bars.empty:
                max_p = trade_bars["high"].max()
                min_p = trade_bars["low"].min()
                
                if direction == "LONG":
                    mfe = (max_p - entry_price) * qty * multiplier
                    mae = (min_p - entry_price) * qty * multiplier
                else:
                    mfe = (entry_price - min_p) * qty * multiplier
                    mae = (entry_price - max_p) * qty * multiplier
                    
                df.at[idx, "mfe"] = float(mfe if mfe > 0 else 0.0)
                df.at[idx, "mae"] = float(mae if mae < 0 else 0.0)
        except Exception:
            pass
            
        # Entry Volatility (14-period C2C on the underlying)
        try:
            locs = df_sym.index.get_indexer([entry], method="pad")
            if len(locs) > 0 and locs[0] >= 14:
                entry_idx = locs[0]
                window = df_sym.iloc[entry_idx - 14 : entry_idx + 1]
                vol = window["close"].pct_change().std()
                df.at[idx, "entry_volatility"] = float(vol)
        except Exception:
            pass

        # PnL Decay (Forward PnL)
        for minutes in horizons:
            target_time = entry + pd.Timedelta(minutes=minutes)
            col_name = f"pnl_decay_{minutes}m"
            
            try:
                locs = df_sym.index.get_indexer([target_time], method="pad")
                if len(locs) > 0 and locs[0] >= 0:
                    hypo_price = df_sym.iloc[locs[0]]["close"]
                    hypo_gross = sign * (hypo_price - entry_price) * qty * multiplier
                    df.at[idx, col_name] = float(hypo_gross - costs)
            except Exception:
                pass
            
    return df
