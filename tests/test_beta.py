import pandas as pd
import numpy as np
from scipy import stats
from src.backtest_engine.analytics.dashboard.core.data_layer import load_result_bundle

def debug_beta():
    bundle = load_result_bundle()
    history = bundle.history
    instrument_closes = bundle.instrument_closes
    trades = bundle.trades
    initial_cap = float(history["total_value"].iloc[0])

    print(f"Initial Cap: {initial_cap}")

    for strat_name in trades["strategy"].unique():
        sub = trades[trades["strategy"] == strat_name]
        symbols = sub["symbol"].value_counts().index.tolist()
        if not symbols: continue
        target_sym = symbols[0]
        
        # Find the slot_id
        str_id = None
        for k, v in bundle.slots.items():
            if v == strat_name:
                str_id = str(k)
                break
                
        if not str_id: continue
        
        slot_weights = getattr(bundle, "slot_weights", {})
        strat_weight = float(slot_weights.get(str_id, 1.0))
        strat_initial_cap = initial_cap * strat_weight
        print(f"Strat {str_id} initial cap = {strat_initial_cap}")
        
        strat_pnl_col = f"slot_{str_id}_pnl"
        if strat_pnl_col in history.columns and target_sym in instrument_closes.columns:
            strat_daily_pnl = history[strat_pnl_col].diff().fillna(0.0).resample("1D").sum()
            strat_rets = strat_daily_pnl / strat_initial_cap
            
            inst_close = instrument_closes[target_sym]
            inst_rets = inst_close.pct_change(fill_method=None).fillna(0.0)

            aligned = pd.concat([strat_rets, inst_rets], axis=1, join='inner').dropna()
            aligned.columns = ["strat_ret", "inst_ret"]
            aligned = aligned[aligned["inst_ret"] != 0.0]
            
            if len(aligned) > 2:
                y = aligned["strat_ret"].values
                x = aligned["inst_ret"].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                print(f"--- {strat_name} on {target_sym} ---")
                print(f"  N: {len(x)}")
                print(f"  Beta (slope): {slope:.6f}")
                print(f"  Beta Sig (p-val): {p_value:.6f}")
                print(f"  Alpha (Ann): {intercept * 252 * 100:.6f}%")
                
                n_ann = len(x)
                std_err_int = std_err * np.sqrt(np.mean(x**2) / np.var(x)) if np.var(x) > 0 else float('inf')
                t_alpha = intercept / std_err_int
                alpha_pval = stats.t.sf(np.abs(t_alpha), n_ann - 2) * 2
                print(f"  Alpha Sig: {alpha_pval:.6f}")
                
                print(f"  Avg Strat Ret: {np.mean(y):.6f}")
                print(f"  Avg Inst Ret: {np.mean(x):.6f}")
                print(f"  Var Inst Ret: {np.var(x):.6f}")

debug_beta()
