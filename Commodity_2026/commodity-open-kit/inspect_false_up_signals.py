"""Inspect false UP signals in Gold and Silver."""

import sys
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
from swing_dataset.scoring2 import signals as SIG, score as S
from swing_dataset.scoring2.prices import PriceSeries

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"

prices = PriceSeries.from_paths(
    PACK_BASE / "mcx" / "mcx_daily.parquet",
    PACK_BASE / "labels" / "designations.parquet",
    PACK_BASE / "mcx" / "mcx_sessions.parquet",
)
inventory = pd.read_parquet(PACK_BASE / "swings" / "swings.parquet")
window = S.resolve_window("holdout", prices)

sig_path = RESULTS_DIR / "test_corrected_sig_0.38_0.00.parquet"
nosig_path = RESULTS_DIR / "test_corrected_nosig_0.38_0.00.parquet"

decisions = SIG.load(sig_path, challenge=1, no_signal_path=nosig_path)
res = S.challenge1(decisions, inventory, prices, window, n_boot=100, seed=42)
graded = res["graded"]

for asset in ["gold", "silver"]:
    df = pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet")
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    
    g_a = graded[(graded["asset"] == asset) & (graded["direction"] == "up") & (graded["decided"] == True)]
    merged = pd.merge(g_a, df, left_on=["asset", "signal_utc"], right_on=["asset", "cutoff_utc"], how="left")
    
    corr = merged[merged["correct"] == True]
    inc = merged[merged["correct"] == False]
    
    print("=" * 80)
    print(f"ASSET: {asset.upper()} - UP SIGNALS: {len(corr)} Correct vs {len(inc)} Incorrect")
    print("=" * 80)
    
    print("\nDates of Incorrect UP signals (Sample of 10):")
    cols_to_print = [c for c in ["signal_utc", "p0", "mfe_pct", "mae_pct", "ret_1", "ret_5", "ret_20", "range_expansion_ratio"] if c in inc.columns]
    print(inc[cols_to_print].head(10))
    
    print("\nMean Feature Comparison (Correct UP vs Incorrect UP):")
    feats = ["ret_1", "ret_5", "ret_20", "rv_10", "rv_20", "range_expansion_ratio", "vol_surge_10d", "close_pos_in_range", "rsi_14", "cot_net_smart_money"]
    for f in feats:
        if f in merged.columns:
            c_val = corr[f].mean()
            i_val = inc[f].mean()
            print(f"  {f:<25} | Correct UP: {c_val:+.4f} | Incorrect UP: {i_val:+.4f} | Diff: {i_val - c_val:+.4f}")
