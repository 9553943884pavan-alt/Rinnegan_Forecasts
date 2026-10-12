"""Analyze confidence distributions and directional filters for Gold, Silver, Crude, and Copper.

Goal:
1. Compare p_up and p_down on Correct vs Incorrect signals.
2. Find optimal asset-specific thresholds (tau_up, tau_down) that push Correctness >= 60%
   while preserving >= 7/10 top large swings.
"""

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

print("=" * 80)
print("     CONFIDENCE DISTRIBUTION: CORRECT VS INCORRECT SIGNALS")
print("=" * 80)

for asset in ["gold", "silver", "copper", "crude"]:
    g_a = graded[graded["asset"] == asset]
    dec = g_a[g_a["decided"].fillna(False).astype(bool)]
    corr = dec[dec["correct"].fillna(False).astype(bool)]
    inc = dec[~dec["correct"].fillna(False).astype(bool)]
    
    print(f"\n--- {asset.upper()} ---")
    print(f"Total Decided: {len(dec)} | Correct: {len(corr)} ({len(corr)/len(dec)*100:.1f}%) | Incorrect: {len(inc)}")
    
    for d in ["up", "down"]:
        corr_d = corr[corr["direction"] == d]
        inc_d = inc[inc["direction"] == d]
        print(f"  Direction: {d.upper()}")
        print(f"    Correct count:   {len(corr_d):2d} | Avg Confidence: {corr_d['confidence'].mean():.4f} | Med: {corr_d['confidence'].median():.4f} | Min: {corr_d['confidence'].min():.4f}")
        if len(inc_d) > 0:
            print(f"    Incorrect count: {len(inc_d):2d} | Avg Confidence: {inc_d['confidence'].mean():.4f} | Med: {inc_d['confidence'].median():.4f} | Min: {inc_d['confidence'].min():.4f}")
        else:
            print(f"    Incorrect count:  0")
