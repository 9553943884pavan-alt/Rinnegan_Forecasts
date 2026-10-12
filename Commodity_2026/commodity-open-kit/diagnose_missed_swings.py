"""Diagnose Missed Swings in Task 1 Holdout."""

import sys
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
from swing_dataset.scoring2 import signals as SIG, score as S, swings as SW
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

sig_f = RESULTS_DIR / "task1_vectorized_opt_signals.parquet"
nosig_f = RESULTS_DIR / "task1_vectorized_opt_no_signals.parquet"

decisions = SIG.load(sig_f, challenge=1, no_signal_path=nosig_f)
res = S.challenge1(decisions, inventory, prices, window, n_boot=10, seed=42)

caught = res["caught"]
graded = res["graded"]

missed = caught[~caught["caught"].fillna(False).astype(bool)].reset_index(drop=True)
print("=" * 80)
print(f"DIAGNOSING {len(missed)} MISSED SWINGS OUT OF {len(caught)} TOTAL SWINGS")
print("=" * 80)

reasons = {
    "no_signals_in_window": 0,
    "candidate_signal_incorrect": 0,
    "wrong_direction_only": 0,
    "not_enough_room_ahead": 0,
    "swing_duration_too_short": 0,
}

durations_missed = []
moves_missed = []

for _, sw in missed.iterrows():
    asset = sw["asset"]
    dirn = sw["direction"]
    t_turn = sw["turn_utc"]
    t_ext = sw["extreme_utc"]
    thresh = sw["threshold_pct"]
    
    # Calculate duration in hours
    dur = (pd.Timestamp(t_ext) - pd.Timestamp(t_turn)).total_seconds() / 3600.0
    durations_missed.append(dur)
    moves_missed.append(sw["move_pct"])
    
    # Find signals in this window
    sig_window = graded[
        (graded["asset"] == asset)
        & (graded["signal_utc"] > t_turn)
        & (graded["signal_utc"] <= t_ext)
    ]
    
    if len(sig_window) == 0:
        reasons["no_signals_in_window"] += 1
        if dur < 24.0:
            reasons["swing_duration_too_short"] += 1
    else:
        sig_dir = sig_window[sig_window["direction"] == dirn]
        if len(sig_dir) == 0:
            reasons["wrong_direction_only"] += 1
        else:
            sig_corr = sig_dir[sig_dir["correct"] == True]
            if len(sig_corr) == 0:
                reasons["candidate_signal_incorrect"] += 1
            else:
                reasons["not_enough_room_ahead"] += 1

print("\nBreakdown of Reasons for Missed Swings:")
for k, v in reasons.items():
    print(f"  {k:<32}: {v:3d} ({v/len(missed)*100:.1f}%)")

print(f"\nMedian duration of missed swings: {np.median(durations_missed):.1f} hours ({np.median(durations_missed)/24:.1f} days)")
print(f"Share of missed swings under 24 hours: {(np.array(durations_missed) < 24.0).mean()*100:.1f}%")
print(f"Share of missed swings under 48 hours: {(np.array(durations_missed) < 48.0).mean()*100:.1f}%")
print(f"Median move of missed swings: {np.median(np.abs(moves_missed)):.2f}%")
