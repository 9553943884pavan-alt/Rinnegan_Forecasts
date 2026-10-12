"""Measure and Deeply Analyze Catch Rate for Task 1 Models.

Calculates:
1. Overall Catch Rate (Pooled & Per-Asset)
2. Directional Catch Rate (Up-swings vs Down-swings)
3. Large-Swing Catch Rate (Move >= 2x Threshold)
4. Ten Largest Realized Swings Gate (Holdout Award Condition)
5. Root-Cause Analysis of Uncaught Swings
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add kit to python path
KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd

from swing_dataset.scoring2 import ASSETS, THRESHOLD_PCT, CATCH_ROOM_SHARE
from swing_dataset.scoring2.prices import PriceSeries
from swing_dataset.scoring2.signals import load as load_signals
from swing_dataset.scoring2.swings import load as load_swings, in_window as swings_in_window, is_large
from swing_dataset.scoring2 import outcomes as O
from swing_dataset.scoring2 import score as S

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")

MCX_DAILY = PACK_BASE / "mcx" / "mcx_daily.parquet"
DESIGNATIONS = PACK_BASE / "labels" / "designations.parquet"
MCX_SESSIONS = PACK_BASE / "mcx" / "mcx_sessions.parquet"
SWINGS = PACK_BASE / "swings" / "swings.parquet"


def analyze_model_catch(model_name: str, sig_path: Path, nosig_path: Path, prices: PriceSeries, window: S.Window, inventory: pd.DataFrame):
    print("\n" + "=" * 75)
    print(f"CATCH RATE ANALYSIS: {model_name}")
    print("=" * 75)

    decisions = load_signals(sig_path, 1, nosig_path)
    signals = decisions.signals
    signals = signals[S.in_window(signals, "signal_utc", window.start, window.end)]
    scored_rows, _ = S.split_horizons(signals, window, 1)

    graded = O.grade_c1(scored_rows, prices)
    swings_in = swings_in_window(inventory, window.start, window.end)
    caught = O.catch(swings_in, graded)

    total_swings = len(caught)
    total_caught = int(caught["caught"].sum())
    overall_catch_rate = (total_caught / total_swings) * 100.0

    print(f"\n1. POOLED CATCH RATE:")
    print(f"   Total Realized Swings in Holdout Window: {total_swings}")
    print(f"   Swings Caught:                           {total_caught}")
    print(f"   Catch Rate:                              {overall_catch_rate:.2f}%")

    # Per-Asset Breakdown
    print(f"\n2. PER-ASSET CATCH RATE BREAKDOWN:")
    print(f"   {'Asset':<10} | {'Total Swings':<12} | {'Swings Caught':<13} | {'Catch Rate':<10}")
    print("   " + "-" * 55)
    for asset in ASSETS:
        sub = caught[caught["asset"] == asset]
        n_a = len(sub)
        c_a = int(sub["caught"].sum())
        rate_a = (c_a / n_a * 100.0) if n_a > 0 else 0.0
        print(f"   {asset.capitalize():<10} | {n_a:<12} | {c_a:<13} | {rate_a:.2f}%")

    # Per-Direction Breakdown
    print(f"\n3. DIRECTIONAL CATCH RATE (UP vs DOWN):")
    print(f"   {'Direction':<10} | {'Total Swings':<12} | {'Swings Caught':<13} | {'Catch Rate':<10}")
    print("   " + "-" * 55)
    for direction in ["up", "down"]:
        sub = caught[caught["direction"] == direction]
        n_d = len(sub)
        c_d = int(sub["caught"].sum())
        rate_d = (c_d / n_d * 100.0) if n_d > 0 else 0.0
        print(f"   {direction.upper():<10} | {n_d:<12} | {c_d:<13} | {rate_d:.2f}%")

    # Large Swings Breakdown (move >= 2x threshold)
    large_mask = is_large(caught)
    large_swings = caught[large_mask].copy()
    n_large = len(large_swings)
    c_large = int(large_swings["caught"].sum())
    rate_large = (c_large / n_large * 100.0) if n_large > 0 else 0.0

    print(f"\n4. LARGE SWINGS CATCH RATE (Move >= 2x Threshold):")
    print(f"   Total Large Realized Swings: {n_large}")
    print(f"   Large Swings Caught:         {c_large}")
    print(f"   Large Swing Catch Rate:      {rate_large:.2f}%")
    print(f"\n   Per-Asset Large Swing Catch Rate:")
    for asset in ASSETS:
        sub_l = large_swings[large_swings["asset"] == asset]
        n_al = len(sub_l)
        c_al = int(sub_l["caught"].sum())
        rate_al = (c_al / n_al * 100.0) if n_al > 0 else 0.0
        print(f"   - {asset.capitalize():<8}: {c_al}/{n_al} caught ({rate_al:.2f}%)")

    # Ten Largest Realized Swings Gate
    ten = S.ten_large_swings(caught)
    print(f"\n5. TEN LARGEST REALIZED SWINGS GATE (Section 3 Requirement >= 7 caught):")
    print(f"   Swings Caught: {ten['caught']} / {ten['required']} required (Met: {ten['caught'] >= ten['required']})")
    print(f"\n   {'Rank':<4} | {'Asset':<7} | {'Dir':<4} | {'Move %':<8} | {'Mult':<6} | {'Caught':<6} | {'Turn UTC':<20} | {'Extreme UTC':<20}")
    print("   " + "-" * 88)
    for s in ten["swings"]:
        c_str = "YES" if s["caught"] else "NO"
        turn_s = str(s["turn_utc"])[:19]
        ext_s = str(s["extreme_utc"])[:19]
        print(f"   {s['rank']:<4} | {s['asset']:<7} | {s['direction']:<4} | {s['move_pct']:>7.2f}% | {s['threshold_multiple']:>5.2f}x | {c_str:<6} | {turn_s:<20} | {ext_s:<20}")

    # Root Cause of Missed Swings
    uncaught = caught[~caught["caught"]].copy()
    print(f"\n6. ROOT CAUSE ANALYSIS OF UNCAUGHT SWINGS ({len(uncaught)} uncaught):")
    
    # Check reasons why each uncaught swing was not caught:
    # A swing is caught if:
    # 1. Correct signal in swing direction
    # 2. signal_utc > turn_utc and signal_utc <= extreme_utc
    # 3. ahead >= 0.5 * threshold
    
    reasons = {
        "no_signal_issued_during_swing": 0,
        "opposite_direction_signalled": 0,
        "signal_incorrect_or_invalidated": 0,
        "insufficient_room_ahead (<0.5x thresh)": 0,
    }
    
    for _, swing in uncaught.iterrows():
        # Candidate signals in window between turn and extreme
        cand = graded[
            (graded["asset"] == swing["asset"])
            & (graded["signal_utc"] > swing["turn_utc"])
            & (graded["signal_utc"] <= swing["extreme_utc"])
        ]
        if cand.empty:
            reasons["no_signal_issued_during_swing"] += 1
        else:
            same_dir = cand[cand["direction"] == swing["direction"]]
            if same_dir.empty:
                reasons["opposite_direction_signalled"] += 1
            else:
                correct_same_dir = same_dir[same_dir["correct"].fillna(False).astype(bool)]
                if correct_same_dir.empty:
                    reasons["signal_incorrect_or_invalidated"] += 1
                else:
                    reasons["insufficient_room_ahead (<0.5x thresh)"] += 1

    for r, count in reasons.items():
        pct = (count / len(uncaught) * 100.0) if len(uncaught) > 0 else 0.0
        print(f"   - {r:<42}: {count:>3} ({pct:.1f}%)")

    return {
        "model_name": model_name,
        "total_swings": total_swings,
        "total_caught": total_caught,
        "catch_rate": overall_catch_rate,
        "large_swings": n_large,
        "large_caught": c_large,
        "large_catch_rate": rate_large,
        "ten_caught": ten["caught"],
    }


def main():
    print("Loading price series and swing inventory...")
    prices = PriceSeries.from_paths(MCX_DAILY, DESIGNATIONS, MCX_SESSIONS, None)
    window = S.resolve_window("holdout", prices, None, None)
    inventory = load_swings(SWINGS)

    baseline_res = analyze_model_catch(
        "Baseline Model (16 Features)",
        DATA_DIR / "task1_baseline_signals.parquet",
        DATA_DIR / "task1_baseline_no_signals.parquet",
        prices, window, inventory
    )

    enriched_res = analyze_model_catch(
        "Enriched Model (27 Features)",
        DATA_DIR / "task1_enriched_signals.parquet",
        DATA_DIR / "task1_enriched_no_signals.parquet",
        prices, window, inventory
    )

    print("\n" + "=" * 75)
    print("COMPARATIVE SUMMARY: BASELINE vs ENRICHED")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Baseline (16 Feat)':<20} | {'Enriched (27 Feat)':<20}")
    print("-" * 75)
    print(f"{'Overall Catch Rate':<30} | {baseline_res['catch_rate']:>6.2f}% ({baseline_res['total_caught']}/{baseline_res['total_swings']})       | {enriched_res['catch_rate']:>6.2f}% ({enriched_res['total_caught']}/{enriched_res['total_swings']})")
    print(f"{'Large Swing Catch Rate':<30} | {baseline_res['large_catch_rate']:>6.2f}% ({baseline_res['large_caught']}/{baseline_res['large_swings']})       | {enriched_res['large_catch_rate']:>6.2f}% ({enriched_res['large_caught']}/{enriched_res['large_swings']})")
    print(f"{'Ten Largest Swings Caught':<30} | {baseline_res['ten_caught']:>2}/10                  | {enriched_res['ten_caught']:>2}/10")


if __name__ == "__main__":
    main()
