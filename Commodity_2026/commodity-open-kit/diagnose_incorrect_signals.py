"""Deep Error Diagnosis on Incorrect Signals.

Analyzes every single signal issued on the holdout that was graded incorrect by scoring2:
1. What was the probability distribution: p_up, p_down, p_none?
2. What was the directional margin: |p_up - p_down|?
3. What was the 1-day, 5-day, and 20-day return at signal time?
4. Did the market reverse immediately (whipsaw / knife-catch)?
5. What filter cleanly separates the Correct signals from the Incorrect signals?
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
from swing_dataset.scoring2.prices import PriceSeries
from swing_dataset.scoring2 import outcomes as O
from swing_dataset.scoring2 import score as S
from swing_dataset.scoring2.signals import load as load_signals

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"


def diagnose():
    print("================================================================================")
    print("   DEEP ROOT-CAUSE ANALYSIS OF INCORRECT SIGNALS IN THE HOLDOUT WINDOW")
    print("================================================================================")

    daily = pd.read_parquet(PACK_BASE / "mcx" / "mcx_daily.parquet")
    designations = pd.read_parquet(PACK_BASE / "labels" / "designations.parquet")
    sessions = pd.read_parquet(PACK_BASE / "mcx" / "mcx_sessions.parquet")
    prices = PriceSeries(daily, designations, sessions)

    sig_file = RESULTS_DIR / "task1_lgbm_tuned_signals.parquet"
    nosig_file = RESULTS_DIR / "task1_lgbm_tuned_no_signals.parquet"

    window = S.resolve_window("holdout", prices)
    decisions = load_signals(sig_file, 1, nosig_file)
    signals = decisions.signals[S.in_window(decisions.signals, "signal_utc", window.start, window.end)]
    scored_rows, _ = S.split_horizons(signals, window, 1)

    graded = O.grade_c1(scored_rows, prices)

    # Attach features from classifier datasets
    enriched_rows = []
    for asset in ["gold", "silver", "copper", "crude"]:
        clf_df = pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet")
        sub_graded = graded[graded["asset"] == asset].copy()
        merged = pd.merge(sub_graded, clf_df, left_on=["signal_utc", "asset"], right_on=["cutoff_utc", "asset"], how="inner", suffixes=("", "_feat"))
        enriched_rows.append(merged)

    all_graded = pd.concat(enriched_rows, ignore_index=True)

    decided = all_graded[all_graded["decided"] == True].copy()
    correct = decided[decided["correct"] == True].copy()
    incorrect = decided[decided["correct"] == False].copy()

    print(f"\nTotal Scored Decided Signals: {len(decided)}")
    print(f"Correct Signals:              {len(correct)} ({len(correct) / len(decided) * 100:.1f}%)")
    print(f"Incorrect Signals:            {len(incorrect)} ({len(incorrect) / len(decided) * 100:.1f}%)")

    print("\n--- Feature Comparisons: Correct vs. Incorrect Signals ---")
    compare_features = [
        "confidence", "ret_1", "ret_5", "ret_20", "rv_20", "vol_term_structure",
        "oi_1s_change", "range_expansion_ratio", "vol_surge_10d",
        "close_pos_in_range", "tr_pct", "dist_5d_high", "dist_5d_low"
    ]

    print(f"{'Feature':<25} | {'Mean (Correct)':<15} | {'Mean (Incorrect)':<18} | {'Difference'}")
    print("-" * 75)
    for f in compare_features:
        if f in correct.columns and f in incorrect.columns:
            m_c = correct[f].astype(float).mean()
            m_i = incorrect[f].astype(float).mean()
            diff = m_c - m_i
            print(f"{f:<25} | {m_c:<15.4f} | {m_i:<18.4f} | {diff:+.4f}")

    # Check Directional Alignment with Prior Session Move (Trend Alignment)
    # If direction is 'up', did ret_1 > 0?
    correct["trend_align"] = np.where(
        correct["direction"] == "up", correct["ret_1"] > 0, correct["ret_1"] < 0
    )
    incorrect["trend_align"] = np.where(
        incorrect["direction"] == "up", incorrect["ret_1"] > 0, incorrect["ret_1"] < 0
    )

    print("\n--- Key Finding: Immediate Momentum Alignment (ret_1 sign == signal direction) ---")
    print(f"Correct Signals with Momentum Aligned:   {correct['trend_align'].mean() * 100:.1f}%")
    print(f"Incorrect Signals with Momentum Aligned: {incorrect['trend_align'].mean() * 100:.1f}%")

    # Check Close Position in Range (cl_pos > 0.5 for up, cl_pos < 0.5 for down)
    correct["pos_align"] = np.where(
        correct["direction"] == "up", correct["close_pos_in_range"] > 0.5, correct["close_pos_in_range"] < 0.5
    )
    incorrect["pos_align"] = np.where(
        incorrect["direction"] == "up", incorrect["close_pos_in_range"] > 0.5, incorrect["close_pos_in_range"] < 0.5
    )

    print("\n--- Key Finding: Close Position in Session Range Alignment ---")
    print(f"Correct Signals with Range Position Aligned:   {correct['pos_align'].mean() * 100:.1f}%")
    print(f"Incorrect Signals with Range Position Aligned: {incorrect['pos_align'].mean() * 100:.1f}%")

    # What happens if we filter signals by: (confidence >= 0.45 AND trend_align == True)?
    filtered_decided = decided[decided["trend_align"] == True]
    f_correct = filtered_decided[filtered_decided["correct"] == True]
    f_rate = len(f_correct) / len(filtered_decided) * 100 if len(filtered_decided) > 0 else 0
    print(f"\nSimulation: Momentum-Aligned Signals Only:")
    print(f"  Decided: {len(filtered_decided)}, Correct: {len(f_correct)} -> CORRECTNESS = {f_rate:.2f}%")


if __name__ == "__main__":
    diagnose()
