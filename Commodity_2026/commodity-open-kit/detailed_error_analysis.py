"""Detailed Error Analysis for Task 1 Holdout.

Evaluates signals, inspects per-commodity breakdowns, analyzes incorrect signals
and missed swings (especially top 10 large swings).
"""

import os
import sys
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
from swing_dataset.scoring2 import (
    ASSETS,
    THRESHOLD_PCT,
    TOP_SWINGS,
    TOP_SWINGS_REQUIRED,
    outcomes as O,
    score as S,
    signals as SIG,
    swings as SW,
)
from swing_dataset.scoring2.prices import PriceSeries

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"

def load_environment():
    prices = PriceSeries.from_paths(
        PACK_BASE / "mcx" / "mcx_daily.parquet",
        PACK_BASE / "labels" / "designations.parquet",
        PACK_BASE / "mcx" / "mcx_sessions.parquet",
    )
    inventory = pd.read_parquet(PACK_BASE / "swings" / "swings.parquet")
    window = S.resolve_window("holdout", prices)
    return prices, inventory, window

def run_analysis(sig_path: Path, nosig_path: Path):
    prices, inventory, window = load_environment()
    decisions = SIG.load(sig_path, challenge=1, no_signal_path=nosig_path)
    
    # Run scoring
    res = S.challenge1(decisions, inventory, prices, window, n_boot=500, seed=42)
    cov = S.coverage(prices, decisions, window)
    
    scores = res["scores"]
    counts = res["counts"]
    graded = res["graded"]
    caught = res["caught"]
    top10 = res["ten_large_swings"]
    
    print("=" * 80)
    print("                      OVERALL HOLDOUT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Signals issued: {counts['signals_in_window']} | Decided: {counts['decided']} | Undecided: {counts['undecided']}")
    print(f"Correctness:    {scores['correctness']['value']:.2f}% (Gate >= 60.0%) - {'PASS' if scores['correctness']['value'] >= 60 else 'FAIL'}")
    print(f"Depth:          {scores['depth']['value']:.2f}% (Gate >= 60.0%) - {'PASS' if scores['depth']['value'] >= 60 else 'FAIL'}")
    print(f"Catch Rate:     {scores['catch_rate']['value']:.2f}% (All swings: {counts['swings_caught']}/{counts['swings_in_window']})")
    print(f"Success Rate:   {scores['success_rate']['value']:.2f}% (Gate >= 70.0%) - {'PASS' if scores['success_rate']['value'] >= 70 else 'FAIL'}")
    print(f"Top 10 Caught:  {top10['caught']}/{top10['required']} (Gate >= 7/10) - {'PASS' if top10['caught'] >= 7 else 'FAIL'}")
    print(f"Coverage Floor: {cov['pooled']['share']:.2f}% (Gate >= 90.0%) - {'PASS' if cov['met'] else 'FAIL'}")
    
    # ---------------------------------------------------------
    # PER ASSET BREAKDOWN
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("                      PER-COMMODITY PERFORMANCE BREAKDOWN")
    print("=" * 80)
    print(f"{'Asset':<10} | {'Signals':<8} | {'Decided':<8} | {'Correct':<8} | {'Correct%':<10} | {'Deep':<6} | {'Deep%':<8} | {'Swings':<8} | {'Caught':<8} | {'Catch%':<8}")
    print("-" * 90)
    
    for a in ASSETS:
        g_a = graded[graded["asset"] == a]
        dec_a = g_a[g_a["decided"].fillna(False).astype(bool)]
        corr_a = dec_a[dec_a["correct"].fillna(False).astype(bool)]
        deep_a = corr_a[corr_a["deep"].fillna(False).astype(bool)]
        
        sw_a = caught[caught["asset"] == a]
        sw_caught_a = sw_a[sw_a["caught"].fillna(False).astype(bool)]
        
        corr_pct = (len(corr_a) / len(dec_a) * 100.0) if len(dec_a) > 0 else 0.0
        deep_pct = (len(deep_a) / len(corr_a) * 100.0) if len(corr_a) > 0 else 0.0
        catch_pct = (len(sw_caught_a) / len(sw_a) * 100.0) if len(sw_a) > 0 else 0.0
        
        print(f"{a:<10} | {len(g_a):<8d} | {len(dec_a):<8d} | {len(corr_a):<8d} | {corr_pct:<9.2f}% | {len(deep_a):<6d} | {deep_pct:<7.2f}% | {len(sw_a):<8d} | {len(sw_caught_a):<8d} | {catch_pct:<7.2f}%")

    # ---------------------------------------------------------
    # TOP 10 LARGE SWINGS DETAILED INSPECTION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("                      TOP 10 LARGE SWINGS INSPECTION")
    print("=" * 80)
    for sw in top10["swings"]:
        status = "CAUGHT" if sw["caught"] else "MISSED"
        by = f"by {sw['caught_by_signal_utc']}" if sw["caught"] else "NO VALID SIGNAL"
        print(f"Rank {sw['rank']:2d}: {sw['asset'].upper():<6} {sw['direction'].upper():<4} | Move: {sw['move_pct']:+6.2f}% ({sw['threshold_multiple']:.1f}x) | Turn: {sw['turn_utc']} -> Extreme: {sw['extreme_utc']} | Status: {status} ({by})")
        
        # If missed, find what happened: did we issue any signal in this window?
        if not sw["caught"]:
            asset_signals = graded[
                (graded["asset"] == sw["asset"])
                & (graded["signal_utc"] >= sw["turn_utc"])
                & (graded["signal_utc"] <= sw["extreme_utc"])
            ]
            if len(asset_signals) == 0:
                print(f"         REASON: Model ABSTAINED (no signals issued between turn and extreme)")
            else:
                for _, srow in asset_signals.iterrows():
                    print(f"         CANDIDATE SIGNAL: {srow['signal_utc']} dir={srow['direction']} correct={srow['correct']} p0={srow['p0']}")

    # ---------------------------------------------------------
    # ERROR ANALYSIS: INCORRECT DECIDED SIGNALS
    # ---------------------------------------------------------
    incorrect = graded[
        graded["decided"].fillna(False).astype(bool)
        & (~graded["correct"].fillna(False).astype(bool))
    ]
    print("\n" + "=" * 80)
    print(f"        ERROR ANALYSIS: {len(incorrect)} INCORRECT SIGNALS OUT OF {counts['decided']} DECIDED")
    print("=" * 80)
    
    # Feature inspection for incorrect signals
    # Load holdout feature datasets to join
    all_ho_feats = []
    for a in ASSETS:
        df_a = pd.read_parquet(DATA_DIR / f"task1_classifier_{a}.parquet")
        df_a["cutoff_utc"] = pd.to_datetime(df_a["cutoff_utc"], utc=True)
        ho_a = df_a[df_a["cutoff_utc"] >= pd.Timestamp("2026-06-01", tz="UTC")].copy()
        all_ho_feats.append(ho_a)
    feats_df = pd.concat(all_ho_feats, ignore_index=True)
    
    merged_inc = pd.merge(
        incorrect,
        feats_df,
        left_on=["asset", "signal_utc"],
        right_on=["asset", "cutoff_utc"],
        how="left"
    )
    
    print("\nIncorrect Signals by Asset:")
    print(merged_inc["asset"].value_counts())
    
    print("\nIncorrect Signals by Direction:")
    print(merged_inc.groupby(["asset", "direction"]).size())
    
    # Analyze MAE and MFE on incorrect signals
    print("\nMFE vs MAE on Incorrect Signals:")
    for a in ASSETS:
        sub = merged_inc[merged_inc["asset"] == a]
        if len(sub) > 0:
            print(f"{a:<10} | Avg MFE (favorable move before stop): {sub['mfe_pct'].mean():.2f}% | Avg MAE (adverse move): {sub['mae_pct'].mean():.2f}%")
            
    # Check if incorrect signals are caught in choppy reversals
    print("\nTop Indicators Correlated with Errors:")
    numeric_cols = [
        "ret_1", "ret_5", "ret_20", "rv_10", "rv_20", "atr_ratio",
        "range_expansion_ratio", "vol_ratio_5d", "bb_pos", "rsi_14", "fng_index",
        "term_spread_active_next", "cot_net_pct", "days_to_expiry"
    ]
    available_cols = [c for c in numeric_cols if c in merged_inc.columns]
    
    correct_df = graded[
        graded["decided"].fillna(False).astype(bool)
        & (graded["correct"].fillna(False).astype(bool))
    ]
    merged_corr = pd.merge(
        correct_df,
        feats_df,
        left_on=["asset", "signal_utc"],
        right_on=["asset", "cutoff_utc"],
        how="left"
    )
    
    print(f"{'Feature':<25} | {'Incorrect Mean':<16} | {'Correct Mean':<16} | {'Difference':<12}")
    print("-" * 75)
    for c in available_cols:
        inc_mean = merged_inc[c].dropna().mean()
        cor_mean = merged_corr[c].dropna().mean()
        diff = inc_mean - cor_mean
        print(f"{c:<25} | {inc_mean:<16.4f} | {cor_mean:<16.4f} | {diff:<+12.4f}")

if __name__ == "__main__":
    sig = RESULTS_DIR / "task1_vectorized_opt_signals.parquet"
    nosig = RESULTS_DIR / "task1_vectorized_opt_no_signals.parquet"
    if not sig.exists():
        sig = RESULTS_DIR / "task1_lgbm_tuned_signals.parquet"
        nosig = RESULTS_DIR / "task1_lgbm_tuned_no_signals.parquet"
    run_analysis(sig, nosig)
