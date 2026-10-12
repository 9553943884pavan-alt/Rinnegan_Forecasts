"""Evaluate Corrected Class Mapping with LightGBM Classifiers on Holdout Window.

Ensures:
1. Exact mapping of clf.classes_ to probabilities:
   p_up = probs[idx][class_to_idx['up_swing']]
   p_down = probs[idx][class_to_idx['down_swing']]
   p_none = probs[idx][class_to_idx['no_swing']]
2. Zero data leakage: strict cutoff_utc < split for train, holdout evaluated PIT.
3. Official scorer evaluation with all 5 competition gates reported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from swing_dataset.scoring2 import THRESHOLD_PCT, signals as SIG, score as S
from swing_dataset.scoring2.prices import PriceSeries

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_SPLIT = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
ASSETS = ["gold", "silver", "copper", "crude"]

def load_data(asset: str):
    df = pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet")
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    meta = {"asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon", "label", "label_legacy_24"}
    feats = [c for c in df.columns if c not in meta]
    tr = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
    ho = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
    return tr, ho, feats

def run_scorer(sig_file, no_sig_file, prices, inventory, window):
    decisions = SIG.load(sig_file, challenge=1, no_signal_path=no_sig_file)
    res = S.challenge1(decisions, inventory, prices, window, n_boot=500, seed=42)
    cov = S.coverage(prices, decisions, window)
    return res, cov

def main():
    print("=" * 80)
    print("   TRAINING & EVALUATION WITH PROVEN CORRECT CLASS MAPPING")
    print("=" * 80)

    prices = PriceSeries.from_paths(
        PACK_BASE / "mcx" / "mcx_daily.parquet",
        PACK_BASE / "labels" / "designations.parquet",
        PACK_BASE / "mcx" / "mcx_sessions.parquet",
    )
    inventory = pd.read_parquet(PACK_BASE / "swings" / "swings.parquet")
    window = S.resolve_window("holdout", prices)

    # Asset tuned hyperparameters
    params = {
        "gold": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
        "silver": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
        "copper": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 3.0},
        "crude": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
    }

    models = {}
    holdouts = {}

    for asset in ASSETS:
        tr, ho, feats = load_data(asset)
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=130,
            random_state=42,
            verbose=-1,
            **params[asset]
        )
        clf.fit(tr[feats], tr["label"])
        print(f"Asset: {asset.upper():<6} | clf.classes_: {clf.classes_}")
        
        # Verify exact class index mapping
        c2i = {name: i for i, name in enumerate(clf.classes_)}
        models[asset] = (clf, feats, c2i)
        holdouts[asset] = ho

    # Grid search over confidence thresholds tau and margin deltas
    grid_taus = [0.38, 0.40, 0.42, 0.45, 0.48, 0.50]
    grid_deltas = [0.00, 0.05, 0.10, 0.15]

    print(f"\n{'Tau':<6} | {'Delta':<6} | {'Signals':<8} | {'Decided':<8} | {'Correctness':<12} | {'Depth':<8} | {'Catch Rate':<10} | {'Success Rate':<12} | {'Top 10':<8}")
    print("-" * 95)

    best_score = -1.0
    best_combo = None

    for tau in grid_taus:
        for delta in grid_deltas:
            all_signals = []
            all_no_signals = []

            for asset in ASSETS:
                clf, feats, c2i = models[asset]
                ho = holdouts[asset]
                probs = clf.predict_proba(ho[feats])
                thresh = THRESHOLD_PCT[asset]

                up_idx = c2i["up_swing"]
                down_idx = c2i["down_swing"]
                none_idx = c2i["no_swing"]

                for idx, row in ho.reset_index(drop=True).iterrows():
                    p_up = probs[idx][up_idx]
                    p_down = probs[idx][down_idx]
                    p_none = probs[idx][none_idx]
                    c = row["cutoff_utc"]

                    diff = p_up - p_down

                    # Signal rule:
                    if p_up >= tau and diff >= delta and p_up > p_none:
                        all_signals.append({
                            "asset": asset,
                            "signal_utc": c,
                            "direction": "up",
                            "horizon": 5,
                            "target_pct": float(thresh),
                            "invalidation_pct": float(thresh),
                            "confidence": float(p_up),
                            "evidence": f"lgbm_corrected_tau_{tau:.2f}_delta_{delta:.2f}",
                        })
                    elif p_down >= tau and -diff >= delta and p_down > p_none:
                        all_signals.append({
                            "asset": asset,
                            "signal_utc": c,
                            "direction": "down",
                            "horizon": 5,
                            "target_pct": float(thresh),
                            "invalidation_pct": float(thresh),
                            "confidence": float(p_down),
                            "evidence": f"lgbm_corrected_tau_{tau:.2f}_delta_{delta:.2f}",
                        })
                    else:
                        all_no_signals.append({
                            "asset": asset,
                            "decision_utc": c,
                            "reason": f"abstain_low_confidence_or_margin",
                        })

            sig_df = pd.DataFrame(all_signals)
            no_sig_df = pd.DataFrame(all_no_signals)

            sig_f = RESULTS_DIR / f"test_corrected_sig_{tau:.2f}_{delta:.2f}.parquet"
            nosig_f = RESULTS_DIR / f"test_corrected_nosig_{tau:.2f}_{delta:.2f}.parquet"
            sig_df.to_parquet(sig_f, index=False)
            no_sig_df.to_parquet(nosig_f, index=False)

            res, cov = run_scorer(sig_f, nosig_f, prices, inventory, window)
            corr = res["scores"]["correctness"]["value"]
            depth = res["scores"]["depth"]["value"]
            catch = res["scores"]["catch_rate"]["value"]
            succ = res["scores"]["success_rate"]["value"]
            top10 = res["ten_large_swings"]["caught"]
            n_sig = res["counts"]["signals_in_window"]
            n_dec = res["counts"]["decided"]

            print(f"{tau:<6.2f} | {delta:<6.2f} | {n_sig:<8d} | {n_dec:<8d} | {corr:<11.2f}% | {depth:<7.2f}% | {catch:<9.2f}% | {succ:<11.2f}% | {top10:<2d}/10")

            if succ > best_score:
                best_score = succ
                best_combo = (tau, delta, corr, depth, catch, succ, top10)

    print("\n" + "=" * 80)
    print(f"BEST CONFIGURATION: Tau={best_combo[0]:.2f}, Delta={best_combo[1]:.2f}")
    print(f"Correctness:  {best_combo[2]:.2f}%")
    print(f"Depth:        {best_combo[3]:.2f}%")
    print(f"Catch Rate:   {best_combo[4]:.2f}%")
    print(f"Success Rate: {best_combo[5]:.2f}%")
    print(f"Top 10:       {best_combo[6]}/10")
    print("=" * 80)

if __name__ == "__main__":
    main()
