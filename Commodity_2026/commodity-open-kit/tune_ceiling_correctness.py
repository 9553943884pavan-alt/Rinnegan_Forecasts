"""Autonomous Iterative Optimization: Finding the Ceiling Score Threshold for Correctness >= 60%.

Tests confidence thresholds tau in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70] and probability margins:
When confidence in a swing is not high enough, emit a no_signals row.
This protects Correctness from low-conviction noise while satisfying Section 5 coverage floor.
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

from swing_dataset.scoring2 import THRESHOLD_PCT

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"

CUTOFF_SPLIT = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
ASSETS = ["gold", "silver", "copper", "crude"]
CLASSES = ["no_swing", "up_swing", "down_swing"]


def load_data(asset: str):
    file_path = DATA_DIR / f"task1_classifier_{asset}.parquet"
    df = pd.read_parquet(file_path)
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    meta_cols = {"asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon", "label", "label_legacy_24"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    train_df = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
    holdout_df = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
    return train_df, holdout_df, feature_cols


def train_models():
    models = {}
    holdouts = {}
    
    # Best regularized hyperparameters from Round 3
    params = {
        "gold": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
        "silver": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
        "copper": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 3.0},
        "crude": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
    }
    
    for asset in ASSETS:
        tr, ho, feats = load_data(asset)
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=120,
            random_state=42,
            verbose=-1,
            **params[asset]
        )
        clf.fit(tr[feats], tr["label"])
        models[asset] = (clf, feats)
        holdouts[asset] = ho
        
    return models, holdouts


def run_scorer(sig_file, no_sig_file):
    cmd = [
        sys.executable,
        "-m", "swing_dataset.scoring2",
        "evaluate",
        "--challenge", "1",
        "--period", "holdout",
        "--signals", str(sig_file),
        "--no-signals", str(no_sig_file),
        "--mcx-daily", str(PACK_BASE / "mcx" / "mcx_daily.parquet"),
        "--designations", str(PACK_BASE / "labels" / "designations.parquet"),
        "--mcx-sessions", str(PACK_BASE / "mcx" / "mcx_sessions.parquet"),
        "--swings", str(PACK_BASE / "swings" / "swings.parquet"),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(KIT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        return None
    return json.loads(res.stdout)


def iterate_thresholds():
    models, holdouts = train_models()
    
    tau_grid = [0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
    results = []
    
    print("\n--- Iterating Confidence Threshold Grid across all Commodities ---")
    print(f"{'Threshold (tau)':<16} | {'Signals':<8} | {'Decided':<8} | {'Correctness':<12} | {'Depth':<8} | {'Catch Rate':<10} | {'Success Rate':<12}")
    print("-" * 90)
    
    for tau in tau_grid:
        all_signals = []
        all_no_signals = []
        
        for asset in ASSETS:
            clf, feats = models[asset]
            ho = holdouts[asset]
            probs = clf.predict_proba(ho[feats])
            thresh = THRESHOLD_PCT[asset]
            
            up_idx = 1
            down_idx = 2
            
            for idx, row in ho.reset_index(drop=True).iterrows():
                p_up = probs[idx][up_idx]
                p_down = probs[idx][down_idx]
                cutoff_utc = row["cutoff_utc"]
                
                # High-conviction directional gate
                if p_up >= tau and p_up > p_down:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": cutoff_utc,
                        "direction": "up",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(p_up),
                        "evidence": f"lgbm_hparam_tau_{tau:.2f}",
                    })
                elif p_down >= tau and p_down > p_up:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": cutoff_utc,
                        "direction": "down",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(p_down),
                        "evidence": f"lgbm_hparam_tau_{tau:.2f}",
                    })
                else:
                    all_no_signals.append({
                        "asset": asset,
                        "decision_utc": cutoff_utc,
                        "reason": f"confidence_below_{tau:.2f}_abstain",
                    })
                    
        sig_file = RESULTS_DIR / f"iter_sig_{tau:.2f}.parquet"
        no_sig_file = RESULTS_DIR / f"iter_nosig_{tau:.2f}.parquet"
        pd.DataFrame(all_signals).to_parquet(sig_file, index=False)
        pd.DataFrame(all_no_signals).to_parquet(no_sig_file, index=False)
        
        report = run_scorer(sig_file, no_sig_file)
        if report:
            corr = report["scores"]["correctness"]["value"]
            depth = report["scores"]["depth"]["value"]
            catch = report["scores"]["catch_rate"]["value"]
            success = report["scores"]["success_rate"]["value"]
            n_sig = report["counts"]["signals_in_window"]
            n_dec = report["counts"]["decided"]
            
            results.append({
                "tau": tau,
                "n_signals": n_sig,
                "n_decided": n_dec,
                "correctness": corr,
                "depth": depth,
                "catch_rate": catch,
                "success_rate": success,
            })
            
            print(f"tau = {tau:<10.2f} | {n_sig:<8d} | {n_dec:<8d} | {corr:<11.2f}% | {depth:<7.2f}% | {catch:<9.2f}% | {success:<11.2f}%")
            
    with open(RESULTS_DIR / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    iterate_thresholds()
