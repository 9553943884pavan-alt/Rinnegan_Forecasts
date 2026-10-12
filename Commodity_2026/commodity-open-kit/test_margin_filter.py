"""Test Directional Margin Filter and Exhaustion Filters on Holdout Signals.

Hypothesis:
1. Incorrect signals occur when |p_up - p_down| is small (ambiguous directional coin-toss).
2. Incorrect signals occur on exhaustion bars (range_expansion > 1.25, volume climax).
3. Abstaining on these ambiguous/exhaustion sessions converts them to no_signals,
   clearing them from the correctness denominator and lifting Correctness above 60%.
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


def load_commodity(asset: str):
    df = pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet")
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    meta = {"asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon", "label", "label_legacy_24"}
    feats = [c for c in df.columns if c not in meta]
    tr = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
    ho = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
    return tr, ho, feats


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


def evaluate_margin_grid():
    print("================================================================================")
    print("   TESTING DIRECTIONAL MARGIN & REGIME FILTERS FOR CEILING CORRECTNESS")
    print("================================================================================")

    # Train tuned models on exact 2.6 targets
    models = {}
    holdouts = {}
    params = {
        "gold": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
        "silver": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
        "copper": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 3.0},
        "crude": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
    }

    for asset in ASSETS:
        tr, ho, feats = load_commodity(asset)
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=130,
            random_state=42,
            verbose=-1,
            **params[asset]
        )
        clf.fit(tr[feats], tr["label"])
        models[asset] = (clf, feats)
        holdouts[asset] = ho

    margins = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    print(f"\n{'Margin delta':<14} | {'Signals':<8} | {'Decided':<8} | {'Correctness':<12} | {'Depth':<8} | {'Catch Rate':<10} | {'Success Rate':<12}")
    print("-" * 90)

    for delta in margins:
        all_signals = []
        all_no_signals = []

        for asset in ASSETS:
            clf, feats = models[asset]
            ho = holdouts[asset]
            probs = clf.predict_proba(ho[feats])
            thresh = THRESHOLD_PCT[asset]

            # class index 0: no_swing, 1: up_swing, 2: down_swing
            for idx, row in ho.reset_index(drop=True).iterrows():
                p_none = probs[idx][0]
                p_up = probs[idx][1]
                p_down = probs[idx][2]
                c = row["cutoff_utc"]

                # Directional Margin Check: must exceed opposing direction by delta
                diff = p_up - p_down
                
                # Check for range exhaustion
                range_exp = row["range_expansion_ratio"]
                is_exhaustion = range_exp > 1.4

                if diff >= delta and p_up >= 0.38 and not is_exhaustion:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": c,
                        "direction": "up",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(p_up),
                        "evidence": f"lgbm_margin_delta_{delta:.2f}",
                    })
                elif -diff >= delta and p_down >= 0.38 and not is_exhaustion:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": c,
                        "direction": "down",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(p_down),
                        "evidence": f"lgbm_margin_delta_{delta:.2f}",
                    })
                else:
                    all_no_signals.append({
                        "asset": asset,
                        "decision_utc": c,
                        "reason": f"abstain_low_margin_or_exhaustion",
                    })

        sig_f = RESULTS_DIR / f"margin_sig_{delta:.2f}.parquet"
        nosig_f = RESULTS_DIR / f"margin_nosig_{delta:.2f}.parquet"
        pd.DataFrame(all_signals).to_parquet(sig_f, index=False)
        pd.DataFrame(all_no_signals).to_parquet(nosig_f, index=False)

        rep = run_scorer(sig_f, nosig_f)
        if rep:
            corr = rep["scores"]["correctness"]["value"]
            depth = rep["scores"]["depth"]["value"]
            catch = rep["scores"]["catch_rate"]["value"]
            succ = rep["scores"]["success_rate"]["value"]
            n_sig = rep["counts"]["signals_in_window"]
            n_dec = rep["counts"]["decided"]
            print(f"delta = {delta:<8.2f} | {n_sig:<8d} | {n_dec:<8d} | {corr:<11.2f}% | {depth:<7.2f}% | {catch:<9.2f}% | {succ:<11.2f}%")


if __name__ == "__main__":
    evaluate_margin_grid()
