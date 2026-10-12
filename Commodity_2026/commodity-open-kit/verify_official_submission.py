"""Verify Official Competition Submission and Gates for Task 1 Holdout."""

import json
import os
import sys
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import pandas as pd
from swing_dataset.scoring2 import score as S, signals as SIG
from swing_dataset.scoring2.prices import PriceSeries

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"

def verify():
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
    res = S.challenge1(decisions, inventory, prices, window, n_boot=500, seed=42)
    cov = S.coverage(prices, decisions, window)

    scores = res["scores"]
    counts = res["counts"]
    top10 = res["ten_large_swings"]

    report = {
        "task": "Task 1: Pre-open Directional Swing Prediction",
        "rules_version": "2.6",
        "period": "holdout",
        "model_architecture": "Boss (LightGBM Multiclass Classifier with Balanced Weights) + Employee (LightGBM Huber Regressor)",
        "features": "60 Leak-Free Input Features (Microstructure, COT Smart Money, Hyperliquid Crypto/Commodity, FBIL FX, Calendar, FinBERT Sentiment)",
        "gates": {
            "correctness": {
                "score": round(scores["correctness"]["value"], 2),
                "threshold": 60.0,
                "status": "PASS" if scores["correctness"]["value"] >= 60.0 else "FAIL"
            },
            "depth": {
                "score": round(scores["depth"]["value"], 2),
                "threshold": 60.0,
                "status": "PASS" if scores["depth"]["value"] >= 60.0 else "FAIL"
            },
            "top_10_large_swings": {
                "caught": top10["caught"],
                "required": top10["required"],
                "total": top10["top"],
                "status": "PASS" if top10["caught"] >= top10["required"] else "FAIL"
            },
            "coverage_floor": {
                "share": round(cov["pooled"]["share"] * 100, 2) if cov["pooled"]["share"] < 1 else round(cov["pooled"]["share"], 2),
                "threshold": 90.0,
                "status": "PASS" if cov["met"] else "FAIL"
            },
            "success_rate": {
                "score": round(scores["success_rate"]["value"], 2),
                "threshold": 70.0,
                "status": "PASS" if scores["success_rate"]["value"] >= 70.0 else "FAIL"
            }
        },
        "counts": {
            "signals_issued": counts["signals_in_window"],
            "signals_decided": counts["decided"],
            "signals_undecided": counts["undecided"],
            "signals_correct": counts["correct"],
            "swings_total": counts["swings_in_window"],
            "swings_caught": counts["swings_caught"],
            "catch_rate": round(scores["catch_rate"]["value"], 2),
        },
        "per_asset_performance": {}
    }

    graded = res["graded"]
    caught_df = res["caught"]

    for a in ["gold", "silver", "copper", "crude"]:
        g_a = graded[graded["asset"] == a]
        dec_a = g_a[g_a["decided"].fillna(False).astype(bool)]
        corr_a = dec_a[dec_a["correct"].fillna(False).astype(bool)]
        sw_a = caught_df[caught_df["asset"] == a]
        sw_c_a = sw_a[sw_a["caught"].fillna(False).astype(bool)]

        report["per_asset_performance"][a] = {
            "signals": len(g_a),
            "decided": len(dec_a),
            "correct": len(corr_a),
            "correctness_pct": round(len(corr_a) / len(dec_a) * 100, 2) if len(dec_a) > 0 else 0.0,
            "swings": len(sw_a),
            "swings_caught": len(sw_c_a),
            "catch_rate_pct": round(len(sw_c_a) / len(sw_a) * 100, 2) if len(sw_a) > 0 else 0.0,
        }

    # Save to json artifact
    out_file = DATA_DIR / "task1_verified_award_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("               TASK 1 COMPETITION VERIFICATION SUMMARY")
    print("=" * 80)
    print(json.dumps(report["gates"], indent=2))
    print("\nPer-Asset Breakdown:")
    print(json.dumps(report["per_asset_performance"], indent=2))
    print(f"\nSaved verified report to: {out_file}")

if __name__ == "__main__":
    verify()
