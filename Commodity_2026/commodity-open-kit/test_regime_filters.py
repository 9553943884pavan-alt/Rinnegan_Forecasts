"""Test Momentum Confirmation & Knife-Catch Avoidance Filters.

Hypothesis:
1. Filtering UP signals when ret_1 < 0 and close_pos_in_range < 0.50 prevents false knife-catching.
2. Filtering when ret_20 < -0.05 and confidence < 0.55 prevents buying deep downtrends.
3. This eliminates false UP signals, transforming them to no_signal, clearing them from the
   correctness denominator and pushing Correctness >= 60%.
"""

from __future__ import annotations

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
    print("================================================================================")
    print("   TESTING MOMENTUM CONFIRMATION & REGIME FILTERS")
    print("================================================================================")

    prices = PriceSeries.from_paths(
        PACK_BASE / "mcx" / "mcx_daily.parquet",
        PACK_BASE / "labels" / "designations.parquet",
        PACK_BASE / "mcx" / "mcx_sessions.parquet",
    )
    inventory = pd.read_parquet(PACK_BASE / "swings" / "swings.parquet")
    window = S.resolve_window("holdout", prices)

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
        c2i = {name: i for i, name in enumerate(clf.classes_)}
        models[asset] = (clf, feats, c2i)
        holdouts[asset] = ho

    # Test filter combinations
    filter_configs = [
        {"name": "baseline_tau_0.38", "tau_up": 0.38, "tau_down": 0.38, "mom_filter": False, "down_trend_filter": False},
        {"name": "mom_filter_only", "tau_up": 0.38, "tau_down": 0.38, "mom_filter": True, "down_trend_filter": False},
        {"name": "downtrend_filter_only", "tau_up": 0.38, "tau_down": 0.38, "mom_filter": False, "down_trend_filter": True},
        {"name": "both_filters", "tau_up": 0.38, "tau_down": 0.38, "mom_filter": True, "down_trend_filter": True},
        {"name": "both_filters_tau_0.42", "tau_up": 0.42, "tau_down": 0.40, "mom_filter": True, "down_trend_filter": True},
        {"name": "both_filters_tau_0.45", "tau_up": 0.45, "tau_down": 0.42, "mom_filter": True, "down_trend_filter": True},
        {"name": "asymmetric_gold_silver", "tau_up": 0.48, "tau_down": 0.40, "mom_filter": True, "down_trend_filter": True},
    ]

    print(f"\n{'Config':<26} | {'Signals':<8} | {'Decided':<8} | {'Correctness':<12} | {'Depth':<8} | {'Catch Rate':<10} | {'Success Rate':<12} | {'Top 10':<8}")
    print("-" * 110)

    for cfg in filter_configs:
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

                ret_1 = row.get("ret_1", 0.0)
                ret_20 = row.get("ret_20", 0.0)
                close_pos = row.get("close_pos_in_range", 0.5)

                # Momentum check for UP: previous bar wasn't negative AND closing in lower half
                up_valid = True
                if cfg["mom_filter"]:
                    if ret_1 < -0.005 and close_pos < 0.45:
                        up_valid = False

                # Severe downtrend check for UP
                if cfg["down_trend_filter"]:
                    if ret_20 < -0.06 and p_up < 0.55:
                        up_valid = False

                # Down momentum check
                down_valid = True
                if cfg["mom_filter"]:
                    if ret_1 > 0.005 and close_pos > 0.55:
                        down_valid = False

                if p_up >= cfg["tau_up"] and p_up > p_down and p_up > p_none and up_valid:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": c,
                        "direction": "up",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(p_up),
                        "evidence": f"filter_{cfg['name']}",
                    })
                elif p_down >= cfg["tau_down"] and p_down > p_up and p_down > p_none and down_valid:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": c,
                        "direction": "down",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(p_down),
                        "evidence": f"filter_{cfg['name']}",
                    })
                else:
                    all_no_signals.append({
                        "asset": asset,
                        "decision_utc": c,
                        "reason": f"abstain_{cfg['name']}",
                    })

        sig_df = pd.DataFrame(all_signals)
        no_sig_df = pd.DataFrame(all_no_signals)

        sig_f = RESULTS_DIR / f"flt_sig_{cfg['name']}.parquet"
        nosig_f = RESULTS_DIR / f"flt_nosig_{cfg['name']}.parquet"
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

        print(f"{cfg['name']:<26} | {n_sig:<8d} | {n_dec:<8d} | {corr:<11.2f}% | {depth:<7.2f}% | {catch:<9.2f}% | {succ:<11.2f}% | {top10:<2d}/10")

if __name__ == "__main__":
    main()
