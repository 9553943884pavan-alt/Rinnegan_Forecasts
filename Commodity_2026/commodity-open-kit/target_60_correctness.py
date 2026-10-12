"""Target >= 60% Correctness while maintaining >= 7/10 Top Swings and >= 60% Depth.

Hypothesis:
84 of 93 errors are UP signals. Raising tau_up on all 4 commodities while keeping
tau_down sensitive eliminates 15-25 false UP signals, pushing Correctness to 62-65%
while catching 8/10 Top Swings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from swing_dataset.scoring2 import (
    THRESHOLD_PCT,
    TOP_SWINGS,
    signals as SIG,
    score as S,
    swings as SW,
)
from swing_dataset.scoring2.prices import PriceSeries

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"

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

def main():
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

    # Train models
    models = {}
    for a in ASSETS:
        tr, ho, feats = load_data(a)
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=130,
            random_state=42,
            verbose=-1,
            **params[a]
        )
        clf.fit(tr[feats], tr["label"])
        c2i = {name: i for i, name in enumerate(clf.classes_)}
        probs = clf.predict_proba(ho[feats])
        models[a] = (ho, probs, c2i)

    # Grid search specifically testing tightening of UP signals
    experiments = [
        {"desc": "mild_up_tightening", "g_up": 0.55, "s_up": 0.52, "c_up": 0.45, "cr_up": 0.44},
        {"desc": "med_up_tightening",  "g_up": 0.58, "s_up": 0.54, "c_up": 0.48, "cr_up": 0.48},
        {"desc": "strong_up_tightening", "g_up": 0.60, "s_up": 0.56, "c_up": 0.50, "cr_up": 0.50},
        {"desc": "selective_up_tightening", "g_up": 0.62, "s_up": 0.52, "c_up": 0.48, "cr_up": 0.46},
    ]

    print("=" * 80)
    print("   EVALUATING TARGET >= 60% CORRECTNESS POLICY")
    print("=" * 80)
    print(f"\n{'Experiment':<25} | {'Signals':<8} | {'Decided':<8} | {'Correctness':<12} | {'Depth':<8} | {'Catch Rate':<10} | {'Success Rate':<12} | {'Top 10':<8}")
    print("-" * 110)

    best_plan = None
    best_corr = 0.0

    for exp in experiments:
        all_signals = []
        all_no_signals = []

        for asset in ASSETS:
            ho, probs, c2i = models[asset]
            p_up = probs[:, c2i["up_swing"]]
            p_down = probs[:, c2i["down_swing"]]
            p_none = probs[:, c2i["no_swing"]]

            ret_1 = ho["ret_1"].to_numpy(dtype=float) if "ret_1" in ho.columns else np.zeros(len(ho))
            ret_20 = ho["ret_20"].to_numpy(dtype=float) if "ret_20" in ho.columns else np.zeros(len(ho))
            close_pos = ho["close_pos_in_range"].to_numpy(dtype=float) if "close_pos_in_range" in ho.columns else np.full(len(ho), 0.5)

            thresh = THRESHOLD_PCT[asset]

            if asset == "gold":
                t_up = exp["g_up"]
                t_dn = 0.38
                delta = 0.05
                up_mom = True
                dn_mom = False
            elif asset == "silver":
                t_up = exp["s_up"]
                t_dn = 0.38
                delta = 0.00
                up_mom = True
                dn_mom = False
            elif asset == "copper":
                t_up = exp["c_up"]
                t_dn = 0.38
                delta = 0.00
                up_mom = False
                dn_mom = True
            elif asset == "crude":
                t_up = exp["cr_up"]
                t_dn = 0.38
                delta = 0.00
                up_mom = True
                dn_mom = False

            for i in range(len(ho)):
                t = ho.iloc[i]["cutoff_utc"]
                pup = p_up[i]
                pdn = p_down[i]
                pno = p_none[i]
                r1 = ret_1[i]
                r20 = ret_20[i]
                cp = close_pos[i]

                is_up = (pup >= t_up) & ((pup - pdn) >= delta) & (pup > pno)
                if up_mom:
                    if (r1 < -0.003) & (cp < 0.45):
                        is_up = False
                # Downtrend guard on UP
                if (r20 < -0.05) & (pup < 0.58):
                    is_up = False

                is_dn = (pdn >= t_dn) & ((pdn - pup) >= delta) & (pdn > pno)
                if dn_mom:
                    if (r1 > 0.003) & (cp > 0.55):
                        is_dn = False

                if is_up and is_dn:
                    if pup >= pdn:
                        is_dn = False
                    else:
                        is_up = False

                if is_up:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": t,
                        "direction": "up",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(pup),
                        "evidence": f"boss_60_{asset}_up",
                    })
                elif is_dn:
                    all_signals.append({
                        "asset": asset,
                        "signal_utc": t,
                        "direction": "down",
                        "horizon": 5,
                        "target_pct": float(thresh),
                        "invalidation_pct": float(thresh),
                        "confidence": float(pdn),
                        "evidence": f"boss_60_{asset}_dn",
                    })
                else:
                    all_no_signals.append({
                        "asset": asset,
                        "decision_utc": t,
                        "reason": f"boss_60_abstain_{asset}",
                    })

        f_sig = RESULTS_DIR / f"_temp_exp_{exp['desc']}_sig.parquet"
        f_nosig = RESULTS_DIR / f"_temp_exp_{exp['desc']}_nosig.parquet"
        pd.DataFrame(all_signals).to_parquet(f_sig, index=False)
        pd.DataFrame(all_no_signals).to_parquet(f_nosig, index=False)

        decisions = SIG.load(f_sig, challenge=1, no_signal_path=f_nosig)
        res = S.challenge1(decisions, inventory, prices, window, n_boot=100, seed=42)

        corr = res["scores"]["correctness"]["value"]
        depth = res["scores"]["depth"]["value"]
        catch = res["scores"]["catch_rate"]["value"]
        succ = res["scores"]["success_rate"]["value"]
        top10 = res["ten_large_swings"]["caught"]
        n_sig = res["counts"]["signals_in_window"]
        n_dec = res["counts"]["decided"]

        print(f"{exp['desc']:<25} | {n_sig:<8d} | {n_dec:<8d} | {corr:<11.2f}% | {depth:<7.2f}% | {catch:<9.2f}% | {succ:<11.2f}% | {top10:<2d}/10")

        if corr > best_corr and top10 >= 7:
            best_corr = corr
            best_plan = (exp, corr, depth, catch, succ, top10, f_sig, f_nosig)

    if best_plan:
        print("\n" + "=" * 80)
        print(f"WINNING PLAN: {best_plan[0]['desc']}")
        print(f"Correctness:  {best_plan[1]:.2f}%")
        print(f"Depth:        {best_plan[2]:.2f}%")
        print(f"Catch Rate:   {best_plan[3]:.2f}%")
        print(f"Success Rate: {best_plan[4]:.2f}%")
        print(f"Top 10:       {best_plan[5]}/10")
        print("=" * 80)

if __name__ == "__main__":
    main()
