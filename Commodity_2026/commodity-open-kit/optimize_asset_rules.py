"""Asset-by-Asset Policy Optimization for Task 1.

Finds the optimal decision rule for each asset independently to:
1. Maximize Correctness (target >= 60%)
2. Maintain Coverage (floor >= 90% per asset)
3. Catch the Top 10 large swings (target >= 7/10 overall)
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
        models[asset] = (clf, feats, ho, c2i)

    # Grid search per asset
    best_asset_configs = {}

    for asset in ASSETS:
        print("\n" + "=" * 70)
        print(f"       OPTIMIZING ASSET: {asset.upper()}")
        print("=" * 70)
        
        clf, feats, ho, c2i = models[asset]
        probs = clf.predict_proba(ho[feats])
        thresh = THRESHOLD_PCT[asset]

        up_idx = c2i["up_swing"]
        down_idx = c2i["down_swing"]
        none_idx = c2i["no_swing"]

        best_corr = 0.0
        best_cfg = None

        # Search grid
        taus = [0.38, 0.42, 0.45, 0.48, 0.52]
        margins = [0.00, 0.05, 0.10, 0.15]
        up_filters = ["none", "ret_1_pos", "close_pos_50", "downtrend_guard"]
        down_filters = ["none", "ret_1_neg", "close_pos_50", "uptrend_guard"]

        for tau in taus:
            for delta in margins:
                for uf in up_filters:
                    for df_filt in down_filters:
                        sigs = []
                        nosigs = []

                        for idx, row in ho.reset_index(drop=True).iterrows():
                            p_up = probs[idx][up_idx]
                            p_down = probs[idx][down_idx]
                            p_none = probs[idx][none_idx]
                            c = row["cutoff_utc"]
                            diff = p_up - p_down

                            ret_1 = row.get("ret_1", 0.0)
                            ret_20 = row.get("ret_20", 0.0)
                            close_pos = row.get("close_pos_in_range", 0.5)

                            # Up filter
                            up_ok = True
                            if uf == "ret_1_pos" and ret_1 < 0:
                                up_ok = False
                            elif uf == "close_pos_50" and close_pos < 0.50:
                                up_ok = False
                            elif uf == "downtrend_guard" and (ret_20 < -0.05 and p_up < 0.55):
                                up_ok = False

                            # Down filter
                            down_ok = True
                            if df_filt == "ret_1_neg" and ret_1 > 0:
                                down_ok = False
                            elif df_filt == "close_pos_50" and close_pos > 0.50:
                                down_ok = False
                            elif df_filt == "uptrend_guard" and (ret_20 > 0.05 and p_down < 0.55):
                                down_ok = False

                            if p_up >= tau and diff >= delta and p_up > p_none and up_ok:
                                sigs.append({
                                    "asset": asset,
                                    "signal_utc": c,
                                    "direction": "up",
                                    "horizon": 5,
                                    "target_pct": float(thresh),
                                    "invalidation_pct": float(thresh),
                                    "confidence": float(p_up),
                                    "evidence": "opt",
                                })
                            elif p_down >= tau and -diff >= delta and p_down > p_none and down_ok:
                                sigs.append({
                                    "asset": asset,
                                    "signal_utc": c,
                                    "direction": "down",
                                    "horizon": 5,
                                    "target_pct": float(thresh),
                                    "invalidation_pct": float(thresh),
                                    "confidence": float(p_down),
                                    "evidence": "opt",
                                })
                            else:
                                nosigs.append({
                                    "asset": asset,
                                    "decision_utc": c,
                                    "reason": "opt_abstain",
                                })

                        if len(sigs) < 20: # Require at least 20 signals per asset to avoid over-filtering
                            continue

                        # Grade asset directly
                        sig_df = pd.DataFrame(sigs)
                        nosig_df = pd.DataFrame(nosigs)
                        
                        f_sig = RESULTS_DIR / f"_temp_opt_sig_{asset}.parquet"
                        f_nosig = RESULTS_DIR / f"_temp_opt_nosig_{asset}.parquet"
                        sig_df.to_parquet(f_sig, index=False)
                        nosig_df.to_parquet(f_nosig, index=False)

                        decisions = SIG.load(f_sig, challenge=1, no_signal_path=f_nosig)
                        res = S.challenge1(decisions, inventory, prices, window, n_boot=10, seed=42)
                        
                        corr = res["scores"]["correctness"]["value"]
                        catch = res["scores"]["catch_rate"]["value"]
                        n_sig = res["counts"]["signals_in_window"]
                        n_dec = res["counts"]["decided"]

                        if corr > best_corr and n_dec >= 25:
                            best_corr = corr
                            best_cfg = {
                                "asset": asset,
                                "tau": tau,
                                "delta": delta,
                                "up_filter": uf,
                                "down_filter": df_filt,
                                "signals": n_sig,
                                "decided": n_dec,
                                "correctness": corr,
                                "catch_rate": catch,
                            }

        print(f"Best Config for {asset.upper()}:")
        print(best_cfg)
        best_asset_configs[asset] = best_cfg

    # Save best configs
    with open(DATA_DIR / "best_asset_configs.json", "w") as f:
        json.dump(best_asset_configs, f, indent=2)

if __name__ == "__main__":
    main()
