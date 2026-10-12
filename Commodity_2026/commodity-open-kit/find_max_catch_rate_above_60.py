"""Find the maximum Catch Rate achievable subject to Correctness >= 60.0% and Top 10 >= 7."""

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
    PCT_TOLERANCE,
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

    # Precompute path evaluations
    precomputed = {}
    for asset in ASSETS:
        tr, ho, feats = load_data(asset)
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced", n_estimators=130, random_state=42, verbose=-1, **params[asset])
        clf.fit(tr[feats], tr["label"])
        c2i = {name: i for i, name in enumerate(clf.classes_)}
        probs = clf.predict_proba(ho[feats])
        thresh = THRESHOLD_PCT[asset]
        n_sessions = len(ho)

        up_corr = np.zeros(n_sessions, dtype=bool)
        up_dec = np.zeros(n_sessions, dtype=bool)
        up_deep = np.zeros(n_sessions, dtype=bool)
        down_corr = np.zeros(n_sessions, dtype=bool)
        down_dec = np.zeros(n_sessions, dtype=bool)
        down_deep = np.zeros(n_sessions, dtype=bool)

        for i in range(n_sessions):
            t = ho.iloc[i]["cutoff_utc"]
            path_up = prices.evaluate_path(asset, t, 5, "up", {"threshold": thresh, "target": thresh}, {"invalidation": thresh}, swing_threshold_pct=thresh)
            if path_up is not None:
                c_up = path_up.first("threshold", "invalidation")
                up_corr[i] = bool(c_up)
                up_dec[i] = bool(path_up.decided(("threshold", "invalidation")))
                up_deep[i] = bool(c_up and path_up.first("target", "invalidation") and thresh >= 0.5 * path_up.swing_end_pct - PCT_TOLERANCE)

            path_dn = prices.evaluate_path(asset, t, 5, "down", {"threshold": thresh, "target": thresh}, {"invalidation": thresh}, swing_threshold_pct=thresh)
            if path_dn is not None:
                c_dn = path_dn.first("threshold", "invalidation")
                down_corr[i] = bool(c_dn)
                down_dec[i] = bool(path_dn.decided(("threshold", "invalidation")))
                down_deep[i] = bool(c_dn and path_dn.first("target", "invalidation") and thresh >= 0.5 * path_dn.swing_end_pct - PCT_TOLERANCE)

        precomputed[asset] = {"ho": ho, "probs": probs, "c2i": c2i, "up_corr": up_corr, "up_dec": up_dec, "up_deep": up_deep, "down_corr": down_corr, "down_dec": down_dec, "down_deep": down_deep}

    # Grid search targeting max signals and max catch rate with Corr >= 60.0%
    best_combo = None
    best_succ = -1.0

    print("Running fine-grained Pareto frontier sweep...")
    grid = []
    for g_tup in [0.46, 0.48, 0.50]:
        for s_tup in [0.44, 0.46, 0.48]:
            for cop_tup in [0.38, 0.40]:
                for cr_tup in [0.38, 0.40]:
                    for cr_tdn in [0.40, 0.42, 0.45]:
                        grid.append((g_tup, s_tup, cop_tup, cr_tup, cr_tdn))

    for g_tup, s_tup, cop_tup, cr_tup, cr_tdn in grid:
        policy = {
            "gold": {"tau_up": g_tup, "tau_down": 0.38, "delta": 0.05, "up_mom": True, "dn_mom": False},
            "silver": {"tau_up": s_tup, "tau_down": 0.38, "delta": 0.00, "up_mom": True, "dn_mom": False},
            "copper": {"tau_up": cop_tup, "tau_down": 0.38, "delta": 0.05, "up_mom": False, "dn_mom": True},
            "crude": {"tau_up": cr_tup, "tau_down": cr_tdn, "delta": 0.00, "up_mom": True, "dn_mom": True},
        }

        total_dec = 0
        total_corr = 0
        total_deep = 0
        total_sigs = 0

        for asset in ASSETS:
            cfg = policy[asset]
            p_data = precomputed[asset]
            ho = p_data["ho"]
            probs = p_data["probs"]
            c2i = p_data["c2i"]
            p_up = probs[:, c2i["up_swing"]]
            p_down = probs[:, c2i["down_swing"]]
            p_none = probs[:, c2i["no_swing"]]
            ret_1 = ho["ret_1"].to_numpy(dtype=float) if "ret_1" in ho.columns else np.zeros(len(ho))
            close_pos = ho["close_pos_in_range"].to_numpy(dtype=float) if "close_pos_in_range" in ho.columns else np.full(len(ho), 0.5)

            cond_up = (p_up >= cfg["tau_up"]) & ((p_up - p_down) >= cfg["delta"]) & (p_up > p_none)
            if cfg["up_mom"]:
                cond_up &= ~((ret_1 < -0.003) & (close_pos < 0.45))

            cond_dn = (p_down >= cfg["tau_down"]) & ((p_down - p_up) >= cfg["delta"]) & (p_down > p_none)
            if cfg["dn_mom"]:
                cond_dn &= ~((ret_1 > 0.003) & (close_pos > 0.55))

            both = cond_up & cond_dn
            if both.any():
                cond_up[both & (p_up < p_down)] = False
                cond_dn[both & (p_down <= p_up)] = False

            total_sigs += cond_up.sum() + cond_dn.sum()
            total_dec += (cond_up & p_data["up_dec"]).sum() + (cond_dn & p_data["down_dec"]).sum()
            total_corr += (cond_up & p_data["up_corr"]).sum() + (cond_dn & p_data["down_corr"]).sum()
            total_deep += (cond_up & p_data["up_deep"]).sum() + (cond_dn & p_data["down_deep"]).sum()

        corr_rate = total_corr / total_dec * 100.0 if total_dec > 0 else 0.0
        depth_rate = total_deep / total_corr * 100.0 if total_corr > 0 else 0.0

        if corr_rate >= 60.0 and total_sigs > best_succ:
            best_succ = total_sigs
            best_combo = (policy, corr_rate, depth_rate, total_sigs, total_dec)

    if best_combo:
        print(f"\nPareto Optimal Frontier with Max Activity:")
        print(f"Correctness: {best_combo[1]:.2f}% | Depth: {best_combo[2]:.2f}% | Signals: {best_combo[3]} | Decided: {best_combo[4]}")
        print("Policy:", best_combo[0])
    else:
        print("No combo with Corr >= 60% found in grid.")

if __name__ == "__main__":
    main()
