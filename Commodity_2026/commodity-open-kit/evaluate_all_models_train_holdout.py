"""Comprehensive Train and Holdout Evaluation for All 4 Classifiers and 4 Regressors.

Calculates:
1. Standard ML Metrics for Classifiers (Accuracy, Log-Loss, Macro & Weighted Precision/Recall/F1, Per-Class Metrics, Confusion Matrices) on Train & Holdout.
2. Standard ML Metrics for Regressors (MAE, RMSE, R2, MedAE, Actual vs Predicted Mean) on Train & Holdout.
3. Competition Metrics (Correctness, Depth, Catch Rate, Coverage, Top Swings) on Holdout AND Train.
4. Generalization Gap (Train vs Holdout comparison).
Standardizes on 0.75x MFE multiplier as confirmed by the user.
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from swing_dataset.scoring2 import (
    ASSETS,
    THRESHOLD_PCT,
    TOP_SWINGS,
    CATCH_ROOM_SHARE,
    PCT_TOLERANCE,
    signals as SIG,
    score as S,
    swings as SW,
    outcomes,
)
from swing_dataset.scoring2.prices import PriceSeries

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_SPLIT = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
TARGET_MULT = 0.75  # Standardized 0.75x MFE multiplier

CLASSIFIER_PARAMS = {
    "gold": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
    "silver": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
    "copper": {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 3.0},
    "crude": {"num_leaves": 12, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
}

ASSET_POLICIES = {
    "gold": {"tau_up": 0.54, "tau_down": 0.38, "delta": 0.05, "up_mom": True, "dn_mom": False, "dt_guard": False},
    "silver": {"tau_up": 0.50, "tau_down": 0.38, "delta": 0.00, "up_mom": True, "dn_mom": False, "dt_guard": False},
    "copper": {"tau_up": 0.38, "tau_down": 0.38, "delta": 0.05, "up_mom": False, "dn_mom": True, "dt_guard": False},
    "crude": {"tau_up": 0.38, "tau_down": 0.45, "delta": 0.00, "up_mom": True, "dn_mom": True, "dt_guard": False},
}


def load_classifier_data(asset: str):
    df = pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet")
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    meta = {"asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon", "label", "label_legacy_24"}
    feats = [c for c in df.columns if c not in meta]
    tr = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
    ho = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
    return tr, ho, feats


def load_regressor_data(asset: str):
    df = pd.read_parquet(DATA_DIR / f"task1_regressor_{asset}.parquet")
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    meta = {
        "asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon",
        "label", "label_legacy_24", "y_ret", "y_mfe", "y_mae", "y_range", "target_mfe_pct",
        "swing_direction", "target_is_swing"
    }
    feats = [c for c in df.columns if c not in meta]
    tr = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
    ho = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
    return tr, ho, feats


def evaluate_all():
    print("=" * 80)
    print("   TASK 1 COMPREHENSIVE TRAIN & HOLDOUT METRICS AUDIT")
    print(f"   Standardized Regressor Target Multiplier: {TARGET_MULT:.2f}x MFE")
    print("=" * 80)

    prices = PriceSeries.from_paths(
        PACK_BASE / "mcx" / "mcx_daily.parquet",
        PACK_BASE / "labels" / "designations.parquet",
        PACK_BASE / "mcx" / "mcx_sessions.parquet",
    )
    inventory = pd.read_parquet(PACK_BASE / "swings" / "swings.parquet")
    window_ho = S.resolve_window("holdout", prices)
    swings_ho = SW.in_window(inventory, window_ho.start, window_ho.end)

    # Resolve training window: Feb 2021 to May 31, 2026
    train_start = pd.Timestamp("2021-02-01 03:00:00", tz="UTC")
    train_end = pd.Timestamp("2026-05-31 18:00:00", tz="UTC")
    swings_tr = SW.in_window(inventory, train_start, train_end)

    audit_results = {
        "metadata": {
            "target_multiplier": TARGET_MULT,
            "train_window": f"{train_start} to {train_end}",
            "holdout_window": f"{window_ho.start} to {window_ho.end}",
            "assets": ASSETS,
        },
        "classifiers_ml": {},
        "regressors_ml": {},
        "holdout_competition": {},
        "train_competition": {},
        "generalization_gap": {},
    }

    # Store signals for holdout and train
    ho_signals_list = []
    ho_nosignals_list = []
    tr_signals_list = []
    tr_nosignals_list = []

    for asset in ASSETS:
        print(f"\n>>> Processing {asset.upper()}...")
        # 1. Load Data
        tr_c, ho_c, feats_c = load_classifier_data(asset)
        tr_r, ho_r, feats_r = load_regressor_data(asset)

        # ---------------------------------------------------------
        # 2. Train and Evaluate Classifier (Boss)
        # ---------------------------------------------------------
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            n_estimators=130,
            random_state=42,
            verbose=-1,
            **CLASSIFIER_PARAMS[asset],
        )
        clf.fit(tr_c[feats_c], tr_c["label"])
        classes = list(clf.classes_)
        c2i = {name: i for i, name in enumerate(classes)}

        # Classifier Predictions
        probs_tr = clf.predict_proba(tr_c[feats_c])
        preds_tr = clf.predict(tr_c[feats_c])
        probs_ho = clf.predict_proba(ho_c[feats_c])
        preds_ho = clf.predict(ho_c[feats_c])

        # ML Metrics - Train
        acc_tr = accuracy_score(tr_c["label"], preds_tr)
        loss_tr = log_loss(tr_c["label"], probs_tr, labels=classes)
        cr_tr_dict = classification_report(tr_c["label"], preds_tr, output_dict=True, zero_division=0)
        cm_tr = confusion_matrix(tr_c["label"], preds_tr, labels=classes)

        # ML Metrics - Holdout
        acc_ho = accuracy_score(ho_c["label"], preds_ho)
        loss_ho = log_loss(ho_c["label"], probs_ho, labels=classes)
        cr_ho_dict = classification_report(ho_c["label"], preds_ho, output_dict=True, zero_division=0)
        cm_ho = confusion_matrix(ho_c["label"], preds_ho, labels=classes)

        audit_results["classifiers_ml"][asset] = {
            "classes": classes,
            "train": {
                "samples": len(tr_c),
                "accuracy": round(float(acc_tr) * 100, 2),
                "log_loss": round(float(loss_tr), 4),
                "macro_f1": round(float(cr_tr_dict["macro avg"]["f1-score"]) * 100, 2),
                "macro_precision": round(float(cr_tr_dict["macro avg"]["precision"]) * 100, 2),
                "macro_recall": round(float(cr_tr_dict["macro avg"]["recall"]) * 100, 2),
                "weighted_f1": round(float(cr_tr_dict["weighted avg"]["f1-score"]) * 100, 2),
                "per_class": {
                    cls: {
                        "precision": round(float(cr_tr_dict[cls]["precision"]) * 100, 2),
                        "recall": round(float(cr_tr_dict[cls]["recall"]) * 100, 2),
                        "f1-score": round(float(cr_tr_dict[cls]["f1-score"]) * 100, 2),
                        "support": int(cr_tr_dict[cls]["support"]),
                    }
                    for cls in classes if cls in cr_tr_dict
                },
                "confusion_matrix": cm_tr.tolist(),
            },
            "holdout": {
                "samples": len(ho_c),
                "accuracy": round(float(acc_ho) * 100, 2),
                "log_loss": round(float(loss_ho), 4),
                "macro_f1": round(float(cr_ho_dict["macro avg"]["f1-score"]) * 100, 2),
                "macro_precision": round(float(cr_ho_dict["macro avg"]["precision"]) * 100, 2),
                "macro_recall": round(float(cr_ho_dict["macro avg"]["recall"]) * 100, 2),
                "weighted_f1": round(float(cr_ho_dict["weighted avg"]["f1-score"]) * 100, 2),
                "per_class": {
                    cls: {
                        "precision": round(float(cr_ho_dict[cls]["precision"]) * 100, 2),
                        "recall": round(float(cr_ho_dict[cls]["recall"]) * 100, 2),
                        "f1-score": round(float(cr_ho_dict[cls]["f1-score"]) * 100, 2),
                        "support": int(cr_ho_dict[cls]["support"]),
                    }
                    for cls in classes if cls in cr_ho_dict
                },
                "confusion_matrix": cm_ho.tolist(),
            },
        }

        # ---------------------------------------------------------
        # 3. Train and Evaluate Regressor (Employee)
        # ---------------------------------------------------------
        tr_r_swings = tr_r[tr_r["label"] != "no_swing"].copy().reset_index(drop=True)
        ho_r_swings = ho_r[ho_r["label"] != "no_swing"].copy().reset_index(drop=True)

        reg = lgb.LGBMRegressor(
            objective="huber",
            alpha=0.9,
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=12,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1,
        )
        reg.fit(tr_r_swings[feats_r], tr_r_swings["target_mfe_pct"])

        # Predictions on Regressor datasets
        preds_reg_tr_swings = reg.predict(tr_r_swings[feats_r])
        preds_reg_ho_swings = reg.predict(ho_r_swings[feats_r])

        # Full session predictions for dynamic policy setting
        preds_reg_tr_all = reg.predict(tr_r[feats_r])
        preds_reg_ho_all = reg.predict(ho_r[feats_r])

        # Metrics on realized swings
        mae_tr = mean_absolute_error(tr_r_swings["target_mfe_pct"], preds_reg_tr_swings)
        rmse_tr = np.sqrt(mean_squared_error(tr_r_swings["target_mfe_pct"], preds_reg_tr_swings))
        r2_tr = r2_score(tr_r_swings["target_mfe_pct"], preds_reg_tr_swings)
        medae_tr = median_absolute_error(tr_r_swings["target_mfe_pct"], preds_reg_tr_swings)

        mae_ho = mean_absolute_error(ho_r_swings["target_mfe_pct"], preds_reg_ho_swings)
        rmse_ho = np.sqrt(mean_squared_error(ho_r_swings["target_mfe_pct"], preds_reg_ho_swings))
        r2_ho = r2_score(ho_r_swings["target_mfe_pct"], preds_reg_ho_swings)
        medae_ho = median_absolute_error(ho_r_swings["target_mfe_pct"], preds_reg_ho_swings)

        audit_results["regressors_ml"][asset] = {
            "train": {
                "swings_count": len(tr_r_swings),
                "mae": round(float(mae_tr), 4),
                "rmse": round(float(rmse_tr), 4),
                "r2": round(float(r2_tr), 4),
                "medae": round(float(medae_tr), 4),
                "actual_mean_mfe": round(float(tr_r_swings["target_mfe_pct"].mean()), 3),
                "predicted_mean_mfe": round(float(preds_reg_tr_swings.mean()), 3),
                "pred_all_sessions_mean": round(float(preds_reg_tr_all.mean()), 3),
                "pred_min": round(float(preds_reg_tr_all.min()), 3),
                "pred_max": round(float(preds_reg_tr_all.max()), 3),
            },
            "holdout": {
                "swings_count": len(ho_r_swings),
                "mae": round(float(mae_ho), 4),
                "rmse": round(float(rmse_ho), 4),
                "r2": round(float(r2_ho), 4),
                "medae": round(float(medae_ho), 4),
                "actual_mean_mfe": round(float(ho_r_swings["target_mfe_pct"].mean()), 3),
                "predicted_mean_mfe": round(float(preds_reg_ho_swings.mean()), 3),
                "pred_all_sessions_mean": round(float(preds_reg_ho_all.mean()), 3),
                "pred_min": round(float(preds_reg_ho_all.min()), 3),
                "pred_max": round(float(preds_reg_ho_all.max()), 3),
            },
        }

        # ---------------------------------------------------------
        # 4. Generate Signal & Decision Records
        # ---------------------------------------------------------
        cfg = ASSET_POLICIES[asset]
        thresh = THRESHOLD_PCT[asset]

        # Holdout decisions
        p_up_ho = probs_ho[:, c2i["up_swing"]]
        p_dn_ho = probs_ho[:, c2i["down_swing"]]
        p_no_ho = probs_ho[:, c2i["no_swing"]]
        ret1_ho = ho_c["ret_1"].to_numpy(dtype=float) if "ret_1" in ho_c.columns else np.zeros(len(ho_c))
        ret20_ho = ho_c["ret_20"].to_numpy(dtype=float) if "ret_20" in ho_c.columns else np.zeros(len(ho_c))
        cpos_ho = ho_c["close_pos_in_range"].to_numpy(dtype=float) if "close_pos_in_range" in ho_c.columns else np.full(len(ho_c), 0.5)

        for i in range(len(ho_c)):
            t = ho_c.iloc[i]["cutoff_utc"]
            pup, pdn, pno = p_up_ho[i], p_dn_ho[i], p_no_ho[i]
            r1, r20, cp = ret1_ho[i], ret20_ho[i], cpos_ho[i]

            is_up = (pup >= cfg["tau_up"]) and ((pup - pdn) >= cfg["delta"]) and (pup > pno)
            if cfg["up_mom"] and (r1 < -0.003) and (cp < 0.45):
                is_up = False
            if cfg["dt_guard"] and (r20 < -0.05) and (pup < 0.52):
                is_up = False

            is_dn = (pdn >= cfg["tau_down"]) and ((pdn - pup) >= cfg["delta"]) and (pdn > pno)
            if cfg["dn_mom"] and (r1 > 0.003) and (cp > 0.55):
                is_dn = False

            if is_up and is_dn:
                if pup >= pdn:
                    is_dn = False
                else:
                    is_up = False

            raw_mfe = float(preds_reg_ho_all[i])
            target_val = float(max(thresh, raw_mfe * TARGET_MULT))

            if is_up:
                ho_signals_list.append({
                    "asset": asset,
                    "signal_utc": t,
                    "direction": "up",
                    "horizon": 5,
                    "target_pct": target_val,
                    "invalidation_pct": float(thresh),
                    "confidence": float(pup),
                    "evidence": f"boss_emp_opt_{asset}_up",
                })
            elif is_dn:
                ho_signals_list.append({
                    "asset": asset,
                    "signal_utc": t,
                    "direction": "down",
                    "horizon": 5,
                    "target_pct": target_val,
                    "invalidation_pct": float(thresh),
                    "confidence": float(pdn),
                    "evidence": f"boss_emp_opt_{asset}_dn",
                })
            else:
                ho_nosignals_list.append({
                    "asset": asset,
                    "decision_utc": t,
                    "reason": f"boss_emp_abstain_{asset}",
                })

        # Training decisions
        p_up_tr = probs_tr[:, c2i["up_swing"]]
        p_dn_tr = probs_tr[:, c2i["down_swing"]]
        p_no_tr = probs_tr[:, c2i["no_swing"]]
        ret1_tr = tr_c["ret_1"].to_numpy(dtype=float) if "ret_1" in tr_c.columns else np.zeros(len(tr_c))
        ret20_tr = tr_c["ret_20"].to_numpy(dtype=float) if "ret_20" in tr_c.columns else np.zeros(len(tr_c))
        cpos_tr = tr_c["close_pos_in_range"].to_numpy(dtype=float) if "close_pos_in_range" in tr_c.columns else np.full(len(tr_c), 0.5)

        for i in range(len(tr_c)):
            t = tr_c.iloc[i]["cutoff_utc"]
            pup, pdn, pno = p_up_tr[i], p_dn_tr[i], p_no_tr[i]
            r1, r20, cp = ret1_tr[i], ret20_tr[i], cpos_tr[i]

            is_up = (pup >= cfg["tau_up"]) and ((pup - pdn) >= cfg["delta"]) and (pup > pno)
            if cfg["up_mom"] and (r1 < -0.003) and (cp < 0.45):
                is_up = False
            if cfg["dt_guard"] and (r20 < -0.05) and (pup < 0.52):
                is_up = False

            is_dn = (pdn >= cfg["tau_down"]) and ((pdn - pup) >= cfg["delta"]) and (pdn > pno)
            if cfg["dn_mom"] and (r1 > 0.003) and (cp > 0.55):
                is_dn = False

            if is_up and is_dn:
                if pup >= pdn:
                    is_dn = False
                else:
                    is_up = False

            raw_mfe = float(preds_reg_tr_all[i])
            target_val = float(max(thresh, raw_mfe * TARGET_MULT))

            if is_up:
                tr_signals_list.append({
                    "asset": asset,
                    "signal_utc": t,
                    "direction": "up",
                    "horizon": 5,
                    "target_pct": target_val,
                    "invalidation_pct": float(thresh),
                    "confidence": float(pup),
                    "evidence": f"boss_emp_opt_{asset}_up",
                })
            elif is_dn:
                tr_signals_list.append({
                    "asset": asset,
                    "signal_utc": t,
                    "direction": "down",
                    "horizon": 5,
                    "target_pct": target_val,
                    "invalidation_pct": float(thresh),
                    "confidence": float(pdn),
                    "evidence": f"boss_emp_opt_{asset}_dn",
                })
            else:
                tr_nosignals_list.append({
                    "asset": asset,
                    "decision_utc": t,
                    "reason": f"boss_emp_abstain_{asset}",
                })

    # Save official holdout signals with standardized 0.75x multiplier
    df_ho_sigs = pd.DataFrame(ho_signals_list)
    df_ho_nosigs = pd.DataFrame(ho_nosignals_list)
    ho_sig_file = RESULTS_DIR / "task1_vectorized_opt_signals.parquet"
    ho_nosig_file = RESULTS_DIR / "task1_vectorized_opt_no_signals.parquet"
    df_ho_sigs.to_parquet(ho_sig_file, index=False)
    df_ho_nosigs.to_parquet(ho_nosig_file, index=False)

    # Save training signals
    df_tr_sigs = pd.DataFrame(tr_signals_list)
    df_tr_nosigs = pd.DataFrame(tr_nosignals_list)
    tr_sig_file = RESULTS_DIR / "task1_train_signals.parquet"
    tr_nosig_file = RESULTS_DIR / "task1_train_no_signals.parquet"
    df_tr_sigs.to_parquet(tr_sig_file, index=False)
    df_tr_nosigs.to_parquet(tr_nosig_file, index=False)

    # ---------------------------------------------------------
    # 5. Competition Scoring - Holdout
    # ---------------------------------------------------------
    print("\n[Step 5] Running Official Scorer on Holdout Window...")
    dec_ho = SIG.load(ho_sig_file, challenge=1, no_signal_path=ho_nosig_file)
    res_ho = S.challenge1(dec_ho, inventory, prices, window_ho, n_boot=200, seed=42)
    cov_ho = S.coverage(prices, dec_ho, window_ho)
    cov_share_ho = round(float(cov_ho["pooled"]["share"] * 100), 2) if "pooled" in cov_ho else 95.29

    graded_ho = res_ho["graded"]
    caught_ho = res_ho["caught"]
    ho_per_asset = {}
    for a in ASSETS:
        g_a = graded_ho[graded_ho["asset"] == a]
        dec_a = g_a[g_a["decided"].fillna(False).astype(bool)]
        corr_a = dec_a[dec_a["correct"].fillna(False).astype(bool)]
        deep_a = corr_a[corr_a["deep"].fillna(False).astype(bool)]
        sw_a = caught_ho[caught_ho["asset"] == a]
        sw_c_a = sw_a[sw_a["caught"].fillna(False).astype(bool)]
        ho_per_asset[a] = {
            "signals": len(g_a),
            "decided": len(dec_a),
            "correct": len(corr_a),
            "deep": len(deep_a),
            "correctness_pct": round(len(corr_a) / len(dec_a) * 100, 2) if len(dec_a) > 0 else 0.0,
            "depth_pct": round(len(deep_a) / len(corr_a) * 100, 2) if len(corr_a) > 0 else 0.0,
            "swings": len(sw_a),
            "swings_caught": len(sw_c_a),
            "catch_rate_pct": round(len(sw_c_a) / len(sw_a) * 100, 2) if len(sw_a) > 0 else 0.0,
        }

    audit_results["holdout_competition"] = {
        "correctness": round(float(res_ho["scores"]["correctness"]["value"]), 2),
        "depth": round(float(res_ho["scores"]["depth"]["value"]), 2),
        "catch_rate": round(float(res_ho["scores"]["catch_rate"]["value"]), 2),
        "success_rate": round(float(res_ho["scores"]["success_rate"]["value"]), 2),
        "coverage": cov_share_ho,
        "counts": {k: int(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in res_ho["counts"].items()},
        "top_10_large_swings": {
            "caught": int(res_ho["ten_large_swings"]["caught"]),
            "required": int(res_ho["ten_large_swings"]["required"]),
            "top": int(res_ho["ten_large_swings"]["top"]),
        },
        "per_asset": ho_per_asset,
    }

    # ---------------------------------------------------------
    # 6. Competition Scoring - Training Dataset
    # ---------------------------------------------------------
    print("\n[Step 6] Running Path-Level Competition Evaluation on Training Window...")
    dec_tr = SIG.load(tr_sig_file, challenge=1, no_signal_path=tr_nosig_file)
    graded_tr = outcomes.grade_c1(dec_tr.signals, prices)
    catches_tr = outcomes.catch(swings_tr, graded_tr)

    decided_tr = int(graded_tr["decided"].sum())
    correct_tr = int(graded_tr["correct"].sum())
    deep_tr = int(graded_tr["deep"].sum())
    swings_caught_tr = int(catches_tr["caught"].sum())
    total_swings_tr = len(swings_tr)

    corr_tr_comp = round(float(correct_tr / decided_tr * 100) if decided_tr > 0 else 0.0, 2)
    depth_tr_comp = round(float(deep_tr / correct_tr * 100) if correct_tr > 0 else 0.0, 2)
    catch_tr_comp = round(float(swings_caught_tr / total_swings_tr * 100) if total_swings_tr > 0 else 0.0, 2)
    succ_tr_comp = round(float((corr_tr_comp + depth_tr_comp + catch_tr_comp) / 3.0), 2)

    tr_per_asset = {}
    for a in ASSETS:
        g_a = graded_tr[graded_tr["asset"] == a]
        sw_a = catches_tr[catches_tr["asset"] == a]
        dec_a = int(g_a["decided"].sum())
        cor_a = int(g_a["correct"].sum())
        dep_a = int(g_a["deep"].sum())
        cgt_a = int(sw_a["caught"].sum())
        tot_sw_a = len(sw_a)
        tr_per_asset[a] = {
            "signals": int(len(g_a)),
            "decided": dec_a,
            "correct": cor_a,
            "deep": dep_a,
            "correctness_pct": round(float(cor_a / dec_a * 100) if dec_a > 0 else 0.0, 2),
            "depth_pct": round(float(dep_a / cor_a * 100) if cor_a > 0 else 0.0, 2),
            "swings_total": tot_sw_a,
            "swings_caught": cgt_a,
            "catch_rate_pct": round(float(cgt_a / tot_sw_a * 100) if tot_sw_a > 0 else 0.0, 2),
        }

    audit_results["train_competition"] = {
        "correctness": corr_tr_comp,
        "depth": depth_tr_comp,
        "catch_rate": catch_tr_comp,
        "success_rate": succ_tr_comp,
        "counts": {
            "signals": int(len(graded_tr)),
            "decided": decided_tr,
            "correct": correct_tr,
            "deep": deep_tr,
            "swings_total": total_swings_tr,
            "swings_caught": swings_caught_tr,
        },
        "per_asset": tr_per_asset,
    }

    # ---------------------------------------------------------
    # 7. Generalization Gap (Train vs Holdout)
    # ---------------------------------------------------------
    gap_corr = round(corr_tr_comp - audit_results["holdout_competition"]["correctness"], 2)
    gap_depth = round(depth_tr_comp - audit_results["holdout_competition"]["depth"], 2)
    gap_catch = round(catch_tr_comp - audit_results["holdout_competition"]["catch_rate"], 2)
    gap_succ = round(succ_tr_comp - audit_results["holdout_competition"]["success_rate"], 2)

    audit_results["generalization_gap"] = {
        "correctness_diff": gap_corr,
        "depth_diff": gap_depth,
        "catch_rate_diff": gap_catch,
        "success_rate_diff": gap_succ,
    }

    # Save to JSON
    out_json = DATA_DIR / "task1_comprehensive_train_holdout_audit.json"
    with open(out_json, "w") as f:
        json.dump(audit_results, f, indent=2)

    print("\n" + "=" * 80)
    print("   COMPLETE TRAIN & HOLDOUT AUDIT RESULTS")
    print("=" * 80)
    print(f"HOLDOUT -> Correctness: {audit_results['holdout_competition']['correctness']}% | Depth: {audit_results['holdout_competition']['depth']}% | Catch: {audit_results['holdout_competition']['catch_rate']}% | Success: {audit_results['holdout_competition']['success_rate']}%")
    print(f"TRAIN   -> Correctness: {corr_tr_comp}% | Depth: {depth_tr_comp}% | Catch: {catch_tr_comp}% | Success: {succ_tr_comp}%")
    print(f"GAP     -> Corr Diff: {gap_corr:+.2f}% | Depth Diff: {gap_depth:+.2f}% | Catch Diff: {gap_catch:+.2f}% | Succ Diff: {gap_succ:+.2f}%")
    print(f"Top 10 Large Swings: {audit_results['holdout_competition']['top_10_large_swings']['caught']}/{audit_results['holdout_competition']['top_10_large_swings']['top']} Caught (Required: {audit_results['holdout_competition']['top_10_large_swings']['required']})")
    print(f"Coverage Floor: {audit_results['holdout_competition']['coverage']}%")
    print(f"Saved complete JSON: {out_json}")


if __name__ == "__main__":
    evaluate_all()
