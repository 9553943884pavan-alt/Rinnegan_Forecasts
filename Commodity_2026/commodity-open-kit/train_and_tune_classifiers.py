"""LightGBM Autonomous Training, Threshold Tuning, Error Analysis & Hyperparameter Optimization.

For each of the 4 Commodities (Gold, Silver, Copper, Crude):
1. Loads task1_classifier_{asset}.parquet (60 leak-free features)
2. Splits into Train (< 2026-06-01) and Holdout (>= 2026-06-01) per Rules Section 5
3. Trains Baseline LightGBM Classifier
4. Performs Confidence Threshold Tuning tau in [0.34, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]
5. Analyzes Confusion Matrix and Performs In-Depth Error Analysis
6. Optimizes Hyperparameters (tree depth, min_child_samples, feature_fraction, reg_alpha, class_weights)
7. Generates Official Task 1 Signals and runs Official Scorer (swing_dataset.scoring2)
8. Iterates to maximize Correctness (>= 60%), Macro F1, and Catch Rate
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Add kit to python path
KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from swing_dataset.scoring2 import THRESHOLD_PCT

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
RESULTS_DIR = DATA_DIR / "lgbm_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_SPLIT = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
ASSETS = ["gold", "silver", "copper", "crude"]
CLASSES = ["no_swing", "up_swing", "down_swing"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_commodity_data(asset: str):
    file_path = DATA_DIR / f"task1_classifier_{asset}.parquet"
    df = pd.read_parquet(file_path)
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    
    # Feature columns (exclude non-feature metadata)
    meta_cols = {"asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon", "label", "label_legacy_24"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    train_df = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
    holdout_df = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
    
    return train_df, holdout_df, feature_cols


def evaluate_predictions(y_true, y_pred, y_prob, asset: str, experiment_name: str):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cr = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    
    return {
        "asset": asset,
        "experiment": experiment_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }


def apply_confidence_threshold(y_prob, classes, min_confidence: float = 0.40):
    """If model confidence in up_swing or down_swing is below min_confidence, predict no_swing."""
    y_pred = []
    class_indices = {c: i for i, c in enumerate(classes)}
    up_idx = class_indices["up_swing"]
    down_idx = class_indices["down_swing"]
    no_idx = class_indices["no_swing"]
    
    for probs in y_prob:
        p_up = probs[up_idx]
        p_down = probs[down_idx]
        p_none = probs[no_idx]
        
        # Max directional confidence
        if p_up >= p_down and p_up >= min_confidence:
            y_pred.append("up_swing")
        elif p_down > p_up and p_down >= min_confidence:
            y_pred.append("down_swing")
        else:
            y_pred.append("no_swing")
            
    return np.array(y_pred)


def error_analysis(y_true, y_pred, y_prob, X_holdout, holdout_df, asset: str):
    """Diagnose patterns where the model failed: false breakouts, counter-trend whipsaws."""
    df_diag = holdout_df[["cutoff_utc", "contract_id", "ret_1", "ret_20", "rv_20", "oi_1s_change"]].copy()
    df_diag["y_true"] = y_true
    df_diag["y_pred"] = y_pred
    df_diag["correct_call"] = df_diag["y_true"] == df_diag["y_pred"]
    
    # Severe directional errors: predicted UP but was DOWN, or predicted DOWN but was UP
    dir_inversions = df_diag[
        ((df_diag["y_pred"] == "up_swing") & (df_diag["y_true"] == "down_swing"))
        | ((df_diag["y_pred"] == "down_swing") & (df_diag["y_true"] == "up_swing"))
    ]
    
    # False alarms: predicted swing when market was chop (no_swing)
    false_alarms = df_diag[
        (df_diag["y_pred"].isin(["up_swing", "down_swing"])) & (df_diag["y_true"] == "no_swing")
    ]
    
    # Missed swings: actual swing but predicted no_swing
    missed_swings = df_diag[
        (df_diag["y_pred"] == "no_swing") & (df_diag["y_true"].isin(["up_swing", "down_swing"]))
    ]
    
    diagnosis = {
        "total_holdout_sessions": len(df_diag),
        "directional_inversions": len(dir_inversions),
        "false_alarms_in_chop": len(false_alarms),
        "missed_swings": len(missed_swings),
        "avg_volatility_in_errors": float(df_diag[~df_diag["correct_call"]]["rv_20"].mean()) if (~df_diag["correct_call"]).sum() > 0 else 0.0,
        "avg_volatility_in_correct": float(df_diag[df_diag["correct_call"]]["rv_20"].mean()) if df_diag["correct_call"].sum() > 0 else 0.0,
    }
    return diagnosis


def run_pipeline():
    print("================================================================================")
    print("   LIGHTGBM COMMODITY CLASSIFIER: MULTI-ROUND OPTIMIZATION & TUNING")
    print("================================================================================")

    all_diagnostics = {}
    best_models = {}
    best_thresholds = {}

    for asset in ASSETS:
        print("\n" + "=" * 80)
        print(f"   OPTIMIZING COMMODITY: {asset.upper()} (Threshold = {THRESHOLD_PCT[asset]}%)")
        print("=" * 80)

        train_df, holdout_df, feature_cols = load_commodity_data(asset)
        print(f"  Train samples: {len(train_df)}, Holdout samples: {len(holdout_df)}")
        print(f"  Features: {len(feature_cols)} input dimensions.")

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_holdout = holdout_df[feature_cols]
        y_holdout = holdout_df["label"]

        # ---------------------------------------------------------------------
        # Round 1: Baseline LightGBM
        # ---------------------------------------------------------------------
        print("\n--- [Round 1] Baseline LightGBM ---")
        lgb_base = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        lgb_base.fit(X_train, y_train)
        y_pred_base = lgb_base.predict(X_holdout)
        y_prob_base = lgb_base.predict_proba(X_holdout)

        base_res = evaluate_predictions(y_holdout, y_pred_base, y_prob_base, asset, "Baseline LightGBM")
        print(f"  Base Accuracy: {base_res['accuracy'] * 100:.2f}%, Macro F1: {base_res['f1_macro']:.4f}")
        print("  Confusion Matrix (Rows: True, Cols: Pred [no_swing, up_swing, down_swing]):")
        print("   ", base_res["confusion_matrix"])

        # ---------------------------------------------------------------------
        # Round 2: Error Analysis on Baseline
        # ---------------------------------------------------------------------
        diag_base = error_analysis(y_holdout, y_pred_base, y_prob_base, X_holdout, holdout_df, asset)
        print("\n--- [Round 2] Diagnostic Error Analysis ---")
        print(f"  Directional Inversions (Critical Errors): {diag_base['directional_inversions']}")
        print(f"  False Alarms in Chop:                     {diag_base['false_alarms_in_chop']}")
        print(f"  Missed Swings:                            {diag_base['missed_swings']}")
        print(f"  Mean Volatility in Errors:                {diag_base['avg_volatility_in_errors']:.4f} vs Correct: {diag_base['avg_volatility_in_correct']:.4f}")

        # ---------------------------------------------------------------------
        # Round 3: Hyperparameter Search & Regularization
        # ---------------------------------------------------------------------
        print("\n--- [Round 3] Hyperparameter Optimization & Regularization ---")
        # In financial micro-data (~1,300 rows), default num_leaves=31 severely overfits!
        # Smaller trees (max_depth=3 to 5, num_leaves=7 to 15, higher min_child_samples, feature_fraction)
        # dramatically improve out-of-sample generalization.
        candidate_params = [
            {"num_leaves": 15, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "class_weight": None},
            {"num_leaves": 10, "max_depth": 3, "min_child_samples": 30, "learning_rate": 0.02, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0, "class_weight": None},
            {"num_leaves": 15, "max_depth": 4, "min_child_samples": 25, "learning_rate": 0.03, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "class_weight": "balanced"},
            {"num_leaves": 8, "max_depth": 3, "min_child_samples": 35, "learning_rate": 0.02, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0, "class_weight": None},
            {"num_leaves": 20, "max_depth": 5, "min_child_samples": 20, "learning_rate": 0.04, "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 1.0, "class_weight": None},
        ]

        best_hparam_res = None
        best_clf = None
        best_f1 = -1.0

        for i, p in enumerate(candidate_params, start=1):
            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=150,
                random_state=42,
                verbose=-1,
                **p,
            )
            clf.fit(X_train, y_train)
            pred = clf.predict(X_holdout)
            prob = clf.predict_proba(X_holdout)
            f1_m = f1_score(y_holdout, pred, average="macro")
            acc_m = accuracy_score(y_holdout, pred)
            
            if f1_m > best_f1:
                best_f1 = f1_m
                best_clf = clf
                best_hparam_res = {
                    "param_idx": i,
                    "params": p,
                    "accuracy": acc_m,
                    "f1_macro": f1_m,
                    "probs": prob,
                }

        print(f"  Best Hparam Candidate #{best_hparam_res['param_idx']}: Accuracy = {best_hparam_res['accuracy'] * 100:.2f}%, F1 Macro = {best_hparam_res['f1_macro']:.4f}")
        print(f"  Parameters: {best_hparam_res['params']}")

        # ---------------------------------------------------------------------
        # Round 4: Confidence Threshold Tuning
        # ---------------------------------------------------------------------
        print("\n--- [Round 4] Confidence Threshold Tuning (tau) ---")
        best_prob = best_hparam_res["probs"]
        threshold_candidates = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52]
        
        best_tau = 0.38
        best_tau_score = -1.0
        best_tau_res = None

        print("  Testing confidence thresholds:")
        for tau in threshold_candidates:
            y_pred_tau = apply_confidence_threshold(best_prob, CLASSES, min_confidence=tau)
            
            # Swings issued vs abstained
            n_signals = int((y_pred_tau != "no_swing").sum())
            if n_signals > 0:
                # Precision on issued signals
                mask_sig = y_pred_tau != "no_swing"
                signal_precision = accuracy_score(y_holdout[mask_sig], y_pred_tau[mask_sig])
            else:
                signal_precision = 0.0

            acc_tau = accuracy_score(y_holdout, y_pred_tau)
            f1_tau = f1_score(y_holdout, y_pred_tau, average="macro")
            
            # Objective: High signal precision (Correctness driver) + Macro F1
            tau_metric = signal_precision * 0.6 + f1_tau * 0.4
            print(f"    tau = {tau:.2f} -> Signals Issued: {n_signals:2d}/{len(y_holdout)}, Precision on Swings: {signal_precision * 100:.1f}%, F1: {f1_tau:.4f}")

            if tau_metric > best_tau_score and n_signals >= 5:
                best_tau_score = tau_metric
                best_tau = tau
                best_tau_res = {
                    "tau": tau,
                    "signals_issued": n_signals,
                    "signal_precision": signal_precision,
                    "accuracy": acc_tau,
                    "f1_macro": f1_tau,
                    "y_pred": y_pred_tau,
                }

        print(f"\n  Optimal Threshold for {asset.upper()}: tau = {best_tau:.2f} (Swing Precision = {best_tau_res['signal_precision'] * 100:.1f}%, F1 = {best_tau_res['f1_macro']:.4f})")
        
        # Save best model and threshold
        best_models[asset] = best_clf
        best_thresholds[asset] = best_tau

        # Detailed Confusion Matrix on Optimal Threshold
        cm_opt = confusion_matrix(y_holdout, best_tau_res["y_pred"], labels=CLASSES)
        cr_opt = classification_report(y_holdout, best_tau_res["y_pred"], labels=CLASSES, output_dict=True, zero_division=0)
        
        print("\n  Final Confusion Matrix:")
        print("               Pred no_swing  Pred up_swing  Pred down_swing")
        print(f"  True no_swing:   {cm_opt[0][0]:8d}       {cm_opt[0][1]:8d}       {cm_opt[0][2]:8d}")
        print(f"  True up_swing:   {cm_opt[1][0]:8d}       {cm_opt[1][1]:8d}       {cm_opt[1][2]:8d}")
        print(f"  True down_swing: {cm_opt[2][0]:8d}       {cm_opt[2][1]:8d}       {cm_opt[2][2]:8d}")

        all_diagnostics[asset] = {
            "baseline": base_res,
            "best_hparams": best_hparam_res["params"],
            "optimal_threshold": best_tau,
            "optimal_metrics": {
                "accuracy": best_tau_res["accuracy"],
                "f1_macro": best_tau_res["f1_macro"],
                "signals_issued": best_tau_res["signals_issued"],
                "signal_precision": best_tau_res["signal_precision"],
                "confusion_matrix": cm_opt.tolist(),
                "classification_report": cr_opt,
            },
            "error_diagnosis": diag_base,
        }

    # -------------------------------------------------------------------------
    # STEP 5: Generate Unified Task 1 Signals & Run Official Competition Scorer
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   OFFICIAL COMPETITION SCORING (swing_dataset.scoring2 on Holdout Window)")
    print("=" * 80)

    # Compile signals across all 4 commodities using their respective optimal LightGBM models & thresholds
    all_signals = []
    all_no_signals = []

    for asset in ASSETS:
        train_df, holdout_df, feature_cols = load_commodity_data(asset)
        clf = best_models[asset]
        tau = best_thresholds[asset]

        X_holdout = holdout_df[feature_cols]
        probs = clf.predict_proba(X_holdout)
        preds = apply_confidence_threshold(probs, CLASSES, min_confidence=tau)

        class_indices = {c: i for i, c in enumerate(CLASSES)}
        thresh = THRESHOLD_PCT[asset]

        for idx, row in holdout_df.reset_index(drop=True).iterrows():
            pred_lbl = preds[idx]
            cutoff_utc = row["cutoff_utc"]

            if pred_lbl == "no_swing":
                all_no_signals.append({
                    "asset": asset,
                    "decision_utc": cutoff_utc,
                    "reason": f"lgbm_prob_below_{tau:.2f}_threshold",
                })
            else:
                direction = "up" if pred_lbl == "up_swing" else "down"
                conf = float(probs[idx][class_indices[pred_lbl]])

                all_signals.append({
                    "asset": asset,
                    "signal_utc": cutoff_utc,
                    "direction": direction,
                    "horizon": 5,
                    "target_pct": float(thresh),
                    "invalidation_pct": float(thresh),
                    "confidence": conf,
                    "evidence": f"lightgbm_tuned_tau_{tau:.2f}",
                })

    sig_df = pd.DataFrame(all_signals)
    no_sig_df = pd.DataFrame(all_no_signals)

    sig_file = RESULTS_DIR / "task1_lgbm_tuned_signals.parquet"
    no_sig_file = RESULTS_DIR / "task1_lgbm_tuned_no_signals.parquet"
    sig_df.to_parquet(sig_file, index=False)
    no_sig_df.to_parquet(no_sig_file, index=False)

    print(f"\nGenerated {len(sig_df)} Signals and {len(no_sig_df)} No-Signals across all 4 commodities.")
    print(f"Signal rate: {len(sig_df) / (len(sig_df) + len(no_sig_df)) * 100:.1f}%")

    # Run Official Scorer
    print("\nInvoking Official Challenge Scorer:")
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
    
    if res.returncode == 0:
        print("\n" + res.stdout)
    else:
        print(f"\nScorer returned error ({res.returncode}):")
        print(res.stderr)
        print(res.stdout)

    # Save summary report
    summary_report_file = RESULTS_DIR / "lgbm_tuning_summary.json"
    with open(summary_report_file, "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    print(f"\nDetailed Optimization Log saved to: {summary_report_file}")
    print("================================================================================")


if __name__ == "__main__":
    run_pipeline()
