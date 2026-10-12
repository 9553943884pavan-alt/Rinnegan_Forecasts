"""Evaluation Script for Task 1 Classifier with Enriched Features.

Compares:
1. Original 16-feature baseline model
2. Enriched 27-feature model (FinBERT sentiment + News activity + Advanced Volume/COT/Vol features)

Evaluates:
- Standard ML Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, Per-Asset Accuracy)
- Feature Importance (Permutation Importance on Holdout)
- Official Competition Metrics via swing_dataset.scoring2 (Correctness, Depth, Catch Rate)
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

# Add kit to python path
KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from swing_dataset.scoring2 import THRESHOLD_PCT

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
DATASET_PATH = DATA_DIR / "task1_classifier_dataset.parquet"


def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df)} total rows. Columns: {list(df.columns)}")

    # 1. Feature sets
    baseline_features = [
        "z_1", "z_5", "z_20", "stretch_20", "log_rv_20",
        "ret_1", "ret_5", "ret_20", "rv_20", "sigma_c",
        "oi_change_pct", "range_ratio", "log_volume",
        "cot_net_smart_money", "usdinr_ret_5d", "oi_momentum_interaction",
    ]

    new_features = [
        # News & FinBERT NLP features
        "is_news_exist",
        "news_count_24h",
        "finbert_sentiment_mean_24h",
        "finbert_sentiment_sum_24h",
        # Technical & Market Structure features
        "range_expansion_ratio",
        "vol_surge_10d",
        "oi_growth_5d",
        "cot_weekly_change",
        "cot_crowded_trade",
        "ret_trend_agreement",
        "volatility_regime",
    ]

    all_features = baseline_features + new_features
    print(f"\nFeature Set Configuration:")
    print(f"- Baseline Features ({len(baseline_features)}): {baseline_features}")
    print(f"- New Enriched Features ({len(new_features)}): {new_features}")
    print(f"- Total Features: {len(all_features)}")

    # Check for any missing values in features
    null_counts = df[all_features].isnull().sum()
    if null_counts.sum() > 0:
        print("\nWARNING: Found null values in features:")
        print(null_counts[null_counts > 0])
    else:
        print("\nAll 27 features are 100% non-null.")

    # 2. Rules Section 5 Time-based Holdout Split
    cutoff_split = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
    train_mask = df["cutoff_utc"] < cutoff_split
    train_df = df[train_mask].copy()
    holdout_df = df[~train_mask].copy()

    print(f"\nDataset Splits:")
    print(f"- Train Set (< 2026-06-01): {len(train_df)} rows")
    print(f"- Holdout Set (>= 2026-06-01): {len(holdout_df)} rows")

    # Evaluate both models: Baseline (16 features) vs Enriched (27 features)
    for model_name, feature_list in [("Baseline (16 Features)", baseline_features), ("Enriched (27 Features)", all_features)]:
        print("\n" + "=" * 60)
        print(f"EVALUATING MODEL: {model_name}")
        print("=" * 60)

        X_train, y_train = train_df[feature_list], train_df["label"]
        X_holdout, y_holdout = holdout_df[feature_list], holdout_df["label"]

        clf = HistGradientBoostingClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_holdout)
        y_prob = clf.predict_proba(X_holdout)
        acc = accuracy_score(y_holdout, y_pred)

        print(f"\nOverall Holdout ML Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_holdout, y_pred, digits=4))

        print("Confusion Matrix:")
        labels_order = ["down_swing", "no_swing", "up_swing"]
        cm = confusion_matrix(y_holdout, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in labels_order], columns=[f"Pred_{c}" for c in labels_order])
        print(cm_df)

        # Per-Asset Accuracy
        print("\nPer-Asset Accuracy:")
        for asset in ["gold", "silver", "copper", "crude"]:
            mask_a = holdout_df["asset"] == asset
            acc_a = accuracy_score(y_holdout[mask_a], y_pred[mask_a])
            print(f"- {asset.capitalize():<8}: {acc_a*100:.2f}% ({mask_a.sum()} rows)")

        # News Availability Window Analysis (July 16 - Sept 25)
        news_mask = holdout_df["is_news_exist"] == 1
        if news_mask.sum() > 0:
            acc_news = accuracy_score(y_holdout[news_mask], y_pred[news_mask])
            acc_pre_news = accuracy_score(y_holdout[~news_mask], y_pred[~news_mask])
            print(f"\nNews Window Sub-Breakdown:")
            print(f"- With News Exist (July 16+): {acc_news*100:.2f}% ({news_mask.sum()} sessions)")
            print(f"- Pre-News (June 1 - July 15): {acc_pre_news*100:.2f}% ({(~news_mask).sum()} sessions)")

        # Feature Importance for Enriched Model
        if model_name.startswith("Enriched"):
            print("\nComputing Permutation Feature Importance on Holdout...")
            perm = permutation_importance(clf, X_holdout, y_holdout, n_repeats=10, random_state=42)
            fi_df = pd.DataFrame({
                "feature": feature_list,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False)
            print("\nTop 15 Features by Holdout Permutation Importance:")
            print(fi_df.head(15).to_string(index=False))

        # Format into official submission format
        signals_records = []
        no_signals_records = []

        class_indices = {c: i for i, c in enumerate(clf.classes_)}

        for idx, row in holdout_df.reset_index(drop=True).iterrows():
            pred_label = y_pred[idx]
            asset = row["asset"]
            cutoff_utc = row["cutoff_utc"]
            thresh = THRESHOLD_PCT[asset]

            if pred_label == "no_swing":
                no_signals_records.append({
                    "asset": asset,
                    "decision_utc": cutoff_utc,
                    "reason": "classifier_predicted_no_swing",
                })
            else:
                direction = "up" if pred_label == "up_swing" else "down"
                prob_idx = class_indices[pred_label]
                conf = float(y_prob[idx][prob_idx])

                signals_records.append({
                    "asset": asset,
                    "signal_utc": cutoff_utc,
                    "direction": direction,
                    "horizon": 5,
                    "target_pct": float(thresh),
                    "invalidation_pct": float(thresh),
                    "confidence": conf,
                    "evidence": f"hist_gb_{model_name.split()[0].lower()}",
                })

        sig_df = pd.DataFrame(signals_records)
        no_sig_df = pd.DataFrame(no_signals_records)

        tag = "baseline" if model_name.startswith("Baseline") else "enriched"
        sig_file = DATA_DIR / f"task1_{tag}_signals.parquet"
        no_sig_file = DATA_DIR / f"task1_{tag}_no_signals.parquet"

        sig_df.to_parquet(sig_file, index=False)
        no_sig_df.to_parquet(no_sig_file, index=False)
        print(f"\nWrote {len(sig_df)} signals to {sig_file}")
        print(f"Wrote {len(no_sig_df)} no_signals to {no_sig_file}")

        # Run Official Scorer
        print(f"\nRunning Official Competition Scorer for {model_name}...")
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
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(KIT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode == 0:
            print(res.stdout)
        else:
            print(f"Scorer error (code {res.returncode}):")
            print(res.stderr)
            print(res.stdout)


if __name__ == "__main__":
    main()
