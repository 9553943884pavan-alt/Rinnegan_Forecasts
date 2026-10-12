"""Test Confidence Thresholding to Maximize Competition Correctness.

In Rules Section 5:
- Coverage floor is 90% across (signals + no_signals).
- Signals that fall below a confidence threshold can be converted to no_signal.
- This tests confidence thresholds [0.34, 0.38, 0.40, 0.42, 0.45, 0.50] to see how Correctness changes.
"""

from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from swing_dataset.scoring2 import THRESHOLD_PCT

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
DATASET_PATH = DATA_DIR / "task1_classifier_dataset.parquet"

df = pd.read_parquet(DATASET_PATH)

baseline_features = [
    "z_1", "z_5", "z_20", "stretch_20", "log_rv_20",
    "ret_1", "ret_5", "ret_20", "rv_20", "sigma_c",
    "oi_change_pct", "range_ratio", "log_volume",
    "cot_net_smart_money", "usdinr_ret_5d", "oi_momentum_interaction",
]

new_features = [
    "is_news_exist", "news_count_24h", "finbert_sentiment_mean_24h", "finbert_sentiment_sum_24h",
    "range_expansion_ratio", "vol_surge_10d", "oi_growth_5d", "cot_weekly_change",
    "cot_crowded_trade", "ret_trend_agreement", "volatility_regime",
]

all_features = baseline_features + new_features

cutoff_split = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
train_df = df[df["cutoff_utc"] < cutoff_split].copy()
holdout_df = df[df["cutoff_utc"] >= cutoff_split].copy()

# Fit enriched model
clf = HistGradientBoostingClassifier(random_state=42)
clf.fit(train_df[all_features], train_df["label"])

y_prob = clf.predict_proba(holdout_df[all_features])
classes = list(clf.classes_)
down_idx = classes.index("down_swing")
up_idx = classes.index("up_swing")
no_idx = classes.index("no_swing")

env = dict(os.environ)
env["PYTHONPATH"] = str(KIT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

print("Evaluating confidence threshold sweeps for Enriched Model...")
for min_conf in [0.34, 0.38, 0.40, 0.42, 0.45]:
    sig_records = []
    no_sig_records = []

    for idx, row in holdout_df.reset_index(drop=True).iterrows():
        p_up = y_prob[idx][up_idx]
        p_down = y_prob[idx][down_idx]
        p_no = y_prob[idx][no_idx]
        asset = row["asset"]
        cutoff = row["cutoff_utc"]
        thresh = THRESHOLD_PCT[asset]

        # Check if either up or down beats both no_swing and the confidence hurdle
        if p_up > p_no and p_up > p_down and p_up >= min_conf:
            sig_records.append({
                "asset": asset, "signal_utc": cutoff, "direction": "up",
                "horizon": 5, "target_pct": float(thresh), "invalidation_pct": float(thresh),
                "confidence": float(p_up), "evidence": "enriched_thresh",
            })
        elif p_down > p_no and p_down > p_up and p_down >= min_conf:
            sig_records.append({
                "asset": asset, "signal_utc": cutoff, "direction": "down",
                "horizon": 5, "target_pct": float(thresh), "invalidation_pct": float(thresh),
                "confidence": float(p_down), "evidence": "enriched_thresh",
            })
        else:
            no_sig_records.append({
                "asset": asset, "decision_utc": cutoff, "reason": "below_confidence_hurdle",
            })

    sig_file = DATA_DIR / f"test_sig_{min_conf}.parquet"
    no_sig_file = DATA_DIR / f"test_nosig_{min_conf}.parquet"
    pd.DataFrame(sig_records).to_parquet(sig_file, index=False)
    pd.DataFrame(no_sig_records).to_parquet(no_sig_file, index=False)

    cmd = [
        sys.executable, "-m", "swing_dataset.scoring2", "evaluate",
        "--challenge", "1", "--period", "holdout",
        "--signals", str(sig_file), "--no-signals", str(no_sig_file),
        "--mcx-daily", str(PACK_BASE / "mcx" / "mcx_daily.parquet"),
        "--designations", str(PACK_BASE / "labels" / "designations.parquet"),
        "--mcx-sessions", str(PACK_BASE / "mcx" / "mcx_sessions.parquet"),
        "--swings", str(PACK_BASE / "swings" / "swings.parquet"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    import json
    try:
        report = json.loads(res.stdout)
        corr = report["scores"]["correctness"]["value"]
        depth = report["scores"]["depth"]["value"]
        n_sig = report["counts"]["signals_in_window"]
        n_dec = report["counts"]["decided"]
        n_corr = report["counts"]["correct"]
        print(f"Conf >= {min_conf:.2f} -> Signals: {n_sig}, Decided: {n_dec}, Correct: {n_corr}, Correctness: {corr:.2f}%, Depth: {depth:.2f}%")
    except Exception as e:
        print(f"Conf >= {min_conf:.2f} -> Error: {e}\n{res.stderr}")
