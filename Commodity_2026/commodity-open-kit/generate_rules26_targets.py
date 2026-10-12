"""Re-label Dataset Targets with Exact Rules 2.6 Ground Truth Outcomes.

Eliminates the ~30% label mismatch between legacy Rules 2.4 (1-sigma barrier)
and Rules 2.6 (official fixed asset threshold barrier: Gold 1.5%, Silver 2.0%, Copper 1.5%, Crude 3.0%).
Uses swing_dataset.scoring2.prices.PriceSeries.evaluate_path to simulate the exact first-touch race.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
from swing_dataset.scoring2.prices import PriceSeries
from swing_dataset.scoring2 import THRESHOLD_PCT

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")

ASSETS = ["gold", "silver", "copper", "crude"]


def generate_exact_26_targets():
    print("================================================================================")
    print("   GENERATING EXACT RULES 2.6 GROUND TRUTH LABELS (ZERO LABEL MISMATCH)")
    print("================================================================================")

    print("\nLoading MCX daily prices, designations and session calendars...")
    daily = pd.read_parquet(PACK_BASE / "mcx" / "mcx_daily.parquet")
    designations = pd.read_parquet(PACK_BASE / "labels" / "designations.parquet")
    sessions = pd.read_parquet(PACK_BASE / "mcx" / "mcx_sessions.parquet")
    prices = PriceSeries(daily, designations, sessions)
    print("PriceSeries engine initialized.")

    for asset in ASSETS:
        t0 = time.time()
        clf_file = DATA_DIR / f"task1_classifier_{asset}.parquet"
        reg_file = DATA_DIR / f"task1_regressor_{asset}.parquet"
        df_clf = pd.read_parquet(clf_file)
        df_reg = pd.read_parquet(reg_file)

        thresh = float(THRESHOLD_PCT[asset])
        print(f"\nProcessing {asset.upper()} ({len(df_clf)} rows, fixed threshold = {thresh}%)...")

        true_labels = []
        old_labels = df_clf["label"].tolist()
        mismatches = 0

        for idx, row in df_clf.iterrows():
            c = row["cutoff_utc"]
            
            # Evaluate 5-session path for UP signal
            path_up = prices.evaluate_path(
                asset, c, 5, "up",
                {"threshold": thresh},
                {"invalidation": thresh}
            )
            # Evaluate 5-session path for DOWN signal
            path_down = prices.evaluate_path(
                asset, c, 5, "down",
                {"threshold": thresh},
                {"invalidation": thresh}
            )

            up_hit = path_up.first("threshold", "invalidation") if path_up else False
            down_hit = path_down.first("threshold", "invalidation") if path_down else False

            if up_hit and not down_hit:
                real_label = "up_swing"
            elif down_hit and not up_hit:
                real_label = "down_swing"
            elif up_hit and down_hit:
                # Both hit: compare touch positions
                touch_up = path_up.touches.get("threshold")
                touch_down = path_down.touches.get("threshold")
                if touch_up is not None and touch_down is not None:
                    real_label = "up_swing" if touch_up < touch_down else "down_swing"
                else:
                    real_label = "up_swing" if up_hit else "down_swing"
            else:
                real_label = "no_swing"

            true_labels.append(real_label)
            if real_label != old_labels[idx]:
                mismatches += 1

        print(f"  Fixed {mismatches} mismatched labels out of {len(df_clf)} ({mismatches / len(df_clf) * 100:.1f}%) in {time.time() - t0:.1f}s!")
        print("  New True Class Distribution:")
        counts = pd.Series(true_labels).value_counts()
        for k, v in counts.items():
            print(f"    - {k:12s}: {v:5d} ({v / len(df_clf) * 100:.1f}%)")

        # Save corrected labels into classifier and regressor datasets
        df_clf["label_legacy_24"] = df_clf["label"]
        df_clf["label"] = true_labels
        df_clf.to_parquet(clf_file, index=False)

        if "label" in df_reg.columns:
            df_reg["label_legacy_24"] = df_reg["label"]
        df_reg["label"] = true_labels
        df_reg["is_swing"] = [1 if l in ["up_swing", "down_swing"] else 0 for l in true_labels]
        df_reg["swing_direction"] = [
            "up" if l == "up_swing" else ("down" if l == "down_swing" else "none")
            for l in true_labels
        ]
        df_reg.to_parquet(reg_file, index=False)

    # Also update unified datasets
    all_clf = []
    all_reg = []
    for asset in ASSETS:
        all_clf.append(pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet"))
        all_reg.append(pd.read_parquet(DATA_DIR / f"task1_regressor_{asset}.parquet"))

    pd.concat(all_clf, ignore_index=True).to_parquet(DATA_DIR / "task1_classifier_unified.parquet", index=False)
    pd.concat(all_reg, ignore_index=True).to_parquet(DATA_DIR / "task1_regressor_unified.parquet", index=False)
    print("\nUnified datasets updated with zero label mismatch.")
    print("================================================================================")


if __name__ == "__main__":
    generate_exact_26_targets()
