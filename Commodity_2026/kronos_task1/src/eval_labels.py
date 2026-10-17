"""Secondary 3-Class Diagnostic Evaluator.

Matches model predictions against `labels/swing_labels.parquet` (k_multiplier=1.0).
Evaluates 3-class directional accuracy:
- `up` -> `up_swing`
- `down` -> `down_swing`
- `no_signal` -> `no_swing`

Outputs:
- 3x3 Confusion Matrix
- Per-class Precision, Recall, F1
- Macro F1 and directional win-rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


CLASSES = ["up_swing", "down_swing", "no_swing"]
DIR_MAP = {"up": "up_swing", "down": "down_swing"}


def evaluate_predictions(
    predictions_path: Path | str,
    pack_path: Path | str,
    horizon: int = 5,
    k_multiplier: float = 1.0,
) -> dict:
    predictions_p = Path(predictions_path)
    pack_p = Path(pack_path)

    preds = pd.read_parquet(predictions_p)
    labels_file = pack_p / "labels" / "swing_labels.parquet"
    if not labels_file.exists():
        raise FileNotFoundError(f"Missing {labels_file}")

    labels = pd.read_parquet(labels_file)

    # Filter labels
    labels = labels[(labels["horizon"] == horizon) & (labels["k_multiplier"] == k_multiplier)].copy()
    labels["cutoff_utc"] = pd.to_datetime(labels["cutoff_utc"], utc=True)
    labels = labels[labels["label"].isin(CLASSES)]

    # Decision instant for signal rows is in signal_utc; for no_signal rows in decision_utc
    instant = preds["signal_utc"].fillna(preds["decision_utc"])
    preds["decision_instant"] = pd.to_datetime(instant, utc=True)
    preds_h = preds[preds["horizon"] == horizon].copy() if "horizon" in preds.columns else preds.copy()

    # Map predictions to 3 classes
    preds_h["pred_label"] = preds_h["direction"].map(DIR_MAP).fillna("no_swing")

    # Merge on (asset, cutoff_utc == decision_instant)
    merged = labels.merge(
        preds_h,
        left_on=["asset", "cutoff_utc"],
        right_on=["asset", "decision_instant"],
        how="inner",
    )

    if merged.empty:
        return {
            "error": "No overlapping records found between predictions and labels.",
            "predictions_count": len(preds_h),
            "labels_count": len(labels),
        }

    y_true = merged["label"].tolist()
    y_pred = merged["pred_label"].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    report_dict = classification_report(
        y_true, y_pred, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0
    )

    # Directional precision (when signaling up or down, how often was it correct)
    signal_mask = merged["pred_label"].isin(["up_swing", "down_swing"])
    signal_accuracy = (
        float(np.mean(merged.loc[signal_mask, "pred_label"] == merged.loc[signal_mask, "label"]))
        if signal_mask.any()
        else 0.0
    )

    results = {
        "samples_evaluated": len(merged),
        "horizon": horizon,
        "k_multiplier": k_multiplier,
        "macro_f1": float(macro_f1),
        "directional_signal_accuracy": float(signal_accuracy),
        "signals_issued": int(signal_mask.sum()),
        "abstentions": int((~signal_mask).sum()),
        "confusion_matrix": {
            "classes": CLASSES,
            "matrix": cm.tolist(),
        },
        "classification_report": report_dict,
    }

    return results


def print_report(results: dict) -> None:
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("=" * 65)
    print(f"3-Class Diagnostic Evaluation (Horizon {results['horizon']})")
    print("=" * 65)
    print(f"Total Sessions Evaluated: {results['samples_evaluated']}")
    print(f"Signals Issued (up/down): {results['signals_issued']} | Abstentions: {results['abstentions']}")
    print(f"Directional Signal Accuracy: {results['directional_signal_accuracy'] * 100:.1f}%")
    print(f"Macro F1-Score: {results['macro_f1']:.4f}\n")

    print("Confusion Matrix (Rows: Ground Truth, Cols: Model Predictions):")
    classes = results["confusion_matrix"]["classes"]
    cm = np.array(results["confusion_matrix"]["matrix"])
    header = f"{'':15}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row_str = f"{c:15}" + "".join(f"{cm[i, j]:>12}" for j in range(len(classes)))
        print(row_str)

    print("\nDetailed Metrics:")
    cr = results["classification_report"]
    for c in classes:
        print(
            f"  {c:12} -> Precision: {cr[c]['precision'] * 100:5.1f}% | "
            f"Recall: {cr[c]['recall'] * 100:5.1f}% | F1: {cr[c]['f1-score']:.3f} | Support: {cr[c]['support']}"
        )
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against 3-class swing labels")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions parquet")
    parser.add_argument("--pack", type=Path, required=True, help="Path to pack root")
    parser.add_argument("--horizon", type=int, default=5, choices=(5, 10))
    parser.add_argument("--out", type=Path, help="Optional JSON path to save report")
    args = parser.parse_args()

    results = evaluate_predictions(args.predictions, args.pack, horizon=args.horizon)
    print_report(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
