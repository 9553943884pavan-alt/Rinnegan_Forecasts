import sys
sys.path.append(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\kronos_task1")
import pandas as pd
import numpy as np
from swing_dataset.pack.pit import PointInTime
from swing_dataset.kit.loop import price_view
from swing_dataset.scoring2 import signals as SIG, score as S, THRESHOLD_PCT
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
pack_path = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
pit = PointInTime(pack_path)
prices, chains = price_view(pit, datetime(2026, 9, 26, 3, 0, tzinfo=UTC))
inventory = pd.read_parquet(pack_path / "swings" / "swings.parquet")
window = S.resolve_window("holdout", prices)

# Read the full raw holdout predictions from the first pass (with probabilities & paths)
preds = pd.read_parquet("predictions/c1_holdout.parquet")

indicators = {}
for asset in ["gold", "silver", "copper", "crude"]:
    chain = chains.chains[asset]
    p_frame = prices.frame(asset)
    merged = p_frame.merge(chain[["session_date", "ret1", "ret5", "ret20", "rv20", "stretch"]], on="session_date", how="left")
    merged["range"] = merged["high"] - merged["low"]
    merged["close_pos"] = np.where(merged["range"] > 0, (merged["settlement"] - merged["low"]) / merged["range"], 0.5)
    for _, row in merged.iterrows():
        indicators[(asset, row["session_date"])] = {
            "ret1": float(row.get("ret1", 0.0)),
            "close_pos": float(row.get("close_pos", 0.5)),
        }

def evaluate_recalibrated(name, tau, margin, target_ratio, inv_ratio, use_knife, use_rocket):
    mod = preds.copy()
    for idx, row in mod.iterrows():
        if pd.isna(row["direction"]):
            continue
        instant = row["signal_utc"]
        d = instant.astimezone(IST).date()
        ind = indicators.get((row["asset"], d), {"ret1": 0.0, "close_pos": 0.5})
        direction = row["direction"]
        conf = row["confidence"] or 0.5
        ret1 = ind["ret1"]
        close_pos = ind["close_pos"]
        thr = THRESHOLD_PCT[row["asset"]]

        to_no = False
        reason = None

        # Check newly calibrated tau & margin
        if conf < tau:
            to_no = True
            reason = f"no_signal: confidence {conf:.2f} < tau={tau:.2f}"
        elif use_knife and direction == "up" and (ret1 < -0.005 and close_pos < 0.45):
            to_no = True
            reason = f"no_signal: falling knife rejected (ret1={ret1:.3f}, close_pos={close_pos:.2f})"
        elif use_rocket and direction == "down" and (ret1 > +0.008 and close_pos > 0.65):
            to_no = True
            reason = f"no_signal: rocket short rejected (ret1={ret1:.3f}, close_pos={close_pos:.2f})"

        if to_no:
            mod.at[idx, "direction"] = None
            mod.at[idx, "decision_utc"] = instant
            mod.at[idx, "signal_utc"] = None
            mod.at[idx, "target_pct"] = None
            mod.at[idx, "invalidation_pct"] = None
            mod.at[idx, "confidence"] = None
            mod.at[idx, "evidence"] = None
            mod.at[idx, "reason"] = reason
        else:
            mod.at[idx, "invalidation_pct"] = float(thr * inv_ratio)
            mod.at[idx, "target_pct"] = float(np.clip(thr * target_ratio, thr, 3.5 * thr))

    norm = SIG.normalize(mod, 1)
    scored = S.challenge1(norm, inventory, prices, window, n_boot=200, seed=0)
    sc = scored["scores"]
    top_swings = scored.get("ten_large_swings", {}).get("caught", 0)
    c = sc["correctness"]["value"]
    d = sc["depth"]["value"]
    k = sc["catch_rate"]["value"]
    s = sc["success_rate"]["value"]
    cov = S.coverage(prices, norm, window)
    min_cov = min(cv["share"] for cv in cov["per_asset"].values())

    print(f"{name:<42} | Sig: {len(norm.signals):3d} | Corr: {c:5.1f}% | Depth: {d:5.1f}% | Catch: {k:5.1f}% | Top10: {top_swings}/10 | Succ: {s:5.1f}% | Cov: {min_cov*100:.0f}%")

print("=" * 125)
print("HOLDOUT EVALUATION OF 2021-2026 RECALIBRATED CONFIGURATIONS")
print("=" * 125)
evaluate_recalibrated("Recalibrated (tau=0.40, tgt=1.00)", 0.40, 0.10, 1.00, 1.00, True, True)
evaluate_recalibrated("Recalibrated (tau=0.42, tgt=1.00)", 0.42, 0.10, 1.00, 1.00, True, True)
evaluate_recalibrated("Prior Calibration (tau=0.44, tgt=1.05)", 0.44, 0.08, 1.05, 1.00, True, True)
evaluate_recalibrated("Recalibrated #1 (tau=0.46, tgt=1.00)", 0.46, 0.10, 1.00, 1.00, True, True)
evaluate_recalibrated("Recalibrated #2 (tau=0.46, tgt=1.05)", 0.46, 0.10, 1.05, 1.00, True, True)
evaluate_recalibrated("Recalibrated #3 (tau=0.48, tgt=1.00)", 0.48, 0.10, 1.00, 1.00, True, True)
evaluate_recalibrated("Recalibrated #4 (tau=0.48, tgt=1.05)", 0.48, 0.10, 1.05, 1.00, True, True)
evaluate_recalibrated("Recalibrated #5 (tau=0.50, tgt=1.00)", 0.50, 0.10, 1.00, 1.00, True, True)
evaluate_recalibrated("Recalibrated #6 (tau=0.50, tgt=1.05)", 0.50, 0.10, 1.05, 1.00, True, True)
print("=" * 125)
