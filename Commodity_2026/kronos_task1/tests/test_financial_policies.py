import sys
sys.path.append(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\kronos_task1")
import pandas as pd
import numpy as np
from swing_dataset.pack.pit import PointInTime
from swing_dataset.kit.loop import price_view
from swing_dataset.scoring2 import signals as SIG, score as S, THRESHOLD_PCT, TOP_SWINGS, swings as SW
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
pack_path = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
pit = PointInTime(pack_path)
prices, chains = price_view(pit, datetime(2026, 9, 26, 3, 0, tzinfo=UTC))
inventory = pd.read_parquet(pack_path / "swings" / "swings.parquet")
window = S.resolve_window("holdout", prices)

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
            "ret20": float(row.get("ret20", 0.0)),
            "rv20": float(row.get("rv20", 0.01)),
            "close_pos": float(row.get("close_pos", 0.5)),
        }

def evaluate_policy(name, min_conf_up, min_conf_dn, filter_knife, filter_downtrend, filter_uptrend, target_mult):
    modified = preds.copy()
    for idx, row in modified.iterrows():
        if pd.isna(row["direction"]):
            continue
        instant = row["signal_utc"]
        d = instant.astimezone(IST).date()
        ind = indicators.get((row["asset"], d), {"ret1": 0.0, "ret20": 0.0, "rv20": 0.01, "close_pos": 0.5})
        direction = row["direction"]
        conf = row["confidence"] or 0.5
        ret1 = ind["ret1"]
        ret20 = ind["ret20"]
        close_pos = ind["close_pos"]

        to_no_signal = False
        reason = None

        if direction == "up":
            if conf < min_conf_up:
                to_no_signal = True
                reason = f"conf {conf:.2f} < {min_conf_up}"
            elif filter_knife and ret1 < -0.005 and close_pos < 0.45:
                to_no_signal = True
                reason = f"knife catch: ret1={ret1:.3f}, close_pos={close_pos:.2f}"
            elif filter_downtrend and ret20 < -0.025 and conf < 0.65:
                to_no_signal = True
                reason = f"downtrend counter-trade: ret20={ret20:.3f}"
        elif direction == "down":
            if conf < min_conf_dn:
                to_no_signal = True
                reason = f"conf {conf:.2f} < {min_conf_dn}"
            elif filter_uptrend and ret20 > 0.035 and conf < 0.65:
                to_no_signal = True
                reason = f"bull trend counter-trade: ret20={ret20:.3f}"

        if to_no_signal:
            modified.at[idx, "direction"] = None
            modified.at[idx, "decision_utc"] = instant
            modified.at[idx, "signal_utc"] = None
            modified.at[idx, "target_pct"] = None
            modified.at[idx, "invalidation_pct"] = None
            modified.at[idx, "confidence"] = None
            modified.at[idx, "evidence"] = None
            modified.at[idx, "reason"] = reason
        elif target_mult != 1.0:
            thr = THRESHOLD_PCT[row["asset"]]
            cur_target = row["target_pct"]
            new_target = np.clip(cur_target * target_mult, thr, 3.5 * thr)
            modified.at[idx, "target_pct"] = new_target

    norm = SIG.normalize(modified, 1)
    sig_cnt = len(norm.signals)
    scored = S.challenge1(norm, inventory, prices, window, n_boot=200, seed=0)
    scores = scored["scores"]
    c_val = scores["correctness"]["value"]
    d_val = scores["depth"]["value"]
    k_val = scores["catch_rate"]["value"]
    s_val = scores["success_rate"]["value"]

    cov = S.coverage(prices, norm, window)
    min_cov = min(c["share"] for c in cov["per_asset"].values())

    print(f"{name:<38} | Sig: {sig_cnt:3d} | Corr: {c_val:5.1f}% | Depth: {d_val:5.1f}% | Catch: {k_val:5.1f}% | Succ: {s_val:5.1f}% | Cov: {min_cov*100:.0f}%")

print("=" * 115)
print("FINANCIAL RULES & KRONOS HYBRID POLICY EXPERIMENTS")
print("=" * 115)
evaluate_policy("0. Raw Kronos Baseline", 0.0, 0.0, False, False, False, 1.0)
evaluate_policy("1. Knife-Catch Filter", 0.0, 0.0, True, False, False, 1.0)
evaluate_policy("2. Downtrend Knife + Trend", 0.0, 0.0, True, True, True, 1.0)
evaluate_policy("3. Asym Conf (up=0.55, dn=0.45)", 0.55, 0.45, True, True, True, 1.0)
evaluate_policy("4. High Conviction (up=0.60, dn=0.50)", 0.60, 0.50, True, True, True, 1.0)
evaluate_policy("5. Ultra Conviction (up=0.65, dn=0.55)", 0.65, 0.55, True, True, True, 1.0)
evaluate_policy("6. Ultra + Target Mult=0.85", 0.65, 0.55, True, True, True, 0.85)
evaluate_policy("7. Ultra + Target Mult=0.75", 0.65, 0.55, True, True, True, 0.75)
print("=" * 115)
