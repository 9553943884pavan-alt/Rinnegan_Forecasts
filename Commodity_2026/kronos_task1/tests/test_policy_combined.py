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
            "ret5": float(row.get("ret5", 0.0)),
            "ret20": float(row.get("ret20", 0.0)),
            "rv20": float(row.get("rv20", 0.01)),
            "stretch": float(row.get("stretch", 0.0)),
            "close_pos": float(row.get("close_pos", 0.5)),
        }

def run_grid():
    results = []
    # Grid search over:
    # inv_mult: [0.95, 1.0]
    # min_conf_up: [0.45, 0.50, 0.55, 0.60]
    # min_conf_dn: [0.40, 0.45, 0.50]
    # knife_filter: [True, False]
    # rocket_filter: [True, False]
    # target_mult: [0.70, 0.75, 0.80, 0.85, 1.0]
    
    experiments = [
        # (name, inv_mult, min_conf_up, min_conf_dn, knife, rocket, trend_filter, target_mult)
        ("Baseline (inv=0.75)", 0.75, 0.0, 0.0, False, False, False, 1.0),
        ("Inv=1.0 alone", 1.0, 0.0, 0.0, False, False, False, 1.0),
        ("Inv=1.0 + Knife Filter", 1.0, 0.0, 0.0, True, False, False, 1.0),
        ("Inv=1.0 + Knife + Rocket Filter", 1.0, 0.0, 0.0, True, True, False, 1.0),
        ("Inv=1.0 + Knife + Rocket + Target 0.80", 1.0, 0.0, 0.0, True, True, False, 0.80),
        ("Inv=1.0 + Knife + Rocket + Target 0.75", 1.0, 0.0, 0.0, True, True, False, 0.75),
        ("Inv=1.0 + Knife + Rocket + Target 0.70", 1.0, 0.0, 0.0, True, True, False, 0.70),
        ("Inv=1.0 + Conf(0.52/0.45) + Knife + Rocket + T0.75", 1.0, 0.52, 0.45, True, True, False, 0.75),
        ("Inv=1.0 + Conf(0.55/0.45) + Knife + Rocket + T0.75", 1.0, 0.55, 0.45, True, True, False, 0.75),
        ("Inv=1.0 + Conf(0.55/0.45) + Knife + Rocket + T0.70", 1.0, 0.55, 0.45, True, True, False, 0.70),
        ("Inv=1.0 + Conf(0.55/0.48) + Knife + Rocket + Trend + T0.75", 1.0, 0.55, 0.48, True, True, True, 0.75),
        ("Inv=1.0 + Conf(0.58/0.48) + Knife + Rocket + Trend + T0.75", 1.0, 0.58, 0.48, True, True, True, 0.75),
        ("Inv=1.0 + Conf(0.60/0.50) + Knife + Rocket + Trend + T0.75", 1.0, 0.60, 0.50, True, True, True, 0.75),
        ("Inv=1.0 + Conf(0.60/0.50) + Knife + Rocket + Trend + T0.70", 1.0, 0.60, 0.50, True, True, True, 0.70),
        ("Inv=1.0 + Conf(0.62/0.50) + Knife + Rocket + Trend + T0.70", 1.0, 0.62, 0.50, True, True, True, 0.70),
    ]

    for name, inv_m, up_c, dn_c, k_filt, r_filt, t_filt, t_mult in experiments:
        mod = preds.copy()
        for idx, row in mod.iterrows():
            if pd.isna(row["direction"]):
                continue
            instant = row["signal_utc"]
            d = instant.astimezone(IST).date()
            ind = indicators.get((row["asset"], d), {"ret1": 0.0, "ret5": 0.0, "ret20": 0.0, "rv20": 0.01, "stretch": 0.0, "close_pos": 0.5})
            direction = row["direction"]
            conf = row["confidence"] or 0.5
            ret1 = ind["ret1"]
            ret20 = ind["ret20"]
            stretch = ind["stretch"]
            close_pos = ind["close_pos"]
            thr = THRESHOLD_PCT[row["asset"]]

            to_no = False
            reason = None

            if direction == "up":
                if conf < up_c:
                    to_no = True
                    reason = f"low conf {conf:.2f} < {up_c:.2f}"
                elif k_filt and (ret1 < -0.005 and close_pos < 0.45):
                    to_no = True
                    reason = f"knife catch: ret1={ret1:.3f}, close_pos={close_pos:.2f}"
                elif t_filt and (ret20 < -0.03 and conf < 0.65):
                    to_no = True
                    reason = f"counter-trend up against ret20={ret20:.3f}"
            elif direction == "down":
                if conf < dn_c:
                    to_no = True
                    reason = f"low conf {conf:.2f} < {dn_c:.2f}"
                elif r_filt and (ret1 > +0.008 and close_pos > 0.65):
                    to_no = True
                    reason = f"rocket catch short: ret1={ret1:.3f}, close_pos={close_pos:.2f}"
                elif t_filt and (ret20 > +0.04 and conf < 0.65):
                    to_no = True
                    reason = f"counter-trend down against ret20={ret20:.3f}"

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
                mod.at[idx, "invalidation_pct"] = thr * inv_m
                if t_mult != 1.0:
                    raw_tgt = row["target_pct"]
                    mod.at[idx, "target_pct"] = float(np.clip(raw_tgt * t_mult, thr, 3.5 * thr))

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

        print(f"{name:<56} | Sig: {len(norm.signals):3d} | Corr: {c:5.1f}% | Depth: {d:5.1f}% | Catch: {k:5.1f}% | Top10: {top_swings}/10 | Succ: {s:5.1f}% | Cov: {min_cov*100:.0f}%")

if __name__ == "__main__":
    print("=" * 135)
    run_grid()
    print("=" * 135)
