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

def test_regime(name, up_stretch_max, dn_stretch_min, target_mult):
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
        stretch = ind["stretch"]
        close_pos = ind["close_pos"]
        thr = THRESHOLD_PCT[row["asset"]]

        to_no = False
        reason = None

        if direction == "up":
            if ret1 < -0.005 and close_pos < 0.45:
                to_no = True
                reason = f"knife catch: ret1={ret1:.3f}, close_pos={close_pos:.2f}"
            elif stretch > up_stretch_max:
                to_no = True
                reason = f"overbought stretch={stretch:.2f} > {up_stretch_max}"
        elif direction == "down":
            if ret1 > +0.008 and close_pos > 0.65:
                to_no = True
                reason = f"rocket catch short: ret1={ret1:.3f}, close_pos={close_pos:.2f}"
            elif stretch < dn_stretch_min:
                to_no = True
                reason = f"oversold stretch={stretch:.2f} < {dn_stretch_min}"

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
            mod.at[idx, "invalidation_pct"] = thr * 1.0
            mod.at[idx, "target_pct"] = float(np.clip(thr * target_mult, thr, 3.5 * thr))

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
    print(f"{name:<35} | Sig: {len(norm.signals):3d} | Corr: {c:5.1f}% | Depth: {d:5.1f}% | Catch: {k:5.1f}% | Top10: {top_swings}/10 | Succ: {s:5.1f}% | Cov: {min_cov*100:.0f}%")

test_regime("Pure Knife+Rocket (T=1.00)", 99.0, -99.0, 1.00)
test_regime("Pure Knife+Rocket (T=1.05)", 99.0, -99.0, 1.05)
test_regime("Stretch Filter (stretch < 1.5, > -1.5)", 1.5, -1.5, 1.00)
test_regime("Stretch Filter (stretch < 1.8, > -1.8)", 1.8, -1.8, 1.00)
test_regime("Stretch Filter (stretch < 2.0, > -2.0)", 2.0, -2.0, 1.00)
