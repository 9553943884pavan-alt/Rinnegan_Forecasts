import sys
sys.path.append(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\kronos_task1")
import pandas as pd
from swing_dataset.pack.pit import PointInTime
from swing_dataset.kit.loop import price_view
from swing_dataset.scoring2 import signals as SIG, score as S, THRESHOLD_PCT
from datetime import datetime, UTC
from pathlib import Path

pack_path = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
pit = PointInTime(pack_path)
prices, chains = price_view(pit, datetime(2026, 9, 26, 3, 0, tzinfo=UTC))
inventory = pd.read_parquet(pack_path / "swings" / "swings.parquet")
window = S.resolve_window("holdout", prices)

preds = pd.read_parquet("predictions/c1_holdout.parquet")

for inv_mult in [0.75, 0.85, 0.95, 1.0]:
    mod = preds.copy()
    for idx, row in mod.iterrows():
        if pd.notna(row["direction"]):
            thr = THRESHOLD_PCT[row["asset"]]
            mod.at[idx, "invalidation_pct"] = thr * inv_mult
    norm = SIG.normalize(mod, 1)
    scored = S.challenge1(norm, inventory, prices, window, n_boot=200, seed=0)
    scores = scored["scores"]
    c = scores["correctness"]["value"]
    d = scores["depth"]["value"]
    k = scores["catch_rate"]["value"]
    s = scores["success_rate"]["value"]
    print(f"Inv Mult: {inv_mult:.2f} | Corr: {c:5.1f}% | Depth: {d:5.1f}% | Catch: {k:5.1f}% | Succ: {s:5.1f}%")
