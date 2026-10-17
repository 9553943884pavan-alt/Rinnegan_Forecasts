import sys
sys.path.append(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\kronos_task1")
import pandas as pd
import numpy as np
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
norm = SIG.normalize(preds, 1)
scored = S.challenge1(norm, inventory, prices, window, n_boot=200, seed=0)

print(f"Total swings in window: {scored['counts']['swings_in_window']}")
print(f"Swings caught: {scored['counts']['swings_caught']}")

# Check per asset swing counts
swings_df = scored.get("swings_scored")
# Let's inspect the keys of scored
print("Keys in scored report:", scored.keys())
print("Diagnostics:", scored.get("diagnostics", {}))
