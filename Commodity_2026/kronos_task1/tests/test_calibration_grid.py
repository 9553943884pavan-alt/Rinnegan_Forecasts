import sys
sys.path.append(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\kronos_task1")
import itertools
import json
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from swing_dataset.kit import ASSETS
from swing_dataset.kit.loop import price_view
from swing_dataset.pack.pit import PointInTime
from swing_dataset.scoring2 import outcomes as O, signals as SIG

from src.path_rules import PathRulesEngine
from src.inputs import DataPipeline
from src.forecaster import RawPathPredictor, TrajectoryForecaster
from src.calibration import sample_calibration_dates

pack = PointInTime("../commodity-open-pack/v2.0.1")
grading_asof = datetime(2026, 6, 1, 3, 0, tzinfo=ZoneInfo("UTC"))
prices_ref, _ = price_view(pack, grading_asof)

pipeline = DataPipeline(pack, max_context=400)
predictor = RawPathPredictor()
forecaster = TrajectoryForecaster(predictor=predictor, max_pred_len=10, sample_count=20)

calibration_instants = sample_calibration_dates(pack, start_date="2021-04-01", end_date="2026-05-31", step_days=60)
print(f"Sampled {len(calibration_instants)} sessions across 2021-2026")

sim_data = []
for idx, instant in enumerate(calibration_instants):
    for asset in ASSETS:
        series = pipeline.build_series(asset, instant, pred_len=10)
        if series is None or len(series.input_df) < 50:
            continue
        close_arr = series.input_df["close"].to_numpy(dtype=float)
        high_arr = series.input_df["high"].to_numpy(dtype=float)
        low_arr = series.input_df["low"].to_numpy(dtype=float)
        ret1 = float((close_arr[-1] - close_arr[-2]) / close_arr[-2]) if len(close_arr) >= 2 and close_arr[-2] > 0 else 0.0
        bar_range = float(high_arr[-1] - low_arr[-1])
        close_pos = float((close_arr[-1] - low_arr[-1]) / bar_range) if bar_range > 0 else 0.5

        paths_10 = forecaster.get_paths(
            asset=asset,
            as_of=instant,
            series_df=series.input_df,
            x_timestamp=series.x_timestamp,
            y_timestamp=series.y_timestamp,
            horizon=10,
        )
        sim_data.append({
            "instant": instant,
            "asset": asset,
            "paths": paths_10,
            "p0": series.p0,
            "last_close": series.last_close,
            "ret1": ret1,
            "close_pos": close_pos,
        })

print(f"Cached {len(sim_data)} asset-sessions.")

# Search key combinations
results = []
for tau in [0.42, 0.44, 0.46]:
    for margin in [0.06, 0.08, 0.10]:
        for tgt_r in [1.00, 1.05, 1.10]:
            for inv_r in [0.95, 1.00]:
                for k_filt, r_filt in [(True, True), (False, False)]:
                    engine = PathRulesEngine(
                        tau=tau, margin=margin, rho=0.55, target_ratio=tgt_r, inv_ratio=inv_r,
                        filter_knife=k_filt, filter_rocket=r_filt,
                        knife_ret_thresh=-0.005, knife_pos_thresh=0.45,
                        rocket_ret_thresh=0.008, rocket_pos_thresh=0.65
                    )
                    sigs = []
                    for item in sim_data:
                        for h in (5, 10):
                            ev = engine.evaluate(item["paths"][:, :h, :], item["p0"], item["last_close"], item["asset"], h, ret1=item["ret1"], close_pos=item["close_pos"])
                            if ev.direction is not None:
                                sigs.append({
                                    "signal_utc": item["instant"],
                                    "decision_utc": None,
                                    "asset": item["asset"],
                                    "direction": ev.direction,
                                    "horizon": h,
                                    "target_pct": ev.target_pct,
                                    "invalidation_pct": ev.invalidation_pct,
                                    "confidence": ev.confidence,
                                    "evidence": ev.evidence,
                                })
                    if len(sigs) < 10:
                        continue
                    df_sig = pd.DataFrame(sigs)
                    norm = SIG.normalize(df_sig, 1).signals
                    graded = O.grade_c1(norm, prices_ref)
                    decided = graded[graded["decided"].fillna(False).astype(bool)]
                    if len(decided) == 0:
                        continue
                    corr = float(decided["correct"].mean() * 100.0)
                    deep_sub = decided[decided["correct"] == True]
                    depth = float(deep_sub["deep"].mean() * 100.0) if len(deep_sub) > 0 else 0.0
                    results.append((corr, depth, len(sigs), tau, margin, tgt_r, inv_r, k_filt))

filtered_res = [r for r in results if r[7] == True]
unfiltered_res = [r for r in results if r[7] == False]

filtered_res.sort(key=lambda x: (x[0] * 0.6 + x[1] * 0.4), reverse=True)
unfiltered_res.sort(key=lambda x: (x[0] * 0.6 + x[1] * 0.4), reverse=True)

print("\n--- TOP CALIBRATION WITH FINANCIAL FILTERS (2021-2026) ---")
print(f"{'Corr':>6} | {'Depth':>6} | {'Sigs':>5} | {'tau':>5} | {'margin':>6} | {'target':>6} | {'inv':>5}")
print("-" * 55)
for r in filtered_res[:5]:
    print(f"{r[0]:6.1f}% | {r[1]:6.1f}% | {r[2]:5d} | {r[3]:5.2f} | {r[4]:6.2f} | {r[5]:6.2f} | {r[6]:5.2f}")

print("\n--- TOP CALIBRATION UNFILTERED (2021-2026) ---")
print(f"{'Corr':>6} | {'Depth':>6} | {'Sigs':>5} | {'tau':>5} | {'margin':>6} | {'target':>6} | {'inv':>5}")
print("-" * 55)
for r in unfiltered_res[:5]:
    print(f"{r[0]:6.1f}% | {r[1]:6.1f}% | {r[2]:5d} | {r[3]:5.2f} | {r[4]:6.2f} | {r[5]:6.2f} | {r[6]:5.2f}")
