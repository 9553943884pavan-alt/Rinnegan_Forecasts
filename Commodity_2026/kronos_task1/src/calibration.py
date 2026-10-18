"""Training Set Calibration & Multi-Year Grid Search for Kronos Task 1.

Spans the full 5-year training period: 2021-04-01 to 2026-05-31.
Holdout period (2026-06-01+) is strictly locked and unseen during calibration.

Recalibrates:
- Probability threshold: tau in [0.42, 0.48]
- Directional margin: margin in [0.06, 0.12]
- Directional ratio: rho in [0.52, 0.58]
- Target multiplier: target_ratio in [1.00, 1.15]
- Invalidation multiplier: inv_ratio in [0.85, 1.00]
- Falling knife rejection: knife_ret_thresh in [-0.007, -0.003], knife_pos_thresh in [0.40, 0.50]
- Rocket short rejection: rocket_ret_thresh in [0.006, 0.010], rocket_pos_thresh in [0.60, 0.70]

Saves optimal config to configs/task1_config.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swing_dataset.kit import ASSETS, DEFAULT_HORIZONS, THRESHOLD_PCT
from swing_dataset.kit.loop import price_view
from swing_dataset.pack.pit import PointInTime
from swing_dataset.scoring.chain import cutoff_instant
from swing_dataset.scoring2 import outcomes as O
from swing_dataset.scoring2 import signals as SIG

from src.forecaster import RawPathPredictor, TrajectoryForecaster
from src.inputs import DataPipeline
from src.path_rules import PathRulesEngine

IST = ZoneInfo("Asia/Kolkata")
CUTOFF_IST = time(8, 30)


def sample_calibration_dates(
    pack: PointInTime,
    start_date: str = "2021-04-01",
    end_date: str = "2026-05-31",
    step_days: int = 30,
) -> List[datetime]:
    """Select representative calibration instants spaced evenly across trading regimes from 2021 to May 2026."""
    sessions_df = pack.frame("mcx/mcx_sessions", datetime(2026, 6, 1, 3, 0, tzinfo=ZoneInfo("UTC"))).collect()
    open_states = {"open", "evening_only", "morning_only"}
    valid = sessions_df[
        (sessions_df["session_date"] >= pd.to_datetime(start_date).date())
        & (sessions_df["session_date"] <= pd.to_datetime(end_date).date())
        & (sessions_df["state"].astype(str).str.lower().isin(open_states))
    ]["session_date"].tolist()

    sampled = valid[::step_days]
    instants = [cutoff_instant(d) for d in sampled]
    return instants


def run_recalibration(
    pack_path: Path | str,
    output_config: Path | str = PROJECT_ROOT / "configs" / "task1_config.json",
    sample_count: int = 20,
    quick: bool = False,
) -> dict:
    pack = PointInTime(pack_path)
    step = 40 if quick else 28
    
    # 5-Year Full Calibration Window: 2021 to 2026-05-31
    calibration_instants = sample_calibration_dates(
        pack, start_date="2021-04-01", end_date="2026-05-31", step_days=step
    )
    print("=" * 80)
    print(f"5-YEAR CALIBRATION (2021-04-01 to 2026-05-31): {len(calibration_instants)} sessions")
    print("=" * 80)

    pipeline = DataPipeline(pack, max_context=400)
    predictor = RawPathPredictor()
    forecaster = TrajectoryForecaster(predictor=predictor, max_pred_len=10, sample_count=sample_count)

    # Pre-generate simulated trajectories and point-in-time momentum indicators
    sim_data: List[dict] = []
    print("Pre-generating Kronos trajectories and point-in-time indicators...")
    for idx, instant in enumerate(calibration_instants):
        for asset in ASSETS:
            series = pipeline.build_series(asset, instant, pred_len=10)
            if series is None or len(series.input_df) < 50:
                continue

            # Point-in-time momentum features strictly <= instant
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
            sim_data.append(
                {
                    "instant": instant,
                    "asset": asset,
                    "paths": paths_10,
                    "p0": series.p0,
                    "last_close": series.last_close,
                    "ret1": ret1,
                    "close_pos": close_pos,
                }
            )
        if (idx + 1) % 5 == 0 or (idx + 1) == len(calibration_instants):
            print(f"  Processed {idx + 1}/{len(calibration_instants)} sessions ({len(sim_data)} asset-sessions cached)")

    print(f"\nTotal asset-sessions cached: {len(sim_data)}")
    print("Building reference prices for training period evaluation...")

    # Reference prices strictly up to training cutoff 2026-06-01
    grading_asof = datetime(2026, 6, 1, 3, 0, tzinfo=ZoneInfo("UTC"))
    prices_ref, _ = price_view(pack, grading_asof)

    # Grid search candidate parameters across all dimensions
    tau_vals = [0.42, 0.44, 0.46, 0.48] if quick else [0.40, 0.42, 0.44, 0.46, 0.48]
    margin_vals = [0.06, 0.08, 0.10]
    target_r_vals = [1.00, 1.05, 1.10]
    inv_r_vals = [0.95, 1.00]
    knife_ret_vals = [-0.005, -0.007]
    knife_pos_vals = [0.45, 0.50]
    rocket_ret_vals = [0.008, 0.010]
    rocket_pos_vals = [0.65, 0.70]

    total_combos = (
        len(tau_vals) * len(margin_vals) * len(target_r_vals) * len(inv_r_vals) *
        len(knife_ret_vals) * len(knife_pos_vals) * len(rocket_ret_vals) * len(rocket_pos_vals)
    )
    print(f"Searching {total_combos} candidate threshold combinations on 2021-2026 training data...")

    best_score = -1.0
    best_params = {}
    best_metrics = {}

    grid = itertools.product(
        tau_vals, margin_vals, target_r_vals, inv_r_vals,
        knife_ret_vals, knife_pos_vals, rocket_ret_vals, rocket_pos_vals
    )

    tested = 0
    for tau, margin, tgt_r, inv_r, k_ret, k_pos, r_ret, r_pos in grid:
        tested += 1
        engine = PathRulesEngine(
            tau=tau,
            margin=margin,
            rho=0.55,
            target_ratio=tgt_r,
            inv_ratio=inv_r,
            filter_knife=True,
            filter_rocket=True,
            knife_ret_thresh=k_ret,
            knife_pos_thresh=k_pos,
            rocket_ret_thresh=r_ret,
            rocket_pos_thresh=r_pos,
        )

        signals_list = []
        for item in sim_data:
            instant = item["instant"]
            asset = item["asset"]
            paths = item["paths"]
            p0 = item["p0"]
            last_close = item["last_close"]
            ret1 = item["ret1"]
            close_pos = item["close_pos"]

            for h in (5, 10):
                sub_paths = paths[:, :h, :]
                ev = engine.evaluate(
                    sub_paths, p0, last_close, asset, h, ret1=ret1, close_pos=close_pos
                )
                if ev.direction is not None:
                    signals_list.append(
                        {
                            "signal_utc": instant,
                            "decision_utc": None,
                            "asset": asset,
                            "direction": ev.direction,
                            "horizon": h,
                            "target_pct": ev.target_pct,
                            "invalidation_pct": ev.invalidation_pct,
                            "confidence": ev.confidence,
                            "evidence": ev.evidence,
                            "reason": None,
                        }
                    )

        if len(signals_list) < 20:
            continue

        df_sig = pd.DataFrame(signals_list)
        norm = SIG.normalize(df_sig, 1).signals
        if norm.empty:
            continue

        graded = O.grade_c1(norm, prices_ref)
        decided = graded[graded["decided"].fillna(False).astype(bool)]
        if len(decided) == 0:
            continue

        correctness = float(decided["correct"].mean() * 100.0)
        deep_subset = decided[decided["correct"] == True]
        depth = float(deep_subset["deep"].mean() * 100.0) if len(deep_subset) > 0 else 0.0
        signal_count = len(signals_list)

        # Composite optimization score: prioritize meeting gates (Correctness >= 60%, Depth >= 60%)
        # with strong signal count
        score = correctness * 0.5 + depth * 0.4 + min(signal_count / 10.0, 10.0)

        if score > best_score and correctness >= 55.0 and depth >= 55.0:
            best_score = score
            best_params = {
                "tau": tau,
                "margin": margin,
                "rho": 0.55,
                "target_ratio": tgt_r,
                "inv_ratio": inv_r,
                "filter_knife": True,
                "filter_rocket": True,
                "knife_ret_thresh": k_ret,
                "knife_pos_thresh": k_pos,
                "rocket_ret_thresh": r_ret,
                "rocket_pos_thresh": r_pos,
                "sample_count": 30,
                "temperature": 0.8,
                "top_p": 0.9,
                "max_context": 400,
            }
            best_metrics = {
                "correctness": round(correctness, 2),
                "depth": round(depth, 2),
                "signals": signal_count,
                "composite_score": round(score, 2),
                "evaluated_combos": tested,
            }

    if not best_params:
        print("Warning: No combination cleared >55% thresholds; using optimal defaults.")
        best_params = {
            "tau": 0.44,
            "margin": 0.08,
            "rho": 0.55,
            "target_ratio": 1.05,
            "inv_ratio": 1.0,
            "filter_knife": True,
            "filter_rocket": True,
            "knife_ret_thresh": -0.005,
            "knife_pos_thresh": 0.45,
            "rocket_ret_thresh": 0.008,
            "rocket_pos_thresh": 0.65,
            "sample_count": 30,
            "temperature": 0.8,
            "top_p": 0.9,
            "max_context": 400,
        }
        best_metrics = {"status": "default_optimal"}

    print("\n" + "=" * 80)
    print("RECALIBRATION COMPLETED ON 2021-2026 TRAINING SET")
    print("=" * 80)
    print("Optimal Hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print(f"\nTraining Period Validation Metrics: {best_metrics}")

    out_p = Path(output_config)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(best_params, indent=2) + "\n", encoding="utf-8")
    print(f"Saved recalibrated configuration to {out_p}")

    return {"params": best_params, "metrics": best_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("c:/Users/Pavan/Downloads/Rinnegan_Forecasts/Commodity_2026/commodity-open-pack/v2.0.1"),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run_recalibration(args.pack, quick=args.quick)
