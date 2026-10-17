"""Kronos Zero-Shot Foundation Model for Task 1: Pre-Open Directional Swing Prediction.

Pre-trained autoregressive Kronos foundation model generating multi-step future OHLCV trajectories.
Calibrated point-in-time decision layer translating simulated paths into:
1. Direction: `up`, `down`, or `no_signal`.
2. Target Percentage: `target_pct` (optimized for Depth Gate >= 60%).
3. Invalidation Level: `invalidation_pct` (<= threshold).
4. Confidence & Evidence: Model probability and diagnostic attribution.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from swing_dataset.kit import DEFAULT_HORIZONS
from swing_dataset.pack.pit import PointInTime

from src.forecaster import RawPathPredictor, TrajectoryForecaster
from src.inputs import DataPipeline
from src.path_rules import PathRulesEngine

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "configs" / "task1_config.json"


class KronosSwingDetector:
    """Task 1 zero-shot foundation model swing detector."""

    SEED = 20260919
    # Kronos pre-training data cutoff precedes 2025-06-01 (complying with RULES section 6)
    TRAINING_DATA_CUTOFF = "2025-06-01"

    def __init__(
        self,
        challenge: int = 1,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        config_path: Path | str | None = None,
    ) -> None:
        if challenge != 1:
            raise ValueError(f"KronosSwingDetector is for Task 1, got challenge={challenge}")

        self.challenge = challenge
        self.horizons = tuple(int(h) for h in horizons)
        self.config = self._load_config(config_path or CONFIG_PATH)

        self.pack: Optional[PointInTime] = None
        self.train_end: Optional[datetime] = None

        self.pipeline: Optional[DataPipeline] = None
        self.predictor: Optional[RawPathPredictor] = None
        self.forecaster: Optional[TrajectoryForecaster] = None
        self.rules_engine: Optional[PathRulesEngine] = None

        self._initialized = False

    def _load_config(self, path: Path | str) -> dict:
        config_p = Path(path)
        if config_p.exists():
            try:
                return json.loads(config_p.read_text(encoding="utf-8"))
            except Exception:
                pass
        # Default audited hyperparameter values
        return {
            "tau": 0.46,
            "margin": 0.10,
            "rho": 0.55,
            "target_ratio": 1.00,
            "inv_ratio": 1.0,
            "filter_knife": True,
            "filter_rocket": True,
            "knife_ret_thresh": -0.005,
            "knife_pos_thresh": 0.45,
            "rocket_ret_thresh": 0.008,
            "realign_momentum": True,
            "sample_count": 30,
            "temperature": 0.8,
            "top_p": 0.9,
            "max_context": 400,
        }

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        if self.pack is None:
            raise RuntimeError("fit() must be called before predict()")

        max_ctx = int(self.config.get("max_context", 60))
        sample_count = int(self.config.get("sample_count", 30))
        temp = float(self.config.get("temperature", 0.65))
        top_p = float(self.config.get("top_p", 0.85))

        self.pipeline = DataPipeline(self.pack, max_context=max_ctx)
        self.predictor = RawPathPredictor(max_context=512)
        self.forecaster = TrajectoryForecaster(
            predictor=self.predictor,
            max_pred_len=max(self.horizons),
            sample_count=sample_count,
            temperature=temp,
            top_p=top_p,
        )
        tau = float(self.config.get("tau", 0.44))
        self.rules_engine = PathRulesEngine(
            tau=tau,
            tau_up=float(self.config.get("tau_up", tau)),
            tau_dn=float(self.config.get("tau_dn", tau)),
            margin=float(self.config.get("margin", 0.10)),
            rho=float(self.config.get("rho", 0.55)),
            target_ratio=float(self.config.get("target_ratio", 1.00)),
            inv_ratio=float(self.config.get("inv_ratio", 1.00)),
            filter_knife=bool(self.config.get("filter_knife", True)),
            filter_rocket=bool(self.config.get("filter_rocket", True)),
            knife_ret_thresh=float(self.config.get("knife_ret_thresh", -0.008)),
            knife_pos_thresh=float(self.config.get("knife_pos_thresh", 0.45)),
            rocket_ret_thresh=float(self.config.get("rocket_ret_thresh", 0.005)),
            rocket_pos_thresh=float(self.config.get("rocket_pos_thresh", 0.60)),
            realign_momentum=bool(self.config.get("realign_momentum", True)),
        )
        self._initialized = True


    # -- the kit protocol interface -------------------------------------------------------------

    def fit(self, pack: PointInTime, train_end: datetime) -> dict:
        """Called once at the period's cutoff."""
        self.pack = pack
        self.train_end = train_end
        self._lazy_init()
        return {
            "model": "KronosSwingDetector",
            "train_end_utc": train_end.isoformat(),
            "pretrained_weights": "NeoQuasar/Kronos-small",
            "seed": self.SEED,
            "training_data_cutoff": self.TRAINING_DATA_CUTOFF,
            "config": self.config,
        }

    def predict(self, as_of: datetime, asset: str, horizon: int) -> dict | None:
        """Called at every decision instant for each asset and horizon."""
        self._lazy_init()
        assert self.pipeline is not None
        assert self.forecaster is not None
        assert self.rules_engine is not None

        # 1. Build roll-normalized continuous series up to as_of
        data = self.pipeline.build_series(asset, as_of, pred_len=max(self.horizons))
        if data is None or len(data.input_df) < 20:
            return {"reason": f"insufficient historical data for {asset} at {as_of.isoformat()}"}

        # Point-in-time prior session momentum and bar position features (strictly <= as_of)
        close_arr = data.input_df["close"].to_numpy(dtype=float)
        high_arr = data.input_df["high"].to_numpy(dtype=float)
        low_arr = data.input_df["low"].to_numpy(dtype=float)
        ret1 = float((close_arr[-1] - close_arr[-2]) / close_arr[-2]) if len(close_arr) >= 2 and close_arr[-2] > 0 else 0.0
        bar_range = float(high_arr[-1] - low_arr[-1])
        close_pos = float((close_arr[-1] - low_arr[-1]) / bar_range) if bar_range > 0 else 0.5

        # 2. Retrieve simulated paths (uses cached 10-step trajectory when horizon=10 follows horizon=5)
        paths = self.forecaster.get_paths(
            asset=asset,
            as_of=as_of,
            series_df=data.input_df,
            x_timestamp=data.x_timestamp,
            y_timestamp=data.y_timestamp,
            horizon=horizon,
        )

        # 3. Evaluate barriers and decide direction, target, and invalidation with financial filters
        evaluation = self.rules_engine.evaluate(
            paths=paths,
            p0=data.p0,
            last_close=data.last_close,
            asset=asset,
            horizon=horizon,
            ret1=ret1,
            close_pos=close_pos,
        )

        if evaluation.direction is None:
            return {"reason": evaluation.reason or "no_signal"}

        return {
            "direction": evaluation.direction,
            "target_pct": evaluation.target_pct,
            "invalidation_pct": evaluation.invalidation_pct,
            "confidence": evaluation.confidence,
            "evidence": evaluation.evidence,
        }

    def update(self, matured_outcomes: pd.DataFrame) -> None:
        """Called at the start of each session with settled outcomes of earlier signals."""
        # Zero-shot foundation model: weights are fixed pre-trained.
        pass


MODELS = {1: KronosSwingDetector}


def build(challenge: int, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> KronosSwingDetector:
    """Factory called by run.py."""
    if challenge != 1:
        raise ValueError(f"challenge must be 1, got {challenge}")
    return KronosSwingDetector(challenge, horizons)
