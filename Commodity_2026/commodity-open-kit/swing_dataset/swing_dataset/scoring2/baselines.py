"""The predeclared baselines of sections 3 and 4, as loops refit at every cutoff.

Each one issues a decision record for every asset and every session of the window at that
session's pre-open, from what was published before it: the prior sessions' highs, lows and
settlements, the 20-session mean and realized volatility of the feature chain, and the trailing
distribution of the favorable excursion that matured before the cutoff. Nothing is fitted once and
reused; every cutoff refits on its own past.

Task 1: `range_break` (the prior session's high and low) and `stretch_reversal` (the 20-session
mean and realized volatility).
Task 2: `prevailing_direction` (the null of section 4), `rv_band` (a realized-volatility band with
the trend sign) and `range_break_vol` (the range break with a volatility-scaled target).

Every baseline issues at horizons 5 and 10 (`BASELINE_HORIZONS`), the ones the period scores:
both on the holdout, 5 on the forward window. Task 2 accepts any horizon from 1 to 20, and its
baselines stay at those two.

They show the mechanics and give the report a floor. They clear no prize.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..scoring.chain import FeatureChains
from . import BASELINE_HORIZONS, THRESHOLD_PCT
from .prices import PriceSeries, ns_of

LOOKBACK_SESSIONS = 60
STRETCH_LOOKBACK = 250
STRETCH_QUANTILE = 0.80
BAND_LOW, BAND_HIGH = 0.10, 0.90
MIN_REFIT_ROWS = 10
TREND_SESSIONS = 20
NULL_SESSIONS = 5


@dataclass
class AssetContext:
    """Point-in-time arrays for one asset: the session bars, the chain features, the excursions."""

    asset: str
    prices: PriceSeries
    frame: pd.DataFrame
    features: pd.DataFrame
    horizons: tuple[int, ...]
    mfe: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    matured_ns: dict[int, np.ndarray] = field(default_factory=dict)

    @classmethod
    def build(
        cls, asset: str, prices: PriceSeries, chains: FeatureChains, horizons: tuple[int, ...]
    ) -> AssetContext:
        frame = prices.frame(asset).reset_index(drop=True)
        features = chains.feature_frame(asset, frame["preopen_utc"]).reset_index(drop=True)
        ctx = cls(asset=asset, prices=prices, frame=frame, features=features, horizons=horizons)
        high = frame["high"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
        settle = frame["settlement"].to_numpy(dtype=float)
        avail = frame["available_utc"]
        avail_ns = np.where(
            avail.notna().to_numpy(),
            avail.astype("datetime64[ns, UTC]").astype("int64"),
            np.iinfo("int64").max,
        )
        ctx.avail_ns = np.maximum.accumulate(avail_ns)
        n = len(frame)
        for h in horizons:
            up = np.full(n, np.nan)
            down = np.full(n, np.nan)
            matured = np.full(n, np.iinfo("int64").max, dtype="int64")
            for p in range(n - h):
                base = settle[p]
                window_high = high[p + 1 : p + 1 + h]
                window_low = low[p + 1 : p + 1 + h]
                if not np.isfinite(base) or base <= 0 or not np.isfinite(window_high).any():
                    continue
                up[p] = (np.nanmax(window_high) - base) / base * 100.0
                down[p] = (base - np.nanmin(window_low)) / base * 100.0
                matured[p] = ctx.avail_ns[p + h]
            ctx.mfe[(h, "up")] = up
            ctx.mfe[(h, "down")] = down
            ctx.matured_ns[h] = np.maximum.accumulate(matured)
        return ctx

    def known_pos(self, t_ns: int) -> int:
        """The last session published strictly before ``t_ns``."""
        return int(np.searchsorted(self.avail_ns, t_ns, side="left")) - 1

    def matured_slice(self, t_ns: int, horizon: int) -> slice:
        end = int(np.searchsorted(self.matured_ns[horizon], t_ns, side="left"))
        return slice(max(0, end - LOOKBACK_SESSIONS), end)

    def trailing_mfe(self, t_ns: int, horizon: int, direction: str) -> np.ndarray:
        values = self.mfe[(horizon, direction)][self.matured_slice(t_ns, horizon)]
        return values[np.isfinite(values)]

    def trailing_scaled_mfe(self, t_ns: int, horizon: int, direction: str) -> np.ndarray:
        window = self.matured_slice(t_ns, horizon)
        values = self.mfe[(horizon, direction)][window]
        rv = self.features["rv20"].to_numpy(dtype=float)[window]
        scale = rv * 100.0 * np.sqrt(horizon)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = values / scale
        return ratio[np.isfinite(ratio) & (scale > 0)]

    def trailing_stretch(self, t_ns: int) -> np.ndarray:
        end = self.known_pos(t_ns) + 1
        values = np.abs(
            self.features["stretch"].to_numpy(dtype=float)[max(0, end - STRETCH_LOOKBACK) : end]
        )
        return values[np.isfinite(values)]

    def value(self, column: str, pos: int) -> float:
        if pos < 0 or pos >= len(self.features):
            return float("nan")
        return float(self.features[column].iloc[pos])


def _finite(value) -> bool:
    return value is not None and np.isfinite(value)


def _quantile(values: np.ndarray, q: float) -> float | None:
    if values.size < MIN_REFIT_ROWS:
        return None
    return float(np.quantile(values, q))


class RangeBreak:
    """Task 1: the prior session settled outside the session before it."""

    name = "range_break"
    challenge = 1

    def decide(self, ctx: AssetContext, t_ns: int, horizon: int) -> dict:
        k = ctx.known_pos(t_ns)
        if k < 1:
            return {"kind": "no_signal", "reason": "no prior session published yet"}
        settle = ctx.frame["settlement"].iloc[k]
        high = ctx.frame["high"].iloc[k - 1]
        low = ctx.frame["low"].iloc[k - 1]
        if not (np.isfinite(settle) and np.isfinite(high) and np.isfinite(low)):
            return {"kind": "no_signal", "reason": "prior session bar incomplete"}
        if settle > high:
            direction = "up"
        elif settle < low:
            direction = "down"
        else:
            return {
                "kind": "no_signal",
                "reason": f"settlement {settle:g} inside the prior range {low:g} to {high:g}",
            }
        threshold = THRESHOLD_PCT[ctx.asset]
        median = _quantile(ctx.trailing_mfe(t_ns, horizon, direction), 0.5)
        target = max(threshold, median) if _finite(median) else threshold
        return {
            "kind": "signal",
            "direction": direction,
            "target_pct": target,
            "invalidation_pct": threshold,
            "confidence": 0.5,
            "evidence": (
                f"range break: settlement {settle:g} against the prior session's "
                f"high {high:g} and low {low:g}; target from the trailing median excursion"
            ),
            "fallback": not _finite(median),
        }


class StretchReversal:
    """Task 1: the chain is stretched from its 20-session mean, so lean the other way."""

    name = "stretch_reversal"
    challenge = 1

    def decide(self, ctx: AssetContext, t_ns: int, horizon: int) -> dict:
        k = ctx.known_pos(t_ns)
        stretch = ctx.value("stretch", k)
        if not np.isfinite(stretch):
            return {"kind": "no_signal", "reason": "no 20-session stretch yet"}
        cut = _quantile(ctx.trailing_stretch(t_ns), STRETCH_QUANTILE)
        if cut is None:
            return {"kind": "no_signal", "reason": "not enough history to fit the stretch cut"}
        if stretch >= cut:
            direction = "down"
        elif stretch <= -cut:
            direction = "up"
        else:
            return {
                "kind": "no_signal",
                "reason": f"stretch {stretch:+.2f} inside the fitted band of {cut:.2f}",
            }
        threshold = THRESHOLD_PCT[ctx.asset]
        median = _quantile(ctx.trailing_mfe(t_ns, horizon, direction), 0.5)
        target = max(threshold, median) if _finite(median) else threshold
        return {
            "kind": "signal",
            "direction": direction,
            "target_pct": target,
            "invalidation_pct": threshold,
            "confidence": 0.5,
            "evidence": (
                f"stretch reversal: stretch {stretch:+.2f} against the fitted {STRETCH_QUANTILE:g} "
                f"quantile {cut:.2f} of the last {STRETCH_LOOKBACK} sessions"
            ),
            "fallback": not _finite(median),
        }


class PrevailingDirection:
    """Task 2: section 4's null, the sign of the last five completed sessions' move."""

    name = "prevailing_direction"
    challenge = 2

    def decide(self, ctx: AssetContext, t_ns: int, horizon: int) -> dict:
        # The same call the scorer makes for section 4's null, so the baseline and the null the
        # report measures an entry against are the same rule and not two readings of it.
        move = ctx.prices.trailing_move_pct(ctx.asset, pd.Timestamp(t_ns, tz="UTC"), NULL_SESSIONS)
        if move is None:
            return {"kind": "no_signal", "reason": "fewer than five published sessions"}
        direction = "up" if move >= 0 else "down"
        threshold = THRESHOLD_PCT[ctx.asset]
        band = _band(ctx, t_ns, horizon, direction, threshold)
        return {
            "kind": "signal",
            "direction": direction,
            "expected_pct": threshold,
            "p10_pct": band[0],
            "p90_pct": band[1],
            "invalidation_pct": threshold,
            "evidence": f"prevailing direction: five-session move {move:+.2f} percent",
            "fallback": band[2],
        }


class RvBand:
    """Task 2: a realized-volatility band carrying the sign of the 20-session trend."""

    name = "rv_band"
    challenge = 2

    def decide(self, ctx: AssetContext, t_ns: int, horizon: int) -> dict:
        k = ctx.known_pos(t_ns)
        rv = ctx.value("rv20", k)
        trend = ctx.value("ret20", k)
        if not (np.isfinite(rv) and rv > 0 and np.isfinite(trend)):
            return {"kind": "no_signal", "reason": "no realized volatility or trend yet"}
        direction = "up" if trend >= 0 else "down"
        scale = rv * 100.0 * np.sqrt(horizon)
        kappa = _quantile(ctx.trailing_scaled_mfe(t_ns, horizon, direction), 0.5)
        expected = scale * kappa if _finite(kappa) else scale
        lo = _quantile(ctx.trailing_scaled_mfe(t_ns, horizon, direction), BAND_LOW)
        hi = _quantile(ctx.trailing_scaled_mfe(t_ns, horizon, direction), BAND_HIGH)
        fallback = not (_finite(lo) and _finite(hi))
        return {
            "kind": "signal",
            "direction": direction,
            "expected_pct": float(expected),
            "p10_pct": float(scale * lo) if not fallback else float(expected * 0.5),
            "p90_pct": float(scale * hi) if not fallback else float(expected * 2.0),
            "invalidation_pct": THRESHOLD_PCT[ctx.asset],
            "evidence": (
                f"volatility band: rv20 {rv:.5f} over {horizon} sessions with the "
                f"{TREND_SESSIONS}-session trend {trend:+.4f}"
            ),
            "fallback": fallback,
        }


class RangeBreakVol:
    """Task 2: the range break of Task 1 with a volatility-scaled target."""

    name = "range_break_vol"
    challenge = 2

    def __init__(self) -> None:
        self.trigger = RangeBreak()

    def decide(self, ctx: AssetContext, t_ns: int, horizon: int) -> dict:
        base = self.trigger.decide(ctx, t_ns, horizon)
        if base["kind"] != "signal":
            return base
        direction = base["direction"]
        k = ctx.known_pos(t_ns)
        rv = ctx.value("rv20", k)
        if not (np.isfinite(rv) and rv > 0):
            return {"kind": "no_signal", "reason": "no realized volatility for the target"}
        scale = rv * 100.0 * np.sqrt(horizon)
        kappa = _quantile(ctx.trailing_scaled_mfe(t_ns, horizon, direction), 0.5)
        expected = scale * kappa if _finite(kappa) else scale
        lo = _quantile(ctx.trailing_scaled_mfe(t_ns, horizon, direction), BAND_LOW)
        hi = _quantile(ctx.trailing_scaled_mfe(t_ns, horizon, direction), BAND_HIGH)
        fallback = not (_finite(lo) and _finite(hi))
        return {
            "kind": "signal",
            "direction": direction,
            "expected_pct": float(expected),
            "p10_pct": float(scale * lo) if not fallback else float(expected * 0.5),
            "p90_pct": float(scale * hi) if not fallback else float(expected * 2.0),
            "invalidation_pct": THRESHOLD_PCT[ctx.asset],
            "evidence": base["evidence"] + "; target scaled by rv20",
            "fallback": fallback,
        }


def _band(
    ctx: AssetContext, t_ns: int, horizon: int, direction: str, expected: float
) -> tuple[float, float, bool]:
    values = ctx.trailing_mfe(t_ns, horizon, direction)
    lo = _quantile(values, BAND_LOW)
    hi = _quantile(values, BAND_HIGH)
    if _finite(lo) and _finite(hi):
        return float(lo), float(hi), False
    return float(expected * 0.5), float(expected * 2.0), True


CHALLENGE1 = (RangeBreak, StretchReversal)
CHALLENGE2 = (PrevailingDirection, RvBand, RangeBreakVol)


def baselines_for(challenge: int) -> list:
    return [cls() for cls in (CHALLENGE1 if challenge == 1 else CHALLENGE2)]


def run(
    baseline,
    prices: PriceSeries,
    chains: FeatureChains,
    window,
    assets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Drive one baseline over every asset and session of the window, at each session's pre-open."""
    scored = window.horizons(baseline.challenge)
    horizons = tuple(h for h in BASELINE_HORIZONS if h in scored)
    rows: list[dict] = []
    for asset in assets or prices.assets():
        ctx = AssetContext.build(asset, prices, chains, horizons)
        frame = ctx.frame
        inside = (frame["preopen_utc"] >= window.start) & (frame["close_utc"] <= window.end)
        for pos in np.flatnonzero(inside.to_numpy()):
            cutoff = frame["preopen_utc"].iloc[int(pos)]
            t_ns = ns_of(cutoff)
            issued = False
            reason = None
            for horizon in horizons:
                out = baseline.decide(ctx, t_ns, horizon)
                if out["kind"] != "signal":
                    reason = out["reason"]
                    continue
                issued = True
                rows.append(
                    {
                        "asset": asset,
                        "signal_utc": cutoff,
                        "direction": out["direction"],
                        "horizon": horizon,
                        "invalidation_pct": out["invalidation_pct"],
                        "evidence": out["evidence"],
                        "decision_utc": pd.NaT,
                        "reason": None,
                        "baseline": baseline.name,
                        "fallback": bool(out.get("fallback", False)),
                        **(
                            {
                                "target_pct": out["target_pct"],
                                "confidence": out["confidence"],
                            }
                            if baseline.challenge == 1
                            else {
                                "expected_pct": out["expected_pct"],
                                "p10_pct": out["p10_pct"],
                                "p90_pct": out["p90_pct"],
                            }
                        ),
                    }
                )
            if not issued:
                rows.append(
                    {
                        "asset": asset,
                        "signal_utc": pd.NaT,
                        "direction": None,
                        "horizon": None,
                        "invalidation_pct": np.nan,
                        "evidence": None,
                        "decision_utc": cutoff,
                        "reason": reason or "no signal",
                        "baseline": baseline.name,
                        "fallback": False,
                        **(
                            {"target_pct": np.nan, "confidence": np.nan}
                            if baseline.challenge == 1
                            else {
                                "expected_pct": np.nan,
                                "p10_pct": np.nan,
                                "p90_pct": np.nan,
                            }
                        ),
                    }
                )
    columns = (
        [
            "asset",
            "signal_utc",
            "direction",
            "horizon",
            "target_pct",
            "invalidation_pct",
            "confidence",
            "evidence",
        ]
        if baseline.challenge == 1
        else [
            "asset",
            "signal_utc",
            "direction",
            "horizon",
            "expected_pct",
            "p10_pct",
            "p90_pct",
            "invalidation_pct",
            "evidence",
        ]
    ) + ["decision_utc", "reason", "baseline", "fallback"]
    return pd.DataFrame(rows, columns=columns)
