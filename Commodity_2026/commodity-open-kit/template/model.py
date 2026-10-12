"""The template models: one for each task, at rules 2.6.

Replace this file with your own model. ``run.py`` only needs the methods below and the ``SEED``
attribute; keep their signatures.

- ``fit(pack, train_end)`` is called once. ``pack`` is the point-in-time client; read every table
  through it at ``train_end`` (or earlier) and never open a pack file directly.
- ``predict(as_of, asset, horizon)`` is called at every decision instant, in order. Return a dict
  of signal fields, a list of such dicts (each may carry its own ``signal_utc``), ``{"reason":
  "..."}`` to record that you looked and saw nothing, or ``None`` for the same with a default
  reason. A ``no_signal`` row counts for the section 5 coverage floor; a missing row does not.
- ``update(matured_outcomes)`` is called at the start of a session with the outcomes of this run's
  own earlier signals that have settled since the last call: the signal's own columns plus
  ``p0``, ``mfe_pct``, ``mae_pct``, ``realized`` (or ``correct``), ``in_band`` (or ``deep``) and
  the window it was graded over, from ``scoring2.outcomes``, which is the code the scorer runs. It
  is never called in a frozen run.

What the two models do:

**Task 1, ``SwingDetector``.** The shipped ``range_break`` rule says a swing is starting when
the prior session settled outside the range of the session before it, and sets the target from the
trailing median favorable excursion. This model takes that trigger and keeps only the breaks that
agree with the 20-session trend, on the argument that a break against the trend is more often the
end of a move than the start of one. There is nothing to fit: the rule refits its own quantities
at every instant from rows published before it, which is what section 3's baselines do. ``update``
only keeps score, and writes the result into ``confidence``, which is stored and never graded.

**Task 2, ``SwingForecaster``.** The base call is the sign of the 20-session chain return,
and both of the model's answers are then checked against its own results. Every settled row comes
back with the excursion in the direction it was issued in and the excursion the other way, so the
model can count, per asset and base direction, how often each side reached the asset's threshold,
and invert a call the other side is clearly winning. The band is quoted from the excursions of its
own rows that went the right way, which is the population depth is scored on, with a prior read
out of the pack at the training cutoff until it has enough of its own. The frozen copy never
receives an outcome, so it never inverts and keeps the prior; that difference is the improvement
score.

Nothing here draws random numbers or touches the network; the declared seed exists for the
determinism rule of section 4a.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from swing_dataset.kit import DEFAULT_HORIZONS, THRESHOLD_PCT
from swing_dataset.kit.baselines import BAND_HIGH, BAND_LOW, MarketView, baseline, response_from
from swing_dataset.pack.pit import PointInTime
from swing_dataset.scoring2 import BAND_WIDTH_CAP
from swing_dataset.scoring2.prices import ns_of

#: Outcomes of its own a Task 2 key needs before the model stops using the fitted prior.
MIN_OWN_OUTCOMES = 12
#: Outcomes of its own an (asset, base direction) needs before the model will invert it.
MIN_FLIP_OUTCOMES = 20
#: How far ahead the other side has to be, in share of outcomes reaching the threshold, to invert.
FLIP_MARGIN = 0.15
#: Signals of its own an asset needs before ``confidence`` stops being 0.5.
MIN_CONFIDENCE_ROWS = 10
DIRECTIONS = ("up", "down")

#: The three constants above are a worked example's choices, set on the 2026 holdout and nothing
#: more. Validate your own before you rely on them.


def _other(direction: str) -> str:
    return "down" if direction == "up" else "up"


def _quantile(values: list[float], q: float) -> float | None:
    if len(values) < MIN_OWN_OUTCOMES:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


class SwingDetector:
    """Task 1: the shipped range break, kept only where it agrees with the trend."""

    SEED = 20260919
    TRAINING_DATA_CUTOFF = "none; the rule refits at every instant from rows published before it"

    def __init__(self, challenge: int = 1, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> None:
        if challenge != 1:
            raise ValueError("SwingDetector is the Task 1 model")
        self.horizons = tuple(int(h) for h in horizons)
        self.rule = baseline("range_break")
        self.pack: PointInTime | None = None
        self.train_end: datetime | None = None
        self.scored: dict[str, list[bool]] = {}
        self._view: MarketView | None = None

    # -- the interface ---------------------------------------------------------------------------

    def fit(self, pack: PointInTime, train_end: datetime) -> dict:
        self.pack = pack
        self.train_end = train_end
        return {
            "model": "SwingDetector",
            "train_end_utc": train_end.isoformat(),
            "fitted": False,
            "note": "the range-break trigger and its target refit at every instant",
        }

    def predict(self, as_of: datetime, asset: str, horizon: int) -> dict | None:
        view = self._view_at(as_of)
        decision = response_from(view.decide(self.rule, asset, as_of, horizon))
        if "direction" not in decision:
            return decision
        trend = self._trend(view, asset, as_of)
        if trend is None:
            return {"reason": "no 20-session trend yet, so the break has nothing to agree with"}
        if (decision["direction"] == "up") != (trend >= 0):
            return {
                "reason": (
                    f"{decision['direction']} range break against a 20-session trend of "
                    f"{trend:+.4f}; a break against the trend is skipped"
                )
            }
        decision["confidence"] = self._confidence(asset)
        decision["evidence"] = f"{decision['evidence']}; 20-session trend {trend:+.4f} agrees"
        return decision

    def update(self, matured_outcomes: pd.DataFrame) -> dict:
        for row in matured_outcomes.itertuples():
            self.scored.setdefault(str(row.asset), []).append(bool(row.correct))
        return {"received": len(matured_outcomes), "assets_scored": {a: len(v) for a, v in sorted(self.scored.items())}}

    # -- internals -------------------------------------------------------------------------------

    def _view_at(self, as_of: datetime) -> MarketView:
        """One point-in-time view per instant, so a session reads the pack once, not once per key."""
        if self.pack is None:
            raise RuntimeError("fit() has not been called")
        if self._view is None or self._view.asof != as_of:
            self._view = MarketView.at(self.pack, as_of, horizons=self.horizons)
        return self._view

    def _trend(self, view: MarketView, asset: str, as_of: datetime) -> float | None:
        ctx = view.context(asset)
        if ctx is None:
            return None
        trend = ctx.value("ret20", ctx.known_pos(ns_of(as_of)))
        return None if not math.isfinite(trend) else float(trend)

    def _confidence(self, asset: str) -> float:
        """The model's own correctness on this asset so far; 0.5 until it has seen enough."""
        seen = self.scored.get(asset, [])
        if len(seen) < MIN_CONFIDENCE_ROWS:
            return 0.5
        return round(float(np.mean(seen)), 4)


class SwingForecaster:
    """Task 2: a trend direction the model checks against its own outcomes, with a depth band.

    The base call is the sign of the 20-session chain return. Two things are learned from the
    outcomes that arrive through ``update``, and both use only what ``update`` delivers.

    **Whether the base call is worth issuing.** Every settled row carries ``mfe_pct``, the
    excursion in the direction the row was issued in, and ``mae_pct``, the excursion the other
    way; between them they say how far the price ran in each direction over the window. Per asset
    and base direction the model counts how often the base side reached the asset's threshold and
    how often the other side did, and inverts that call once the other side is clearly ahead over
    enough of its own outcomes. The comparison is between two excursions measured the same way, so
    neither side is flattered by the test.

    **How deep the move goes.** Depth asks whether the realized move landed inside
    ``[p10_pct, p90_pct]``, and it is scored only on the rows that went the right way, so the band
    is quoted from the excursions of the model's own rows that went the right way. ``fit`` reads
    the same quantity out of the pack at the training cutoff as a prior to quote until the model
    has enough of its own.

    The frozen copy never receives an outcome: it never inverts and keeps the fitted prior, and
    the direction it issues where the two differ is what the improvement score measures.
    """

    SEED = 20260919
    TRAINING_DATA_CUTOFF = "the train_end passed to fit(); no pretrained weights"

    def __init__(self, challenge: int = 2, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> None:
        if challenge != 2:
            raise ValueError("SwingForecaster is the Task 2 model")
        self.horizons = tuple(int(h) for h in horizons)
        self.pack: PointInTime | None = None
        self.train_end: datetime | None = None
        self.prior: dict[tuple[str, int, str], tuple[float, float, float]] = {}
        self.own: dict[tuple[str, int, str], list[float]] = {}
        self.reach: dict[tuple[str, str], list[tuple[bool, bool]]] = {}
        self.inverted: set[tuple[str, str]] = set()
        self.base: dict[tuple[str, int, int], str] = {}
        self._view: MarketView | None = None

    # -- the interface ---------------------------------------------------------------------------

    def fit(self, pack: PointInTime, train_end: datetime) -> dict:
        """Read the prior for the favorable excursion out of the pack at the training cutoff.

        One view over the whole history available before ``train_end``, and per asset, horizon and
        direction the 10th, 50th and 90th percentiles of the excursions that reached the asset's
        threshold, which is the population depth is scored on. That is what the model quotes until
        its own outcomes replace it, and it is all the frozen copy ever has.
        """
        self.pack = pack
        self.train_end = train_end
        view = MarketView.at(pack, train_end, horizons=self.horizons, lookback_sessions=None)
        t_ns = ns_of(train_end)
        self.prior.clear()
        for asset in view.assets():
            ctx = view.context(asset)
            if ctx is None:
                continue
            threshold = THRESHOLD_PCT[asset]
            for horizon in self.horizons:
                for direction in DIRECTIONS:
                    sample = ctx.trailing_mfe(t_ns, horizon, direction)
                    sample = sample[sample >= threshold]
                    if sample.size < MIN_OWN_OUTCOMES:
                        continue
                    lo, mid, hi = (float(np.quantile(sample, q)) for q in (BAND_LOW, 0.5, BAND_HIGH))
                    self.prior[(asset, horizon, direction)] = (lo, mid, hi)
        self._view = None
        return {
            "model": "SwingForecaster",
            "train_end_utc": train_end.isoformat(),
            "prior_keys": len(self.prior),
            "assets": view.assets(),
        }

    def predict(self, as_of: datetime, asset: str, horizon: int) -> dict | None:
        if self.pack is None:
            raise RuntimeError("fit() has not been called")
        view = self._view_at(as_of)
        ctx = view.context(asset)
        if ctx is None:
            return {"reason": f"no published {asset} sessions in this view yet"}
        pos = ctx.known_pos(ns_of(as_of))
        trend = ctx.value("ret20", pos)
        rv = ctx.value("rv20", pos)
        if not (math.isfinite(trend) and math.isfinite(rv) and rv > 0):
            return {"reason": "no 20-session trend or realized volatility published yet"}
        base = "up" if trend >= 0 else "down"
        inverted = (asset, base) in self.inverted
        direction = _other(base) if inverted else base
        self.base[(asset, ns_of(as_of), int(horizon))] = base
        lo, mid, hi, source = self._band(asset, horizon, direction, rv)
        lo, hi = self._cap(asset, lo, hi)
        note = ""
        if inverted:
            seen = self.reach[(asset, base)]
            note = f"; inverted: over {len(seen)} of its own outcomes the other side reached the threshold more often"
        return {
            "direction": direction,
            "expected_pct": mid,
            "p10_pct": lo,
            "p90_pct": hi,
            "invalidation_pct": THRESHOLD_PCT[asset],
            "evidence": (
                f"20-session trend {trend:+.4f} with realized volatility {rv:.5f}; "
                f"{source} band over {horizon} sessions{note}"
            ),
        }

    def update(self, matured_outcomes: pd.DataFrame) -> dict:
        """Score both sides of each settled row, then requote the band and revisit the inversions."""
        for row in matured_outcomes.itertuples():
            asset, horizon = str(row.asset), int(row.horizon)
            mfe, mae = float(row.mfe_pct), float(row.mae_pct)
            if not (np.isfinite(mfe) and np.isfinite(mae)):
                continue
            issued = str(row.direction)
            base = self.base.pop((asset, ns_of(row.signal_utc), horizon), issued)
            threshold = THRESHOLD_PCT[asset]
            base_reach, other_reach = (mfe >= threshold, mae >= threshold)
            if issued != base:
                base_reach, other_reach = other_reach, base_reach
            self.reach.setdefault((asset, base), []).append((base_reach, other_reach))
            if bool(row.realized):
                self.own.setdefault((asset, horizon, issued), []).append(mfe)
        for key, seen in self.reach.items():
            ahead = float(np.mean([o for _b, o in seen])) - float(np.mean([b for b, _o in seen]))
            if len(seen) >= MIN_FLIP_OUTCOMES and ahead >= FLIP_MARGIN:
                self.inverted.add(key)
            else:
                self.inverted.discard(key)
        return {
            "received": len(matured_outcomes),
            "own_samples": {f"{a}_h{h}_{d}": len(v) for (a, h, d), v in sorted(self.own.items())},
            "inverted": sorted(f"{a}_{d}" for a, d in self.inverted),
        }

    # -- internals -------------------------------------------------------------------------------

    def _view_at(self, as_of: datetime) -> MarketView:
        if self.pack is None:
            raise RuntimeError("fit() has not been called")
        if self._view is None or self._view.asof != as_of:
            self._view = MarketView.at(self.pack, as_of, horizons=self.horizons)
        return self._view

    def _band(self, asset: str, horizon: int, direction: str, rv: float) -> tuple[float, float, float, str]:
        """The model's own quantiles once it has enough of them, else the fitted prior, else vol."""
        key = (asset, horizon, direction)
        own = self.own.get(key, [])
        lo, mid, hi = (_quantile(own, q) for q in (BAND_LOW, 0.5, BAND_HIGH))
        if lo is not None and mid is not None and hi is not None:
            return lo, mid, hi, f"own ({len(own)} matured outcomes that went the right way)"
        if key in self.prior:
            lo, mid, hi = self.prior[key]
            return lo, mid, hi, "fitted prior"
        scale = rv * 100.0 * math.sqrt(horizon)
        return scale * 0.5, scale, scale * 2.0, "volatility-scaled fallback"

    @staticmethod
    def _cap(asset: str, lo: float, hi: float) -> tuple[float, float]:
        """Keep the band inside ``BAND_WIDTH_CAP`` thresholds, which is what makes a row valid."""
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        widest = BAND_WIDTH_CAP * THRESHOLD_PCT[asset]
        if hi - lo > widest:
            middle = 0.5 * (lo + hi)
            lo, hi = middle - widest / 2, middle + widest / 2
        return float(lo), float(hi)


MODELS = {1: SwingDetector, 2: SwingForecaster}


def build(challenge: int, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> SwingDetector | SwingForecaster:
    """The model for a challenge. ``run.py`` calls ``build(challenge, horizons)`` and nothing else."""
    if challenge not in MODELS:
        raise ValueError("challenge must be 1 or 2")
    return MODELS[challenge](challenge, horizons)
