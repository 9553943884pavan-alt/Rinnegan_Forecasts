"""The five shipped baselines, behind the kit's fit / predict / update protocol.

The baselines of CHALLENGE-SPEC sections 3 and 4 are implemented once, in
``swing_dataset.scoring2.baselines``, and that is what the report scores an entry against. This
module does not reimplement them: it builds the point-in-time view they need out of frames the
client returned and turns ``decide`` into the shape ``kit.loop`` expects, so the baseline a
participant reads here is the exact object they are ranked against.

| challenge | name | rule |
|---|---|---|
| 1 | ``range_break`` | the prior session settled outside the session before it |
| 1 | ``stretch_reversal`` | the chain is past its fitted stretch quantile, so lean the other way |
| 2 | ``prevailing_direction`` | section 4's null: the sign of the last five sessions' move |
| 2 | ``rv_band`` | the 20-session trend sign with a realized-volatility band |
| 2 | ``range_break_vol`` | the range break with the volatility-scaled target and band |

All of them are loops: nothing is fitted once and reused, and every instant refits its own
quantities on rows published before it. ``fit`` therefore has nothing to do and ``update`` has
nothing to learn from; both are kept so that a baseline drops into ``run.py`` unchanged and a
participant can diff their own model against one that works.

They show the mechanics and give the report a floor. They clear no prize.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..scoring.chain import FeatureChains
from ..scoring2.baselines import (
    BAND_HIGH,
    BAND_LOW,
    CHALLENGE1,
    CHALLENGE2,
    LOOKBACK_SESSIONS,
    MIN_REFIT_ROWS,
    NULL_SESSIONS,
    STRETCH_LOOKBACK,
    STRETCH_QUANTILE,
    TREND_SESSIONS,
    AssetContext,
    baselines_for,
)
from ..scoring2.baselines import run as run_baseline
from ..scoring2.prices import PriceSeries, ns_of
from . import ASSETS, DEFAULT_HORIZONS
from .loop import price_view

#: Sessions of history a view keeps. The baselines look back 250 sessions for the stretch cut and
#: 60 for the trailing excursion quantiles, so this is the longest window any of them asks for
#: plus room for the rolling features behind it.
VIEW_LOOKBACK_SESSIONS = 400

BASELINES = {cls.name: cls for cls in (*CHALLENGE1, *CHALLENGE2)}

__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BASELINES",
    "CHALLENGE1",
    "CHALLENGE2",
    "LOOKBACK_SESSIONS",
    "MIN_REFIT_ROWS",
    "NULL_SESSIONS",
    "STRETCH_LOOKBACK",
    "STRETCH_QUANTILE",
    "TREND_SESSIONS",
    "VIEW_LOOKBACK_SESSIONS",
    "AssetContext",
    "BaselineModel",
    "MarketView",
    "baseline",
    "baselines_for",
    "response_from",
    "run_baseline",
]


def baseline(name: str):
    """One shipped baseline by name, as the scorer instantiates it."""
    try:
        return BASELINES[name]()
    except KeyError:
        raise KeyError(f"unknown baseline {name!r}; the shipped ones are {sorted(BASELINES)}") from None


class MarketView:
    """The price path and the feature chain at one instant, from rows available before it.

    ``scoring2.prices.PriceSeries`` and ``scoring.chain.FeatureChains`` both take frames, so the
    kit builds them from what the point-in-time client returned rather than opening a pack file.
    ``context(asset)`` is the ``AssetContext`` the baselines decide on: the session bars, the chain
    features and the trailing excursions, all of them cut off at the instant this view was built
    for.
    """

    def __init__(
        self,
        prices: PriceSeries,
        chains: FeatureChains,
        asof: datetime,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    ) -> None:
        self.prices = prices
        self.chains = chains
        self.asof = asof
        self.horizons = tuple(int(h) for h in horizons)
        self._contexts: dict[str, AssetContext | None] = {}

    @classmethod
    def at(
        cls,
        pit,
        asof: datetime,
        *,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        lookback_sessions: int | None = VIEW_LOOKBACK_SESSIONS,
        quotes: bool = False,
    ) -> MarketView:
        prices, chains = price_view(pit, asof, lookback_sessions=lookback_sessions, quotes=quotes)
        return cls(prices, chains, asof, horizons)

    def context(self, asset: str) -> AssetContext | None:
        """The asset's context, built once per view; None when the view holds no bars for it."""
        if asset not in self._contexts:
            if asset not in self.prices.assets() or asset not in self.chains.assets():
                self._contexts[asset] = None
            else:
                self._contexts[asset] = AssetContext.build(asset, self.prices, self.chains, self.horizons)
        return self._contexts[asset]

    def assets(self) -> list[str]:
        return [a for a in ASSETS if a in self.prices.assets()]

    def decide(self, rule, asset: str, instant: datetime, horizon: int) -> dict:
        """``rule.decide`` at one instant, or a ``no_signal`` dict when the view has no bars."""
        ctx = self.context(asset)
        if ctx is None:
            return {"kind": "no_signal", "reason": f"no published {asset} sessions in this view yet"}
        return rule.decide(ctx, ns_of(instant), int(horizon))


def response_from(decision: dict) -> dict:
    """A ``decide`` result as ``kit.loop`` wants it: the signal fields, or ``{"reason": ...}``."""
    if decision.get("kind") != "signal":
        return {"reason": decision.get("reason") or "no signal"}
    return {k: v for k, v in decision.items() if k not in ("kind", "fallback")}


class BaselineModel:
    """A shipped baseline as a submission model: ``fit``, ``predict``, ``update``.

    The baselines refit at every instant on rows published before it, so ``fit`` records the
    training cutoff and nothing else, and ``update`` has nothing to learn from. A Task 2
    baseline run against its own frozen copy therefore has an improvement score of zero by
    construction, which is what the baseline report says about itself.
    """

    SEED = 0
    TRAINING_DATA_CUTOFF = "none; the shipped baselines refit at every instant from the pack"

    def __init__(self, name: str, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> None:
        self.rule = baseline(name)
        self.name = name
        self.challenge = self.rule.challenge
        self.horizons = tuple(int(h) for h in horizons)
        self.pack = None
        self.train_end: datetime | None = None
        self._view: MarketView | None = None

    def fit(self, pack, train_end: datetime) -> dict:
        self.pack = pack
        self.train_end = train_end
        return {"baseline": self.name, "train_end_utc": train_end.isoformat(), "fitted": False}

    def view_at(self, as_of: datetime) -> MarketView:
        """One view per instant, so a session's reads happen once and not once per horizon."""
        if self.pack is None:
            raise RuntimeError("fit() has not been called")
        if self._view is None or self._view.asof != as_of:
            self._view = MarketView.at(self.pack, as_of, horizons=self.horizons)
        return self._view

    def predict(self, as_of: datetime, asset: str, horizon: int) -> dict:
        return response_from(self.view_at(as_of).decide(self.rule, asset, as_of, horizon))

    def update(self, matured_outcomes: pd.DataFrame) -> dict:
        return {"received": len(matured_outcomes), "refit": False}
