"""Episode-block confidence intervals for scores that live on different unit sets.

Section 5 asks for episode blocks of `h` sessions per asset, 2000 seeded resamples, and an
interval printed for every score. The three scores of a challenge do not share a unit set: the
catch rate counts swings, correctness counts signals, depth counts the correct ones, and the
success rate is an average of all three. Resampling each set on its own would give three intervals
drawn from three different sets of episodes and no interval at all for the average.

So the episodes are drawn once. The universe is every (asset, block) pair any unit falls in; a
replicate draws that many episodes with replacement and recomputes every score on the drawn
episodes, the average included. For a single unit set covering the whole universe this is exactly
``scoring.folds.block_bootstrap_mean`` with the same seed, and the test suite asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..scoring.folds import DEFAULT_BOOTSTRAP, episode_ids, session_calendar
from .prices import PriceSeries, ns_of

DEFAULT_LEVEL = 0.90


def episode_keys(
    prices: PriceSeries,
    window_start,
    horizon: int,
    assets: np.ndarray | list,
    instants,
) -> np.ndarray:
    """``asset:block`` per unit, blocks of ``horizon`` sessions counted from the window start."""
    assets = np.asarray(assets, dtype=object)
    instants = pd.to_datetime(pd.Series(list(instants)), utc=True).reset_index(drop=True)
    out = np.empty(len(assets), dtype=object)
    for asset in pd.unique(assets):
        rows = np.flatnonzero(assets == asset)
        if not len(rows):
            continue
        series = prices.session_frame(str(asset))["preopen_utc"]
        calendar = session_calendar(series)
        anchor = int(np.searchsorted(calendar, ns_of(window_start)))
        anchors = []
        for i in rows:
            pos = prices.session_pos(str(asset), instants.iloc[int(i)])
            if pos is None:
                pos = max(prices.last_session_pos(str(asset), instants.iloc[int(i)]), 0)
            anchors.append(series.iloc[pos])
        blocks = episode_ids(pd.Series(anchors), horizon, calendar, anchor=anchor)
        for i, block in zip(rows, blocks, strict=True):
            out[i] = f"{asset}:{int(block)}"
    return out


@dataclass(frozen=True, slots=True)
class Replicates:
    """Observed ratios and their 2000 resampled copies, on one shared draw of episodes."""

    observed: dict[str, float]
    draws: dict[str, np.ndarray]
    n: dict[str, int]
    n_episodes: dict[str, int]
    universe: int
    n_boot: int
    seed: int
    level: float

    def interval(self, name: str, scale: float = 100.0) -> dict:
        return self.of(
            self.observed[name] * scale,
            self.draws[name] * scale,
            self.n[name],
            self.n_episodes[name],
        )

    def of(self, value: float, samples: np.ndarray, n: int, n_episodes: int) -> dict:
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        alpha = (1.0 - self.level) / 2.0
        lo, hi = (
            np.quantile(finite, [alpha, 1.0 - alpha])
            if finite.size
            else (float("nan"), float("nan"))
        )
        return {
            "value": float(value) if np.isfinite(value) else None,
            "ci_lo": float(lo) if np.isfinite(lo) else None,
            "ci_hi": float(hi) if np.isfinite(hi) else None,
            "level": self.level,
            "n": int(n),
            "n_episodes": int(n_episodes),
            "n_boot": self.n_boot,
            "seed": self.seed,
            "resamples_used": int(finite.size),
        }


@dataclass
class EpisodeBootstrap:
    """Collect unit sets, then resample the shared episode universe once for all of them."""

    n_boot: int = DEFAULT_BOOTSTRAP
    seed: int = 0
    level: float = DEFAULT_LEVEL
    _sets: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def add(self, name: str, values, keys) -> None:
        values = np.asarray(values, dtype=float)
        keys = np.asarray(keys, dtype=object)
        if values.shape != keys.shape:
            raise ValueError(f"{name}: values and episode keys must align")
        self._sets[name] = (values, keys)

    def run(self) -> Replicates:
        universe = (
            np.unique(np.concatenate([k for _, k in self._sets.values()]))
            if any(len(k) for _, k in self._sets.values())
            else np.array([], dtype=object)
        )
        size = len(universe)
        rng = np.random.default_rng(self.seed)
        draws = (
            rng.integers(0, size, size=(self.n_boot, size))
            if size
            else np.zeros((self.n_boot, 0), dtype=int)
        )
        observed, samples, counts, episodes = {}, {}, {}, {}
        for name, (values, keys) in self._sets.items():
            counts[name] = int(values.size)
            if values.size == 0 or size == 0:
                observed[name] = float("nan")
                samples[name] = np.full(self.n_boot, np.nan)
                episodes[name] = 0
                continue
            index = np.searchsorted(universe, keys)
            sums = np.bincount(index, weights=values, minlength=size)
            n_rows = np.bincount(index, minlength=size).astype(float)
            observed[name] = float(values.mean())
            episodes[name] = int(np.count_nonzero(n_rows))
            with np.errstate(invalid="ignore", divide="ignore"):
                samples[name] = sums[draws].sum(axis=1) / n_rows[draws].sum(axis=1)
        return Replicates(
            observed=observed,
            draws=samples,
            n=counts,
            n_episodes=episodes,
            universe=size,
            n_boot=self.n_boot,
            seed=self.seed,
            level=self.level,
        )
