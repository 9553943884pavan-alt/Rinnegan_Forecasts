"""Purged chronological walk-forward folds, horizon episodes, and the episode-block bootstrap.

Time is measured in MCX sessions: the sorted distinct cutoff instants of the frame being split
(one per trading day). A row is (asset, cutoff, horizon) with its label window ending at
``available_utc``, the instant the label became known.

- Episode: block of ``horizon`` consecutive sessions counted from the anchor session. Cutoffs in
  the same episode have overlapping label windows; cutoffs ``horizon`` sessions apart do not.
  ``episode_ids`` returns the calendar block; the evaluator's default ``asset`` scope makes every
  (asset, block) pair its own unit, and its ``calendar`` scope shares the block across assets.
- Purging: for a test fold whose first cutoff is session ``s``, a training row is kept only if
  its label was available at or before the cutoff of session ``s - embargo``. Nothing in a
  training row can then overlap or sit within ``embargo`` sessions of the test window.
- Bootstrap: paired per-row score differences are grouped by episode; episodes are resampled with
  replacement, the row-weighted mean difference recomputed, and the 90 percent interval read
  off the percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_EMBARGO = 20
DEFAULT_BOOTSTRAP = 2000


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def session_calendar(cutoffs: pd.Series) -> np.ndarray:
    """Sorted distinct cutoff instants as int64 nanoseconds."""
    return np.unique(_utc(cutoffs).astype("datetime64[ns, UTC]").astype("int64").to_numpy())


def session_ordinal(cutoffs: pd.Series, calendar: np.ndarray) -> np.ndarray:
    """Position of each cutoff in the calendar; raises if a cutoff is not a calendar session."""
    ns = _utc(cutoffs).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    pos = np.searchsorted(calendar, ns)
    pos_clipped = np.clip(pos, 0, len(calendar) - 1)
    if not np.array_equal(calendar[pos_clipped], ns):
        raise ValueError("a cutoff is not in the session calendar")
    return pos_clipped


def episode_ids(
    cutoffs: pd.Series, horizon: int, calendar: np.ndarray, anchor: int = 0
) -> np.ndarray:
    """Episode index per row: floor((session - anchor) / horizon)."""
    ordinal = session_ordinal(cutoffs, calendar)
    return (ordinal - anchor) // horizon


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start_utc: pd.Timestamp
    test_end_utc: pd.Timestamp
    purge_boundary_utc: pd.Timestamp


def purged_train_mask(
    available_utc: pd.Series,
    test_start_session: int,
    calendar: np.ndarray,
    embargo: int = DEFAULT_EMBARGO,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Rows whose label was known at or before the cutoff ``embargo`` sessions before the test."""
    boundary_session = test_start_session - embargo
    if boundary_session < 0:
        return np.zeros(len(available_utc), dtype=bool), pd.Timestamp(calendar[0], tz="UTC")
    boundary = pd.Timestamp(calendar[boundary_session], tz="UTC")
    avail = _utc(available_utc)
    mask = (avail <= boundary).to_numpy() & avail.notna().to_numpy()
    return mask, boundary


def walk_forward_folds(
    rows: pd.DataFrame,
    n_splits: int,
    embargo: int = DEFAULT_EMBARGO,
    min_train_sessions: int = 0,
) -> list[Fold]:
    """Chronological folds over ``rows`` (needs ``cutoff_utc`` and ``available_utc``).

    The session calendar after the first ``min_train_sessions`` sessions is cut into
    ``n_splits`` contiguous test blocks. Each fold trains on the purged rows before its block.
    """
    calendar = session_calendar(rows["cutoff_utc"])
    ordinal = session_ordinal(rows["cutoff_utc"], calendar)
    usable = np.arange(min_train_sessions, len(calendar))
    if n_splits < 1 or len(usable) < n_splits:
        raise ValueError("not enough sessions for the requested splits")
    blocks = np.array_split(usable, n_splits)
    folds = []
    for i, block in enumerate(blocks):
        start, end = int(block[0]), int(block[-1])
        test_idx = np.flatnonzero((ordinal >= start) & (ordinal <= end))
        mask, boundary = purged_train_mask(rows["available_utc"], start, calendar, embargo)
        train_idx = np.flatnonzero(mask & (ordinal < start))
        folds.append(
            Fold(
                index=i,
                train_idx=train_idx,
                test_idx=test_idx,
                test_start_utc=pd.Timestamp(calendar[start], tz="UTC"),
                test_end_utc=pd.Timestamp(calendar[end], tz="UTC"),
                purge_boundary_utc=boundary,
            )
        )
    return folds


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    level: float
    n_rows: int
    n_episodes: int
    n_boot: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi) and (self.lo > 0 or self.hi < 0))

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "level": self.level,
            "excludes_zero": self.excludes_zero,
            "n_rows": self.n_rows,
            "n_episodes": self.n_episodes,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }


def block_bootstrap_mean(
    values: np.ndarray,
    episodes: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP,
    level: float = 0.90,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile interval for the mean of ``values`` resampling whole episodes.

    Use with paired differences ``score_a - score_b`` computed on identical rows; the interval
    excluding zero is the confidence rule of CHALLENGE-SPEC section 7.
    """
    values = np.asarray(values, dtype=float)
    episodes = np.asarray(episodes)
    keep = np.isfinite(values)
    values, episodes = values[keep], episodes[keep]
    if values.size == 0:
        nan = float("nan")
        return BootstrapCI(nan, nan, nan, level, 0, 0, n_boot, seed)
    codes, inverse = np.unique(episodes, return_inverse=True)
    n_ep = len(codes)
    sums = np.bincount(inverse, weights=values, minlength=n_ep)
    counts = np.bincount(inverse, minlength=n_ep).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_ep, size=(n_boot, n_ep))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return BootstrapCI(
        float(values.mean()), float(lo), float(hi), level, int(values.size), n_ep, n_boot, seed
    )
