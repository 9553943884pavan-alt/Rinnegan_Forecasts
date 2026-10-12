"""The realized-swing inventory of lane `labels2`, read and filtered to the scored window.

One row per completed leg, as `build/swings/swings.parquet` ships it: `asset, direction, turn_utc,
turn_price, extreme_utc, extreme_price, move_pct, sessions_spanned, contract_id, contract_changed,
bar, threshold_pct, available_utc, clock_quality`. Only the first six are needed to score a catch;
the rest are carried through to the report when the file has them.

A swing counts for the catch rate when its whole catch opportunity lies inside the scored window:
the turn at or after the window start and the extreme at or before the window end. Swings that
straddle either edge are counted and reported, never scored, because a model cannot be asked for a
signal it had no window to issue. The ten large swings of rules 2.6 section 3 are ranked from the
same set by ``largest``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import ASSETS, DIRECTIONS, LARGE_SWING_MULTIPLE, THRESHOLD_PCT

REQUIRED = ("asset", "direction", "turn_utc", "turn_price", "extreme_utc", "extreme_price")
OPTIONAL = (
    "move_pct",
    "sessions_spanned",
    "contract_id",
    "contract_changed",
    "bar",
    "threshold_pct",
    "available_utc",
    "clock_quality",
)


def load(path) -> pd.DataFrame:
    return normalize(pq.read_table(path).to_pandas())


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"swing inventory lacks columns {missing}")
    out = frame.reset_index(drop=True).copy()
    out["asset"] = out["asset"].astype(str)
    out["direction"] = out["direction"].astype(str)
    for column in ("turn_utc", "extreme_utc"):
        parsed = pd.to_datetime(out[column])
        if parsed.dt.tz is None:
            raise ValueError(f"{column} must be timezone-aware")
        out[column] = parsed.dt.tz_convert("UTC")
    for column in ("turn_price", "extreme_price"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    if "threshold_pct" not in out.columns:
        out["threshold_pct"] = out["asset"].map(THRESHOLD_PCT).astype(float)
    bad = (
        ~out["asset"].isin(ASSETS)
        | ~out["direction"].isin(DIRECTIONS)
        | ~np.isfinite(out["turn_price"])
        | ~np.isfinite(out["extreme_price"])
        | (out["extreme_utc"] < out["turn_utc"])
    )
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} swing rows are unusable (asset, direction, price, order)"
        )
    keep = list(REQUIRED) + [c for c in OPTIONAL if c in out.columns]
    return out[keep].sort_values(["turn_utc", "asset"]).reset_index(drop=True)


def large_move_pct() -> dict[str, float]:
    """The move a leg must make to count as large, per asset: the threshold times the multiple."""
    return {asset: float(LARGE_SWING_MULTIPLE) * pct for asset, pct in THRESHOLD_PCT.items()}


def move_pct(swings: pd.DataFrame) -> np.ndarray:
    """Per leg, the signed move from turn to extreme in percent of the turn price.

    `move_pct` where the inventory carries it and the turn-to-extreme move otherwise, which is the
    same number to the bit on the shipped file.
    """
    if "move_pct" in swings.columns:
        return pd.to_numeric(swings["move_pct"], errors="coerce").to_numpy(dtype=float)
    turn = swings["turn_price"].to_numpy(dtype=float)
    return (swings["extreme_price"].to_numpy(dtype=float) - turn) / turn * 100.0


def threshold_multiple(swings: pd.DataFrame) -> np.ndarray:
    """Per leg, the size of the move in asset thresholds: ``|move_pct| / threshold_pct``."""
    threshold = pd.to_numeric(swings["threshold_pct"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(move_pct(swings)) / threshold


def is_large(swings: pd.DataFrame) -> np.ndarray:
    """Per leg, whether the realized move is at least `LARGE_SWING_MULTIPLE` asset thresholds.

    Reported, never scored. Section 3's catch rate counts every leg above the threshold and that
    does not change; this picks out the subset the diagnostic of FORMATS2.md is computed on. The
    tolerance is there so that a leg written as exactly the multiple, whose float lands a few ulps
    under it, is still large.
    """
    if not len(swings):
        return np.zeros(0, dtype=bool)
    threshold = pd.to_numeric(swings["threshold_pct"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.abs(move_pct(swings)) >= float(LARGE_SWING_MULTIPLE) * threshold - 1e-12


def largest(swings: pd.DataFrame, n: int) -> pd.DataFrame:
    """The ``n`` legs with the largest move in asset thresholds, largest first.

    Rules 2.6 section 3 ranks by the move from turn to extreme divided by the asset's threshold,
    so that each commodity is measured against its own normal range. Ties go to the earlier
    `turn_utc`, then to the asset name. The frame comes back with `move_pct` and
    `threshold_multiple` filled in and every column it had.
    """
    ranked = swings.reset_index(drop=True).assign(
        move_pct=move_pct(swings), threshold_multiple=threshold_multiple(swings)
    )
    ranked = ranked.sort_values(
        ["threshold_multiple", "turn_utc", "asset"], ascending=[False, True, True], kind="stable"
    )
    return ranked.head(n).reset_index(drop=True)


def in_window(swings: pd.DataFrame, start, end) -> pd.DataFrame:
    """Swings whose turn and extreme both lie inside the scored window, with the edge counts."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    inside = (swings["turn_utc"] >= start) & (swings["extreme_utc"] <= end)
    out = swings[inside].reset_index(drop=True).copy()
    out.attrs["straddles_start"] = int(
        ((swings["turn_utc"] < start) & (swings["extreme_utc"] >= start)).sum()
    )
    out.attrs["straddles_end"] = int(
        ((swings["turn_utc"] <= end) & (swings["extreme_utc"] > end)).sum()
    )
    out.attrs["outside"] = int(
        len(swings) - len(out) - out.attrs["straddles_start"] - out.attrs["straddles_end"]
    )
    return out
