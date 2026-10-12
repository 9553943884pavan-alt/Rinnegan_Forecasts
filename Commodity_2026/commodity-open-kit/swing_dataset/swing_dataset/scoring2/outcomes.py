"""Grading one signal against the price path, and one swing against the graded signals.

Section 3 (Task 1). A signal is correct when the price moves at least the asset's threshold in the
signalled direction within the horizon before it moves `invalidation_pct` against it. It is
decided when the price touched either of those two levels inside the horizon; a signal whose
window completed with neither touched is undecided, and correctness leaves it out. A signal is
deep when it also reached its own `target_pct` before the invalidation and that target was at
least half of the move to the swing's end: from the entry price to the favorable extreme the
price reached before it first turned back a full threshold, or before the window ended
(`swing_end_pct`, measured by `prices.SwingEnd`). A swing is caught by a correct signal in its
direction issued after the turn, at or before the extreme, while at least half the threshold of
the move was still ahead.

Section 4 (Task 2). A signal is directionally realized when the price moves at least the asset's
threshold in the signalled direction within the horizon before it moves the threshold against it:
the same test `grade_c1` decides `correct` on, and the same test the prevailing-direction null is
measured by, so the two tasks are comparable. `expected_pct` and the band are scored in depth,
which is whether the realized move landed inside `[p10_pct, p90_pct]`, the realized move being
the favorable extreme of the window, `mfe_pct`.

Maturity. A signal whose window runs past the end of the data is still scored when its outcome is
already decided inside the part that exists; otherwise it is unmatured, counted, and left out of
both the numerator and the denominator. It still counts as a decision record for coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import CATCH_ROOM_SHARE, DEPTH_TARGET_SHARE, PCT_TOLERANCE
from .prices import PriceSeries

C1_COLUMNS = (
    "p0",
    "price_source",
    "contract_id",
    "window_end_utc",
    "window_complete",
    "bars",
    "ambiguous",
    "mfe_pct",
    "mae_pct",
    "swing_end_pct",
    "threshold_hit",
    "correct",
    "decided",
    "undecided",
    "target_first",
    "deep",
    "scorable",
    "unmatured",
    "no_price",
)
C2_COLUMNS = (
    "p0",
    "price_source",
    "contract_id",
    "window_end_utc",
    "window_complete",
    "bars",
    "ambiguous",
    "mfe_pct",
    "mae_pct",
    "realized",
    "in_band",
    "scorable",
    "unmatured",
    "no_price",
)


def _blank(columns: tuple[str, ...], n: int) -> dict[str, list]:
    return {c: [None] * n for c in columns}


def grade_c1(signals: pd.DataFrame, prices: PriceSeries) -> pd.DataFrame:
    """Add the Task 1 outcome columns to a validated signal frame."""
    out = signals.reset_index(drop=True).copy()
    n = len(out)
    fields = _blank(C1_COLUMNS, n)
    for i in range(n):
        row = out.iloc[i]
        if not row["valid"]:
            _unpriced(fields, i, no_price=False)
            continue
        threshold = float(row["threshold_pct"])
        path = prices.evaluate_path(
            str(row["asset"]),
            row["signal_utc"],
            int(row["horizon"]),
            str(row["direction"]),
            {"threshold": threshold, "target": float(row["target_pct"])},
            {"invalidation": float(row["invalidation_pct"])},
            swing_threshold_pct=threshold,
        )
        if path is None:
            _unpriced(fields, i, no_price=True)
            continue
        correct = path.first("threshold", "invalidation")
        decided = path.decided(("threshold", "invalidation"))
        target_first = path.first("target", "invalidation")
        deep = bool(
            correct
            and target_first
            and float(row["target_pct"]) >= DEPTH_TARGET_SHARE * path.swing_end_pct - PCT_TOLERANCE
        )
        _record(fields, i, path)
        fields["swing_end_pct"][i] = path.swing_end_pct
        fields["threshold_hit"][i] = path.touched("threshold")
        fields["correct"][i] = bool(correct)
        fields["decided"][i] = bool(decided)
        fields["undecided"][i] = bool(not decided and path.window_complete)
        fields["target_first"][i] = bool(target_first)
        fields["deep"][i] = deep
        fields["scorable"][i] = bool(decided or path.window_complete)
        fields["unmatured"][i] = bool(not decided and not path.window_complete)
        fields["no_price"][i] = False
    return out.assign(**fields)


def _unpriced(fields: dict[str, list], i: int, no_price: bool) -> None:
    """An invalid row, or one the series cannot price: scored, never correct, never undecided.

    Neither was decided by the market, and both stay in the correctness denominator anyway,
    because leaving them out would pay a model for issuing rows the rules reject or cannot grade.
    """
    fields["scorable"][i] = True
    fields["correct"][i] = False
    fields["decided"][i] = False
    fields["undecided"][i] = False
    fields["deep"][i] = False
    fields["unmatured"][i] = False
    fields["no_price"][i] = no_price


def grade_c2(signals: pd.DataFrame, prices: PriceSeries) -> pd.DataFrame:
    """Add the Task 2 outcome columns to a validated signal frame."""
    out = signals.reset_index(drop=True).copy()
    n = len(out)
    fields = _blank(C2_COLUMNS, n)
    for i in range(n):
        row = out.iloc[i]
        if not row["valid"]:
            fields["scorable"][i] = True
            fields["realized"][i] = False
            fields["in_band"][i] = False
            fields["unmatured"][i] = False
            fields["no_price"][i] = False
            continue
        threshold = float(row["threshold_pct"])
        path = prices.evaluate_path(
            str(row["asset"]),
            row["signal_utc"],
            int(row["horizon"]),
            str(row["direction"]),
            {"threshold": threshold, "expected": float(row["expected_pct"])},
            {"adverse_threshold": threshold},
        )
        if path is None:
            fields["scorable"][i] = True
            fields["realized"][i] = False
            fields["in_band"][i] = False
            fields["unmatured"][i] = False
            fields["no_price"][i] = True
            continue
        realized = path.first("threshold", "adverse_threshold")
        decided = path.decided(("threshold", "adverse_threshold"))
        _record(fields, i, path)
        fields["realized"][i] = bool(realized)
        fields["in_band"][i] = bool(
            realized and float(row["p10_pct"]) <= path.mfe_pct <= float(row["p90_pct"])
        )
        fields["scorable"][i] = bool(decided or path.window_complete)
        fields["unmatured"][i] = bool(not decided and not path.window_complete)
        fields["no_price"][i] = False
    return out.assign(**fields)


def _record(fields: dict[str, list], i: int, path) -> None:
    fields["p0"][i] = path.p0
    fields["price_source"][i] = path.price_source
    fields["contract_id"][i] = path.contract_id
    fields["window_end_utc"][i] = path.window_end_utc
    fields["window_complete"][i] = path.window_complete
    fields["bars"][i] = path.bars
    fields["ambiguous"][i] = path.ambiguous
    fields["mfe_pct"][i] = path.mfe_pct
    fields["mae_pct"][i] = path.mae_pct


def catch(swings: pd.DataFrame, graded: pd.DataFrame) -> pd.DataFrame:
    """Per swing, whether a correct signal caught it and which one.

    The signal must be in the swing's direction, correct, issued strictly after the turn instant
    and at or before the extreme instant, and far enough from the extreme that at least
    ``CATCH_ROOM_SHARE`` of the asset's threshold of the move was still ahead of its entry price.
    The first such signal in issue order is the catcher.
    """
    out = swings.reset_index(drop=True).copy()
    caught = np.zeros(len(out), dtype=bool)
    catcher: list = [None] * len(out)
    catcher_utc: list = [None] * len(out)
    if len(graded):
        usable = graded[
            graded["valid"].astype(bool)
            & graded["correct"].fillna(False).astype(bool)
            & graded["p0"].notna()
        ]
    else:
        usable = graded
    for i in range(len(out)):
        swing = out.iloc[i]
        if usable.empty:
            continue
        threshold = float(swing["threshold_pct"])
        extreme = float(swing["extreme_price"])
        candidates = usable[
            (usable["asset"] == swing["asset"])
            & (usable["direction"] == swing["direction"])
            & (usable["signal_utc"] > swing["turn_utc"])
            & (usable["signal_utc"] <= swing["extreme_utc"])
        ]
        if candidates.empty:
            continue
        p0 = candidates["p0"].to_numpy(dtype=float)
        ahead = (
            (extreme - p0) / p0 * 100.0
            if swing["direction"] == "up"
            else (p0 - extreme) / p0 * 100.0
        )
        hit = np.flatnonzero(ahead >= CATCH_ROOM_SHARE * threshold - PCT_TOLERANCE)
        if hit.size:
            first = candidates.iloc[int(hit[0])]
            caught[i] = True
            catcher[i] = int(first["signal_index"])
            catcher_utc[i] = first["signal_utc"]
    out["caught"] = caught
    out["caught_by_signal_index"] = catcher
    out["caught_by_signal_utc"] = catcher_utc
    return out
