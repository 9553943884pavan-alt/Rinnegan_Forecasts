"""The report: the three scores with their intervals, coverage, conditions, baselines, identities.

Task 1 also carries `ten_large_swings`, whose caught count is a gate of the minimum on the
holdout, and `diagnostics`, which is reported and never scored. Nothing in `diagnostics` enters a
score or a boolean; it exists so that the one-to-one call of section 4a has the numbers.

The JSON is the thing a participant reads and the organizers re-run, so it carries the inputs by
path, hash and row count, the commit that produced it, and a definition for every number in it.
Serialization is the 2.4 scorer's (`scoring.evaluate.dump_json`), so both reports sanitize the
same way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import common
from ..scoring.evaluate import dump_json, file_identity, sanitize
from . import (
    BAND_WIDTH_CAP,
    CATCH_ROOM_SHARE,
    COVERAGE_FLOOR,
    DEPTH_TARGET_SHARE,
    FORWARD_MAX_HORIZON,
    HORIZONS,
    IMPROVEMENT_MULTIPLIER,
    LARGE_SWING_MULTIPLE,
    MINIMUM_CORRECTNESS,
    MINIMUM_DEPTH,
    MINIMUM_DIRECTION_SCORE,
    MINIMUM_SUCCESS_RATE,
    RULES_VERSION,
    TOP_SWINGS,
    TOP_SWINGS_REQUIRED,
)
from .bootstrap import DEFAULT_LEVEL
from .prices import MIN_TICKS_PER_SESSION
from .score import Window, award_conditions

FRAME_KEYS = ("graded", "caught")


def strip_frames(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in FRAME_KEYS}


def definitions(challenge: int) -> dict:
    task_1 = HORIZONS[1]
    task_2 = HORIZONS[2]
    recorded = (
        "the holdout scores every one; the forward window scores only horizons of at most "
        f"{FORWARD_MAX_HORIZON} sessions, and a valid row with a longer horizon is recorded under "
        "counts.recorded_not_scored_other_horizon and never scored"
    )
    shared = {
        "price_series": (
            "the contract designated for the signal's session in labels/designations.parquet at "
            "horizon 5, held for the whole window so a roll never enters as a calendar spread"
        ),
        "entry_price": (
            "the last price of that contract available at or before signal_utc: the last captured "
            "tick or the last published settlement, whichever became available later"
        ),
        "path_resolution": (
            f"tick by tick where a session carries at least {MIN_TICKS_PER_SESSION} captured "
            "quotes after the signal, session high and low otherwise; inside a session bar the "
            "order of two barriers is unknowable, so the adverse one is taken first and the row "
            "is flagged"
        ),
        "horizons": (
            f"Task 1: {task_1[0]} or {task_1[1]} completed MCX sessions; {recorded}"
            if challenge == 1
            else f"Task 2: any whole number of completed MCX sessions from {task_2[0]} to "
            f"{task_2[-1]}; {recorded}"
        ),
        "horizon_window": (
            "from signal_utc to the close of the h-th session that closes strictly after it; a "
            "signal already decided inside a truncated window is scored, otherwise it is unmatured"
        ),
        "unmatured": "counted, left out of both the numerator and the denominator, still a decision record",
        "invalid_row": (
            "counted as issued and never correct; not a decision record for coverage. Reasons are "
            "listed in counts.invalid_reasons"
        ),
        "episode": (
            "a block of h sessions per asset counted from the window start, h in "
            "window.episode_horizon. Task 1: the longest horizon the period scores, 10 on the "
            f"holdout and {FORWARD_MAX_HORIZON} on the forward window. Task 2: the longest horizon "
            "the entry's valid scored rows use, so an entry at 5 and 10 is drawn in blocks of 10; "
            "with no such row the longest baseline horizon the period scores, 10 on the holdout "
            f"and {FORWARD_MAX_HORIZON} on the forward window"
        ),
        "confidence_interval": (
            f"the {DEFAULT_LEVEL:.0%} percentile interval over 2000 seeded resamples of the shared "
            "episode universe; every score in one report is recomputed on the same draw, so the "
            "success rate's interval is the interval of the average and not an average of intervals"
        ),
        "coverage_floor": (
            f"a valid signal row or a valid no_signal row on at least {COVERAGE_FLOOR:.0%} of the "
            "asset's sessions in the window, per asset and pooled"
        ),
        "success_rate": "the plain average of the three scores; null when any of the three has no units",
    }
    if challenge == 1:
        shared.update(
            {
                "minimum": (
                    "five gates, none averaged with another, each in award_conditions.gates with "
                    f"its value, threshold and result: success rate at or above "
                    f"{MINIMUM_SUCCESS_RATE:g} percent, correctness at or above "
                    f"{MINIMUM_CORRECTNESS:g}, depth at or above {MINIMUM_DEPTH:g}, the coverage "
                    f"floor met, and on the holdout at least {TOP_SWINGS_REQUIRED} of the "
                    f"{TOP_SWINGS} large swings caught (the ten large swings do not apply on the "
                    "forward window). The minimum must hold on the holdout and on the forward window"
                ),
                "catch_rate": (
                    "swings caught divided by realized swings in the window. A swing is caught by "
                    "a correct signal in its direction issued strictly after the turn and at or "
                    f"before the extreme, with at least {CATCH_ROOM_SHARE:g} of the asset's "
                    "threshold of the move still ahead: the move from the entry price to the "
                    "swing's extreme price, in percent of the entry price, is at least half the "
                    "threshold"
                ),
                "correctness": (
                    "correct signals divided by decided signals. Correct means the price moved at "
                    "least the asset's threshold in the signalled direction inside the horizon "
                    "before it moved invalidation_pct against it; decided means the price touched "
                    "the threshold level or the invalidation level inside the horizon. An "
                    "undecided signal is left out and counted apart; an invalid or unpriced row "
                    "stays in as incorrect"
                ),
                "undecided": (
                    "a scored signal whose horizon window completed with neither its threshold "
                    "nor its invalidation level touched: the market did not settle the call. It is "
                    "in no score; counts.undecided holds how many and counts.undecided_share their "
                    "share of the scored signals, beside counts.decided"
                ),
                "depth": (
                    "among correct signals, the share that reached target_pct before "
                    f"invalidation_pct and whose target_pct was at least {DEPTH_TARGET_SHARE:g} of "
                    "the move to the swing's end. A target never reached fails, and so does one "
                    "under half the move"
                ),
                "move_to_swing_end": (
                    "from the entry price to the furthest point the price reached in the "
                    "signalled direction before it first turned back a full threshold from its "
                    "running favorable extreme, or before the horizon ended, whichever came first; "
                    "a later move after such a turn belongs to the next swing. Walked on the same "
                    "prices as the barriers, a session bar favorable side first, and recorded per "
                    "row as swing_end_pct beside mfe_pct, the furthest point of the whole window"
                ),
                "swings_in_window": "swings whose turn and extreme both lie inside the window; the ones straddling either edge are counted separately and never scored",
                "ten_large_swings": (
                    f"the {TOP_SWINGS} realized swings of the window with the largest move from "
                    "turn to extreme divided by the asset's threshold, ties to the earlier turn and "
                    "then the asset name, each with whether it was caught, as for the catch rate, "
                    f"and by which signal. On the holdout at least {TOP_SWINGS_REQUIRED} of the "
                    f"{TOP_SWINGS} must be caught, a gate of award_conditions; on the forward "
                    "window the check does not apply"
                ),
                "large_swing_catch_rate": (
                    "reported and never scored: the catch rate over the legs whose move is at "
                    f"least {LARGE_SWING_MULTIPLE} times the asset's threshold, printed under "
                    "`diagnostics` with its own interval. It is not averaged into the success "
                    "rate and does not appear in `award_conditions`; section 3 and step 4 of "
                    "section 4a say what it is for"
                ),
            }
        )
    else:
        shared.update(
            {
                "minimum": (
                    f"success rate at or above {MINIMUM_SUCCESS_RATE:g} percent on the holdout and "
                    "on the forward window, with the coverage floor met and the hard conditions"
                ),
                "direction": (
                    "signals whose direction was realized divided by signals issued. Realized "
                    "means the price moved at least the asset's threshold in the signalled "
                    "direction inside the horizon before it moved the threshold against it: "
                    "deliberately the ruler Task 1 scores correctness on and the one the "
                    "prevailing-direction null is measured by, so the two tasks compare"
                ),
                "depth": (
                    "among directionally right signals, the share whose realized move landed "
                    "inside [p10_pct, p90_pct]. The realized move is the favorable extreme of the "
                    "window from the entry price (mfe_pct). This is where expected_pct and the "
                    "band are scored, and the band may be at most "
                    f"{BAND_WIDTH_CAP:g} times the asset's threshold wide"
                ),
                "improvement": (
                    f"{IMPROVEMENT_MULTIPLIER:g} times the learning model's direction gain over "
                    "its frozen copy in percentage points, floored at 0 and capped at 100, on the "
                    "rows both issued at the same instant, asset and horizon"
                ),
                "prevailing_direction_null": (
                    "the sign of the last five completed sessions' move, issued at the entry's "
                    "instants with the entry's horizon, expected and invalidation at the asset's "
                    "threshold"
                ),
                "hard_conditions": (
                    "learning beats frozen on both periods; the second half of each period is at "
                    f"least the first; direction alone is at least {MINIMUM_DIRECTION_SCORE:g} "
                    "percent on both periods; the frozen copy beats the null on the holdout"
                ),
            }
        )
    return shared


def assemble(
    result: dict,
    coverage_block: dict,
    window: Window,
    inputs: dict,
    command: str,
    baselines: dict | None = None,
) -> dict:
    report = {
        "challenge": result["challenge"],
        "rules_version": RULES_VERSION,
        "window": {
            **window.to_dict(result["challenge"]),
            "episode_horizon": result["episode_horizon"],
        },
        "scores": result["scores"],
        "counts": result["counts"],
        "coverage": coverage_block,
    }
    for key in (
        "ten_large_swings",
        "diagnostics",
        "hard_conditions",
        "paired_direction",
        "halves",
        "prevailing_direction_null",
        "frozen",
    ):
        if key in result:
            report[key] = result[key]
    report["baselines"] = baselines or {}
    report["award_conditions"] = award_conditions(result, window, coverage_block)
    report["inputs"] = inputs
    report["build"] = {**common.git_identity(), "command": command}
    report["definitions"] = definitions(result["challenge"])
    return report


def identities(paths: dict[str, Path | None]) -> dict:
    return {name: (file_identity(path) if path else None) for name, path in paths.items()}


def write(report: dict, path: Path | None) -> str:
    return dump_json(report, path)


def write_signals(
    frame: pd.DataFrame, path: Path, name: str, notes: list[str], sources: list[dict]
) -> None:
    """Write a baseline's decision records with the SPEC.md sidecar beside them."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = frame.copy()
    for column in ("signal_utc", "decision_utc"):
        out[column] = pd.to_datetime(out[column], utc=True).astype("datetime64[us, UTC]")
    out["horizon"] = out["horizon"].astype("Int16")
    table = pa.Table.from_pandas(out, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    meanings = {
        "asset": "asset key",
        "signal_utc": "instant the signal was issued; null on a no_signal row",
        "direction": "up or down",
        "horizon": "completed MCX sessions: 5 or 10, the horizons every shipped baseline issues at",
        "target_pct": "Task 1: the model's own expected move, at least the asset's threshold",
        "expected_pct": "Task 2: the expected move in the signalled direction",
        "p10_pct": "Task 2: lower bound of the expected move",
        "p90_pct": "Task 2: upper bound of the expected move",
        "invalidation_pct": "how far against the signal the price may go first",
        "confidence": "the model's own confidence; stored, never scored",
        "evidence": "the reason for the signal; stored, never scored, required",
        "decision_utc": "instant of a no_signal decision; null on a signal row",
        "reason": "one line saying why no signal was issued",
        "baseline": "which baseline produced the row",
        "fallback": "true when a refit quantity had too little history and the unconditional form was used",
    }
    columns = [
        {"name": f.name, "type": str(f.type), "meaning": meanings.get(f.name, "")}
        for f in table.schema
    ]
    issued = out["signal_utc"].notna()
    common.write_meta(
        path,
        {
            "name": name,
            "rows": table.num_rows,
            "columns": columns,
            "date_range": {
                "signal_utc_min": str(out.loc[issued, "signal_utc"].min())
                if issued.any()
                else None,
                "signal_utc_max": str(out.loc[issued, "signal_utc"].max())
                if issued.any()
                else None,
            },
            "sources": sources,
            "clock_quality": {
                "file": "exact",
                "note": "every row is decided at its own instant from rows published before it; the instant is the availability of the decision",
            },
            "build": {**common.git_identity(), "command": name},
            "notes": notes,
        },
    )


__all__ = [
    "assemble",
    "definitions",
    "identities",
    "sanitize",
    "strip_frames",
    "write",
    "write_signals",
]
