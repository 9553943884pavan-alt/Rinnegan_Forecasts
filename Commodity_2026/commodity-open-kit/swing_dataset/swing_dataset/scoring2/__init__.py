"""Commodity Open 2026 scoring, rules 2.6: three scores per task over signal rows.

The 2.4 scorer in ``swing_dataset.scoring`` grades probability and distribution files against the
one-sigma barrier labels. Rules 2.6 sections 3 and 4 grade signals instead: a direction, a
target, an invalidation and a horizon, against the price path of the designated contract and the
realized-swing inventory of lane ``labels2``. Task 1 is ``--challenge 1`` and Task 2 is
``--challenge 2``. This package implements that and nothing else. The folds, the episode-block
bootstrap convention and the report plumbing come from ``swing_dataset.scoring`` and are not
modified here.

Every column, score and rule is written out in FORMATS2.md next to this file.
"""

from pathlib import Path

from .. import BUILD_ROOT

LANE = "scoring2"
#: The rules this scorer implements; every report it writes carries it.
RULES_VERSION = "2.6"

ASSETS = ("gold", "silver", "copper", "crude")
DIRECTIONS = ("up", "down")
#: CHALLENGE-SPEC section 2: the horizons a signal may name, in completed MCX sessions, per task.
#: Task 1 takes 5 or 10; Task 2 any whole number from 1 to 20.
HORIZONS = {1: (5, 10), 2: tuple(range(1, 21))}
#: The horizons the shipped baselines issue at, on both tasks, within what a period scores.
BASELINE_HORIZONS = (5, 10)

#: CHALLENGE-SPEC section 2, percent of the price at the turn.
THRESHOLD_PCT = {"gold": 1.5, "silver": 2.0, "copper": 1.5, "crude": 3.0}

PERIODS = ("holdout", "forward")
#: CHALLENGE-SPEC section 5. The holdout runs to the pack's last session and scores every valid
#: horizon; the forward window starts after it and scores only horizons of at most
#: ``FORWARD_MAX_HORIZON`` sessions on both tasks, because a longer signal issued there has not
#: matured by the calls. A valid row with a longer horizon is recorded and counted, never scored.
HOLDOUT_START = "2026-06-01"
FORWARD_MAX_HORIZON = 5

#: Section 3 and 4 minimums, in percent, on both periods: the success rate of either task, and
#: Task 1's correctness and depth on their own. None of them averages with another.
MINIMUM_SUCCESS_RATE = 70.0
MINIMUM_CORRECTNESS = 60.0
MINIMUM_DEPTH = 60.0
#: Section 4's third hard condition: the direction score alone, in percent.
MINIMUM_DIRECTION_SCORE = 60.0
#: Section 5, per asset: the share of sessions carrying a decision record.
COVERAGE_FLOOR = 0.90
#: Section 3's catch rule: a swing is caught by a correct signal with at least this share of the
#: asset's threshold of the move still ahead of its entry price.
CATCH_ROOM_SHARE = 0.5
#: Section 3's depth rule: the target must be at least this share of the move to the swing's end.
DEPTH_TARGET_SHARE = 0.5
#: Section 3's large-swing check, on the holdout only: of the ``TOP_SWINGS`` realized swings with
#: the largest move in asset thresholds, at least ``TOP_SWINGS_REQUIRED`` must be caught.
TOP_SWINGS = 10
TOP_SWINGS_REQUIRED = 7
#: Comparisons against a percentage boundary (half a threshold of room, a turn of a full
#: threshold, half the move) allow this much, so that a value that is exactly on the boundary in
#: decimal and lands a few ulps under it in binary still counts as on it.
PCT_TOLERANCE = 1e-12
#: Section 3's reported diagnostic: a realized swing is large when its move is at least this many
#: asset thresholds, so gold 3.0, silver 4.0, copper 3.0 and crude 6.0 percent. The catch rate
#: restricted to those legs is printed beside the pooled one and decides nothing: it is not in the
#: success rate and not in `award_conditions`. Why two, in FORMATS2.md.
LARGE_SWING_MULTIPLE = 2
#: Section 4's depth rule: ``p90_pct - p10_pct`` may be at most this many asset thresholds wide.
#: A band wide enough to contain every move makes depth free, so it is bounded. The number is set
#: off the shipped baselines rather than invented: their band widths in threshold units have a
#: median of 4, a p99 of 11 and a maximum of 22.26, and the 2,089 realized legs of
#: ``build/swings/swings.parquet`` run 1.95 thresholds at the median and 8.48 at the p99. Twelve
#: therefore sits above both p99s, costs 0.3 to 0.6 percent of baseline rows, and rejects the
#: ``[-99, +99]`` band that scores depth 100 with no model in it.
BAND_WIDTH_CAP = 12
#: Section 4's improvement score: percentage points of direction gain, times this, clipped.
IMPROVEMENT_MULTIPLIER = 10.0

BUILD = Path(BUILD_ROOT) / "build"
MCX_DAILY_PATH = BUILD / "mcx" / "mcx_daily.parquet"
MCX_SESSIONS_PATH = BUILD / "mcx" / "mcx_sessions.parquet"
DESIGNATIONS_PATH = BUILD / "labels" / "designations.parquet"
SWINGS_PATH = BUILD / "swings" / "swings.parquet"
QUOTES_PATH = Path(BUILD_ROOT) / "pack" / "v1.1" / "pack" / "quotes" / "mcx_quotes.parquet"
OUT_DIR = BUILD / LANE

C1_SIGNAL_COLUMNS = (
    "asset",
    "signal_utc",
    "direction",
    "horizon",
    "target_pct",
    "invalidation_pct",
    "confidence",
    "evidence",
)
C2_SIGNAL_COLUMNS = (
    "asset",
    "signal_utc",
    "direction",
    "horizon",
    "expected_pct",
    "p10_pct",
    "p90_pct",
    "invalidation_pct",
    "evidence",
)
NO_SIGNAL_COLUMNS = ("asset", "decision_utc", "reason")


def scored_horizons(challenge: int, period: str) -> tuple[int, ...]:
    """The horizons a period scores for a task: every valid one on the holdout, those of at most
    ``FORWARD_MAX_HORIZON`` sessions on the forward window."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    valid = HORIZONS[challenge]
    if period == "forward":
        return tuple(h for h in valid if h <= FORWARD_MAX_HORIZON)
    return valid
