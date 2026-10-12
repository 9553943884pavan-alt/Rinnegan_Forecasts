"""Participant kit support: the signal-file format and its validator, price-only features, the
shipped baselines behind a participant-facing wrapper, the self-run fit/predict/update loop, and
the organizer-side reproduction helpers.

Everything here codes against CHALLENGE-SPEC.md sections 2 to 7 at rules 2.6. The column tuples,
the thresholds, the horizons and every bound come from ``swing_dataset.scoring2``, which is the
authority: FORMATS2.md next to it is binding, and nothing in this package restates a rule the
scorer already owns or computes a score of its own.
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

from ..scoring2 import (
    ASSETS,
    BAND_WIDTH_CAP,
    BASELINE_HORIZONS,
    C1_SIGNAL_COLUMNS,
    C2_SIGNAL_COLUMNS,
    COVERAGE_FLOOR,
    DIRECTIONS,
    FORWARD_MAX_HORIZON,
    HORIZONS,
    NO_SIGNAL_COLUMNS,
    THRESHOLD_PCT,
)

__all__ = [
    "ASSETS",
    "BAND_WIDTH_CAP",
    "C1_SIGNAL_COLUMNS",
    "C2_SIGNAL_COLUMNS",
    "COVERAGE_FLOOR",
    "DEFAULT_HORIZONS",
    "DIRECTIONS",
    "FORWARD_MAX_HORIZON",
    "HORIZONS",
    "IST",
    "KEY_COLUMNS",
    "NO_SIGNAL_COLUMNS",
    "PREOPEN_IST",
    "SIGNAL_KEY",
    "THRESHOLD_PCT",
]

IST = ZoneInfo("Asia/Kolkata")
PREOPEN_IST = time(8, 30)
"""The MCX pre-open, the one instant outside a session at which a signal may fire (section 3)."""

DEFAULT_HORIZONS = BASELINE_HORIZONS
"""The horizons a run issues at unless told otherwise: 5 and 10, the ones the shipped baselines
and the template use on both tasks. Which horizons are valid is ``HORIZONS``, per task: Task 1 5
or 10, Task 2 any whole number from 1 to 20. The forward window scores only horizons of at most
``FORWARD_MAX_HORIZON`` sessions."""

KEY_COLUMNS = ("asset", "cutoff_utc", "horizon")
"""The designation key ``features.py`` computes on. A signal file is keyed by SIGNAL_KEY."""

SIGNAL_KEY = ("asset", "signal_utc", "direction", "horizon")
"""What identifies a signal row when two files are compared. Repeats are allowed; the scorer
counts every row."""
