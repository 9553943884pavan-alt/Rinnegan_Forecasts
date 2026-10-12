"""Lane ``swings``: the realized-swing inventory of CHALLENGE-SPEC.md section 3.

A swing is a move of at least the asset's threshold, in percent of the price at the turn,
before a move of at least the threshold the other way. Turns sit on hourly highs and lows of
the designated contract; a swing's availability is the settlement publication of the session
its extreme printed in.

Inputs are read-only: the labels lane's ``designations.parquet`` (horizon 5 defines the front
series), the mcx lane's ``mcx_daily.parquet`` and ``mcx_sessions.parquet``, and the pack's
``quotes/mcx_quotes.parquet``. Outputs go to ``build/swings/``. The 1-sigma barrier labels of
version 2.4 under ``build/labels/`` are left alone; they ship as research.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

from .. import BUILD_ROOT

LANE = "swings"
LANE_BUILD = Path(BUILD_ROOT) / "build" / LANE
LABELS_BUILD = Path(BUILD_ROOT) / "build" / "labels"
MCX_BUILD = Path(BUILD_ROOT) / "build" / "mcx"
QUOTES_BUILD = Path(BUILD_ROOT) / "build" / "pack" / "quotes"

ASSETS = ("gold", "silver", "copper", "crude")

#: CHALLENGE-SPEC.md section 2, percent of the price at the turn.
THRESHOLD_PCT = {"gold": 1.5, "silver": 2.0, "copper": 1.5, "crude": 3.0}

#: The designation whose contract defines the front series.
DESIGNATION_HORIZON = 5

#: First session with production intraday quotes; before it the bar is the Bhav Copy session.
HOURLY_START = date(2026, 6, 22)

#: Holdout start, CHALLENGE-SPEC.md section 5.
HOLDOUT_START = date(2026, 6, 1)

IST = ZoneInfo("Asia/Kolkata")
PRE_OPEN_IST = time(8, 30)

BAR_HOURLY = "hourly"
BAR_DAILY = "daily"
BAR_MIXED = "mixed"
BAR_NONE = "none"

UP = "up"
DOWN = "down"
