"""Commodity Open 2026, the superseded 2.4 scoring: metrics, predeclared baselines, purged folds, and the evaluation report.

Column contracts are documented in FORMATS.md next to this file. The label and target files
come from the labels lane (`build/labels/swing_labels.parquet`, `build/labels/return_targets.parquet`);
the price history for the price-only baselines is `build/mcx/mcx_daily.parquet`.
"""

from pathlib import Path

from .. import BUILD_ROOT

LANE = "scoring"
CLASSES = ("up_swing", "down_swing", "no_swing")
PROB_COLS = ("p_up", "p_down", "p_none")
QUANTILE_LEVELS = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
QUANTILE_COLS = (
    "q_025",
    "q_050",
    "q_100",
    "q_250",
    "q_500",
    "q_750",
    "q_900",
    "q_950",
    "q_975",
)
SAMPLE_COUNT = 512
HORIZONS = (5, 10, 20)
SCORED_MULTIPLIER = 1.0
KEY = ["asset", "cutoff_utc", "horizon"]

MCX_DAILY_PATH = Path(BUILD_ROOT) / "build" / "mcx" / "mcx_daily.parquet"
LABELS_DIR = Path(BUILD_ROOT) / "build" / "labels"
SWING_LABELS_PATH = LABELS_DIR / "swing_labels.parquet"
RETURN_TARGETS_PATH = LABELS_DIR / "return_targets.parquet"
