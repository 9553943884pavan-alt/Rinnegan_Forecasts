#!/usr/bin/env python3
"""Check a signal file against the Commodity Open formats and print a report.

    python tools/validate_submission.py predictions/c1_holdout.parquet
    python tools/validate_submission.py signals.parquet --challenge 2 --pack ./pack
    python tools/validate_submission.py signals.parquet --no-signals decisions.parquet

Which rows are valid is decided by the shipped scorer (`swing_dataset.scoring2.signals`), so a
file this tool calls valid is one the scorer calls valid. What the tool adds is the reason in
plain words, the counts per asset and horizon, and, with `--pack`, the section 5 coverage floor:
a decision record on at least 90 percent of each asset's sessions in the window. `no_signal` rows
may sit in the same file (leave `signal_utc` null, fill `decision_utc` and `reason`) or in a
second file passed with `--no-signals`. The scorer reads Parquet only: a CSV or JSON-lines file
has its rows checked and is then reported as `not_parquet`, because it could not be scored.

Exit status: 0 valid, 1 invalid, 2 usage error. Requires the `swing-dataset` package.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from swing_dataset.kit import formats


def _aware(text: str | None) -> datetime | None:
    if text is None:
        return None
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{text!r} needs a timezone offset")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("signals", type=Path, help="the signal file, as Parquet (CSV and JSON lines are checked row by row but cannot be scored)")
    parser.add_argument("--challenge", type=int, choices=(1, 2), help="declare the challenge; detected from target_pct or expected_pct when omitted")
    parser.add_argument("--no-signals", type=Path, help="a separate Parquet file of no_signal rows, when they are not inline")
    parser.add_argument("--pack", type=Path, help="pack root (directory with MANIFEST.json); enables the coverage check")
    parser.add_argument("--period", choices=("holdout", "forward"), default="holdout", help="the window coverage is measured over (default: holdout)")
    parser.add_argument("--start", type=_aware, help="first instant of the window (ISO 8601 with offset)")
    parser.add_argument("--end", type=_aware, help="last instant of the window (ISO 8601 with offset)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of text")
    args = parser.parse_args(argv)
    if not args.signals.exists():
        parser.error(f"{args.signals} does not exist")
    if args.no_signals is not None and not args.no_signals.exists():
        parser.error(f"{args.no_signals} does not exist")
    if args.pack is not None and not (args.pack / "mcx" / "mcx_daily.parquet").exists():
        parser.error(f"{args.pack} has no mcx/mcx_daily.parquet")
    if args.period == "forward" and args.pack is not None and args.start is None:
        parser.error("--period forward has no fixed start in the data; pass --start")
    report = formats.validate_file(
        args.signals,
        args.challenge,
        no_signal_path=args.no_signals,
        pack_root=args.pack,
        period=args.period,
        start=args.start,
        end=args.end,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
