"""CLI for lane swings.

    python -m swing_dataset.swings build [--out DIR] [--labels DIR] [--mcx DIR] [--quotes DIR]
    python -m swing_dataset.swings check [--out DIR] [--labels DIR] [--mcx DIR] [--quotes DIR]

``build`` rewrites swings.parquet, swing_sessions.parquet and stats.json from the read-only
inputs; it is deterministic and idempotent. ``check`` re-derives every leg with an independent
implementation, compares, and writes ``check.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from . import HOURLY_START, LABELS_BUILD, LANE_BUILD, MCX_BUILD, QUOTES_BUILD


def _inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=LANE_BUILD)
    parser.add_argument("--labels", type=Path, default=LABELS_BUILD)
    parser.add_argument("--mcx", type=Path, default=MCX_BUILD)
    parser.add_argument("--quotes", type=Path, default=QUOTES_BUILD)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m swing_dataset.swings")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build swings, swing sessions and stats")
    _inputs(build)
    build.add_argument("--hourly-start", type=date.fromisoformat, default=HOURLY_START)
    build.add_argument(
        "--last-session",
        type=date.fromisoformat,
        default=None,
        help="stop the series here instead of at the last session with a Bhav Copy",
    )

    check = sub.add_parser("check", help="recompute every leg independently and compare")
    _inputs(check)
    check.add_argument("--report", type=Path, default=None, help="default: <out>/check.json")

    args = parser.parse_args(argv)
    started = time.monotonic()
    command = "python -m swing_dataset.swings " + " ".join(argv)
    if args.command == "build":
        from .build import build as run_build

        run_build(
            args.out,
            labels_dir=args.labels,
            mcx_dir=args.mcx,
            quotes_dir=args.quotes,
            hourly_start=args.hourly_start,
            last_session=args.last_session,
            command=command,
        )
        print(f"built in {time.monotonic() - started:.1f}s")
        return 0
    if args.command == "check":
        from .check import run_check

        report = run_check(args.out, args.labels, args.mcx, args.quotes, report_path=args.report)
        print(report.render())
        print(f"checked in {time.monotonic() - started:.1f}s")
        return 0 if report.ok else 1
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
