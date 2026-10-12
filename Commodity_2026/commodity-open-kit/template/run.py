#!/usr/bin/env python3
"""Regenerate the signal files of this submission from the pack.

    python run.py --pack /path/to/pack --challenge 1 --period holdout
    python run.py --pack /path/to/pack --challenge 2 --period holdout
    python run.py --pack /path/to/pack --challenge 2 --period holdout --frozen
    python run.py --pack /path/to/pack --challenge 2 --period forward

Writes, under ``--out`` (default ``predictions/``), for the stem ``c<challenge>_<period>`` with
``_frozen`` appended when ``--frozen`` is set:

    <stem>.parquet          the signal file: signal rows and no_signal rows (FORMATS2.md)
    <stem>.readlog.jsonl    every pack read the run made, through the point-in-time client
    <stem>.run.json         arguments, pack identity, counts and timings

The pack is read only through ``swing_dataset.pack.pit.PointInTime``. ``fit`` happens once at the
period's training cutoff (holdout: 2026-06-01 03:00 UTC; forward: its first session), then every
session of the period is visited in order: the outcomes of this run's own signals that settled
since the previous session go to ``update`` (never with ``--frozen``), and ``predict`` is called
for every asset and horizon at the session's decision instants. The run stops at the first session
the pack does not reach, so a re-run over a longer pack reproduces the earlier rows and appends the
new ones. ``--from`` and ``--to`` narrow the sessions for smoke tests; the organizers run without
them.

Two runs of this file on the same pack produce byte-identical parquets, which is what section 4a
asks for. Nothing here draws a random number, reads the clock into a row, or writes a path into
the signal file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import model as entry

from swing_dataset.kit import DEFAULT_HORIZONS, formats, loop
from swing_dataset.pack.pit import PointInTime, ReadLog
from swing_dataset.scoring2 import RULES_VERSION

HERE = Path(__file__).resolve().parent


def _date(text: str) -> date:
    return date.fromisoformat(text)


def _aware(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{text!r} needs a timezone offset")
    return parsed.astimezone(UTC)


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _pack_identity(pack: Path) -> dict:
    manifest = pack / "MANIFEST.json"
    if not manifest.exists():
        return {"root": str(pack), "manifest": None}
    data = json.loads(manifest.read_text())
    return {
        "root": str(pack),
        "version": data.get("version"),
        "built_utc": data.get("built_utc"),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "files": len(data.get("files", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, required=True, help="pack root (the directory with MANIFEST.json)")
    parser.add_argument("--challenge", type=int, choices=(1, 2), required=True)
    parser.add_argument("--period", choices=sorted(loop.PERIODS), required=True)
    parser.add_argument("--frozen", action="store_true", help="never call update(): the copy the improvement score is measured against")
    parser.add_argument("--out", type=Path, default=HERE / "predictions", help="output directory (default: predictions/ next to this file)")
    parser.add_argument("--from", dest="start", type=_date, help="first session date (smoke tests only)")
    parser.add_argument("--to", dest="end", type=_date, help="last session date (smoke tests only)")
    parser.add_argument("--train-end", type=_aware, help="override the period's fit instant (ISO 8601 with offset)")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="comma-separated horizons (default: 5,10; Task 1 takes 5 or 10, Task 2 any whole number from 1 to 20)",
    )
    args = parser.parse_args(argv)

    if not args.pack.is_dir():
        parser.error(f"{args.pack} is not a directory")
    horizons = tuple(int(h) for h in args.horizons.split(","))
    period = loop.period_for(args.period, args.start, args.end, args.train_end)
    stem = f"c{args.challenge}_{args.period}" + ("_frozen" if args.frozen else "")
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / f"{stem}.readlog.jsonl"
    pit = PointInTime(args.pack, read_log=ReadLog(log_path))
    model = entry.build(args.challenge, horizons)

    started = time.perf_counter()
    result = loop.run(
        model,
        pit,
        challenge=args.challenge,
        period=period,
        horizons=horizons,
        frozen=args.frozen,
        log=lambda m: print(m, file=sys.stderr, flush=True),
    )
    signals_path = formats.write_predictions(result.decisions, args.out / f"{stem}.parquet")
    report = formats.validate(result.decisions, args.challenge, path=str(signals_path))
    manifest = {
        "stem": stem,
        "rules_version": RULES_VERSION,
        "arguments": {k: (str(v) if isinstance(v, (Path, date, datetime)) else v) for k, v in vars(args).items()},
        "submission_commit": _git_commit(HERE),
        "model": {
            "class": type(model).__name__,
            "seed": getattr(model, "SEED", None),
            "training_data_cutoff": getattr(model, "TRAINING_DATA_CUTOFF", None),
        },
        "pack": _pack_identity(args.pack),
        "run": result.summary(),
        "rows": formats.summarize(result.decisions),
        "fit": result.fit_info,
        "read_log": {"path": log_path.name, "entries": pit.read_log.entries if pit.read_log else 0},
        "validation": {
            "ok": report.ok,
            "errors": [f"{p.code}: {p.message}" for p in report.errors],
            "warnings": [f"{p.code}: {p.message}" for p in report.warnings],
        },
        "seconds": round(time.perf_counter() - started, 3),
        "python": sys.version.split()[0],
    }
    (args.out / f"{stem}.run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(report.render())
    print(f"signals: {signals_path}")
    print(f"read log: {log_path} ({manifest['read_log']['entries']} entries)")
    print(
        f"sessions: {len(result.sessions)}, updates: {result.updates}, "
        f"outcomes delivered: {result.outcomes_delivered}, {manifest['seconds']}s"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
