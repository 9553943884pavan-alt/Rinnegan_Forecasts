#!/usr/bin/env python3
"""Score a submission's holdout files with the shipped scorer and print what it decided.

    python tools/score_holdout.py --pack ./pack --predictions predictions/
    python tools/score_holdout.py --pack ./pack --predictions predictions/ --challenge 2

Reads `c1_holdout.parquet`, `c2_holdout.parquet` and `c2_holdout_frozen.parquet` (the improvement
score needs the frozen copy) from `--predictions`, runs `python -m swing_dataset.scoring2 evaluate
--baselines` over the holdout window, writes `report_<challenge>_holdout.json` and
`holdout_scores.json` under `--out` (default: the predictions directory), and prints the three
scores, the success rate, the coverage floor and every condition of the minimum for this period,
each with its value and what it needs. Task 1 (`--challenge 1`): success rate at least 70,
correctness at least 60 (over decided signals; the undecided ones are counted apart), depth at
least 60, the coverage floor, and at least 7 of the 10 large swings of the holdout caught, which
are listed. Task 2 (`--challenge 2`): success rate at least 70, the coverage floor and the five
hard conditions of section 4. The forward window and the section 4a confirmation are decided
elsewhere.

Without `--challenge` both tasks are scored, and both files have to be there. A team that entered
one task passes `--challenge 1` or `--challenge 2`.

Task 1's catch rate is counted against the realized-swing inventory. `--swings` points at it;
by default the tool uses `swings/swings.parquet` inside the pack. The scorer lists the inventory
among a report's inputs for both tasks, so the file has to exist whichever task is scored. Exit
status 0 when every requested file was scored. Exit status 2 when an input is missing (a signal
file, the inventory, or the pack's own tables): a line on stderr says which and what to pass, and
nothing is scored or written then. Requires the `swing-dataset` package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swing_dataset.kit import repro


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, required=True, help="pack root with mcx/ and labels/")
    parser.add_argument("--predictions", type=Path, required=True, help="directory holding c1_holdout.parquet and c2_holdout.parquet")
    parser.add_argument("--challenge", type=int, choices=(1, 2), action="append", help="score one task, 1 or 2 (default: both, which needs both files)")
    parser.add_argument("--swings", type=Path, help="the realized-swing inventory (default: swings/swings.parquet in the pack)")
    parser.add_argument("--out", type=Path, help="where the reports go (default: the predictions directory)")
    parser.add_argument("--no-baselines", dest="baselines", action="store_false", help="skip the shipped baselines, which are most of the work")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from", dest="start", type=repro.date_arg, help="narrow the window (smoke tests only)")
    parser.add_argument("--to", dest="end", type=repro.date_arg, help="narrow the window (smoke tests only)")
    args = parser.parse_args(argv)
    if not (args.pack / "mcx" / "mcx_daily.parquet").exists():
        parser.error(f"{args.pack} has no mcx/mcx_daily.parquet")
    challenges = sorted(set(args.challenge or [1, 2]))
    files = {c: args.predictions / f"{repro.stem_for(c, 'holdout')}.parquet" for c in challenges}
    missing = [c for c in challenges if not files[c].exists()]
    if missing:
        for challenge in missing:
            print(f"missing {files[challenge]}", file=sys.stderr)
        present = [c for c in challenges if c not in missing]
        if args.challenge is None and present:
            print(
                f"nothing was scored: without --challenge both tasks are scored. To score Task {present[0]} alone, pass --challenge {present[0]}",
                file=sys.stderr,
            )
        else:
            print("nothing was scored", file=sys.stderr)
        return 2
    try:
        swings = args.swings or repro.swings_path(args.pack)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not Path(swings).exists():
        print(f"missing the realized-swing inventory {swings}; pass --swings with a copy of swings/swings.parquet", file=sys.stderr)
        return 2
    out = args.out or args.predictions
    period = repro.period_from("holdout", args.start, args.end)
    narrowed = args.start is not None or args.end is not None
    summary: dict = {"period": period.to_dict(), "pack": str(args.pack), "swings": str(swings), "challenges": {}}
    for challenge in challenges:
        signals = files[challenge]
        frozen = args.predictions / f"{repro.stem_for(2, 'holdout', frozen=True)}.parquet" if challenge == 2 else None
        if frozen is not None and not frozen.exists():
            print(f"note: {frozen} not found; without it there is no improvement score and no success rate", file=sys.stderr)
            frozen = None
        report_path = out / f"report_{challenge}_holdout.json"
        report = repro.evaluate(
            args.pack,
            signals,
            challenge=challenge,
            period=period,
            out=report_path,
            frozen=frozen,
            swings=swings,
            baselines=args.baselines,
            narrowed=narrowed,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        print(f"wrote {report_path}", file=sys.stderr)
        summary["challenges"][str(challenge)] = {
            "report": str(report_path),
            "scores": report["scores"],
            "counts": report["counts"],
            "coverage": report["coverage"],
            "award_conditions": report["award_conditions"],
            "baselines": {name: block["scores"] for name, block in (report.get("baselines") or {}).items()},
        }
        for key in ("ten_large_swings", "diagnostics"):
            if report.get(key):
                summary["challenges"][str(challenge)][key] = report[key]
        print(repro.render_report(report))
        print()
    (out / "holdout_scores.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"summary: {out / 'holdout_scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
