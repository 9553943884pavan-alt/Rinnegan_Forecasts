#!/usr/bin/env python3
"""Organizer side: re-run a submission at its committed hash and compare with what it shipped.

    python tools/reproduce.py --repo https://github.com/team/entry --commit abc123 \\
        --pack ./pack --workdir /tmp/repro/team

    # a local checkout, the current interpreter, no install (dry run of the tool itself)
    python tools/reproduce.py --repo ../template --commit HEAD --pack ./pack \\
        --workdir /tmp/repro/template --python "$(which python)" --no-install

Steps, in order: clone and check out the commit; build a fresh virtual environment and install
``requirements.txt``; run ``run.py --period holdout`` for every requested challenge (Task 2
also with ``--frozen``); compare each produced file with the committed one under
``predictions/`` (byte for byte, else every numeric column within ``--tolerance``); check the
committed and the produced read logs (no read past the instant it served, no returned row at or
after its as-of instant, no read after the last instant the model was called at or issued a row
for, fit reads before the training cutoff) and that both logs list the same reads; run the
repository checks of CHALLENGE-SPEC section 7 (license, pinned dependencies,
memo, direct parquet reads outside the client). A logged clock at or after its as-of is repeated
on ``--pack`` and judged by the repeat, because a log written by an earlier client carries the
latest of every timestamp column among the rows returned, not the clock that admitted each row.
Prints a verdict, writes ``verdict.json`` in
the work directory, exits 0 on REPRODUCED and 1 otherwise. Requires the ``swing-dataset``
package in the interpreter that runs this tool.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

try:
    from swing_dataset.kit import loop, repro
except ImportError as exc:  # for example "No module named 'swing_dataset.pack'", left by a reinstall that .gitignore hid files from
    sys.exit(
        f"swing_dataset cannot be imported here ({exc}): replace the whole swing_dataset/ directory of your repository with the one from "
        "kit 2.6.2, then run, from the repository root, pip install --force-reinstall --no-deps ./swing_dataset"
    )
if not hasattr(repro, "day_label"):  # added in kit 2.6.2: a package without it is a 2.6 or 2.6.1 one
    sys.exit(
        "The installed swing_dataset is older than these tools: replace the whole swing_dataset/ directory of your repository with the one from "
        "kit 2.6.2, then run, from the repository root, pip install --force-reinstall --no-deps ./swing_dataset"
    )


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="repository URL or local path")
    parser.add_argument("--commit", required=True, help="the submitted commit hash")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True, help="scratch directory for the clone, the venv and the outputs")
    parser.add_argument("--challenge", type=int, choices=(1, 2), action="append", help="reproduce one challenge (default: both)")
    parser.add_argument("--period", choices=sorted(loop.PERIODS), default="holdout")
    parser.add_argument("--python", help="use this interpreter instead of building a venv")
    parser.add_argument("--no-install", action="store_true", help="do not pip install requirements.txt (with --python)")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--run-args", default="", help="extra arguments passed to run.py (smoke tests: '--from 2026-07-01 --to 2026-07-14')")
    parser.add_argument("--timeout", type=float, default=6 * 3600, help="seconds per run.py invocation")
    args = parser.parse_args(argv)
    if not (args.pack / "labels" / "designations.parquet").exists():
        parser.error(f"{args.pack} has no labels/designations.parquet")

    args.workdir.mkdir(parents=True, exist_ok=True)
    checkout = args.workdir / "repo"
    _log(f"cloning {args.repo} at {args.commit}")
    resolved = repro.clone_at(args.repo, args.commit, checkout)
    _log(f"checked out {resolved}")
    checks = repro.check_repository(checkout)
    verdict: dict = {"repo": args.repo, "commit": resolved, "pack": str(args.pack), "period": args.period, "repository_checks": checks, "runs": {}, "problems": [], "review": []}
    if not checks["run_py"]:
        verdict["problems"].append("run.py is missing")
    if checks["license"] is None:
        verdict["review"].append("no Apache-2.0 or MIT LICENSE file")
    if not checks["memo"]:
        verdict["review"].append("no memo file (memo.md, memo.pdf or docs/memo*)")
    if checks["requirements_pinned"] is False:
        verdict["review"].append("requirements.txt has unpinned lines: " + "; ".join(checks["requirements_unpinned"][:5]))
    if checks["requirements_pinned"] is None and not checks["lockfile"]:
        verdict["review"].append("no requirements.txt and no lockfile")
    if checks["direct_reads"]:
        verdict["review"].append(f"{len(checks['direct_reads'])} direct parquet read(s) outside the client: " + "; ".join(f"{h['file']}:{h['line']}" for h in checks["direct_reads"][:8]))

    python = repro.make_env(args.workdir, checkout / "requirements.txt", python=args.python, install=not args.no_install, log=_log)
    extra = shlex.split(args.run_args)
    period = repro.period_from(args.period)
    reproduced_all = bool(checks["run_py"])
    for challenge in sorted(set(args.challenge or [1, 2])):
        for frozen in ((False,) if challenge == 1 else (False, True)):
            stem = repro.stem_for(challenge, args.period, frozen)
            out = args.workdir / "out"
            _log(f"running run.py for {stem}")
            proc = repro.run_submission(checkout, python, args.pack, challenge=challenge, period=args.period, out=out, frozen=frozen, extra=extra, timeout=args.timeout)
            entry: dict = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
            if proc.returncode != 0:
                entry["status"] = "run_failed"
                verdict["problems"].append(f"{stem}: run.py exited {proc.returncode}")
                reproduced_all = False
                verdict["runs"][stem] = entry
                continue
            produced = out / f"{stem}.parquet"
            committed = checkout / "predictions" / f"{stem}.parquet"
            comparison = repro.compare_predictions(committed, produced, args.tolerance)
            entry["files"] = comparison
            if not comparison["ok"]:
                verdict["problems"].append(f"{stem}: {comparison['status']}")
                reproduced_all = False
            produced_log = repro.load_read_log(repro.read_log_for(produced))
            committed_log = repro.load_read_log(repro.read_log_for(committed))
            decisions = repro.load_predictions(produced) if produced.exists() else None
            entry["read_log"] = {
                "produced": repro.check_read_log(produced_log, decisions, period.train_end, pack=args.pack),
                "committed": repro.check_read_log(committed_log, decisions, period.train_end, pack=args.pack) if committed_log else {"entries": 0, "ok": False, "violations": ["no committed read log"]},
                "same_reads": repro.compare_read_logs(committed_log, produced_log),
            }
            for which in ("produced", "committed"):
                if not entry["read_log"][which]["ok"]:
                    verdict["problems"].append(f"{stem}: {which} read log: " + "; ".join(entry["read_log"][which]["violations"][:5]))
                    reproduced_all = False
            if not entry["read_log"]["same_reads"]["same"]:
                verdict["problems"].append(f"{stem}: the produced read log differs from the committed one at seq {(entry['read_log']['same_reads']['first_difference'] or {}).get('seq')}")
                reproduced_all = False
            entry["status"] = comparison["status"]
            verdict["runs"][stem] = entry
    verdict["verdict"] = "REPRODUCED" if reproduced_all and verdict["runs"] else "NOT REPRODUCED"
    (args.workdir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    print(f"repository {args.repo} at {resolved}")
    for stem, entry in verdict["runs"].items():
        files = entry.get("files", {})
        print(f"  {stem}: {entry['status']}" + (f" (max abs difference {files['max_abs_difference']:.3g})" if "max_abs_difference" in files else "") + f"; read log {'ok' if entry.get('read_log', {}).get('produced', {}).get('ok') else 'PROBLEM'}")
    for problem in verdict["problems"]:
        print(f"  problem: {problem}")
    for item in verdict["review"]:
        print(f"  review: {item}")
    print(f"verdict: {verdict['verdict']}" + (f"; {len(verdict['review'])} item(s) for review" if verdict["review"] else ""))
    print(f"details: {args.workdir / 'verdict.json'}")
    return 0 if verdict["verdict"] == "REPRODUCED" else 1


if __name__ == "__main__":
    sys.exit(main())
