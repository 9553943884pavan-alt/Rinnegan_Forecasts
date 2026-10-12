#!/usr/bin/env python3
"""Organizer side: run every frozen repository over the forward window and score it.

    python tools/forward_run.py --repos repos.json --pack /path/to/extended-pack --workdir ./forward
    python tools/forward_run.py --repos repos.json --pack ... --workdir ... --score-only   # labels matured, rerun the scorer

``repos.json`` is a list of ``{"team": "...", "repo": "<url or path>", "commit": "<hash>",
"challenges": [1, 2]}``. For every entry the tool clones the commit (once; the checkout is
reused on later days), builds its environment from ``requirements.txt``, runs ``run.py
--period forward`` for each challenge (Task 2 also with ``--frozen``), keeps the outputs
under ``<workdir>/<team>/<YYYY-MM-DD>/``, and checks that the rows already produced on an
earlier day are unchanged, the provisional ones described below excepted. When ``--score`` is set (the default once the
forward labels exist in the pack), each file is scored with the shipped scorer over the forward
window (the sessions after the submission close, fit at the first of them) and the section 3
and 4 forward numbers are printed per team; ``forward_scores.json`` collects them.

How the pack grows: during the forward window the organizers rebuild the whole pack before each
session, as of 08:00 IST on the session day: every table the shipped pack carries (MCX
settlements, quotes, news, calendar, macro, Hyperliquid candles and funding, labels and
scoring), with the calendar captures synced in first, then ``python -m swing_dataset.pack
assemble``. A rebuilt pack is strictly additive: it holds every row available before 08:00 IST,
the designations for the next cutoff, and the labels and targets that matured. Running
this tool against the rebuilt pack each day extends every submission's forward file by the
sessions the new designations reach.

``--day`` is the day the pack was built for, ``YYYY-MM-DD`` (default: the UTC date). The outputs
go to ``<workdir>/<team>/<day>/``, and the label is what the check on the earlier rows needs.

The rows of the previous day's file must reappear, unchanged, in today's file, except its
provisional rows. On the run labelled X the pack holds nothing published after 08:00 IST on X, so
a row issued at a later instant was computed without it, and the next run, whose pack has it, may
give another row for the same instant. That is the case for the in-session rows of X (a model may
signal at any instant of a session), for its 08:30 IST pre-open rows (Hyperliquid and the news
run all night, and the pre-open reads thirty minutes of rows the pack lacks) and for its
``no_signal`` rows (the loop writes one only when no instant of the session signalled). When X is
not a trading day, the rows of the next session are in the same case. The check ignores those
rows of the previous file and nothing else: a changed row, a dropped row or a row of an earlier
session is reported, whether it was issued at the pre-open or inside the session. The rows that
were provisional on X are final in the file of the next run and are checked against the run
after it. They stay in the files: ``check_read_log`` reports a read after the last instant the
model was called at or issued a row for, and counts no call in a session the file has no row for,
so the reads of a session whose rows were removed are reported. The summary gives, for each
comparison, the rows set aside and the rows looked for; the first run of a window compares none.

The last run is made on a pack built after the window's last session, labelled with a later day
and stopped at that session (``--to``, and ``--to`` inside ``--run-args``). Every row of its file
was issued before 08:00 IST on its label, so none is provisional, and that is the file to score.
The rows of the last session in it have not been compared with an earlier file; one more run
under the next day's label compares them.

The forward window scores only horizons of at most 5 sessions on both tasks, because a longer
signal issued in the test week does not mature by the calls (CHALLENGE-SPEC section 5); those
rows are recorded and reported, not scored, and Task 1's ten large swings do not apply there.
Requires the ``swing-dataset`` package.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from swing_dataset.kit import repro
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


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "team"


def _day(text: str) -> str:
    try:
        return repro.day_label(text).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _previous_run(team_dir: Path, today: str) -> Path | None:
    """The directory of the latest run before ``today``; a directory whose name is not a day is not a run."""
    earlier = []
    for path in team_dir.iterdir():
        try:
            day = repro.day_label(path.name)
        except ValueError:
            continue
        if path.is_dir() and day < repro.day_label(today):
            earlier.append((day, path))
    return max(earlier)[1] if earlier else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos", type=Path, required=True, help="JSON list of {team, repo, commit, challenges}")
    parser.add_argument("--pack", type=Path, required=True, help="the extended pack (rebuilt after the latest session)")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--day", default=datetime.now(UTC).date().isoformat(), type=_day, help="the day the pack was built for, YYYY-MM-DD, which labels today's outputs (default: the UTC date)")
    parser.add_argument("--python", help="use this interpreter for every repository instead of building venvs")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--score", dest="score", action="store_true", default=True, help="score the forward window (default)")
    parser.add_argument("--no-score", dest="score", action="store_false")
    parser.add_argument("--score-only", action="store_true", help="skip the runs; score the latest outputs")
    parser.add_argument("--swings", type=Path, help="the realized-swing inventory (default: swings/swings.parquet in the pack)")
    parser.add_argument("--run-args", default="", help="extra arguments for run.py (smoke tests)")
    parser.add_argument("--from", dest="start", type=repro.date_arg, help="narrow the scored window (smoke tests only)")
    parser.add_argument("--to", dest="end", type=repro.date_arg, help="narrow the scored window (smoke tests only)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=6 * 3600)
    args = parser.parse_args(argv)
    if not (args.pack / "labels" / "designations.parquet").exists():
        parser.error(f"{args.pack} has no labels/designations.parquet")
    entries = json.loads(args.repos.read_text())
    if not isinstance(entries, list) or not entries:
        parser.error(f"{args.repos} must hold a non-empty JSON list")
    period = repro.period_from("forward", args.start, args.end)
    extra = shlex.split(args.run_args)
    args.workdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"day": args.day, "pack": str(args.pack), "period": period.to_dict(), "teams": {}}

    for entry in entries:
        team = _slug(str(entry["team"]))
        team_dir = args.workdir / team
        team_dir.mkdir(parents=True, exist_ok=True)
        checkout = team_dir / "repo"
        record: dict = {"team": entry["team"], "repo": entry["repo"], "commit": entry["commit"], "runs": {}, "conditions": {}}
        challenges = sorted({int(c) for c in entry.get("challenges", [1, 2])})
        out = team_dir / args.day
        if not args.score_only:
            if checkout.exists() and (checkout / ".git").exists():
                _log(f"{team}: reusing checkout {checkout}")
                record["resolved_commit"] = entry["commit"]
            else:
                _log(f"{team}: cloning {entry['repo']} at {entry['commit']}")
                record["resolved_commit"] = repro.clone_at(entry["repo"], entry["commit"], checkout)
            record["repository_checks"] = {k: v for k, v in repro.check_repository(checkout).items() if k != "direct_reads"}
            python = repro.make_env(team_dir, checkout / "requirements.txt", python=args.python, install=not args.no_install, reuse=True, log=_log)
            previous = _previous_run(team_dir, args.day)
            out.mkdir(parents=True, exist_ok=True)
            for challenge in challenges:
                for frozen in ((False,) if challenge == 1 else (False, True)):
                    stem = repro.stem_for(challenge, "forward", frozen)
                    _log(f"{team}: run.py {stem}")
                    proc = repro.run_submission(checkout, python, args.pack, challenge=challenge, period="forward", out=out, frozen=frozen, extra=extra, timeout=args.timeout)
                    run: dict = {"returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:]}
                    produced = out / f"{stem}.parquet"
                    if proc.returncode == 0 and produced.exists():
                        decisions = repro.load_predictions(produced)
                        run["rows"] = len(decisions)
                        run["read_log"] = repro.check_read_log(repro.load_read_log(repro.read_log_for(produced)), decisions, period.train_end, pack=args.pack)
                        run["overlap_with_previous_day"] = repro.overlap_unchanged(previous / f"{stem}.parquet", produced, previous.name) if previous else {"checked": False}
                    record["runs"][stem] = run
        if args.score:
            reports_dir = out / "scores"
            swings = args.swings or repro.swings_path(args.pack)
            for challenge in challenges:
                signals = out / f"{repro.stem_for(challenge, 'forward')}.parquet"
                if not signals.exists():
                    record["conditions"][str(challenge)] = {"error": f"{signals} missing"}
                    continue
                frozen = out / f"{repro.stem_for(2, 'forward', frozen=True)}.parquet" if challenge == 2 else None
                if frozen is not None and not frozen.exists():
                    frozen = None
                try:
                    report = repro.evaluate(args.pack, signals, challenge=challenge, period=period, out=reports_dir / f"report_{challenge}_forward.json", frozen=frozen, swings=swings, n_boot=args.n_boot, seed=args.seed)
                except (ValueError, RuntimeError, SystemExit) as exc:
                    record["conditions"][str(challenge)] = {"error": f"scoring failed: {exc}"}
                    _log(f"{team}: scoring challenge {challenge} failed: {exc}")
                    continue
                record["conditions"][str(challenge)] = {
                    "report": str(reports_dir / f"report_{challenge}_forward.json"),
                    "scores": report["scores"],
                    "coverage": report["coverage"],
                    "award_conditions": report["award_conditions"],
                }
                print(f"[{team}] " + repro.render_report(report).replace("\n", f"\n[{team}] "))
        summary["teams"][team] = record
    (args.workdir / f"forward_scores_{args.day}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.workdir / "forward_scores.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    problems = [(team, stem, run) for team, rec in summary["teams"].items() for stem, run in rec["runs"].items() if run.get("returncode") or not run.get("read_log", {}).get("ok", True) or run.get("overlap_with_previous_day", {}).get("ok") is False]
    for team, stem, run in problems:
        print(f"problem: {team} {stem}: exit {run.get('returncode')}, read log {run.get('read_log', {}).get('violations')}, overlap {run.get('overlap_with_previous_day')}")
    print(f"summary: {args.workdir / 'forward_scores.json'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
