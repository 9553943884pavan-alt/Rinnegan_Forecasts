"""Organizer-side helpers: scoring a period with the shipped scorer, read-log checks, file
comparison, and running a submission repository in a fresh environment.

The score definitions, the minimums of CHALLENGE-SPEC sections 3 and 4 and the coverage floor of
section 5 belong to ``swing_dataset.scoring2``: ``evaluate`` runs its command line and returns the
report it wrote, and ``render_report`` prints what that report already decided, down to each gate
of Task 1's minimum. Nothing here recomputes a score or restates a condition.

The tools in ``challenge/kit/tools/`` (``score_holdout.py``, ``reproduce.py``, ``forward_run.py``)
are thin command lines over this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import venv
from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..pack.pit import PointInTime, ReadLog
from ..scoring2 import COVERAGE_FLOOR, MINIMUM_SUCCESS_RATE, SWINGS_PATH
from ..scoring2.__main__ import main as scoring2_main
from ..scoring2.prices import IST, ist_instant
from .loop import PERIODS, Period, period_for, preopen_instant

DIRECT_READ_PATTERNS = (
    r"\bread_table\(",
    r"\bread_parquet\(",
    r"\bParquetFile\(",
    r"\bParquetDataset\(",
    r"pyarrow\.dataset",
    r"\bpl\.(scan|read)_parquet\(",
    r"\bduckdb\b",
)
PREDICTION_STEMS = {1: ("c1_{period}",), 2: ("c2_{period}", "c2_{period}_frozen")}
#: The scored window ends after the last session's close; MCX closes at 23:30 or 23:55 IST.
END_OF_DAY_IST = time(23, 59, 59)


def stem_for(challenge: int, period: str, frozen: bool = False) -> str:
    return f"c{challenge}_{period}" + ("_frozen" if frozen else "")


def swings_path(pack: Path | None = None) -> Path:
    """The realized-swing inventory Task 1's catch rate is counted against.

    It ships inside the pack as ``swings/swings.parquet``. It is also a lane output at
    ``build/swings/swings.parquet``, so ``SWINGS_PATH`` under the build root is used when no
    pack is given; a pack that does not carry the file is an error rather than a silent
    substitution, because scoring against a different inventory gives a different score.
    """
    if pack is None:
        return Path(SWINGS_PATH)
    path = Path(pack) / "swings" / "swings.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: the pack does not carry the realized-swing inventory. "
            "Pass --swings to point at a copy of swings.parquet."
        )
    return path


def window_bounds(period: Period) -> tuple[str, str]:
    """The period's instant bounds for ``--test-start`` and ``--test-end``."""
    return (
        preopen_instant(period.start).isoformat(),
        ist_instant(period.end, END_OF_DAY_IST).isoformat(),
    )


# -- scoring --------------------------------------------------------------------------------


def evaluate(
    pack: Path,
    signals: Path,
    *,
    challenge: int,
    period: Period,
    out: Path,
    frozen: Path | None = None,
    no_signals: Path | None = None,
    swings: Path | None = None,
    baselines: bool = True,
    narrowed: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Run ``python -m swing_dataset.scoring2 evaluate`` over one period and return its report.

    Every input comes from ``pack`` so that a participant scores against the same files the
    organizers do. ``narrowed`` passes the period's own bounds as ``--test-start`` and
    ``--test-end``; without it the holdout keeps the scorer's own bounds (2026-06-01 to the pack's
    last session) and only the forward window, which has no fixed start in the data, is bounded.
    """
    pack = Path(pack)
    start, end = window_bounds(period)
    argv = [
        "evaluate",
        "--challenge", str(challenge),
        "--period", period.name,
        "--signals", str(signals),
        "--swings", str(swings or swings_path(pack)),
        "--mcx-daily", str(pack / "mcx" / "mcx_daily.parquet"),
        "--designations", str(pack / "labels" / "designations.parquet"),
        "--mcx-sessions", str(pack / "mcx" / "mcx_sessions.parquet"),
        "--quotes", str(pack / "pack" / "quotes" / "mcx_quotes.parquet"),
        "--n-boot", str(n_boot),
        "--seed", str(seed),
        "--out", str(out),
    ]  # fmt: skip
    if baselines:
        argv.append("--baselines")
    if frozen is not None:
        argv += ["--frozen", str(frozen)]
    if no_signals is not None:
        argv += ["--no-signals", str(no_signals)]
    if narrowed or period.name == "forward":
        argv += ["--test-start", start, "--test-end", end]
    out.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        code = scoring2_main(argv)
    if code:
        raise RuntimeError(f"scoring2 returned {code} for {signals}")
    return json.loads(out.read_text())


def _pct(entry: dict | None) -> str:
    if entry is None or entry.get("value") is None:
        return "n/a"
    lo, hi = entry.get("ci_lo"), entry.get("ci_hi")
    band = f" [{lo:.1f}, {hi:.1f}]" if lo is not None and hi is not None else ""
    return f"{entry['value']:.1f}{band}"


#: How each Task 1 gate of ``award_conditions`` is named and its numbers printed.
GATE_LINES = {
    "success_rate": ("success rate", "{:.1f}", "{:g}"),
    "correctness": ("correctness", "{:.1f}", "{:g}"),
    "depth": ("depth", "{:.1f}", "{:g}"),
    "coverage_floor": ("coverage floor, least covered asset", "{:.3f}", "{:.2f}"),
    "ten_large_swings": ("ten large swings caught", "{:d}", "{:d}"),
}


def _gate_line(name: str, gate: dict) -> str:
    label, value_format, threshold_format = GATE_LINES[name]
    if not gate["applicable"]:
        return f"{label}: does not apply to this period"
    value = "n/a" if gate["value"] is None else value_format.format(gate["value"])
    needs = threshold_format.format(gate["threshold"])
    return f"{label} {value}, needs {needs}: {'yes' if gate['met'] else 'no'}"


def _large_swing_lines(block: dict) -> list[str]:
    lines = [
        (
            f"  ten large swings: {block['caught']} of {block['swings_ranked']} caught, "
            f"at least {block['required']} needed on the holdout"
        )
    ]
    for swing in block["swings"]:
        by = swing["caught_by_signal_utc"]
        lines.append(
            f"    {swing['rank']:>2}. {swing['asset']:<6} {swing['direction']:<4} "
            f"{str(swing['turn_utc'])[:16]} to {str(swing['extreme_utc'])[:16]}  "
            f"{swing['move_pct']:+.2f} percent, {swing['threshold_multiple']:.2f} thresholds  "
            + (f"caught by the signal of {str(by)[:19]}" if swing["caught"] else "not caught")
        )
    return lines


def render_report(report: dict) -> str:
    """The report's own numbers as a block, with nothing recomputed."""
    window = report["window"]
    scores = report["scores"]
    task_1 = report["challenge"] == 1
    names = ("catch_rate", "correctness", "depth") if task_1 else ("direction", "depth", "improvement")
    heading = (
        f"Task {report['challenge']}, {window['period']} window "
        f"{window['start_utc'][:10]} to {window['end_utc'][:10]}, horizons {window['horizons_scored']}"
    )
    lines = [heading]
    lines.append("  " + "  ".join(f"{n.replace('_', ' ')} {_pct(scores.get(n))}" for n in names))
    lines.append(f"  success rate {_pct(scores.get('success_rate'))} (minimum {MINIMUM_SUCCESS_RATE:.0f})")
    counts = report["counts"]
    detail = [
        f"signals {counts['signals_in_window']}",
        f"invalid {counts['invalid']}",
        f"unmatured {counts['unmatured']}",
        f"scored {counts['scored']}",
    ]
    if task_1:
        share = counts.get("undecided_share")
        detail.append(
            f"undecided {counts['undecided']}"
            + (f" ({100 * share:.1f} percent of scored, left out of correctness)" if share is not None else "")
        )
    detail.append(f"no_signal {counts['no_signal_rows']}")
    if task_1:
        detail += [f"swings {counts['swings_in_window']}", f"caught {counts['swings_caught']}"]
    lines.append("  " + ", ".join(detail))
    cover = report["coverage"]
    shares = " ".join(
        f"{a}={(v['share'] if v['share'] is not None else float('nan')):.3f}" for a, v in sorted(cover["per_asset"].items())
    )
    lines.append(f"  coverage (floor {COVERAGE_FLOOR:.2f}): {shares} -> {'met' if cover['met'] else 'NOT met'}")
    award = report["award_conditions"]
    if task_1 and report.get("ten_large_swings") and award["gates"]["ten_large_swings"]["applicable"]:
        lines += _large_swing_lines(report["ten_large_swings"])
    if report["challenge"] == 2:
        conditions = award.get("hard_conditions", {})
        state = {True: "yes", False: "no", None: "not decided here"}
        lines.append("  hard conditions: " + ", ".join(f"{k} {state[v]}" for k, v in conditions.items()))
        null = report.get("prevailing_direction_null", {})
        if null.get("direction_at_entry_instants") is not None:
            lines.append(f"  prevailing-direction null at the same instants: {null['direction_at_entry_instants']:.1f}")
    for name, block in (report.get("baselines") or {}).items():
        lines.append(f"  baseline {name}: success rate {_pct(block['scores'].get('success_rate'))}")
    lines.append(
        f"  minimum on the {award['period']} period: "
        f"{'met' if award['minimum_met_this_period'] else 'NOT met'} "
        "(both periods and the section 4a gate decide the prize)"
    )
    if task_1:
        lines += [f"    {_gate_line(name, gate)}" for name, gate in award["gates"].items()]
    return "\n".join(lines)


# -- read log checks ------------------------------------------------------------------------


def _iso(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def last_instant(decisions: pd.DataFrame | None) -> datetime | None:
    """The latest instant any row of a decision file carries, signal or ``no_signal``."""
    if decisions is None or not len(decisions):
        return None
    stamps = []
    for column in ("signal_utc", "decision_utc"):
        if column in decisions.columns:
            values = pd.to_datetime(decisions[column], utc=True, errors="coerce").dropna()
            if len(values):
                stamps.append(values.max())
    return max(stamps).to_pydatetime() if stamps else None


def _end_of_day(instant: datetime) -> datetime:
    """The end of the Indian calendar day ``instant`` falls on, after the close of any session of it."""
    return ist_instant(instant.astimezone(IST).date(), END_OF_DAY_IST).to_pydatetime()


class _Repeats:
    """The reads of a log repeated on a pack with this package's client, each distinct read once."""

    def __init__(self, pack: Path | str) -> None:
        self.pack = pack
        self.pit: PointInTime | None = None
        self.lines: dict[str, dict | None] = {}

    def of(self, entry: dict, asof: datetime) -> dict | None:
        """The line the logged read writes on the pack, or None when the pack cannot repeat it."""
        columns = entry.get("columns")
        if asof.tzinfo is None or (columns is not None and not (isinstance(columns, list) and all(isinstance(c, str) for c in columns))):
            return None
        opt_in = bool(entry.get("include_unvintaged"))
        key = json.dumps([entry["file"], asof.isoformat(), opt_in, columns])
        if key not in self.lines:
            if self.pit is None:
                self.pit = PointInTime(self.pack)
            frame = self.pit.frame(entry["file"], asof, include_unvintaged=opt_in) if entry["file"] in self.pit.files else None
            self.lines[key] = None if frame is None else frame.log_entry(columns)
        return self.lines[key]


def check_read_log(
    entries: list[dict],
    decisions: pd.DataFrame | None = None,
    train_end: datetime | None = None,
    pack: Path | str | None = None,
) -> dict:
    """Every read served a phase and an instant; none looked past the instant it served.

    Violations: an ``asof_utc`` after the instant or fit cutoff in its context, a returned row
    whose clock is at or after ``asof_utc``, a read returning rows after the last instant the model
    was called at or issued a row for, or an unvintaged read (allowed only when the memo declares
    it, so it is listed, not failed).

    That last instant is the later of two: the last instant any row of ``decisions`` carries, and
    the latest ``cutoff_utc`` of a read made for a ``predict`` call. The second is there because a
    model that defines ``instants`` is called at each of them and may signal at only some: the loop
    writes a ``no_signal`` row, at the pre-open, only when no instant of the session issued a row,
    so a call that issued nothing leaves nothing in the file. A call counts only when it falls on or
    before the Indian calendar day of the file's last row. The loop writes a row for every session
    it calls the model in, so a log that runs past its file is not evidence of a call. ``update``,
    ``fit`` and the loop's eligibility reads are not calls of ``predict``, and a read of theirs
    after the last call is a violation whatever instant it was served.

    A log written by an earlier client carries a clock that is the latest of every timestamp
    column among the rows returned, which for a vintage published long before it was captured is
    after the as-of although every row was available. With ``pack`` given, a logged clock that
    fails the test is not trusted: the read (file, as-of, opt-in, columns, as logged) is repeated
    on the pack with this package's client and judged by the clock it returns, so only a returned
    row whose own clock is at or after the as-of is a violation. A repeat that returns another
    number of rows than the log holds is not the logged read and clears nothing, and neither does a
    read the pack cannot repeat. ``clocks_rederived`` counts the reads repeated. Without ``pack``
    the logged clock stands.

    The loop's own end-of-pack probe, an eligibility-phase read of ``labels/designations`` at the
    next session's pre-open, is not counted as a read after the last instant: it returns only rows
    available before its instant, and every other check applies to it.

    The last check is what proves the section 4a rule at the row level: a signal may fire at any
    instant inside its session, and no read in the log returned a row it could not have seen.
    """
    violations: list[str] = []
    unvintaged = 0
    rederived = 0
    repeats = None if pack is None else _Repeats(pack)
    files: dict[str, int] = {}
    latest_asof: datetime | None = None
    last = last_instant(decisions)
    covered = None if last is None else _end_of_day(last)
    called: datetime | None = None
    for entry in entries:
        seq = entry.get("seq")
        asof = _iso(entry.get("asof_utc"))
        if asof is None:
            violations.append(f"seq {seq}: no asof_utc")
            continue
        files[entry["file"]] = files.get(entry["file"], 0) + 1
        probe = entry["file"] == "labels/designations" and (entry.get("context") or {}).get("phase") == "eligibility"
        if entry.get("rows", 0) and not probe and (latest_asof is None or asof > latest_asof):
            latest_asof = asof
        context = entry.get("context") or {}
        served = _iso(context.get("cutoff_utc"))
        if served is not None and asof > served:
            violations.append(f"seq {seq}: {entry['file']} read at {asof.isoformat()} for the instant {served.isoformat()}")
        if context.get("phase") == "predict" and served is not None and covered is not None and served <= covered:
            called = served if called is None else max(called, served)
        fit_end = _iso(context.get("train_end_utc"))
        if fit_end is not None and asof > fit_end:
            violations.append(f"seq {seq}: fit read at {asof.isoformat()} after train_end {fit_end.isoformat()}")
        if context.get("phase") == "fit" and train_end is not None and asof > train_end:
            violations.append(f"seq {seq}: fit read at {asof.isoformat()} after the period's train_end {train_end.isoformat()}")
        clock_max = _iso(entry.get("clock_max_utc"))
        if clock_max is not None and clock_max >= asof:
            again = None if repeats is None else repeats.of(entry, asof)
            if again is None:
                violations.append(f"seq {seq}: returned row clock {clock_max.isoformat()} at or after asof {asof.isoformat()}")
            else:
                rederived += 1
                repeated = _iso(again["clock_max_utc"])
                if repeated is not None and repeated >= asof:
                    violations.append(f"seq {seq}: returned row clock {repeated.isoformat()} at or after asof {asof.isoformat()}")
                elif entry.get("rows") is not None and entry["rows"] != again["rows"]:
                    violations.append(
                        f"seq {seq}: returned row clock {clock_max.isoformat()} at or after asof {asof.isoformat()}, "
                        f"and the same read on the pack returns {again['rows']} rows, not the {entry['rows']} logged"
                    )
        if entry.get("include_unvintaged"):
            unvintaged += 1
    bound = last if called is None or last is None else max(last, called)
    if bound is not None and latest_asof is not None and latest_asof > bound:
        violations.append(
            f"latest read returning rows, {latest_asof.isoformat()}, is after the last instant "
            f"the model was called at or issued a row for, {bound.isoformat()}"
        )
    return {
        "entries": len(entries),
        "files": files,
        "latest_asof_utc": latest_asof.isoformat() if latest_asof else None,
        "last_row_instant_utc": last.isoformat() if last else None,
        "last_call_instant_utc": called.isoformat() if called else None,
        "unvintaged_reads": unvintaged,
        "clocks_rederived": rederived,
        "violations": violations,
        "ok": not violations,
    }


def compare_read_logs(a: list[dict], b: list[dict]) -> dict:
    """The same reads in the same order (file, asof, columns, rows), ignoring wall-clock stamps."""

    def key(entry: dict) -> tuple:
        return (
            entry.get("file"),
            entry.get("asof_utc"),
            entry.get("include_unvintaged"),
            tuple(entry.get("columns") or ()),
            entry.get("rows"),
            json.dumps(entry.get("context"), sort_keys=True),
        )

    ka, kb = [key(e) for e in a], [key(e) for e in b]
    first = next((i for i, (x, y) in enumerate(zip(ka, kb, strict=False)) if x != y), None)
    return {
        "entries": (len(a), len(b)),
        "same": ka == kb,
        "first_difference": None if first is None else {"seq": first + 1, "submitted": ka[first], "reproduced": kb[first]},
    }


# -- signal file comparison ---------------------------------------------------------------


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in a canonical order, so two files that hold the same rows compare equal.

    A signal file may repeat a key and the order of two rows at one instant is not part of the
    submission, so the comparison sorts on every column rather than on a key.
    """
    out = frame.copy()
    for column in ("signal_utc", "decision_utc"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce").dt.as_unit("us")
    columns = sorted(out.columns)
    keys = out[columns].astype(str)
    order = keys.sort_values(columns, kind="stable").index
    return out.loc[order].reset_index(drop=True)


def compare_predictions(submitted: Path, reproduced: Path, tolerance: float = 1e-9) -> dict:
    """Byte-identical, else equal within ``tolerance`` on every numeric column over the same rows."""
    result: dict = {"submitted": str(submitted), "reproduced": str(reproduced), "tolerance": tolerance}
    if not Path(submitted).exists() or not Path(reproduced).exists():
        result.update(status="missing", ok=False)
        return result
    if sha256_of(submitted) == sha256_of(reproduced):
        result.update(status="identical", ok=True, max_abs_difference=0.0)
        return result
    a = _sorted(pq.read_table(submitted).to_pandas())
    b = _sorted(pq.read_table(reproduced).to_pandas())
    if sorted(a.columns) != sorted(b.columns):
        result.update(status="different_columns", ok=False, columns=(sorted(a.columns), sorted(b.columns)))
        return result
    if len(a) != len(b):
        result.update(status="different_rows", ok=False, rows=(len(a), len(b)))
        return result
    worst = 0.0
    for column in sorted(a.columns):
        if pd.api.types.is_numeric_dtype(a[column]) and not pd.api.types.is_bool_dtype(a[column]):
            x = pd.to_numeric(a[column], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(b[column], errors="coerce").to_numpy(dtype=float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                result.update(status="different_values", ok=False, column=column)
                return result
            both = ~(np.isnan(x) & np.isnan(y))
            if both.any():
                worst = max(worst, float(np.abs(x[both] - y[both]).max()))
        elif not a[column].astype(str).equals(b[column].astype(str)):
            result.update(status="different_values", ok=False, column=column)
            return result
    result.update(
        status="within_tolerance" if worst <= tolerance else "different_values",
        ok=worst <= tolerance,
        max_abs_difference=worst,
    )
    return result


def _row_keys(frame: pd.DataFrame) -> list[tuple]:
    columns = sorted(frame.columns)
    text = _sorted(frame)[columns].astype(str)
    return [tuple(row) for row in text.itertuples(index=False, name=None)]


#: The organizers build each forward day's pack as of 08:00 IST on that day (``tools/forward_run.py``).
PACK_ASOF_IST = time(8, 0)
DAY_LABEL = re.compile(r"\d{4}-\d{2}-\d{2}")


def day_label(value: str | date) -> date:
    """The day a forward run is labelled with: a date, or ``YYYY-MM-DD`` text, the name of its directory."""
    if isinstance(value, str) and DAY_LABEL.fullmatch(value):
        with contextlib.suppress(ValueError):
            return date.fromisoformat(value)
    elif isinstance(value, date) and not isinstance(value, datetime):
        return value
    raise ValueError(f"{value!r} is not a date or YYYY-MM-DD text")


def _issued_at(frame: pd.DataFrame) -> pd.Series:
    """The instant of each row: ``signal_utc``, or ``decision_utc`` on a ``no_signal`` row; NaT with neither."""
    out = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[us, UTC]")
    for column in ("signal_utc", "decision_utc"):
        if column in frame.columns:
            out = out.fillna(pd.to_datetime(frame[column], utc=True, errors="coerce").dt.as_unit("us"))
    return out


def overlap_unchanged(previous: Path, current: Path, previous_day: str | date) -> dict:
    """The rows of an earlier forward file that were final must reappear, unchanged, in a later one.

    ``previous_day`` is the label the earlier run was made under (``forward_run.py --day``). Its
    pack was built as of 08:00 IST on that day (``PACK_ASOF_IST``), so a row issued at a later
    instant (``signal_utc``, or ``decision_utc`` on a ``no_signal`` row) was computed without
    what was published after the cut, and a run whose pack holds it may give another row for the
    same instant. Those rows are provisional:

    - the day's in-session rows: a model may signal at any instant of a session, and the pack
      holds none of that day's data;
    - its 08:30 IST pre-open rows: sources that run all night, Hyperliquid and the news, add rows
      in the thirty minutes between the cut and the pre-open;
    - its ``no_signal`` rows: the loop writes one only when no instant of the session signalled,
      and an in-session instant may signal on the next run;
    - every row of a later session, when the day is not a trading day and the pack already holds
      the next session's designations.

    The check ignores the provisional rows of the earlier file and nothing else. Every other row
    must reappear: a changed row, a dropped one, a row of an earlier session, whether it was issued
    at the pre-open or inside the session. The ignored rows are final in the later run's own file
    and are checked against the run after it. A label later than every row, which is how the last
    run is labelled, leaves no row provisional.

    A signal file has no unique key: a model may issue several signals for one asset at one
    instant, and rules 2.6 counts every one of them, so the rows that must reappear are compared
    as a multiset of whole rows, and the later file may hold more.

    The result says how many rows were set aside (``provisional_rows``) and how many were looked
    for (``compared_rows``): the first run of a window sets all of its rows aside and compares none.
    """
    day = day_label(previous_day)
    if not Path(previous).exists() or not Path(current).exists():
        return {"checked": False}
    cut = ist_instant(day, PACK_ASOF_IST)
    earlier = load_predictions(Path(previous))
    provisional = _issued_at(earlier) > cut
    old = Counter(_row_keys(earlier[~provisional]))
    new = Counter(_row_keys(load_predictions(Path(current))))
    missing = old - new
    return {
        "checked": True,
        "ok": not missing,
        "rows": (len(earlier), sum(new.values())),
        "provisional_rows": int(provisional.sum()),
        "compared_rows": sum(old.values()),
        "missing_rows": sum(missing.values()),
        "first_missing": list(next(iter(missing))) if missing else None,
        "previous": str(previous),
        "previous_day": day.isoformat(),
        "provisional_after_utc": cut.isoformat(),
    }


# -- repository checks ------------------------------------------------------------------------


def scan_direct_reads(repo: Path) -> list[dict]:
    """Lines in the repository's Python files that read parquet outside the client (for review)."""
    hits = []
    pattern = re.compile("|".join(DIRECT_READ_PATTERNS))
    for path in sorted(Path(repo).rglob("*.py")):
        if any(part in {".venv", "venv", "node_modules", "swing_dataset", "site-packages"} for part in path.parts):
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if pattern.search(line) and not line.lstrip().startswith("#"):
                hits.append({"file": str(path.relative_to(repo)), "line": number, "text": line.strip()[:160]})
    return hits


def _requirement_text(raw: str) -> str:
    """One line of a requirements file without its comment, whole-line or trailing, as pip reads it."""
    return re.sub(r"(^|\s)#.*$", "", raw).strip()


def _repository_directory(repo: Path, requirement: str) -> bool:
    """A requirement that is a path to a directory inside the repository.

    pip installs it from the checkout, so the commit pins it and it needs no version: the
    template's own ``./swing_dataset`` line is one. A bare name such as ``swing_dataset`` is a
    package name for an index, a path that leaves the repository is not in the commit, and a path
    to a directory the repository does not hold installs nothing, so none of those is one.
    """
    if requirement not in (".", "..") and not requirement.startswith(("./", "../")):
        return False
    target = (repo / requirement).resolve()
    return target.is_dir() and target.is_relative_to(repo.resolve())


def check_repository(repo: Path) -> dict:
    """Presence checks from CHALLENGE-SPEC section 7: run.py, pinned dependencies, license, memo.

    A requirements line is pinned when it has ``==``, or a commit for a git URL, or is a directory
    of the repository itself (``_repository_directory``). Any other line is listed in
    ``requirements_unpinned``; comments are not lines.
    """
    repo = Path(repo)
    requirements = repo / "requirements.txt"
    lock = [p for p in ("uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock") if (repo / p).exists()]
    pinned = None
    unpinned: list[str] = []
    if requirements.exists():
        lines = [text for text in map(_requirement_text, requirements.read_text().splitlines()) if text]
        for ln in lines:
            if ln.startswith("-") or _repository_directory(repo, ln):
                continue
            if "git+" in ln or "://" in ln:
                if not re.search(r"@[^#/\s]+(#|$)", ln.split("git+", 1)[-1]):
                    unpinned.append(ln)
            elif "==" not in ln:
                unpinned.append(ln)
        pinned = not unpinned if lines else False
    license_text = ""
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        if (repo / name).exists():
            license_text = (repo / name).read_text(errors="replace")
            break
    license_kind = (
        "Apache-2.0"
        if "Apache License" in license_text
        else ("MIT" if "MIT License" in license_text or "Permission is hereby granted, free of charge" in license_text else None)
    )
    memo = sorted(
        str(p.relative_to(repo))
        for pattern in ("memo*", "docs/memo*")
        for p in repo.glob(pattern)
        if p.is_file() and not p.name.startswith("memo_template")
    )
    return {
        "run_py": (repo / "run.py").exists(),
        "model_py": (repo / "model.py").exists(),
        "requirements_pinned": pinned,
        "requirements_unpinned": unpinned,
        "lockfile": lock,
        "license": license_kind,
        "memo": memo,
        "predictions_dir": (repo / "predictions").is_dir(),
        "direct_reads": scan_direct_reads(repo),
    }


# -- running a submission -----------------------------------------------------------------------


def clone_at(repo: str, commit: str, dest: Path) -> str:
    """Clone (or copy a local directory) and check out ``commit``; returns the resolved commit."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    source = Path(repo)
    if source.exists() and (source / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", str(source), str(dest)], check=True)
    else:
        subprocess.run(["git", "clone", "--quiet", repo, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", commit], check=True)
    return subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()


def make_env(
    dest: Path,
    requirements: Path | None,
    *,
    python: str | None = None,
    install: bool = True,
    reuse: bool = False,
    log=lambda _m: None,
) -> str:
    """A fresh virtual environment with the repository's pinned dependencies; returns its python.

    ``python`` skips the venv and uses that interpreter as is; ``reuse`` keeps an environment built
    on an earlier day (the forward window runs the same commit every day).
    """
    if python:
        return python
    env_dir = (Path(dest) / ".venv").resolve()
    interpreter = str(env_dir / "bin" / "python")
    if reuse and Path(interpreter).exists():
        return interpreter
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    if install and requirements is not None and requirements.exists():
        log(f"pip install -r {requirements}")
        # The kit's requirements.txt pins ./swing_dataset, a path relative to the repository, not
        # to the caller's directory, so pip runs from the requirements file's own parent. The venv
        # interpreter is absolute, so it stays valid under that working directory.
        req = requirements.resolve()
        subprocess.run(
            [interpreter, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", str(req)],
            check=True,
            cwd=req.parent,
        )
    return interpreter


def run_submission(
    repo: Path,
    python: str,
    pack: Path,
    *,
    challenge: int,
    period: str,
    out: Path,
    frozen: bool = False,
    extra: Sequence[str] = (),
    timeout: float = 6 * 3600,
) -> subprocess.CompletedProcess:
    # Resolve everything to absolute paths: the tools accept relative --workdir/--pack, and a
    # relative script path would otherwise resolve against cwd=repo and double the directory.
    repo = Path(repo).resolve()
    pack = Path(pack).resolve()
    out = Path(out).resolve()
    argv = [python, str(repo / "run.py"), "--pack", str(pack), "--challenge", str(challenge), "--period", period, "--out", str(out), *extra]
    if frozen:
        argv.append("--frozen")
    return subprocess.run(argv, cwd=str(repo), capture_output=True, text=True, timeout=timeout, check=False)


def load_predictions(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def read_log_for(predictions: Path) -> Path:
    return Path(str(predictions).removesuffix(".parquet") + ".readlog.jsonl")


def period_from(name: str, start=None, end=None, train_end: datetime | None = None) -> Period:
    """The named period, narrowed to [start, end] for smoke tests; see ``loop.period_for``."""
    return period_for(name, start, end, train_end)


def load_read_log(path: Path) -> list[dict]:
    return ReadLog.read(path) if Path(path).exists() else []


def python_for(argv_python: str | None) -> str:
    return argv_python or sys.executable


def date_arg(text: str) -> date:
    return date.fromisoformat(text)


__all__ = [
    "COVERAGE_FLOOR",
    "DIRECT_READ_PATTERNS",
    "END_OF_DAY_IST",
    "MINIMUM_SUCCESS_RATE",
    "PACK_ASOF_IST",
    "PERIODS",
    "PREDICTION_STEMS",
    "check_read_log",
    "check_repository",
    "clone_at",
    "compare_predictions",
    "compare_read_logs",
    "date_arg",
    "day_label",
    "evaluate",
    "last_instant",
    "load_predictions",
    "load_read_log",
    "make_env",
    "overlap_unchanged",
    "period_from",
    "python_for",
    "read_log_for",
    "render_report",
    "run_submission",
    "scan_direct_reads",
    "sha256_of",
    "stem_for",
    "swings_path",
    "window_bounds",
]
