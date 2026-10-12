"""Signal-file formats and their validation (CHALLENGE-SPEC 2.6 sections 3, 4, 5 and 7).

A submission is a set of decision records over the scored window. A signal row says which way the
price is about to go and how far; a ``no_signal`` row says the model looked and saw nothing. The
two kinds may live in one file, where a ``no_signal`` row leaves ``signal_utc`` null and fills
``decision_utc`` and ``reason``, or in two.

- Task 1: ``asset, signal_utc, direction, horizon, target_pct, invalidation_pct,
  confidence, evidence``
- Task 2: ``asset, signal_utc, direction, horizon, expected_pct, p10_pct, p90_pct,
  invalidation_pct, evidence``
- ``no_signal``: ``asset, decision_utc, reason``

**Which rows are valid is not decided here.** The columns, the thresholds, the band cap and every
validity rule belong to ``swing_dataset.scoring2``; ``validate`` hands the frame to
``scoring2.signals.normalize`` and turns what comes back into the messages below, so a file this
module calls valid is one the scorer calls valid by construction rather than by agreement. What
this module adds is the part a participant wants before submitting: readable messages, the counts
per asset and horizon, and the section 5 coverage floor measured with the scorer's own
``scoring2.score.coverage``.

Files may be parquet (the interchange format, and the only one the scorer reads), CSV or JSON
lines. A CSV or JSON-lines file is read for its rows, so ``validate_file`` can name what is wrong
with them, but it is reported as ``not_parquet`` all the same: ``scoring2.signals.load`` cannot open
it, so it could never be scored. The canonical in-memory form is a pandas frame with ``signal_utc``
and ``decision_utc`` as UTC timestamps.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..scoring2 import (
    BAND_WIDTH_CAP,
    C1_SIGNAL_COLUMNS,
    C2_SIGNAL_COLUMNS,
    COVERAGE_FLOOR,
    DIRECTIONS,
    HORIZONS,
    NO_SIGNAL_COLUMNS,
    THRESHOLD_PCT,
)
from ..scoring2 import score as S
from ..scoring2 import signals as SIG
from ..scoring2.prices import PriceSeries
from . import ASSETS

TIMESTAMP_COLUMNS = ("signal_utc", "decision_utc")

#: One friendly line per invalid-row reason ``scoring2.signals`` can return. The keys are the
#: scorer's, not ours: the organizers' test suite, which the kit does not ship, reads the reasons
#: out of ``scoring2.signals`` and fails when this table does not cover them, so a validity rule
#: added to the scorer cannot slip past the kit unnoticed.
REASON_MESSAGES: dict[str, str] = {
    "unknown_asset": f"asset is not one of {list(ASSETS)}",
    "bad_direction": f"direction is not one of {list(DIRECTIONS)}",
    "bad_horizon": (
        f"horizon is not a horizon of the task: Task 1 takes {HORIZONS[1][0]} or {HORIZONS[1][1]} "
        f"completed MCX sessions, Task 2 any whole number from {HORIZONS[2][0]} to {HORIZONS[2][-1]}"
    ),
    "missing_evidence": (
        "evidence is empty; a signal without a stated reason is not valid (section 3). Write a "
        "short line or a list of pack row identifiers"
    ),
    "bad_target_pct": "target_pct is null, not a number, or not positive",
    "bad_expected_pct": "expected_pct is null, not a number, or not positive",
    "target_below_threshold": (
        "target_pct is below the asset's swing threshold "
        f"({', '.join(f'{a} {v:g}' for a, v in THRESHOLD_PCT.items())} percent)"
    ),
    "bad_invalidation_pct": "invalidation_pct is null, not a number, or not positive",
    "invalidation_above_threshold": (
        "invalidation_pct is above the asset's swing threshold; Task 1 caps it there"
    ),
    "bad_band": "p10_pct or p90_pct is null or not a number",
    "band_inverted": "p10_pct is above p90_pct",
    "band_too_wide": (
        f"p90_pct - p10_pct is more than BAND_WIDTH_CAP = {BAND_WIDTH_CAP} asset thresholds wide; "
        "a band that contains every move scores depth without saying anything"
    ),
}


class FormatError(ValueError):
    """The file cannot even be read into the canonical frame."""


@dataclass(slots=True)
class Problem:
    severity: str  # error | warning
    code: str
    message: str
    count: int = 1


@dataclass
class Report:
    path: str | None
    challenge: int | None
    rows: int
    signal_rows: int
    no_signal_rows: int
    valid_signal_rows: int = 0
    invalid_signal_rows: int = 0
    valid_no_signal_rows: int = 0
    by_horizon: dict[int, int] = field(default_factory=dict)
    by_asset: dict[str, int] = field(default_factory=dict)
    coverage: dict | None = None
    problems: list[Problem] = field(default_factory=list)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ok"] = self.ok
        return out

    def render(self) -> str:
        lines = [f"signal file: {self.path or '<frame>'}"]
        lines.append(f"challenge: {self.challenge if self.challenge else 'unknown'}")
        lines.append(
            f"rows: {self.rows}  signals: {self.signal_rows} "
            f"({self.valid_signal_rows} valid, {self.invalid_signal_rows} invalid)  "
            f"no_signal: {self.no_signal_rows} ({self.valid_no_signal_rows} valid)"
        )
        if self.by_horizon:
            lines.append("horizons: " + " ".join(f"h{h}={n}" for h, n in sorted(self.by_horizon.items())))
        if self.by_asset:
            lines.append("assets: " + " ".join(f"{a}={n}" for a, n in sorted(self.by_asset.items())))
        if self.coverage:
            per = self.coverage["per_asset"]
            shares = " ".join(f"{a}={(v['share'] if v['share'] is not None else float('nan')):.3f}" for a, v in sorted(per.items()))
            met = "met" if self.coverage["met"] else "NOT met"
            lines.append(f"coverage (floor {COVERAGE_FLOOR:.2f}, per asset): {shares} -> {met}")
        for p in self.problems:
            suffix = f" (x{p.count})" if p.count > 1 else ""
            lines.append(f"{p.severity}: {p.code}: {p.message}{suffix}")
        lines.append(f"errors: {len(self.errors)}  warnings: {len(self.warnings)}")
        lines.append("result: " + ("VALID" if self.ok else "INVALID"))
        return "\n".join(lines)


# -- reading -----------------------------------------------------------------------------


def _parse_aware(value: object) -> datetime | None:
    """A timezone-aware datetime from a string or datetime, None when naive or unparseable."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return None if value.tzinfo is None else value.to_pydatetime()
    if isinstance(value, datetime):
        return None if value.tzinfo is None else value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return None if parsed.tzinfo is None else parsed
    return None


def _is_null(value: object) -> bool:
    """A value JSON lines leave out: NaN, NaT, None or pandas' NA."""
    return value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


def read_predictions(path: Path | str) -> pd.DataFrame:
    """Read a signal file into the canonical frame. Raises FormatError when unreadable."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            frame = pq.read_table(path).to_pandas()
        except (pa.ArrowException, OSError) as exc:
            raise FormatError(f"{path}: not a readable Parquet file ({exc})") from exc
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in (".jsonl", ".ndjson", ".json"):
        records = []
        with path.open() as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FormatError(f"{path}:{number}: not JSON ({exc.msg})") from exc
                if not isinstance(record, dict):
                    raise FormatError(f"{path}:{number}: each line must be a JSON object")
                records.append(record)
        frame = pd.DataFrame.from_records(records)
    else:
        raise FormatError(f"{path}: unsupported suffix {suffix!r}; use .parquet, .csv or .jsonl")
    return canonicalize(frame)


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the instant columns as UTC timestamps and ``horizon`` nullable integer.

    Naive or unparseable instants become NaT and are reported by ``validate``; nothing is assumed
    to be UTC. The scorer rejects a naive timestamp outright, so the kit never invents an offset.
    """
    out = frame.copy()
    for column in TIMESTAMP_COLUMNS:
        if column not in out.columns:
            continue
        col = out[column]
        if isinstance(col.dtype, pd.DatetimeTZDtype):
            out[column] = col.dt.tz_convert("UTC")
            continue
        if pd.api.types.is_datetime64_any_dtype(col.dtype):
            out[column] = pd.Series([pd.NaT] * len(out), index=out.index, dtype="datetime64[us, UTC]")
            out.attrs[f"naive_{column}"] = int(col.notna().sum())
            continue
        parsed = [_parse_aware(v) for v in col.tolist()]
        raw = col.tolist()
        naive = sum(
            1
            for v, r in zip(parsed, raw, strict=True)
            if v is None and r is not None and not (isinstance(r, float) and math.isnan(r)) and str(r).strip() != ""
        )
        out.attrs[f"naive_{column}"] = naive
        out[column] = pd.to_datetime(pd.Series(parsed, index=out.index), utc=True)
    if "horizon" in out.columns:
        out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype("Int64")
    return out


def write_predictions(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write the canonical frame as parquet, CSV or JSON lines by suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pa.Table.from_pandas(frame, preserve_index=False)
        for column in TIMESTAMP_COLUMNS:
            if column in table.column_names:
                idx = table.schema.get_field_index(column)
                table = table.set_column(idx, column, pc.cast(table[column], pa.timestamp("us", tz="UTC")))
        pq.write_table(table, path, compression="zstd")
    elif suffix == ".csv":
        out = frame.copy()
        for column in TIMESTAMP_COLUMNS:
            if column in out.columns:
                out[column] = out[column].map(lambda t: t.isoformat() if pd.notna(t) else "")
        out.to_csv(path, index=False)
    elif suffix in (".jsonl", ".ndjson"):
        with path.open("w") as handle:
            for rec in frame.to_dict(orient="records"):
                rec = {k: v for k, v in rec.items() if not _is_null(v)}
                for column in TIMESTAMP_COLUMNS:
                    if isinstance(rec.get(column), pd.Timestamp):
                        rec[column] = rec[column].isoformat()
                if rec.get("horizon") is not None:
                    rec["horizon"] = int(rec["horizon"])
                handle.write(json.dumps(rec, default=str) + "\n")
    else:
        raise FormatError(f"{path}: unsupported suffix {suffix!r}; use .parquet, .csv or .jsonl")
    return path


# -- pack lookups -------------------------------------------------------------------------


DAILY_COLUMNS = (
    "session_date",
    "asset",
    "contract_id",
    "high",
    "low",
    "close",
    "settlement",
    "volume_lots",
    "available_utc",
)
SESSION_COLUMNS = ("session_date", "state", "morning_open_ist", "evening_close_ist")


def load_sessions(pack_root: Path | str) -> PriceSeries:
    """The session calendar the coverage floor is counted on, as the scorer builds it.

    Section 5 counts a decision record against the asset's own sessions, and which sessions those
    are is a fact of ``mcx/mcx_daily`` and ``mcx/mcx_sessions`` that ``scoring2.prices.PriceSeries``
    already resolves. The validator reads the three files directly: which sessions exist and when
    they open is a fixed fact of the historical pack, not a point-in-time question. Only the
    columns the files carry are read, so a pack missing an optional one still validates.
    """
    root = Path(pack_root)

    def frame(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
        present = set(pq.read_schema(path).names)
        return pq.read_table(path, columns=[c for c in columns if c in present]).to_pandas()

    return PriceSeries(
        frame(root / "mcx" / "mcx_daily.parquet", DAILY_COLUMNS),
        frame(root / "labels" / "designations.parquet", ("asset", "cutoff_date", "horizon", "contract_id")),
        frame(root / "mcx" / "mcx_sessions.parquet", SESSION_COLUMNS),
    )


# -- validation --------------------------------------------------------------------------


def detect_challenge(frame: pd.DataFrame) -> int | None:
    """The challenge implied by the columns; None when both or neither are present.

    Task 1 rows carry ``target_pct`` and Task 2 rows ``expected_pct``; that one column
    is the whole difference in the key of the two formats.
    """
    cols = set(frame.columns)
    has_target = "target_pct" in cols
    has_expected = "expected_pct" in cols
    if has_target and not has_expected:
        return 1
    if has_expected and not has_target:
        return 2
    return None


def _split_kinds(frame: pd.DataFrame) -> tuple[int, int]:
    """(signal rows, no_signal rows) by the rule FORMATS2.md states: a ``no_signal`` row leaves
    ``signal_utc`` null and fills ``decision_utc``."""
    if "signal_utc" not in frame.columns:
        return 0, len(frame) if "decision_utc" in frame.columns else 0
    is_signal = frame["signal_utc"].notna()
    return int(is_signal.sum()), int((~is_signal).sum())


def validate(
    frame: pd.DataFrame,
    challenge: int | None = None,
    *,
    no_signal: pd.DataFrame | None = None,
    sessions: PriceSeries | None = None,
    period: str = "holdout",
    window: tuple[datetime, datetime] | None = None,
    path: str | None = None,
) -> Report:
    """Check a canonical frame against the 2.6 signal format and return a Report.

    The validity of each row is decided by ``scoring2.signals.normalize`` and reported through
    ``REASON_MESSAGES``. ``sessions`` (a ``PriceSeries`` from ``load_sessions``) adds the section 5
    coverage floor and flags instants that fall outside every published session; ``period`` and
    ``window`` set the window those are measured over.
    """
    problems: list[Problem] = []
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate takes a pandas frame")
    frame = canonicalize(frame)
    signal_rows, no_signal_rows = _split_kinds(frame)
    # canonicalize turns a signal_utc without an offset into NaT, which _split_kinds takes for a
    # no_signal row. It is a signal row, and one the scorer rejects, so it is counted as both.
    naive_signals = int(frame.attrs.get("naive_signal_utc", 0))
    signal_rows += naive_signals
    no_signal_rows -= naive_signals
    if no_signal is not None:
        no_signal = canonicalize(no_signal)
        no_signal_rows += len(no_signal)
    report = Report(
        path=path,
        challenge=challenge,
        rows=len(frame) + (len(no_signal) if no_signal is not None else 0),
        signal_rows=signal_rows,
        no_signal_rows=no_signal_rows,
        problems=problems,
    )
    if report.rows == 0:
        problems.append(Problem("error", "empty", "the file has no rows"))
        return report

    detected = detect_challenge(frame)
    if challenge is None:
        challenge = detected
        report.challenge = challenge
    if challenge is None:
        problems.append(
            Problem(
                "error",
                "ambiguous_challenge",
                "columns match neither Task 1 (target_pct) nor Task 2 (expected_pct), "
                "or carry both",
            )
        )
        return report
    if detected is not None and detected != challenge:
        problems.append(
            Problem(
                "error",
                "wrong_challenge",
                f"columns look like Challenge {detected} but the file is declared for Challenge {challenge}",
            )
        )
        return report

    required = C1_SIGNAL_COLUMNS if challenge == 1 else C2_SIGNAL_COLUMNS
    missing = [c for c in required if c not in frame.columns]
    if missing:
        problems.append(
            Problem("error", "missing_columns", f"Challenge {challenge} needs {list(required)}; missing {missing}")
        )
        return report
    for column in TIMESTAMP_COLUMNS:
        naive = int(frame.attrs.get(f"naive_{column}", 0)) + (
            int(no_signal.attrs.get(f"naive_{column}", 0)) if no_signal is not None else 0
        )
        if naive:
            problems.append(
                Problem(
                    "error",
                    "naive_timestamp",
                    f"{column} values without a timezone offset (or unparseable); the scorer "
                    "rejects them. Write ISO 8601 with an offset, for example "
                    "2026-07-01T03:00:00+00:00",
                    naive,
                )
            )
    if problems and any(p.code == "naive_timestamp" for p in problems):
        report.invalid_signal_rows = naive_signals  # the rows are counted by kind; nothing else is judged yet
        return report

    try:
        decisions = SIG.normalize(frame, challenge, no_signal)
    except ValueError as exc:
        problems.append(Problem("error", "rejected_by_the_scorer", str(exc)))
        return report

    counts = decisions.counts
    report.valid_signal_rows = counts["valid_signal_rows"]
    report.invalid_signal_rows = counts["invalid_signal_rows"]
    report.no_signal_rows = counts["no_signal_rows"]
    report.valid_no_signal_rows = counts["valid_no_signal_rows"]
    signals = decisions.signals
    if len(signals):
        report.by_horizon = {
            int(h): int(n) for h, n in pd.to_numeric(signals["horizon"], errors="coerce").dropna().value_counts().items()
        }
        report.by_asset = {str(a): int(n) for a, n in signals["asset"].value_counts().items()}
    for reason, count in counts["invalid_reasons"].items():
        problems.append(
            Problem("error", reason, REASON_MESSAGES.get(reason, f"{reason}: see FORMATS2.md"), count)
        )
    empty_reasons = counts["no_signal_rows"] - counts["valid_no_signal_rows"]
    if empty_reasons:
        problems.append(
            Problem(
                "error",
                "no_signal_without_a_reason",
                "a no_signal row needs an asset of the four and a one-line reason; without one it "
                "is not a decision record and does not count for the coverage floor",
                empty_reasons,
            )
        )
    if report.signal_rows and not report.valid_signal_rows and not report.valid_no_signal_rows:
        problems.append(
            Problem("error", "nothing_valid", "no row in the file is a decision record the rules recognise")
        )

    if sessions is not None:
        _coverage_problems(report, decisions, sessions, period, window, problems)
    return report


def _coverage_problems(
    report: Report,
    decisions: SIG.Decisions,
    prices: PriceSeries,
    period: str,
    window: tuple[datetime, datetime] | None,
    problems: list[Problem],
) -> None:
    """Section 5, measured with ``scoring2.score.coverage`` over the window the scorer resolves."""
    start = pd.Timestamp(window[0]) if window else None
    end = pd.Timestamp(window[1]) if window else None
    try:
        resolved = S.resolve_window(period, prices, start, end)
    except ValueError as exc:
        problems.append(Problem("warning", "window", str(exc)))
        return
    block = S.coverage(prices, decisions, resolved)
    report.coverage = block
    outside = int(block["decision_records_outside_a_session"])
    if outside:
        problems.append(
            Problem(
                "warning",
                "outside_a_session",
                "decision records whose instant is in no published MCX session and is not a "
                "pre-open; section 3 issues signals inside a session or at the 08:30 IST pre-open, "
                "and a record outside one covers nothing",
                outside,
            )
        )
    short = {a: v for a, v in block["per_asset"].items() if not v["met"]}
    if short:
        detail = ", ".join(
            f"{a} {v['sessions_with_a_decision']}/{v['sessions']}" for a, v in sorted(short.items())
        )
        problems.append(
            Problem(
                "error",
                "coverage_floor",
                f"section 5 wants a decision record on at least {COVERAGE_FLOOR:.0%} of each "
                f"asset's sessions in the {resolved.period} window; short on {detail}",
                len(short),
            )
        )


def _scorer_cannot_open(path: Path) -> str | None:
    """Why ``scoring2.signals.load`` cannot open ``path``, or None when it can.

    The scorer opens every signal file with ``pq.read_table`` whatever it is called, so this makes
    the same call. A file with the ``.parquet`` suffix has been read that way already, by
    ``read_predictions``.
    """
    if path.suffix.lower() == ".parquet":
        return None
    try:
        pq.read_table(path)
    except (pa.ArrowException, OSError) as exc:
        return str(exc)
    return None


def validate_file(
    path: Path | str,
    challenge: int | None = None,
    *,
    no_signal_path: Path | str | None = None,
    pack_root: Path | str | None = None,
    period: str = "holdout",
    start: datetime | None = None,
    end: datetime | None = None,
) -> Report:
    """Read and validate a file; with ``pack_root`` the coverage floor is measured too."""
    path = Path(path)
    try:
        frame = read_predictions(path)
        extra = read_predictions(no_signal_path) if no_signal_path is not None else None
    except FormatError as exc:
        return Report(
            path=str(path),
            challenge=challenge,
            rows=0,
            signal_rows=0,
            no_signal_rows=0,
            problems=[Problem("error", "unreadable", str(exc))],
        )
    sessions = load_sessions(pack_root) if pack_root is not None else None
    window = (start, end) if (start is not None and end is not None) else None
    report = validate(
        frame,
        challenge,
        no_signal=extra,
        sessions=sessions,
        period=period,
        window=window,
        path=str(path),
    )
    # A CSV or JSON-lines file has passed the row checks above and still cannot be scored.
    unopened = [Path(p) for p in (path, no_signal_path) if p is not None and _scorer_cannot_open(Path(p))]
    for other in reversed(unopened):
        report.problems.insert(
            0,
            Problem(
                "error",
                "not_parquet",
                f"{other.name} is not Parquet, and the scorer reads Parquet only, so it could not be "
                "scored as it is; its rows are checked below all the same. Write it with pandas "
                "`to_parquet` or `formats.write_predictions(frame, 'signals.parquet')` and validate that file",
            ),
        )
    return report


def split_decisions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frame's signal rows and its ``no_signal`` rows, in the columns each kind carries."""
    frame = canonicalize(frame)
    if "signal_utc" not in frame.columns:
        signals = frame.iloc[:0]
        rest = frame
    else:
        is_signal = frame["signal_utc"].notna()
        signals, rest = frame[is_signal], frame[~is_signal]
    challenge = detect_challenge(frame)
    columns = C1_SIGNAL_COLUMNS if challenge == 1 else C2_SIGNAL_COLUMNS
    signals = signals[[c for c in columns if c in signals.columns]].reset_index(drop=True)
    rest = rest[[c for c in NO_SIGNAL_COLUMNS if c in rest.columns]].reset_index(drop=True)
    return signals, rest


def summarize(frame: pd.DataFrame) -> dict:
    """Row counts a run manifest records: signals and ``no_signal`` rows, per asset and horizon."""
    signals, no_signals = split_decisions(frame)
    horizon = pd.to_numeric(signals["horizon"], errors="coerce").dropna() if len(signals) else pd.Series(dtype=float)
    return {
        "rows": len(frame),
        "signals": len(signals),
        "no_signals": len(no_signals),
        "by_horizon": {int(h): int(n) for h, n in horizon.value_counts().items()},
        "by_asset": {str(a): int(n) for a, n in signals["asset"].value_counts().items()} if len(signals) else {},
        "by_direction": {str(d): int(n) for d, n in signals["direction"].value_counts().items()} if len(signals) else {},
        "no_signals_by_asset": {str(a): int(n) for a, n in no_signals["asset"].value_counts().items()}
        if len(no_signals)
        else {},
    }
