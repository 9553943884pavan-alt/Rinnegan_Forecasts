"""Point-in-time client over an assembled pack (or a lane build directory).

``PointInTime(root).available(asof_utc)`` returns one lazy frame per pack file. Collecting a
frame yields only rows whose availability clock is strictly before ``asof_utc``:

- ``exact`` rows: the exact clock column the sidecar names is before ``asof_utc`` (``fetched_utc``
  in the quote and news files, ``captured_utc`` in ``calendar/release_vintages``)
- ``published`` rows: the published clock column the sidecar names is before ``asof_utc``
  (``available_utc``, and ``release_utc`` in ``calendar/release_schedule``)
- ``unvintaged`` rows: excluded unless ``include_unvintaged=True``; then rows carrying a
  ``raw_file_date`` are admitted once the UTC day the feed wrote them has ended
  (``raw_file_date + 1 day <= asof_utc``), and rows without one are admitted as-is.
- files whose sidecar resolves no clock column at all (for example a calendar table) are
  treated as unvintaged as a whole.

The clock columns and the per-row quality column come from each file's ``.meta.json``
(``clock_quality.clocks`` and ``clock_quality.row_column``); files with a uniform quality
use that quality for every row.

``PointInTime(root, read_log=ReadLog(path))`` writes one JSON line per collected frame: the
file, the as-of instant, the columns, the rows returned, the latest availability clock among
them, and whatever context the caller set (``read_log.context = {"phase": "predict",
"cutoff_utc": ...}``). A submission ships that log next to its prediction files; the
organizers check every line against the cutoff it served.

The latest availability clock measures each returned row by the clock that admitted it: an
``exact`` row by its fetch clock, a ``published`` row by its availability instant, an
``unvintaged`` row (returned only on opt-in) by its own ``available_utc`` when it has one. The
timestamps a row carries beyond that say when its bytes were captured, which for a vintage
published years ago and captured last week is no statement about when the row was available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .clock import normalize_clock_quality

QUALITIES = ("exact", "published", "unvintaged")
READ_LOG_VERSION = 1
#: The timestamp an unvintaged row can carry that says when it became available. The filter never
#: reads it, because an unvintaged row is admitted on opt-in; a read logs it so the log says
#: what the opt-in returned.
UNVINTAGED_CLOCK = "available_utc"


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("asof_utc must be timezone-aware")
    return ts.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class PackFile:
    name: str
    path: Path
    meta_path: Path
    file_quality: str  # exact | published | unvintaged | mixed
    row_column: str | None
    clocks: dict[str, str]

    @classmethod
    def from_meta(cls, path: Path, meta: dict, name: str) -> PackFile:
        if "clock_quality" not in meta:
            raise ValueError(f"{path}: meta lacks clock_quality")
        cq = normalize_clock_quality(meta, pq.read_schema(path).names)
        return cls(
            name=name,
            path=path,
            meta_path=path.with_suffix(".meta.json"),
            file_quality=str(cq["file"]),
            row_column=cq.get("row_column"),
            clocks=dict(cq.get("clocks") or {}),
        )

    @property
    def unclocked(self) -> bool:
        """True when no availability clock column can be resolved: treated as unvintaged."""
        return not self.clocks

    def clock_for(self, quality: str) -> str | None:
        return self.clocks.get(quality)


class ReadLog:
    """JSON-lines record of every point-in-time read made through one client.

    Every ``LazyFrame.table()`` or ``collect()`` appends one line with ``seq``, ``file``,
    ``asof_utc``, ``include_unvintaged``, ``columns`` (null for all), ``rows`` (returned),
    ``rows_in_file``, ``clock_max_utc`` (the latest clock among the returned rows, each measured
    by the clock that admitted it, so before ``asof_utc`` for every row the filter admits by its
    class clock), ``logged_utc`` and ``context``. Set ``context`` to a JSON-serializable dict
    before a group of reads; the run template records the phase and the cutoff the reads serve.
    """

    def __init__(self, path: Path | str, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context: dict = {}
        self.entries = 0
        if not append:
            self.path.write_text("")

    def record(self, entry: dict) -> None:
        self.entries += 1
        line = {"version": READ_LOG_VERSION, "seq": self.entries, **entry, "logged_utc": datetime.now(UTC).isoformat(), "context": dict(self.context)}
        with self.path.open("a") as handle:
            handle.write(json.dumps(line, default=str) + "\n")

    @staticmethod
    def read(path: Path | str) -> list[dict]:
        entries = []
        with Path(path).open() as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{number}: not JSON ({exc.msg})") from exc
        return entries


def _qualities(table: pa.Table, file: PackFile) -> pa.Array | pa.ChunkedArray:
    """Each row's clock class: the sidecar's per-row column when the table has it, else the file's."""
    if file.row_column and file.row_column in table.column_names:
        return table[file.row_column]
    return pa.array([file.file_quality] * table.num_rows, type=pa.string())


def _class_clocks(file: PackFile, table: pa.Table) -> list[tuple[str, str]]:
    """The (class, column) pairs the filter admits rows by, for the clock columns the table holds."""
    pairs = []
    for quality in ("exact", "published"):
        clock = file.clock_for(quality)
        if clock is not None and clock in table.column_names:
            pairs.append((quality, clock))
    return pairs


def _clock_max(table: pa.Table, file: PackFile) -> datetime | None:
    """The latest clock among the rows a read returned, each measured by the clock that admitted it.

    ``exact`` rows count by their fetch clock and ``published`` rows by their availability
    instant, the two columns the filter compares with the as-of. An ``unvintaged`` row, which only
    an opt-in returns, counts by its own ``UNVINTAGED_CLOCK`` when it has one. Every other timestamp
    a row carries is left out: ``captured_utc`` on a vintage that was published in 2021 and
    captured in September 2026 says nothing about when the row was available.
    """
    if not table.num_rows:
        return None
    quality = _qualities(table, file)
    pairs = _class_clocks(file, table)
    if UNVINTAGED_CLOCK in table.column_names:
        pairs.append(("unvintaged", UNVINTAGED_CLOCK))
    latest: datetime | None = None
    for q, clock in pairs:
        rows = table.select([clock]).filter(pc.fill_null(pc.equal(quality, q), False))
        value = pc.max(rows[clock]).as_py()
        if value is not None and (latest is None or value > latest):
            latest = value
    return latest


class LazyFrame:
    """A deferred point-in-time filter over one parquet file."""

    def __init__(self, file: PackFile, asof: datetime, include_unvintaged: bool, read_log: ReadLog | None = None) -> None:
        self.file = file
        self.asof = asof
        self.include_unvintaged = include_unvintaged
        self.read_log = read_log

    def __repr__(self) -> str:
        return f"LazyFrame({self.file.name}, asof={self.asof.isoformat()})"

    def _read(self, columns: list[str] | None) -> tuple[pa.Table, int]:
        """The rows the filter admits, with the columns it and the log need kept, and the file's row count."""
        needed = set(columns or [])
        schema = pq.read_schema(self.file.path)
        wanted = None
        if columns is not None:
            for extra in (self.file.row_column, *self.file.clocks.values(), "raw_file_date", UNVINTAGED_CLOCK):
                if extra and extra in schema.names:
                    needed.add(extra)
            wanted = [c for c in schema.names if c in needed]
        table = pq.read_table(self.file.path, columns=wanted)
        rows_in_file = table.num_rows
        return table.filter(self._mask(table)), rows_in_file

    def _entry(self, table: pa.Table, columns: list[str] | None, rows_in_file: int) -> dict:
        clock_max = _clock_max(table, self.file)
        return {
            "file": self.file.name,
            "asof_utc": self.asof.isoformat(),
            "include_unvintaged": self.include_unvintaged,
            "columns": list(columns) if columns is not None else None,
            "rows": table.num_rows,
            "rows_in_file": rows_in_file,
            "clock_max_utc": clock_max.isoformat() if clock_max is not None else None,
        }

    def table(self, columns: list[str] | None = None) -> pa.Table:
        table, rows_in_file = self._read(columns)
        if self.read_log is not None:
            self.read_log.record(self._entry(table, columns, rows_in_file))
        if columns is not None:
            table = table.select([c for c in columns if c in table.column_names])
        return table

    def log_entry(self, columns: list[str] | None = None) -> dict:
        """The line ``table(columns)`` writes to a read log, without writing it.

        The organizers repeat a logged read with this to check the clock the log carries.
        """
        table, rows_in_file = self._read(columns)
        return self._entry(table, columns, rows_in_file)

    def collect(self, columns: list[str] | None = None) -> pd.DataFrame:
        return self.table(columns).to_pandas()

    def _mask(self, table: pa.Table) -> pa.Array:
        n = table.num_rows
        asof = pa.scalar(self.asof, type=pa.timestamp("us", tz="UTC"))
        quality = _qualities(table, self.file)
        mask = pa.array([False] * n, type=pa.bool_())
        if self.file.unclocked:
            return mask if not self.include_unvintaged else pa.array([True] * n, type=pa.bool_())
        for q, clock in _class_clocks(self.file, table):
            before = pc.fill_null(pc.less(table[clock], asof), False)
            mask = pc.or_(mask, pc.and_(pc.equal(quality, q), before))
        if self.include_unvintaged:
            is_u = pc.fill_null(pc.equal(quality, "unvintaged"), False)
            if "raw_file_date" in table.column_names:
                # admitted once the UTC day the feed wrote the row has ended
                end_of_day = pc.add(pc.cast(table["raw_file_date"], pa.timestamp("us")), pa.scalar(timedelta(days=1)))
                asof_naive = pa.scalar(self.asof.replace(tzinfo=None), type=pa.timestamp("us"))
                ended = pc.fill_null(pc.less_equal(end_of_day, asof_naive), False)
                is_u = pc.and_(is_u, ended)
            mask = pc.or_(mask, is_u)
        return pc.fill_null(mask, False)


class PointInTime:
    def __init__(self, root: Path | str, *, read_log: ReadLog | None = None) -> None:
        self.root = Path(root)
        self.read_log = read_log
        self.files: dict[str, PackFile] = {}
        manifest = self.root / "MANIFEST.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            for entry in data.get("files", []):
                path = self.root / entry["path"]
                meta = json.loads((self.root / entry["meta_path"]).read_text())
                name = entry["path"].removesuffix(".parquet")
                self.files[name] = PackFile.from_meta(path, meta, name)
        else:
            for path in sorted(self.root.rglob("*.parquet")):
                meta_path = path.with_suffix(".meta.json")
                if not meta_path.exists():
                    continue
                name = str(path.relative_to(self.root)).removesuffix(".parquet")
                self.files[name] = PackFile.from_meta(path, json.loads(meta_path.read_text()), name)
        if not self.files:
            raise FileNotFoundError(f"no parquet files with meta sidecars under {self.root}")

    def names(self) -> list[str]:
        return sorted(self.files)

    def available(self, asof_utc: datetime, *, include_unvintaged: bool = False) -> dict[str, LazyFrame]:
        asof = _aware(asof_utc)
        return {name: LazyFrame(f, asof, include_unvintaged, self.read_log) for name, f in sorted(self.files.items())}

    def frame(self, name: str, asof_utc: datetime, *, include_unvintaged: bool = False) -> LazyFrame:
        return LazyFrame(self.files[name], _aware(asof_utc), include_unvintaged, self.read_log)


def clock_of_row(file: PackFile, row: dict) -> datetime | date | None:
    """The availability instant the PIT filter applies to one row (for tests and audits)."""
    quality = row.get(file.row_column) if file.row_column else file.file_quality
    if quality in ("exact", "published"):
        clock = file.clock_for(str(quality))
        return row.get(clock) if clock else None
    if quality == "unvintaged" and row.get("raw_file_date") is not None:
        return row["raw_file_date"]
    return None
