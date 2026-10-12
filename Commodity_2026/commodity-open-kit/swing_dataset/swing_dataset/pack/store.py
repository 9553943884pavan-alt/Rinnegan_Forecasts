"""Batched parquet writing with a fixed schema, plus meta sidecar helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .. import BUILD_ROOT
from ..common import git_identity, sha256_bytes, write_meta
from .provenance import build_label, capture_label

PACK_BUILD = Path(BUILD_ROOT) / "build" / "pack"
RAW_FEED = Path(BUILD_ROOT) / "raw" / "feed"
PULL_LOG = Path(BUILD_ROOT) / "logs" / "pull-raw.log"

TS = pa.timestamp("us", tz="UTC")
DATE = pa.date32()
F64 = pa.float64()
I64 = pa.int64()
STR = pa.string()
BOOL = pa.bool_()


@dataclass(slots=True)
class Column:
    name: str
    type: pa.DataType
    meaning: str
    dictionary: bool = False
    clock: str | None = None  # "exact" | "published" | None


def schema_of(columns: list[Column]) -> pa.Schema:
    return pa.schema([pa.field(c.name, c.type, nullable=True) for c in columns])


def column_docs(columns: list[Column]) -> list[dict[str, str]]:
    return [{"name": c.name, "type": str(c.type), "meaning": c.meaning} for c in columns]


@dataclass
class TableWriter:
    """Accumulates rows column-wise and flushes row groups to a parquet file."""

    path: Path
    columns: list[Column]
    batch_rows: int = 250_000
    rows: int = 0
    _buffers: dict[str, list] = field(default_factory=dict)
    _writer: pq.ParquetWriter | None = None
    _tmp: Path | None = None
    _min_ts: dict[str, datetime] = field(default_factory=dict)
    _max_ts: dict[str, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._buffers = {c.name: [] for c in self.columns}
        dictionary = [c.name for c in self.columns if c.dictionary]
        self._writer = pq.ParquetWriter(
            self._tmp,
            schema_of(self.columns),
            compression="zstd",
            compression_level=9,
            use_dictionary=dictionary or False,
            write_statistics=True,
        )

    def append(self, row: dict[str, object]) -> None:
        for c in self.columns:
            value = row.get(c.name)
            self._buffers[c.name].append(value)
            if value is not None and c.type == TS:
                lo = self._min_ts.get(c.name)
                hi = self._max_ts.get(c.name)
                if lo is None or value < lo:
                    self._min_ts[c.name] = value
                if hi is None or value > hi:
                    self._max_ts[c.name] = value
        self.rows += 1
        if self.rows % self.batch_rows == 0:
            self.flush()

    def flush(self) -> None:
        if not self._buffers or not self._buffers[self.columns[0].name]:
            return
        arrays = []
        for c in self.columns:
            arrays.append(pa.array(self._buffers[c.name], type=c.type))
            self._buffers[c.name] = []
        assert self._writer is not None
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=schema_of(self.columns)))

    def close(self) -> Path:
        self.flush()
        assert self._writer is not None and self._tmp is not None
        self._writer.close()
        os.replace(self._tmp, self.path)
        return self.path

    def date_range(self, column: str) -> dict[str, str | None]:
        lo = self._min_ts.get(column)
        hi = self._max_ts.get(column)
        return {
            "column": column,
            "start": lo.isoformat() if lo else None,
            "end": hi.isoformat() if hi else None,
        }


def build_identity(command: str | None = None) -> dict[str, str]:
    """Commit and command, without the branch: a lane branch name resolves nowhere for a reader."""
    ident = {k: v for k, v in git_identity(Path(__file__).resolve().parent).items() if k != "branch"}
    ident["command"] = command or " ".join(["python", "-m", "swing_dataset.pack", *sys.argv[1:]])
    ident["built_utc"] = datetime.now(UTC).isoformat()
    return ident


def file_source(path: Path, url: str, fetched_at_utc: str | None) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "url": url,
        "fetched_at_utc": fetched_at_utc,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pull_times() -> dict[str, str]:
    """``ok <source> <time>`` lines from the pull log, keyed by source (``pg/<table>`` for dumps)."""
    out: dict[str, str] = {}
    if not PULL_LOG.exists():
        return out
    for line in PULL_LOG.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "ok":
            out[parts[1]] = parts[-1]
    return out


def raw_source_entries(source: str, files: list[Path], raw_root: Path = RAW_FEED) -> list[dict[str, object]]:
    """One entry per captured file, labelled ``feed-capture:<source>/<asset>/<day>.csv``.

    The label names the collector and the file it filed the rows under, which is what a reader
    needs to follow the provenance. Where that file sat, on whose machine, is not part of it.
    """
    fetched = pull_times().get(source) if raw_root == RAW_FEED else None
    entries = []
    for file in files:
        rel = file.relative_to(raw_root / source)
        entries.append(
            {
                "url": capture_label(source, rel.as_posix()),
                "fetched_at_utc": fetched,
                "sha256": file_sha256(file),
                "bytes": file.stat().st_size,
            }
        )
    return entries


def capture_ref(path: Path, raw_root: Path = RAW_FEED) -> str:
    """The capture label for a raw file, taken from its position under ``raw_root``."""
    try:
        rel = path.relative_to(raw_root).parts
    except ValueError:
        return build_label(path)
    return capture_label(rel[0], "/".join(rel[1:]))


def clock_meta(columns: list[Column], row_counts: dict[str, int] | None, rule: str) -> dict:
    clocks = {c.clock: c.name for c in columns if c.clock}
    qualities = sorted(k for k, v in (row_counts or {}).items() if v)
    if row_counts is None:
        level = "exact" if clocks.get("exact") and not clocks.get("published") else "mixed"
    elif len(qualities) == 1:
        level = qualities[0]
    else:
        level = "mixed"
    return {
        "file": level,
        "row_column": "clock_quality" if row_counts is not None else None,
        "rows": row_counts,
        "clocks": clocks,
        "rule": rule,
    }


def write_sidecar(
    parquet_path: Path,
    *,
    name: str,
    rows: int,
    columns: list[Column],
    date_range: dict,
    sources: list[dict],
    clock_quality: dict,
    notes: list[str],
    extra: dict | None = None,
) -> Path:
    meta = {
        "name": name,
        "rows": rows,
        "columns": column_docs(columns),
        "date_range": date_range,
        "sources": sources,
        "clock_quality": clock_quality,
        "build": build_identity(),
        "notes": notes,
    }
    if extra:
        meta.update(extra)
    return write_meta(parquet_path, meta)


def load_meta(parquet_path: Path) -> dict:
    return json.loads(parquet_path.with_suffix(".meta.json").read_text())


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def as_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
