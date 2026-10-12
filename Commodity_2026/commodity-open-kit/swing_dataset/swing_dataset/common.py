"""Shared helpers: output roots, sidecar metadata, raw capture index."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import BUILD_ROOT

RAW_FEED = Path(BUILD_ROOT) / "raw" / "feed"
CLOCK_QUALITIES = ("exact", "published", "unvintaged")


def sidecar_path(path: str | Path | None) -> str | None:
    """``path`` as a sidecar may record it: relative to the build root, never absolute.

    Sidecars travel with the pack, so a build path would put the operator's machine into every
    published file. Anything outside the build root is recorded by name alone.
    """
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(BUILD_ROOT).resolve()))
    except ValueError:
        return p.name


def lane_root(lane: str) -> Path:
    root = Path(BUILD_ROOT) / "build" / lane
    (root / "raw").mkdir(parents=True, exist_ok=True)
    return root


def now_utc() -> datetime:
    return datetime.now(UTC)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _complaints_but_no_checkout(stderr: str) -> str:
    """What git wrote to stderr, less its "not a git repository" notice.

    The notice is one line, or two when git also says it stopped at a filesystem boundary. Every
    other line stays, whatever came before or after the notice.
    """
    kept: list[str] = []
    after_notice = False
    for line in stderr.splitlines(keepends=True):
        if line.startswith("fatal: not a git repository"):
            after_notice = True
        elif after_notice and line.startswith("Stopping at filesystem boundary"):
            after_notice = False
        else:
            after_notice = False
            kept.append(line)
    return "".join(kept)


def git_identity(cwd: Path | None = None) -> dict[str, str]:
    """The branch and commit of the checkout at ``cwd``; ``unknown`` for each where there is none.

    A directory that is no checkout is the usual case for a team that unzipped the kit, so git's
    notice that it is not a repository is not passed on. Anything else git says (a configuration it
    cannot read, a directory it will not trust) goes to stderr as git wrote it, and the value is
    still ``unknown``.
    """

    def run(*args: str) -> str:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=cwd,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
                # git's own words whatever the user's locale: _complaints_but_no_checkout reads them
                env={**os.environ, "LC_ALL": "C"},
            )
        except FileNotFoundError:
            return "unknown"
        if done.returncode:
            sys.stderr.write(_complaints_but_no_checkout(done.stderr))
            return "unknown"
        return done.stdout.strip()

    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD")}


def record_capture(
    lane: str, path: Path, url: str, fetched_at: datetime, data: bytes, http_status: int
) -> None:
    """Append one raw capture to build/<lane>/raw/INDEX.csv (never overwrite)."""
    index = lane_root(lane) / "raw" / "INDEX.csv"
    new = not index.exists()
    with index.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "url", "fetched_at_utc", "sha256", "bytes", "http_status"])
        w.writerow(
            [str(path), url, fetched_at.isoformat(), sha256_bytes(data), len(data), http_status]
        )


def write_meta(parquet_path: Path, meta: dict) -> Path:
    """Write X.meta.json beside X.parquet. Caller supplies the SPEC.md fields."""
    required = {
        "name",
        "rows",
        "columns",
        "date_range",
        "sources",
        "clock_quality",
        "build",
        "notes",
    }
    missing = required - set(meta)
    if missing:
        raise ValueError(f"meta missing {sorted(missing)}")
    out = parquet_path.with_suffix(".meta.json")
    out.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    return out
