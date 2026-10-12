"""A sidecar's ``clock_quality`` in its dict form.

The point-in-time client and the organizers' pack builders both read sidecars through this
module, which depends on nothing else in the package.
"""

from __future__ import annotations

CLOCK_CANDIDATES = {"exact": ("fetched_utc", "captured_utc"), "published": ("available_utc", "release_utc", "published_utc")}


def normalize_clock_quality(meta: dict, column_names: list[str]) -> dict:
    """Return the dict form of a sidecar's clock_quality.

    SPEC.md allows a plain class string per file; lanes may also write the dict form
    (``file``, ``row_column``, ``rows``, ``clocks``, ``rule``). Clock columns not named by the
    sidecar are resolved from conventional column names present in the parquet.
    """
    raw = meta.get("clock_quality")
    if isinstance(raw, str):
        out: dict = {"file": raw, "row_column": None, "rows": None, "clocks": {}, "rule": None}
    elif isinstance(raw, dict):
        out = {"file": raw.get("file"), "row_column": raw.get("row_column"), "rows": raw.get("rows"), "clocks": dict(raw.get("clocks") or {}), "rule": raw.get("rule")}
    else:
        raise TypeError("meta clock_quality must be a class string or a dict with a file-level class")
    if out["file"] not in ("exact", "published", "unvintaged", "mixed"):
        raise ValueError(f"clock_quality.file {out['file']!r} is not exact|published|unvintaged|mixed")
    if out["row_column"] is None and "clock_quality" in column_names:
        out["row_column"] = "clock_quality"
    for quality, candidates in CLOCK_CANDIDATES.items():
        if quality not in out["clocks"]:
            found = next((c for c in candidates if c in column_names), None)
            if found:
                out["clocks"][quality] = found
    if out["file"] == "published" and "published" not in out["clocks"] and "exact" in out["clocks"]:
        out["clocks"]["published"] = out["clocks"]["exact"]
    return out
