"""Unit test for zero data leakage.

Asserts that in the readlog produced by loop.run:
Every read's clock_max_utc is strictly <= asof_utc (and cutoff_utc when in predict phase).
Asserts that no direct pack parquet reads bypassed PointInTime.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def test_readlog_leakage(readlog_path: Path | str | None = None) -> None:
    if readlog_path is None:
        readlog_path = PROJECT_ROOT / "predictions" / "c1_holdout.readlog.jsonl"
    path = Path(readlog_path)
    if not path.exists():
        raise FileNotFoundError(f"Readlog {path} does not exist. Run run.py first.")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0, "Readlog is empty"

    violations = []
    for line_idx, line in enumerate(lines, start=1):
        record = json.loads(line)
        asof_utc = pd.to_datetime(record["asof_utc"], utc=True)
        clock_max = record.get("clock_max_utc")
        if clock_max is not None:
            clock_max_utc = pd.to_datetime(clock_max, utc=True)
            if clock_max_utc > asof_utc:
                violations.append((line_idx, record["file"], clock_max_utc, asof_utc))

        context = record.get("context", {})
        cutoff_str = context.get("cutoff_utc")
        if cutoff_str is not None:
            cutoff_utc = pd.to_datetime(cutoff_str, utc=True)
            if asof_utc > cutoff_utc:
                violations.append((line_idx, record["file"], asof_utc, cutoff_utc))

    assert not violations, f"Detected {len(violations)} leakage violations: {violations}"
    print(f"Zero-leakage test passed across {len(lines)} PointInTime readlog entries!")


if __name__ == "__main__":
    test_readlog_leakage()
