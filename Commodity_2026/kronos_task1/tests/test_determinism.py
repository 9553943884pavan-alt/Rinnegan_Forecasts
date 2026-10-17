"""Unit test for determinism and reproducibility.

Runs two consecutive passes of `run.py` over the same period and verifies
that the resulting Parquet files have identical SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PYTHON = sys.executable
PACK_PATH = Path("c:/Users/Pavan/Downloads/Rinnegan_Forecasts/Commodity_2026/commodity-open-pack/v2.0.1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_determinism() -> None:
    out1 = PROJECT_ROOT / "predictions" / "det_run1"
    out2 = PROJECT_ROOT / "predictions" / "det_run2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    cmd1 = [
        PYTHON,
        str(PROJECT_ROOT / "run.py"),
        "--pack",
        str(PACK_PATH),
        "--challenge",
        "1",
        "--period",
        "holdout",
        "--from",
        "2026-06-01",
        "--to",
        "2026-06-01",
        "--out",
        str(out1),
    ]

    cmd2 = [
        PYTHON,
        str(PROJECT_ROOT / "run.py"),
        "--pack",
        str(PACK_PATH),
        "--challenge",
        "1",
        "--period",
        "holdout",
        "--from",
        "2026-06-01",
        "--to",
        "2026-06-01",
        "--out",
        str(out2),
    ]

    print("Executing Run 1...")
    subprocess.check_call(cmd1, cwd=str(PROJECT_ROOT))
    print("Executing Run 2...")
    subprocess.check_call(cmd2, cwd=str(PROJECT_ROOT))

    pq1 = out1 / "c1_holdout.parquet"
    pq2 = out2 / "c1_holdout.parquet"

    hash1 = sha256_file(pq1)
    hash2 = sha256_file(pq2)

    print(f"Run 1 SHA-256: {hash1}")
    print(f"Run 2 SHA-256: {hash2}")

    assert hash1 == hash2, f"Determinism failure: {hash1} != {hash2}"
    print("Determinism test PASSED: Byte-identical reproduction confirmed!")


if __name__ == "__main__":
    test_determinism()
