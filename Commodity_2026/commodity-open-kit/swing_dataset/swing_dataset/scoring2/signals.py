"""Signal files and `no_signal` decision rows: reading them and saying which rows are valid.

A submission is a set of decision records. A signal row states a direction, a horizon, how far the
model expects the price to go and how far against it may go first; a `no_signal` row states that
the model saw nothing, with a one-line reason. Both count for the coverage floor of section 5;
only signal rows are scored.

Validation follows sections 2, 3 and 4: a Task 1 horizon is 5 or 10 sessions, a Task 2 horizon
any whole number from 1 to 20. An invalid signal row stays in the denominator and can never be
correct: dropping it would pay a model for issuing rows the rules reject. It does not count as a
decision record, because it is not a decision the rules recognise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import (
    ASSETS,
    BAND_WIDTH_CAP,
    C1_SIGNAL_COLUMNS,
    C2_SIGNAL_COLUMNS,
    DIRECTIONS,
    HORIZONS,
    THRESHOLD_PCT,
)

C1_REQUIRED = tuple(c for c in C1_SIGNAL_COLUMNS)
C2_REQUIRED = tuple(c for c in C2_SIGNAL_COLUMNS)


@dataclass(frozen=True, slots=True)
class Decisions:
    """Signal rows and `no_signal` rows of one submission, normalized and validated."""

    signals: pd.DataFrame
    no_signals: pd.DataFrame
    challenge: int

    @property
    def counts(self) -> dict:
        s = self.signals
        reasons: dict[str, int] = {}
        for row in s["invalid_reasons"]:
            for reason in row:
                reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "signal_rows": len(s),
            "valid_signal_rows": int(s["valid"].sum()) if len(s) else 0,
            "invalid_signal_rows": int((~s["valid"]).sum()) if len(s) else 0,
            "invalid_reasons": dict(sorted(reasons.items())),
            "no_signal_rows": len(self.no_signals),
            "valid_no_signal_rows": int(self.no_signals["valid"].sum())
            if len(self.no_signals)
            else 0,
        }


def _aware(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(pd.Series(values).reset_index(drop=True), errors="coerce")
    if parsed.notna().any() and parsed.dt.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    if parsed.dt.tz is None:
        return parsed
    return parsed.dt.tz_convert("UTC")


def _text(values: pd.Series) -> pd.Series:
    def one(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, np.ndarray)):
            return " ".join(str(v).strip() for v in value if str(v).strip())
        if isinstance(value, float) and not np.isfinite(value):
            return ""
        return str(value).strip()

    return values.map(one)


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def load(path, challenge: int, no_signal_path=None) -> Decisions:
    frame = pq.read_table(path).to_pandas()
    extra = pq.read_table(no_signal_path).to_pandas() if no_signal_path is not None else None
    return normalize(frame, challenge, extra)


def normalize(
    frame: pd.DataFrame, challenge: int, no_signal: pd.DataFrame | None = None
) -> Decisions:
    """Split a submission frame into signal rows and `no_signal` rows, and validate both."""
    required = C1_REQUIRED if challenge == 1 else C2_REQUIRED
    frame = frame.reset_index(drop=True).copy()
    missing = [c for c in required if c not in frame.columns]
    if missing and not (len(frame) == 0 and "signal_utc" not in frame.columns):
        raise ValueError(f"signal file lacks columns {missing}; see FORMATS2.md")
    for column in required:
        if column not in frame.columns:
            frame[column] = None
    frame["signal_utc"] = _aware(frame["signal_utc"], "signal_utc")
    parts = [frame[frame["signal_utc"].notna()].copy()]
    inline = frame[frame["signal_utc"].isna()]
    decision_frames = []
    if len(inline) and "decision_utc" in inline.columns:
        decision_frames.append(inline)
    if no_signal is not None and len(no_signal):
        decision_frames.append(no_signal.reset_index(drop=True).copy())
    signals = _validate_signals(parts[0].reset_index(drop=True), challenge)
    decisions = _validate_no_signals(decision_frames)
    return Decisions(signals=signals, no_signals=decisions, challenge=challenge)


def _validate_signals(frame: pd.DataFrame, challenge: int) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    n = len(out)
    out["asset"] = out["asset"].astype(str) if n else pd.Series(dtype=str)
    out["direction"] = out["direction"].astype(str) if n else pd.Series(dtype=str)
    out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce")
    out["evidence"] = _text(out["evidence"]) if n else pd.Series(dtype=str)
    out["invalidation_pct"] = _number(out, "invalidation_pct")
    if challenge == 1:
        out["target_pct"] = _number(out, "target_pct")
        out["confidence"] = _number(out, "confidence")
    else:
        out["expected_pct"] = _number(out, "expected_pct")
        out["p10_pct"] = _number(out, "p10_pct")
        out["p90_pct"] = _number(out, "p90_pct")
    reasons: list[list[str]] = [[] for _ in range(n)]
    threshold = out["asset"].map(THRESHOLD_PCT) if n else pd.Series(dtype=float)
    out["threshold_pct"] = threshold.astype(float) if n else pd.Series(dtype=float)

    def flag(mask, reason: str) -> None:
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons[int(i)].append(reason)

    if n:
        flag(~out["asset"].isin(ASSETS), "unknown_asset")
        flag(~out["direction"].isin(DIRECTIONS), "bad_direction")
        flag(~out["horizon"].isin(HORIZONS[challenge]), "bad_horizon")
        flag(out["evidence"].str.len() == 0, "missing_evidence")
        inval = out["invalidation_pct"]
        flag(~np.isfinite(inval) | (inval <= 0), "bad_invalidation_pct")
        if challenge == 1:
            target = out["target_pct"]
            flag(~np.isfinite(target) | (target <= 0), "bad_target_pct")
            flag(np.isfinite(target) & (target < out["threshold_pct"]), "target_below_threshold")
            flag(
                np.isfinite(inval) & (inval > out["threshold_pct"]),
                "invalidation_above_threshold",
            )
        else:
            expected = out["expected_pct"]
            p10, p90 = out["p10_pct"], out["p90_pct"]
            flag(~np.isfinite(expected) | (expected <= 0), "bad_expected_pct")
            flag(~np.isfinite(p10) | ~np.isfinite(p90), "bad_band")
            flag(np.isfinite(p10) & np.isfinite(p90) & (p10 > p90), "band_inverted")
            flag(
                np.isfinite(p10)
                & np.isfinite(p90)
                & ((p90 - p10) > BAND_WIDTH_CAP * out["threshold_pct"]),
                "band_too_wide",
            )
    out["invalid_reasons"] = reasons
    out["valid"] = [len(r) == 0 for r in reasons]
    out["signal_index"] = np.arange(n)
    return out.sort_values(["signal_utc", "asset"]).reset_index(drop=True)


def _validate_no_signals(frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns = ["asset", "decision_utc", "reason", "valid"]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    rows = []
    for frame in frames:
        part = frame.reset_index(drop=True).copy()
        if "decision_utc" not in part.columns:
            continue
        part["decision_utc"] = _aware(part["decision_utc"], "decision_utc")
        part = part[part["decision_utc"].notna()]
        if part.empty:
            continue
        part["asset"] = part["asset"].astype(str)
        part["reason"] = _text(part["reason"]) if "reason" in part.columns else ""
        rows.append(part[["asset", "decision_utc", "reason"]])
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    out = pd.concat(rows, ignore_index=True)
    out["valid"] = out["asset"].isin(ASSETS) & (out["reason"].str.len() > 0)
    return out.sort_values(["decision_utc", "asset"]).reset_index(drop=True)


def in_window(frame: pd.DataFrame, column: str, start, end) -> pd.Series:
    when = pd.to_datetime(frame[column], utc=True)
    return (when >= pd.Timestamp(start)) & (when <= pd.Timestamp(end))
