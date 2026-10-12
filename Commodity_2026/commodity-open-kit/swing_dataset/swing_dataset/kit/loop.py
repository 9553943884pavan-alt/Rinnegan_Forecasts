"""The self-run loop (CHALLENGE-SPEC 2.6 sections 3, 4, 5 and 7).

A submission's ``run.py`` drives this loop over the pack: ``fit(pack, train_end)`` once, then
session by session ``update(matured_outcomes)`` with the outcomes of its own signals that matured
since the previous session, and ``predict(as_of, asset, horizon)`` at every instant the model
decides on. Every pack row reaches the model through the point-in-time client, so the client's
read log is a complete record of what the run saw.

**Every session produces a decision record.** Section 5 asks for one on at least 90 percent of the
sessions of each asset, and a decision record is a signal *or* an explicit ``no_signal`` row with
a one-line reason. ``predict`` returning ``None`` therefore writes a ``no_signal`` row rather than
leaving a hole; a model is never required to signal, only to say that it looked. One ``no_signal``
row is written per asset and session, and only when no horizon signalled there, which is the
convention ``scoring2.baselines`` follows.

**Instants.** Rules 2.6 section 3 lets a signal fire at any instant inside a published MCX
session or at the 08:30 IST pre-open. The loop's default cadence is the pre-open, which is what
the five shipped baselines do and what satisfies the coverage floor, but it does not stand in the
way of a model that wants its own instants: define ``instants(session_date, asset)`` on the model
and the loop calls ``predict`` at each of them, or return rows carrying their own ``signal_utc``.
Either way the instant must lie inside that session or at its pre-open and must be at or after
the instant the model was called at, so the read log still proves nothing was read past a row's
own instant.

Periods (section 5): the holdout starts on 2026-06-01 and runs to the pack's last session, with
``fit`` at 2026-06-01 03:00 UTC; the forward window is the sessions after the submission close,
with ``fit`` at its first session. Neither end is a constant in the data: the loop walks the MCX
calendar and stops at the first session the pack does not reach yet, so a re-run over a longer
pack reproduces the earlier rows and appends the new ones.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

import numpy as np
import pandas as pd

from ..pack.pit import PointInTime
from ..scoring.chain import FeatureChains
from ..scoring2 import outcomes as O
from ..scoring2 import signals as SIG
from ..scoring2.prices import (
    DEFAULT_CLOSE_IST,
    DEFAULT_OPEN_IST,
    DESIGNATION_HORIZON,
    PREOPEN_IST,
    PriceSeries,
    ist_instant,
)
from . import ASSETS, C1_SIGNAL_COLUMNS, C2_SIGNAL_COLUMNS, DEFAULT_HORIZONS

DECISION_COLUMNS = {
    1: (*C1_SIGNAL_COLUMNS, "decision_utc", "reason"),
    2: (*C2_SIGNAL_COLUMNS, "decision_utc", "reason"),
}
#: What ``grade_c1`` and ``grade_c2`` add to a signal row, and therefore what ``update`` receives
#: beside the row's own columns.
OUTCOME_COLUMNS = {1: O.C1_COLUMNS, 2: O.C2_COLUMNS}
DEFAULT_NO_SIGNAL_REASON = "the model returned no signal at this instant"
#: Sessions of history the maturity view keeps. A signal matures inside its horizon, at most twenty
#: sessions for Task 2, so this is three times what grading one needs; it is a bound on the work,
#: not on what the model may read.
MATURITY_LOOKBACK_SESSIONS = 60
DAILY_COLUMNS = [
    "session_date",
    "asset",
    "contract_id",
    "expiry_date",
    "open",
    "high",
    "low",
    "close",
    "settlement",
    "volume_lots",
    "open_interest",
    "available_utc",
]
DESIGNATION_COLUMNS = ["asset", "cutoff_date", "cutoff_utc", "horizon", "contract_id"]
SESSION_COLUMNS = ["session_date", "state", "morning_open_ist", "evening_close_ist"]


@dataclass(frozen=True, slots=True)
class Period:
    name: str
    start: date
    end: date
    train_end: datetime

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "train_end_utc": self.train_end.isoformat(),
        }


def preopen_instant(day: date) -> datetime:
    """08:30 Asia/Kolkata on ``day`` as a UTC-aware datetime: the session's pre-open."""
    return ist_instant(day, PREOPEN_IST).to_pydatetime()


#: Kept under its 2.4 name because ``run.py`` and the organizer tools call it; the pre-open is the
#: loop's default decision instant, not the only instant a signal may carry.
cutoff_instant = preopen_instant

#: The far end of each period is the pack's last session, not a date in this table: the loop stops
#: at the first session the pack does not reach. The bounds below only keep a run from walking the
#: exchange calendar forever.
PERIODS: dict[str, Period] = {
    "holdout": Period("holdout", date(2026, 6, 1), date(2026, 12, 31), preopen_instant(date(2026, 6, 1))),
    "forward": Period("forward", date(2026, 10, 26), date(2027, 3, 31), preopen_instant(date(2026, 10, 26))),
}


class Model(Protocol):
    """What ``run`` needs from a submission's model (``template/model.py`` implements it).

    ``predict`` returns one of:

    - ``None`` or ``{}``: no signal here. The loop writes a ``no_signal`` row with the default
      reason.
    - ``{"reason": "..."}``: no signal here, for that reason.
    - a dict of signal fields: one signal row. ``signal_utc`` and ``horizon`` default to the
      instant and horizon it was called with.
    - a list of such dicts: several signal rows, each free to carry its own ``signal_utc``.

    Optionally define ``instants(session_date, asset) -> Sequence[datetime]`` to be called at your
    own instants inside the session instead of only at its pre-open.
    """

    def fit(self, pack: PointInTime, train_end: datetime) -> object: ...

    def predict(self, as_of: datetime, asset: str, horizon: int) -> dict | list[dict] | None: ...

    def update(self, matured_outcomes: pd.DataFrame) -> object: ...


class LoopError(RuntimeError):
    pass


# -- reads through the client ----------------------------------------------------------------


def _utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def trading_days(pit: PointInTime, asof: datetime, start: date, end: date) -> list[date]:
    """MCX trading dates in [start, end] from ``mcx/mcx_sessions`` as known at ``asof``.

    Session rows are published when the exchange issues its holiday and hours circulars, months
    ahead, so a read at the period's first pre-open sees the whole period.
    """
    sessions = pit.frame("mcx/mcx_sessions", asof).collect(columns=["session_date", "state"])
    open_days = sessions.loc[sessions["state"] != "closed", "session_date"]
    return sorted(d for d in open_days.tolist() if start <= d <= end)


def session_bounds(pit: PointInTime, asof: datetime, start: date, end: date) -> dict[date, tuple[datetime, datetime]]:
    """Per trading date in [start, end], the pre-open instant and the close instant.

    A signal may fire anywhere in this range and nowhere else (section 3). The clocks and their
    fallbacks are ``scoring2.prices``': the scorer places a decision record in a session by the
    same pre-open-to-close range.
    """
    sessions = pit.frame("mcx/mcx_sessions", asof).collect(columns=SESSION_COLUMNS)
    out: dict[date, tuple[datetime, datetime]] = {}
    for row in sessions.itertuples():
        if row.state == "closed" or not (start <= row.session_date <= end):
            continue
        close_ist = getattr(row, "evening_close_ist", None) or DEFAULT_CLOSE_IST
        out[row.session_date] = (
            ist_instant(row.session_date, PREOPEN_IST).to_pydatetime(),
            ist_instant(row.session_date, close_ist).to_pydatetime(),
        )
    return out


def designated_assets(pit: PointInTime, instant: datetime) -> list[str]:
    """The assets ``labels/designations`` names a contract for at ``instant``'s session.

    The scorer prices every signal, whatever its horizon, on the contract designated at horizon
    ``DESIGNATION_HORIZON``, so that is the designation asked for here: a Task 2 horizon the labels
    lane designates nothing at (anything but 5, 10 and 20) is still called on. The designation is
    published at the close of the previous session, so it is available strictly before the
    pre-open. An empty list means the pack does not reach this session yet, which is how a run
    over a growing pack knows where to stop.
    """
    des = pit.frame("labels/designations", instant).collect(
        columns=["asset", "cutoff_utc", "horizon", "contract_id"]
    )
    if des.empty:
        return []
    des["cutoff_utc"] = pd.to_datetime(des["cutoff_utc"], utc=True)
    at = des[(des["cutoff_utc"] == _utc(instant)) & (des["horizon"] == DESIGNATION_HORIZON)]
    at = at[at["asset"].isin(ASSETS) & at["contract_id"].notna()]
    return sorted(str(a) for a in at["asset"].unique())


def price_view(
    pit: PointInTime,
    asof: datetime,
    *,
    lookback_sessions: int | None = None,
    quotes: bool = False,
) -> tuple[PriceSeries, FeatureChains]:
    """The scorer's own price and feature objects, built from rows available before ``asof``.

    ``PriceSeries`` and ``FeatureChains`` both take frames, so the kit never opens a pack file: it
    collects ``mcx/mcx_daily``, ``labels/designations`` and ``mcx/mcx_sessions`` through the client
    and hands them over. ``lookback_sessions`` keeps only that many of the most recent sessions,
    which is what makes a per-session rebuild cheap; pass ``None`` for the whole history.
    """
    daily = pit.frame("mcx/mcx_daily", asof).collect(columns=DAILY_COLUMNS)
    daily = daily[daily["asset"].isin(ASSETS)].reset_index(drop=True)
    if lookback_sessions is not None and len(daily):
        dates = np.array(sorted(daily["session_date"].unique()))
        keep = set(dates[-int(lookback_sessions) :].tolist())
        daily = daily[daily["session_date"].isin(keep)].reset_index(drop=True)
    designations = pit.frame("labels/designations", asof).collect(columns=DESIGNATION_COLUMNS)
    sessions = pit.frame("mcx/mcx_sessions", asof).collect(columns=SESSION_COLUMNS)
    ticks = None
    if quotes:
        ticks = pit.frame("pack/quotes/mcx_quotes", asof).collect(
            columns=["asset", "contract_id", "available_utc", "ltp"]
        )
    return PriceSeries(daily, designations, sessions, ticks), FeatureChains(daily)


def matured_outcomes(
    pit: PointInTime,
    asof: datetime,
    issued: pd.DataFrame,
    challenge: int,
    *,
    lookback_sessions: int = MATURITY_LOOKBACK_SESSIONS,
) -> pd.DataFrame:
    """The outcomes of the run's own signals that are decided on rows available before ``asof``.

    Rules 2.6 scores a signal against the price path of the designated contract, not against the
    one-sigma barrier labels of 2.4, so this is what a model learns from: its own rows graded by
    ``scoring2.outcomes``, the same code the scorer runs. A row is returned once its outcome is
    settled inside the data that exists at ``asof`` (``scorable`` and not ``unmatured``); an
    undecided row comes back at a later session instead. A settled row the price series could not
    price carries ``no_price`` and no outcome; it is returned so the caller knows it is done with.

    The returned frame carries the signal's own columns plus the outcome columns of the challenge:
    ``p0``, ``mfe_pct``, ``mae_pct``, ``correct``/``realized``, ``deep``/``in_band``,
    ``window_end_utc`` and the price provenance.
    """
    if issued is None or not len(issued):
        return pd.DataFrame(columns=[*DECISION_COLUMNS[challenge], *OUTCOME_COLUMNS[challenge]])
    prices, _ = price_view(pit, asof, lookback_sessions=lookback_sessions)
    normalized = SIG.normalize(issued.reset_index(drop=True), challenge).signals
    if normalized.empty:
        return pd.DataFrame(columns=[*DECISION_COLUMNS[challenge], *OUTCOME_COLUMNS[challenge]])
    graded = (O.grade_c1 if challenge == 1 else O.grade_c2)(normalized, prices)
    settled = graded["scorable"].fillna(False).astype(bool) & ~graded["unmatured"].fillna(False).astype(bool)
    out = graded[settled]
    return out.drop(columns=["invalid_reasons"], errors="ignore").reset_index(drop=True)


# -- turning a predict() response into rows ------------------------------------------------------

_SIGNAL_FIELDS = {
    1: ("direction", "target_pct", "invalidation_pct", "confidence", "evidence"),
    2: ("direction", "expected_pct", "p10_pct", "p90_pct", "invalidation_pct", "evidence"),
}
_REQUIRED = {
    1: ("direction", "target_pct", "invalidation_pct", "evidence"),
    2: ("direction", "expected_pct", "p10_pct", "p90_pct", "invalidation_pct", "evidence"),
}


def _no_signal(asset: str, instant: datetime, reason: str, challenge: int) -> dict:
    row = {c: None for c in DECISION_COLUMNS[challenge]}
    row["asset"] = asset
    row["signal_utc"] = pd.NaT
    row["decision_utc"] = pd.Timestamp(instant)
    row["reason"] = reason
    return row


def _signal(
    response: dict,
    asset: str,
    instant: datetime,
    horizon: int,
    challenge: int,
    bounds: tuple[datetime, datetime] | None,
) -> dict:
    missing = [k for k in _REQUIRED[challenge] if response.get(k) is None]
    if missing:
        raise LoopError(
            f"Challenge {challenge} predict() must return {list(_REQUIRED[challenge])}; "
            f"missing {missing} in {sorted(response)}"
        )
    when = _utc(response.get("signal_utc") or instant)
    if when < _utc(instant):
        raise LoopError(
            f"signal_utc {when.isoformat()} is before the instant predict was called at "
            f"({_utc(instant).isoformat()}); a row may not be dated earlier than the reads behind it"
        )
    if bounds is not None and not (_utc(bounds[0]) <= when <= _utc(bounds[1])):
        raise LoopError(
            f"signal_utc {when.isoformat()} is outside its session: section 3 issues a signal "
            f"inside a published MCX session ({_utc(bounds[0]).isoformat()} to "
            f"{_utc(bounds[1]).isoformat()}) or at its 08:30 IST pre-open"
        )
    row = {c: None for c in DECISION_COLUMNS[challenge]}
    row["asset"] = asset
    row["signal_utc"] = when
    row["horizon"] = int(response.get("horizon", horizon))
    row["decision_utc"] = pd.NaT
    for field_name in _SIGNAL_FIELDS[challenge]:
        value = response.get(field_name)
        if field_name in ("direction", "evidence"):
            row[field_name] = None if value is None else str(value)
        else:
            row[field_name] = None if value is None else float(value)
    return row


def rows_from(
    response: dict | list[dict] | None,
    asset: str,
    instant: datetime,
    horizon: int,
    challenge: int,
    bounds: tuple[datetime, datetime] | None = None,
) -> tuple[list[dict], str | None]:
    """(signal rows, the reason when there are none) for one ``predict`` response."""
    if response is None:
        return [], DEFAULT_NO_SIGNAL_REASON
    if isinstance(response, dict):
        if not response:
            return [], DEFAULT_NO_SIGNAL_REASON
        if "direction" not in response:
            reason = str(response.get("reason") or DEFAULT_NO_SIGNAL_REASON).strip()
            return [], reason or DEFAULT_NO_SIGNAL_REASON
        response = [response]
    if not isinstance(response, (list, tuple)):
        raise LoopError(f"predict() returned {type(response).__name__}; expected a dict, a list of dicts or None")
    if not response:
        return [], DEFAULT_NO_SIGNAL_REASON
    return [_signal(dict(r), asset, instant, horizon, challenge, bounds) for r in response], None


# -- the driver ---------------------------------------------------------------------------------


@dataclass
class RunResult:
    challenge: int
    period: Period
    frozen: bool
    horizons: tuple[int, ...]
    decisions: pd.DataFrame
    sessions: list[date] = field(default_factory=list)
    updates: int = 0
    outcomes_delivered: int = 0
    stopped_at: date | None = None
    fit_seconds: float = 0.0
    loop_seconds: float = 0.0
    fit_info: object = None

    @property
    def signals(self) -> pd.DataFrame:
        frame = self.decisions
        return frame[frame["signal_utc"].notna()] if len(frame) else frame

    def summary(self) -> dict:
        signals = len(self.signals)
        return {
            "challenge": self.challenge,
            "period": self.period.to_dict(),
            "frozen": self.frozen,
            "horizons": list(self.horizons),
            "sessions": len(self.sessions),
            "first_session": self.sessions[0].isoformat() if self.sessions else None,
            "last_session": self.sessions[-1].isoformat() if self.sessions else None,
            "rows": len(self.decisions),
            "signals": signals,
            "no_signals": len(self.decisions) - signals,
            "updates": self.updates,
            "outcomes_delivered": self.outcomes_delivered,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "fit_seconds": round(self.fit_seconds, 3),
            "loop_seconds": round(self.loop_seconds, 3),
        }


def _instants_for(model: Model, day: date, asset: str, bounds: tuple[datetime, datetime]) -> list[datetime]:
    hook = getattr(model, "instants", None)
    if hook is None:
        return [bounds[0]]
    raw = hook(day, asset)
    if raw is None:
        return [bounds[0]]
    if isinstance(raw, (datetime, pd.Timestamp)):
        raw = [raw]
    if not isinstance(raw, Sequence):
        raise LoopError("instants() must return a sequence of timezone-aware datetimes")
    out = []
    for value in raw:
        when = _utc(value)
        if not (_utc(bounds[0]) <= when <= _utc(bounds[1])):
            raise LoopError(
                f"instants() returned {when.isoformat()} for {asset} on {day}, outside that "
                "session's pre-open to close range"
            )
        out.append(when.to_pydatetime())
    if not out:
        return [bounds[0]]
    return sorted(out)


def run(
    model: Model,
    pit: PointInTime,
    *,
    challenge: int,
    period: Period,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    frozen: bool = False,
    log: Callable[[str], None] = lambda _m: None,
) -> RunResult:
    """Fit once, then walk the period's sessions with interleaved updates; returns the rows.

    ``frozen`` skips every ``update`` call: the model stays at its fit, which is the copy section 4
    measures the improvement score against. The read log context names the phase and the instant
    each read serves.
    """
    if challenge not in (1, 2):
        raise ValueError("challenge must be 1 or 2")
    horizons = tuple(int(h) for h in horizons)
    reads = pit.read_log
    train_end = _utc(period.train_end).to_pydatetime()

    def context(**fields: object) -> None:
        if reads is not None:
            reads.context = {"period": period.name, **fields}

    context(phase="calendar")
    first = preopen_instant(period.start)
    days = trading_days(pit, first, period.start, period.end)
    if not days:
        raise LoopError(f"no MCX trading days between {period.start} and {period.end} in mcx/mcx_sessions")
    bounds = session_bounds(pit, first, period.start, period.end)
    log(f"{len(days)} trading days from {days[0]} to {days[-1]}; fit at {train_end.isoformat()}")

    context(phase="fit", train_end_utc=train_end.isoformat())
    started = time.perf_counter()
    fit_info = model.fit(pit, train_end)
    fit_seconds = time.perf_counter() - started
    log(f"fit done in {fit_seconds:.1f}s")

    rows: list[dict] = []
    sessions: list[date] = []
    updates = 0
    delivered = 0
    stopped_at: date | None = None
    pending: list[dict] = []
    started = time.perf_counter()
    for i, day in enumerate(days):
        preopen = preopen_instant(day)
        context(phase="eligibility", cutoff_utc=preopen.isoformat())
        assets = designated_assets(pit, preopen)
        if not assets:
            stopped_at = day
            log(f"{day}: the pack has no designation for this session; stopping")
            break
        if not frozen and pending:
            context(phase="update", cutoff_utc=preopen.isoformat())
            matured = matured_outcomes(pit, preopen, pd.DataFrame(pending), challenge)
            if len(matured):
                settled = set(matured["signal_index"].astype(int))
                pending = [r for j, r in enumerate(pending) if j not in settled]
                priced = matured[~matured["no_price"].fillna(False).astype(bool)]
                if len(priced):
                    model.update(priced.reset_index(drop=True))
                    updates += 1
                    delivered += len(priced)
        window = bounds.get(day, (preopen, preopen))
        for asset in assets:
            issued_here: list[dict] = []
            reason: str | None = None
            for instant in _instants_for(model, day, asset, window):
                context(phase="predict", cutoff_utc=_utc(instant).isoformat())
                for horizon in horizons:
                    response = model.predict(instant, asset, horizon)
                    produced, why = rows_from(response, asset, instant, horizon, challenge, window)
                    issued_here.extend(produced)
                    if why is not None and reason is None:
                        reason = why
            if issued_here:
                rows.extend(issued_here)
                pending.extend(issued_here)
            else:
                rows.append(_no_signal(asset, window[0], reason or DEFAULT_NO_SIGNAL_REASON, challenge))
        sessions.append(day)
        if (i + 1) % 20 == 0 or i + 1 == len(days):
            log(f"{i + 1}/{len(days)} sessions, {len(rows)} rows, {delivered} outcomes delivered")
    context(phase="done")
    decisions = pd.DataFrame(rows, columns=list(DECISION_COLUMNS[challenge]))
    if len(decisions):
        for column in ("signal_utc", "decision_utc"):
            decisions[column] = pd.to_datetime(decisions[column], utc=True).dt.as_unit("us")
        decisions["horizon"] = pd.to_numeric(decisions["horizon"], errors="coerce").astype("Int64")
        order = decisions["signal_utc"].fillna(decisions["decision_utc"])
        decisions = (
            decisions.assign(_instant=order, _kind=decisions["signal_utc"].isna())
            .sort_values(["_instant", "asset", "_kind", "horizon"], kind="stable")
            .drop(columns=["_instant", "_kind"])
            .reset_index(drop=True)
        )
    return RunResult(
        challenge,
        period,
        frozen,
        horizons,
        decisions,
        sessions,
        updates,
        delivered,
        stopped_at,
        fit_seconds,
        time.perf_counter() - started,
        fit_info,
    )


def period_for(name: str, start: date | None = None, end: date | None = None, train_end: datetime | None = None) -> Period:
    """The named period, narrowed to [start, end] for smoke tests.

    The holdout's fit instant never moves (2026-06-01 03:00 UTC). The forward window's fit instant
    is its first session, so narrowing its start moves it to the new first session.
    """
    base = PERIODS[name]
    if train_end is None:
        train_end = preopen_instant(start) if (name == "forward" and start is not None) else base.train_end
    return Period(base.name, start or base.start, end or base.end, _utc(train_end).to_pydatetime())


__all__ = [
    "DECISION_COLUMNS",
    "DEFAULT_CLOSE_IST",
    "DEFAULT_NO_SIGNAL_REASON",
    "DEFAULT_OPEN_IST",
    "MATURITY_LOOKBACK_SESSIONS",
    "OUTCOME_COLUMNS",
    "PERIODS",
    "LoopError",
    "Model",
    "Period",
    "RunResult",
    "cutoff_instant",
    "designated_assets",
    "matured_outcomes",
    "period_for",
    "preopen_instant",
    "price_view",
    "rows_from",
    "run",
    "session_bounds",
    "trading_days",
]
