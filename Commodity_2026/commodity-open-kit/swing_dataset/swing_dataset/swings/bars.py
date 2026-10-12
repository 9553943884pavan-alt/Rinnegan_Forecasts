"""The designated-contract bar series the zigzag runs on.

One bar is an hourly high and low of the designated contract, bucketed on the quote's
``available_utc`` floored to the Asia/Kolkata hour, so that a bar boundary falls on the 09:00
IST session open rather than mid-bar. Before 2026-06-22 there are no intraday quotes and the
bar is the Bhav Copy session high and low, marked ``daily``. The same fallback covers the
sessions after 2026-06-22 where the feed had not yet started polling the contract the
designation rule had already moved to.

A quote only counts when its ``available_utc`` lies inside the session's own trading window
from ``mcx_sessions``. The polling era (2026-06-22 to 2026-07-14) kept writing rows around the
clock and on closed days; those rows carry the last traded price of a session that had already
ended and are not part of any bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..mcx.sessions import session_close_utc
from . import (
    ASSETS,
    BAR_DAILY,
    BAR_HOURLY,
    BAR_NONE,
    DESIGNATION_HORIZON,
    HOURLY_START,
    IST,
    LABELS_BUILD,
    MCX_BUILD,
    QUOTES_BUILD,
)


class InputError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Session:
    session_date: date
    state: str
    full_day: bool
    #: None when the exchange has announced the session but not yet its timings (a Muhurat
    #: sitting notified "subsequently"). Such a session still counts in a session span.
    open_utc: datetime | None
    close_utc: datetime | None

    def window(self) -> tuple[datetime, datetime]:
        if self.open_utc is None or self.close_utc is None:
            raise InputError(f"session {self.session_date} has no announced trading hours")
        return self.open_utc, self.close_utc


@dataclass(frozen=True, slots=True)
class Bar:
    session_date: date
    contract_id: str
    start_utc: datetime
    high: float
    low: float
    kind: str
    ticks: int


@dataclass(slots=True)
class SessionRow:
    session_date: date
    contract_id: str | None
    designation_eligible: bool
    ineligible_reason: str | None
    bar: str
    bars: int
    ticks: int
    session_high: float | None
    session_low: float | None
    session_settlement: float | None


@dataclass(slots=True)
class Series:
    asset: str
    bars: list[Bar] = field(default_factory=list)
    sessions: list[SessionRow] = field(default_factory=list)


@dataclass(slots=True)
class Calendar:
    """Every non-closed MCX session, in date order."""

    sessions: list[Session]
    index: dict[date, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {s.session_date: i for i, s in enumerate(self.sessions)}

    def span(self, first: date, last: date) -> int:
        """Trading sessions from ``first`` to ``last`` inclusive."""
        return self.index[last] - self.index[first] + 1

    def get(self, day: date) -> Session | None:
        i = self.index.get(day)
        return None if i is None else self.sessions[i]


def load_calendar(mcx_dir: Path = MCX_BUILD) -> Calendar:
    rows = pq.read_table(mcx_dir / "mcx_sessions.parquet").to_pylist()
    rows.sort(key=lambda r: r["session_date"])
    out: list[Session] = []
    for row in rows:
        if row["state"] == "closed":
            continue
        open_ist = row["morning_open_ist"] or row["evening_open_ist"]
        close = session_close_utc(row)
        open_utc = (
            None
            if open_ist is None
            else datetime.combine(row["session_date"], open_ist, tzinfo=IST).astimezone(UTC)
        )
        out.append(
            Session(
                row["session_date"],
                row["state"],
                row["state"] == "open" and row["holiday_name"] is None,
                open_utc,
                close,
            )
        )
    return Calendar(out)


def load_designations(
    labels_dir: Path = LABELS_BUILD, horizon: int = DESIGNATION_HORIZON
) -> pd.DataFrame:
    """Horizon-``horizon`` designation rows: ``asset, session_date, contract_id, eligible, reason``.

    A row with a contract but ``eligible = false`` still designates: the only ineligibility in
    this data is ``insufficient_vol_history``, and the volatility scale the 2.4 labels needed
    has no part in a percent threshold.
    """
    table = pq.read_table(
        labels_dir / "designations.parquet",
        columns=["asset", "horizon", "cutoff_date", "contract_id", "eligible", "ineligible_reason"],
    )
    frame = table.to_pandas()
    frame = frame[frame["horizon"] == horizon].drop(columns=["horizon"])
    frame = frame.rename(columns={"cutoff_date": "session_date"})
    frame = frame[frame["asset"].isin(ASSETS)]
    return frame.sort_values(["asset", "session_date"]).reset_index(drop=True)


def load_daily(mcx_dir: Path = MCX_BUILD) -> pd.DataFrame:
    table = pq.read_table(
        mcx_dir / "mcx_daily.parquet",
        columns=["session_date", "asset", "contract_id", "high", "low", "settlement"],
    )
    frame = table.to_pandas()
    return frame[frame["asset"].isin(ASSETS)].reset_index(drop=True)


def load_quotes(quotes_dir: Path = QUOTES_BUILD) -> pd.DataFrame:
    table = pq.read_table(
        quotes_dir / "mcx_quotes.parquet", columns=["available_utc", "asset", "contract_id", "ltp"]
    )
    frame = table.to_pandas()
    return frame[frame["asset"].isin(ASSETS)].reset_index(drop=True)


def hourly_bars(quotes: pd.DataFrame, calendar: Calendar) -> tuple[pd.DataFrame, dict[str, int]]:
    """Hourly highs and lows per (asset, contract, session), plus what the window filter dropped."""
    frame = quotes.copy()
    ist = frame["available_utc"].dt.tz_convert(IST)
    frame["session_date"] = ist.dt.date
    frame["hour_ist"] = ist.dt.floor("h")
    opens = frame["session_date"].map(lambda d: getattr(calendar.get(d), "open_utc", None))
    closes = frame["session_date"].map(lambda d: getattr(calendar.get(d), "close_utc", None))
    inside = opens.notna() & (frame["available_utc"] >= opens) & (frame["available_utc"] <= closes)
    dropped = {
        "rows": len(frame),
        "outside_session_window": int((~inside).sum()),
        "no_session_window": int(opens.isna().sum()),
    }
    frame = frame[inside]
    bars = (
        frame.groupby(["asset", "contract_id", "session_date", "hour_ist"], observed=True)["ltp"]
        .agg(high="max", low="min", ticks="count")
        .reset_index()
    )
    bars["start_utc"] = bars["hour_ist"].dt.tz_convert(UTC)
    return bars.drop(columns=["hour_ist"]).sort_values(
        ["asset", "contract_id", "session_date", "start_utc"]
    ), dropped


def build_series(
    *,
    labels_dir: Path = LABELS_BUILD,
    mcx_dir: Path = MCX_BUILD,
    quotes_dir: Path = QUOTES_BUILD,
    hourly_start: date = HOURLY_START,
    last_session: date | None = None,
) -> tuple[dict[str, Series], Calendar, dict[str, object]]:
    """The per-asset bar series and session inventory, plus the provenance the sidecars need."""
    calendar = load_calendar(mcx_dir)
    designations = load_designations(labels_dir)
    daily = load_daily(mcx_dir)
    quotes = load_quotes(quotes_dir)
    bars, dropped = hourly_bars(quotes, calendar)

    end = min(daily["session_date"].max(), designations["session_date"].max())
    if last_session is not None:
        end = min(end, last_session)
    designations = designations[designations["session_date"] <= end]

    daily_key = {
        (r.asset, r.contract_id, r.session_date): (r.high, r.low, r.settlement)
        for r in daily.itertuples()
    }
    hourly_key: dict[tuple[str, str, date], list[Bar]] = {}
    for (asset, contract_id, session_date), group in bars.groupby(
        ["asset", "contract_id", "session_date"], observed=True
    ):
        hourly_key[(asset, contract_id, session_date)] = [
            Bar(
                session_date,
                contract_id,
                row.start_utc.to_pydatetime(),
                float(row.high),
                float(row.low),
                BAR_HOURLY,
                int(row.ticks),
            )
            for row in group.itertuples()
        ]

    series: dict[str, Series] = {}
    for asset in ASSETS:
        out = Series(asset)
        rows = designations[designations["asset"] == asset]
        for row in rows.itertuples():
            day, contract_id = row.session_date, _text(row.contract_id)
            record = SessionRow(
                day, contract_id, bool(row.eligible), _text(row.ineligible_reason), BAR_NONE, 0, 0,
                None, None, None,
            )
            if contract_id is not None:
                high, low, settlement = daily_key.get((asset, contract_id, day), (None, None, None))
                record.session_high = _finite(high)
                record.session_low = _finite(low)
                record.session_settlement = _finite(settlement)
                hours = hourly_key.get((asset, contract_id, day)) if day >= hourly_start else None
                if hours:
                    out.bars.extend(hours)
                    record.bar = BAR_HOURLY
                    record.bars = len(hours)
                    record.ticks = sum(bar.ticks for bar in hours)
                elif record.session_high is not None and record.session_low is not None:
                    session = calendar.get(day)
                    if session is None:
                        raise InputError(f"{asset} {day}: designated session is not in the calendar")
                    out.bars.append(
                        Bar(
                            day,
                            contract_id,
                            session.window()[0],
                            record.session_high,
                            record.session_low,
                            BAR_DAILY,
                            0,
                        )
                    )
                    record.bar = BAR_DAILY
                    record.bars = 1
            out.sessions.append(record)
        series[asset] = out

    provenance = {
        "last_session": end,
        "hourly_start": hourly_start,
        "quote_rows": dropped,
        "quote_last_available_utc": quotes["available_utc"].max().to_pydatetime(),
    }
    return series, calendar, provenance


def _text(value: object) -> str | None:
    """An all-null parquet string column comes back from pandas as float NaN."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value)


def _finite(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
