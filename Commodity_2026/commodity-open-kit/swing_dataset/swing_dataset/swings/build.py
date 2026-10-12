"""Build ``swings.parquet``, ``swing_sessions.parquet`` and ``stats.json`` under ``build/swings/``."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from ..common import sidecar_path
from ..pack.store import (
    BOOL,
    DATE,
    F64,
    I64,
    STR,
    TS,
    Column,
    TableWriter,
    build_identity,
    clock_meta,
    file_sha256,
    write_sidecar,
)
from . import (
    ASSETS,
    BAR_DAILY,
    BAR_HOURLY,
    BAR_MIXED,
    BAR_NONE,
    DESIGNATION_HORIZON,
    DOWN,
    HOLDOUT_START,
    HOURLY_START,
    IST,
    LABELS_BUILD,
    LANE_BUILD,
    MCX_BUILD,
    PRE_OPEN_IST,
    QUOTES_BUILD,
    THRESHOLD_PCT,
    UP,
)
from .bars import Bar, Calendar, Series, build_series
from .zigzag import Leg, zigzag

SWING_COLUMNS = [
    Column("asset", STR, "gold | silver | copper | crude", dictionary=True),
    Column("direction", STR, "up | down: the sign of the move from the turn to the extreme", dictionary=True),
    Column("turn_utc", TS, "start instant of the bar the turn printed in (the previous leg's extreme)"),
    Column("turn_price", F64, "price at the turn: that bar's low for an up leg, its high for a down leg, INR per exchange unit"),
    Column("extreme_utc", TS, "start instant of the bar the extreme printed in"),
    Column("extreme_price", F64, "price at the extreme: that bar's high for an up leg, its low for a down leg, INR per exchange unit"),
    Column("move_pct", F64, "signed (extreme_price - turn_price) / turn_price * 100; its magnitude is at least threshold_pct"),
    Column("sessions_spanned", I64, "MCX trading sessions from the turn's session to the extreme's session, inclusive"),
    Column("contract_id", STR, "designated contract the extreme printed on; the turn's contract differs exactly when contract_changed", dictionary=True),
    Column("contract_changed", BOOL, "true when the designation rolled between the turn bar and the extreme bar, so the two prices are of different contracts and the move includes the calendar spread"),
    Column("bar", STR, "hourly | daily | mixed: the bar type behind the leg's bars", dictionary=True),
    Column("threshold_pct", F64, "the asset's swing threshold in percent (CHALLENGE-SPEC section 2)"),
    Column("available_utc", TS, "settlement publication of the session the extreme printed in: that session's close instant", clock="published"),
    Column("confirm_utc", TS, "start instant of the bar whose move completed the leg: the first bar to reverse threshold_pct from the extreme. At or after extreme_utc on every row"),
    Column("confirm_available_utc", TS, "settlement publication of the session the confirming bar printed in, the same convention available_utc uses. The earliest instant this leg is knowable as a swing, and the one a point-in-time consumer must use"),
    Column("clock_quality", STR, "published", dictionary=True),
]

SESSION_COLUMNS = [
    Column("asset", STR, "gold | silver | copper | crude", dictionary=True),
    Column("session_date", DATE, "Asia/Kolkata civil date of the MCX session", dictionary=True),
    Column("contract_id", STR, "designated contract for this session (designations.parquet, horizon 5); null when none qualified", dictionary=True),
    Column("designation_eligible", BOOL, "the designation row's own eligible flag"),
    Column("ineligible_reason", STR, "the designation row's reason; null when eligible", dictionary=True),
    Column("bar", STR, "hourly | daily | none: the bar type this session contributed to the series", dictionary=True),
    Column("bars", I64, "bars the session contributed: hours for an hourly session, 1 for a daily one, 0 for none"),
    Column("ticks", I64, "quotes behind the hourly bars; 0 for a daily session"),
    Column("session_high", F64, "designated contract's Bhav Copy high, INR; null when it printed no trade"),
    Column("session_low", F64, "designated contract's Bhav Copy low, INR; null when it printed no trade"),
    Column("session_settlement", F64, "designated contract's Bhav Copy settlement, INR"),
    Column("turn_count", I64, "swing turns that printed in this session"),
    Column("extreme_count", I64, "swing extremes that printed in this session"),
    Column("has_turn", BOOL, "true when at least one swing turn printed in this session"),
    Column("has_extreme", BOOL, "true when at least one swing extreme printed in this session"),
    Column("cutoff_utc", TS, "08:30 Asia/Kolkata pre-open of this session, in UTC: the earliest instant a signal may carry this session's date"),
    Column("eligible_cutoff", BOOL, "true when the session has a designated contract and a priced bar, so a decision record is expected of a model and the session counts in the coverage denominator"),
    Column("period", STR, "training (before 2026-06-01) | holdout (CHALLENGE-SPEC section 5)", dictionary=True),
    Column("session_state", STR, "open | morning_only | evening_only from mcx_sessions", dictionary=True),
    Column("full_day", BOOL, "true when the session is open with no holiday shortening"),
    Column("session_open_utc", TS, "first trading instant of the session"),
    Column("available_utc", TS, "settlement publication of this session: its close instant", clock="published"),
    Column("clock_quality", STR, "published", dictionary=True),
]


@dataclass(frozen=True, slots=True)
class Swing:
    asset: str
    leg: Leg
    turn: Bar
    extreme: Bar
    sessions_spanned: int
    contract_changed: bool
    bar: str
    available_utc: datetime
    #: Bar whose move completed the leg, and the settlement publication of its session.
    confirm: Bar
    confirm_available_utc: datetime
    #: Trading sessions from the extreme's session to the confirming bar's session; 0 when both
    #: printed in the same session.
    confirmation_lag_sessions: int


@dataclass(frozen=True, slots=True)
class Outputs:
    swings: Path
    sessions: Path
    stats: Path


def bar_kind(bars: list[Bar], first: int, last: int) -> str:
    kinds = {bar.kind for bar in bars[first : last + 1]}
    if len(kinds) == 1:
        return kinds.pop()
    return BAR_MIXED


def swings_of(asset: str, series: Series, calendar: Calendar) -> list[Swing]:
    bars = series.bars
    legs = zigzag([b.high for b in bars], [b.low for b in bars], THRESHOLD_PCT[asset])
    out: list[Swing] = []
    for leg in legs:
        turn, extreme = bars[leg.turn_index], bars[leg.extreme_index]
        confirm = bars[leg.confirm_index]
        contracts = {bar.contract_id for bar in bars[leg.turn_index : leg.extreme_index + 1]}
        session = calendar.get(extreme.session_date)
        confirm_session = calendar.get(confirm.session_date)
        assert session is not None
        assert confirm_session is not None
        out.append(
            Swing(
                asset,
                leg,
                turn,
                extreme,
                calendar.span(turn.session_date, extreme.session_date),
                len(contracts) > 1,
                bar_kind(bars, leg.turn_index, leg.extreme_index),
                session.window()[1],
                confirm,
                confirm_session.window()[1],
                calendar.span(extreme.session_date, confirm.session_date) - 1,
            )
        )
    return out


def cutoff_utc(day: date) -> datetime:
    return datetime.combine(day, PRE_OPEN_IST, tzinfo=IST).astimezone(UTC)


def write_swings(path: Path, swings: dict[str, list[Swing]]) -> TableWriter:
    writer = TableWriter(path, SWING_COLUMNS)
    for asset in ASSETS:
        for swing in swings[asset]:
            leg = swing.leg
            writer.append(
                {
                    "asset": asset,
                    "direction": leg.direction,
                    "turn_utc": swing.turn.start_utc,
                    "turn_price": leg.turn_price,
                    "extreme_utc": swing.extreme.start_utc,
                    "extreme_price": leg.extreme_price,
                    "move_pct": leg.move_pct,
                    "sessions_spanned": swing.sessions_spanned,
                    "contract_id": swing.extreme.contract_id,
                    "contract_changed": swing.contract_changed,
                    "bar": swing.bar,
                    "threshold_pct": THRESHOLD_PCT[asset],
                    "available_utc": swing.available_utc,
                    "confirm_utc": swing.confirm.start_utc,
                    "confirm_available_utc": swing.confirm_available_utc,
                    "clock_quality": "published",
                }
            )
    writer.close()
    return writer


def write_sessions(
    path: Path, series: dict[str, Series], swings: dict[str, list[Swing]], calendar: Calendar
) -> TableWriter:
    writer = TableWriter(path, SESSION_COLUMNS)
    for asset in ASSETS:
        turns: Counter[date] = Counter()
        extremes: Counter[date] = Counter()
        for swing in swings[asset]:
            turns[swing.turn.session_date] += 1
            extremes[swing.extreme.session_date] += 1
        for record in series[asset].sessions:
            session = calendar.get(record.session_date)
            assert session is not None
            open_utc, close_utc = session.window()
            writer.append(
                {
                    "asset": asset,
                    "session_date": record.session_date,
                    "contract_id": record.contract_id,
                    "designation_eligible": record.designation_eligible,
                    "ineligible_reason": record.ineligible_reason,
                    "bar": record.bar,
                    "bars": record.bars,
                    "ticks": record.ticks,
                    "session_high": record.session_high,
                    "session_low": record.session_low,
                    "session_settlement": record.session_settlement,
                    "turn_count": turns[record.session_date],
                    "extreme_count": extremes[record.session_date],
                    "has_turn": turns[record.session_date] > 0,
                    "has_extreme": extremes[record.session_date] > 0,
                    "cutoff_utc": cutoff_utc(record.session_date),
                    "eligible_cutoff": record.contract_id is not None and record.bar != BAR_NONE,
                    "period": "holdout" if record.session_date >= HOLDOUT_START else "training",
                    "session_state": session.state,
                    "full_day": session.full_day,
                    "session_open_utc": open_utc,
                    "available_utc": close_utc,
                    "clock_quality": "published",
                }
            )
    writer.close()
    return writer


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def _confirmation_lag(items: list[Swing]) -> dict:
    """``confirm_available_utc`` minus ``available_utc``, counted in trading sessions.

    Zero means the reversal that completed the leg printed in the extreme's own session, so the
    settlement that made the extreme available also made the swing knowable.
    """
    lags = sorted(s.confirmation_lag_sessions for s in items)
    hours = [
        (s.confirm_available_utc - s.available_utc).total_seconds() / 3600.0 for s in items
    ]
    return {
        "unit": "MCX trading sessions from the extreme's session to the confirming bar's session",
        "legs": len(lags),
        "min": lags[0] if lags else None,
        "median": _median([float(v) for v in lags]),
        "p90": lags[min(len(lags) - 1, int(0.9 * len(lags)))] if lags else None,
        "max": lags[-1] if lags else None,
        "mean": round(sum(lags) / len(lags), 4) if lags else None,
        "same_session": sum(1 for v in lags if v == 0),
        "histogram": {str(v): lags.count(v) for v in sorted(set(lags))},
        "median_hours": _median(hours),
        "max_hours": round(max(hours), 4) if hours else None,
    }


def _leg_stats(items: list[Swing], sessions: int | None) -> dict:
    moves = [abs(s.leg.move_pct) for s in items]
    spans = [float(s.sessions_spanned) for s in items]
    hours = [(s.extreme.start_utc - s.turn.start_utc).total_seconds() / 3600.0 for s in items]
    out = {
        "legs": len(items),
        "up": sum(1 for s in items if s.leg.direction == UP),
        "down": sum(1 for s in items if s.leg.direction == DOWN),
        "contract_changed": sum(1 for s in items if s.contract_changed),
        "median_abs_move_pct": _median(moves),
        "median_sessions_spanned": _median(spans),
        "median_duration_hours": _median(hours),
        "confirmation_lag_sessions": _confirmation_lag(items),
    }
    if sessions is not None:
        out["sessions"] = sessions
        out["legs_per_20_sessions"] = round(len(items) / sessions * 20, 3) if sessions else None
    return out


def _per_year(items: list[Swing]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for swing in items:
        counts[str(swing.extreme.session_date.year)] += 1
    return dict(sorted(counts.items()))


def stats_of(
    series: dict[str, Series],
    swings: dict[str, list[Swing]],
    provenance: dict[str, object],
    command: str,
) -> dict:
    last_session = provenance["last_session"]
    assets: dict[str, dict] = {}
    for asset in ASSETS:
        items = swings[asset]
        session_rows = series[asset].sessions
        by_bar_sessions = Counter(r.bar for r in session_rows)
        block = _leg_stats(items, len([r for r in session_rows if r.bar != BAR_NONE]))
        block["threshold_pct"] = THRESHOLD_PCT[asset]
        block["sessions_designated"] = len(session_rows)
        block["bars"] = {
            BAR_HOURLY: sum(r.bars for r in session_rows if r.bar == BAR_HOURLY),
            BAR_DAILY: sum(r.bars for r in session_rows if r.bar == BAR_DAILY),
        }
        block["per_year"] = _per_year(items)
        block["by_bar"] = {
            kind: _leg_stats(
                [s for s in items if s.bar == kind],
                by_bar_sessions.get(kind) if kind in (BAR_HOURLY, BAR_DAILY) else None,
            )
            | {"per_year": _per_year([s for s in items if s.bar == kind])}
            for kind in (BAR_HOURLY, BAR_DAILY, BAR_MIXED)
        }
        assets[asset] = block

    return {
        "build": build_identity(command),
        "rule": {
            "thresholds_pct": THRESHOLD_PCT,
            "designation_horizon": DESIGNATION_HORIZON,
            "hourly_start": str(provenance["hourly_start"]),
            "bar_clock": "Asia/Kolkata hour of the quote's available_utc",
        },
        "range": {
            "first_session": str(min(series[a].sessions[0].session_date for a in ASSETS)),
            "last_session": str(last_session),
            "quote_last_available_utc": provenance["quote_last_available_utc"].isoformat(),
            "quote_rows": provenance["quote_rows"],
        },
        "assets": assets,
        "totals": {
            "legs": sum(len(swings[a]) for a in ASSETS),
            "sessions": sum(len(series[a].sessions) for a in ASSETS),
        },
    }


def _source(path: Path) -> dict:
    return {
        "url": sidecar_path(path),
        "fetched_at_utc": None,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "local_path": sidecar_path(path),
    }


CLOCK_RULE = (
    "A swing row's availability is the settlement publication of the session its extreme printed "
    "in: that session's close instant from mcx_sessions, the same convention mcx_daily uses. The "
    "exchange publishes at or after the close and we did not capture at that instant, so the class "
    "is published. confirm_available_utc is the same published clock read on the session of the "
    "bar that completed the leg, and it is the column a point-in-time consumer must use; it is "
    "not named under clocks because that mapping holds one column per class and available_utc is "
    "the file's primary availability column. The bar instants are not availability: an hourly bar "
    "carries the quotes' own clocks (mcx_quotes states exact or published per row) and a daily bar "
    "carries the session's opening instant, inside which the Bhav Copy high and low printed at an "
    "unknown time, and confirm_utc is a bar instant of that kind."
)

NOTES = [
    "One row per completed leg of the percent-threshold zigzag of CHALLENGE-SPEC.md section 2, per asset.",
    "Price series: the contract of designations.parquet at horizon 5, session by session. Prices are the contract's own, never spread-adjusted; contract_changed marks a leg whose turn and extreme sit on different contracts, so its move includes the calendar spread.",
    "Bars: hourly highs and lows of ltp from pack/quotes/mcx_quotes.parquet, bucketed on available_utc floored to the Asia/Kolkata hour so a bar boundary falls on the 09:00 IST open. A quote counts only inside its session's own trading window from mcx_sessions; the 2026-06-22 to 2026-07-14 polling era also wrote rows outside trading hours and on closed days, carrying an already-settled last traded price, and those are not part of any bar.",
    "Before 2026-06-22 there are no intraday quotes and the bar is the Bhav Copy session high and low, marked daily. The same fallback covers the sessions after 2026-06-22 where the feed had not yet started polling the contract the designation rule had already moved to (for example silver 2026-08-31 to 2026-09-03 on the December contract).",
    "Within a bar the order of the high and the low is unknown. A bar that both reverses the threshold from the running extreme and makes a new extreme completes the leg: the extreme established in an earlier bar wins.",
    "A leg in progress at the last session is not a swing and is not written; the reversal that would complete it has not happened inside the data.",
    "Availability is the extreme's session close, as the rule states, and available_utc is not the instant the swing became knowable: a leg is only confirmed by the reversal that completes it, which can print sessions later. confirm_utc is the instant of that confirming bar and confirm_available_utc is its session's settlement publication, the same convention available_utc uses. A point-in-time consumer must read confirm_available_utc; available_utc is when the extreme's own settlement published, which is knowable, but the fact that the extreme was a swing's extreme is not knowable until the leg confirms.",
    "Both confirmation columns are non-null on every row, because a leg is only written once it has completed. confirm_utc is at or after extreme_utc and confirm_available_utc is at or after confirm_utc; stats.json carries the distribution of confirm_available_utc minus available_utc in sessions, per asset.",
    "The confirming bar can sit on a different contract from the leg. A roll between the extreme and the reversal leaves contract_changed false on this leg and true on the next one, which is the leg whose own turn-to-extreme span crosses the roll; contract_changed is a statement about the leg's own span and never about the bar that confirmed it.",
    "The 1-sigma barrier labels of version 2.4 under build/labels/ are untouched and ship as research.",
]

SESSION_NOTES = [
    "One row per (asset, session) over the designated-contract series, from the first designation to the last session with a Bhav Copy.",
    "eligible_cutoff is the coverage denominator of CHALLENGE-SPEC.md section 5: the session has a designated contract and a priced bar, so a decision record is expected of a model.",
    "session_high, session_low and session_settlement are the designated contract's own Bhav Copy values, whatever the bar type; on an hourly session they bound the hourly bars but are the exchange's numbers, not ours.",
    "turn_count and extreme_count are of swings.parquet rows; a session can hold several of either.",
]


def build(
    out: Path = LANE_BUILD,
    *,
    labels_dir: Path = LABELS_BUILD,
    mcx_dir: Path = MCX_BUILD,
    quotes_dir: Path = QUOTES_BUILD,
    hourly_start: date = HOURLY_START,
    last_session: date | None = None,
    command: str = "python -m swing_dataset.swings build",
    log: Callable[[str], None] = print,
) -> Outputs:
    series, calendar, provenance = build_series(
        labels_dir=labels_dir,
        mcx_dir=mcx_dir,
        quotes_dir=quotes_dir,
        hourly_start=hourly_start,
        last_session=last_session,
    )
    swings = {asset: swings_of(asset, series[asset], calendar) for asset in ASSETS}
    for asset in ASSETS:
        log(f"{asset}: {len(series[asset].bars)} bars, {len(swings[asset])} swings")

    out.mkdir(parents=True, exist_ok=True)
    sources = [
        _source(labels_dir / "designations.parquet"),
        _source(mcx_dir / "mcx_daily.parquet"),
        _source(mcx_dir / "mcx_sessions.parquet"),
        _source(quotes_dir / "mcx_quotes.parquet"),
    ]
    date_range = {
        "column": "session_date",
        "start": str(min(series[a].sessions[0].session_date for a in ASSETS)),
        "end": str(provenance["last_session"]),
    }

    swings_path = out / "swings.parquet"
    writer = write_swings(swings_path, swings)
    write_sidecar(
        swings_path,
        name="swings",
        rows=writer.rows,
        columns=SWING_COLUMNS,
        date_range=writer.date_range("extreme_utc"),
        sources=sources,
        clock_quality=clock_meta(SWING_COLUMNS, {"published": writer.rows}, CLOCK_RULE),
        notes=NOTES,
        extra={"legs_per_asset": {a: len(swings[a]) for a in ASSETS}},
    )

    sessions_path = out / "swing_sessions.parquet"
    session_writer = write_sessions(sessions_path, series, swings, calendar)
    write_sidecar(
        sessions_path,
        name="swing_sessions",
        rows=session_writer.rows,
        columns=SESSION_COLUMNS,
        date_range=date_range,
        sources=sources,
        clock_quality=clock_meta(SESSION_COLUMNS, {"published": session_writer.rows}, CLOCK_RULE),
        notes=SESSION_NOTES + NOTES[1:5],
    )

    stats_path = out / "stats.json"
    stats = stats_of(series, swings, provenance, command)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True, default=str) + "\n")
    log(f"wrote {swings_path.name}, {sessions_path.name}, {stats_path.name} to {out}")
    return Outputs(swings_path, sessions_path, stats_path)


__all__ = ["Outputs", "Swing", "build", "stats_of", "swings_of"]
