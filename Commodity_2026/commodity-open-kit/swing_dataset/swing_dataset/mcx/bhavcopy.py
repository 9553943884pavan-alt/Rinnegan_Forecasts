"""MCX Bhav Copy: fetch per date, parse, normalize to one row per contract-day.

Source path (found 2026-09-15 by reading the site's ``BhavCopy.js``): the page
``/market-data/bhavcopy`` calls
``GET /market-data/bhavcopy/GetDateWiseBhavCopy?InstrumentName=FUTCOM&fromDate=DD/MM/YYYY``
and renders the JSON; the CSV/XLS buttons export that same JSON client-side, so
the JSON response is the exchange file. The envelope is
``{"IsSuccess", "Message", "Data", "Metadata"}``; a non-session date answers
``IsSuccess=false, Message="No data found.", Data=null``.

Two traps are handled here and must stay handled:

- an unparseable ``fromDate`` silently returns the latest session, so every row's
  date is checked against the requested date;
- untraded contracts print ``Open=High=Low=0`` with a real ``Close``: the
  ``Close`` column is the exchange's daily settlement price, not a trade.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pyarrow as pa

from .. import common
from . import ASSET_BY_SYMBOL, LANE
from .client import ORIGIN, McxClient

PAGE_URL = f"{ORIGIN}/market-data/bhavcopy"
API_URL = f"{ORIGIN}/market-data/bhavcopy/GetDateWiseBhavCopy"
INSTRUMENT = "FUTCOM"
RAW_SUBDIR = "bhavcopy"
NO_DATA_MESSAGE = "No data found."

#: The exact JSON row shape served for every date probed from 2021 to 2026.
ROW_KEYS = frozenset(
    {
        "sLTT",
        "Date",
        "Symbol",
        "ExpiryDate",
        "Open",
        "High",
        "Low",
        "Close",
        "PreviousClose",
        "Volume",
        "VolumeInThousands",
        "Value",
        "OpenInterest",
        "DateDisplay",
        "InstrumentName",
        "StrikePrice",
        "OptionType",
    }
)
ENVELOPE_KEYS = frozenset({"IsSuccess", "Message", "Data", "Metadata"})

_VOLUME_UNITS = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+([A-Za-z]+)\s*$")
_EXPIRY = re.compile(r"^(\d{2})([A-Z]{3})(\d{4})$")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

#: Output schema of mcx_daily.parquet, in column order.
DAILY_SCHEMA = pa.schema(
    [
        ("session_date", pa.date32()),
        ("symbol", pa.string()),
        ("asset", pa.string()),
        ("expiry_date", pa.date32()),
        ("contract_id", pa.string()),
        ("instrument", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("settlement", pa.float64()),
        ("previous_close", pa.float64()),
        ("volume_lots", pa.int64()),
        ("volume_units", pa.float64()),
        ("volume_unit", pa.string()),
        ("value_inr", pa.float64()),
        ("open_interest", pa.int64()),
        ("open_interest_change", pa.int64()),
        ("source_file", pa.string()),
        ("clock_quality", pa.string()),
        ("available_utc", pa.timestamp("us", tz="UTC")),
    ]
)

COLUMN_MEANINGS = {
    "session_date": "MCX trade date (Asia/Kolkata civil date); Bhav Copy `Date`/`DateDisplay`",
    "symbol": "Bhav Copy `Symbol`, trailing padding stripped (GOLD, GOLDM, SILVER, ...)",
    "asset": "dataset asset key for the four required products (gold/silver/copper/crude), null for other symbols",
    "expiry_date": "Bhav Copy `ExpiryDate` (DDMONYYYY) as a date",
    "contract_id": "feed contract id `mcx:<SYMBOL><DD><MON><YY>FUT`",
    "instrument": "Bhav Copy `InstrumentName` (FUTCOM)",
    "open": "Bhav Copy `Open`, INR per quotation unit; null when the exchange prints 0 (no trade)",
    "high": "Bhav Copy `High`; null when the exchange prints 0 (no trade)",
    "low": "Bhav Copy `Low`; null when the exchange prints 0 (no trade)",
    "close": "Bhav Copy `Close` for contract-days with volume_lots > 0, null otherwise. MCX publishes one closing price and it is the daily settlement price, not the last trade",
    "settlement": "Bhav Copy `Close`: the exchange's daily settlement price, present for every listed contract including untraded ones",
    "previous_close": "Bhav Copy `Previous Close`: the prior session's settlement as the exchange restates it",
    "volume_lots": "Bhav Copy `Volume(Lots)`",
    "volume_units": "Bhav Copy `Volume(In 000's)` times 1000, in `volume_unit`",
    "volume_unit": "unit label carried in `VolumeInThousands` (GRMS, KGS, BBL, ...)",
    "value_inr": "Bhav Copy `Value(Lacs)` times 100000, rounded to the rupee (the file carries two decimals of lakhs)",
    "open_interest": "Bhav Copy `Open Interest(Lots)`",
    "open_interest_change": "open_interest minus the same contract's open_interest on the immediately preceding session present in this file; null when the contract has no row on that session (derived, not published)",
    "source_file": "raw capture path relative to build/mcx/raw/",
    "clock_quality": "published: the exchange publishes the file the evening of the trade date; we did not capture it live",
    "available_utc": "the session's close instant in UTC from mcx_sessions (evening close, or the morning close on morning-only days, or the special session's close): the earliest instant the exchange can have published this row",
}


class BhavcopyFormatError(ValueError):
    """The response does not have the shape this parser was written against."""


@dataclass(frozen=True, slots=True)
class ParsedDay:
    session_date: date
    has_data: bool
    message: str
    rows: list[dict]


def api_url(day: date, instrument: str = INSTRUMENT) -> str:
    return f"{API_URL}?InstrumentName={instrument}&fromDate={day.strftime('%d/%m/%Y')}"


def raw_relative_path(day: date) -> str:
    return f"{RAW_SUBDIR}/{day.isoformat()}.json"


def raw_path(day: date, lane_root: Path | None = None) -> Path:
    root = lane_root or common.lane_root(LANE)
    return root / "raw" / raw_relative_path(day)


def fetch_day(client: McxClient, day: date):
    return client.capture(
        api_url(day),
        raw_relative_path(day),
        headers={
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": PAGE_URL,
        },
    )


# -- parsing ----------------------------------------------------------------


def parse_expiry(text: str) -> date:
    match = _EXPIRY.match(text.strip())
    if not match:
        raise BhavcopyFormatError(f"unrecognised ExpiryDate {text!r}")
    day, mon, year = match.groups()
    if mon not in _MONTHS:
        raise BhavcopyFormatError(f"unrecognised month in ExpiryDate {text!r}")
    return date(int(year), _MONTHS.index(mon) + 1, int(day))


def contract_id(symbol: str, expiry: date) -> str:
    return f"mcx:{symbol}{expiry.day:02d}{_MONTHS[expiry.month - 1]}{expiry.year % 100:02d}FUT"


def parse_volume_units(text: str) -> tuple[float, str]:
    match = _VOLUME_UNITS.match(text)
    if not match:
        raise BhavcopyFormatError(f"unrecognised VolumeInThousands {text!r}")
    return float(match.group(1)) * 1000.0, match.group(2)


def _number(row: dict, key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BhavcopyFormatError(f"{key} is {type(value).__name__} {value!r}, expected a number")
    return float(value)


def _integer(row: dict, key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise BhavcopyFormatError(f"{key} is {type(value).__name__} {value!r}, expected an integer")
    return value


def parse_bhavcopy(body: bytes, expected_date: date) -> ParsedDay:
    """Validate the envelope and every row against the shape and the requested date."""
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise BhavcopyFormatError(f"not JSON: {exc}; head={body[:120]!r}") from exc
    if not isinstance(payload, dict) or set(payload) != ENVELOPE_KEYS:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise BhavcopyFormatError(f"envelope keys {keys} != {sorted(ENVELOPE_KEYS)}")
    data = payload["Data"]
    message = str(payload["Message"])
    if data is None:
        if payload["IsSuccess"] is False and message == NO_DATA_MESSAGE:
            return ParsedDay(expected_date, False, message, [])
        raise BhavcopyFormatError(
            f"Data is null with IsSuccess={payload['IsSuccess']!r} Message={message!r}"
        )
    if not isinstance(data, list):
        raise BhavcopyFormatError(f"Data is {type(data).__name__}, expected a list")
    if payload["IsSuccess"] is not True:
        raise BhavcopyFormatError(f"Data present but IsSuccess={payload['IsSuccess']!r}")
    rows: list[dict] = []
    expected_display = expected_date.strftime("%d %b %Y")
    expected_us = expected_date.strftime("%m/%d/%Y")
    for index, row in enumerate(data):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            keys = sorted(row) if isinstance(row, dict) else type(row).__name__
            raise BhavcopyFormatError(f"row {index} keys {keys} != expected {sorted(ROW_KEYS)}")
        if row["DateDisplay"] != expected_display or row["Date"] != expected_us:
            raise BhavcopyFormatError(
                f"row {index} dated {row['DateDisplay']!r}/{row['Date']!r}, requested {expected_date.isoformat()}"
            )
        rows.append(row)
    return ParsedDay(expected_date, True, message, rows)


def normalize_row(row: dict, session_date: date, source_file: str) -> dict:
    instrument = str(row["InstrumentName"]).strip()
    if instrument != INSTRUMENT:
        raise BhavcopyFormatError(f"InstrumentName {instrument!r}, expected {INSTRUMENT}")
    if row["OptionType"] != "-" or _number(row, "StrikePrice") != 0.0:
        raise BhavcopyFormatError(
            f"futures row carries OptionType={row['OptionType']!r} StrikePrice={row['StrikePrice']!r}"
        )
    symbol = str(row["Symbol"]).strip()
    if not symbol or not symbol.isalnum():
        raise BhavcopyFormatError(f"unrecognised Symbol {row['Symbol']!r}")
    expiry = parse_expiry(str(row["ExpiryDate"]))
    volume_lots = _integer(row, "Volume")
    open_interest = _integer(row, "OpenInterest")
    volume_units, volume_unit = parse_volume_units(str(row["VolumeInThousands"]))
    settlement = _number(row, "Close")
    ohl = {key: _number(row, key.capitalize()) for key in ("open", "high", "low")}
    return {
        "session_date": session_date,
        "symbol": symbol,
        "asset": ASSET_BY_SYMBOL.get(symbol),
        "expiry_date": expiry,
        "contract_id": contract_id(symbol, expiry),
        "instrument": instrument,
        "open": ohl["open"] if ohl["open"] != 0.0 else None,
        "high": ohl["high"] if ohl["high"] != 0.0 else None,
        "low": ohl["low"] if ohl["low"] != 0.0 else None,
        "close": settlement if volume_lots > 0 else None,
        "settlement": settlement,
        "previous_close": _number(row, "PreviousClose"),
        "volume_lots": volume_lots,
        "volume_units": volume_units,
        "volume_unit": volume_unit,
        "value_inr": round(_number(row, "Value") * 100_000.0),
        "open_interest": open_interest,
        "open_interest_change": None,
        "source_file": source_file,
        "clock_quality": "published",
    }


def normalize_day(parsed: ParsedDay, source_file: str) -> list[dict]:
    records = [normalize_row(row, parsed.session_date, source_file) for row in parsed.rows]
    seen: set[str] = set()
    for record in records:
        if record["contract_id"] in seen:
            raise BhavcopyFormatError(
                f"duplicate contract {record['contract_id']} on {parsed.session_date}"
            )
        seen.add(record["contract_id"])
    return records


# -- capture bookkeeping ------------------------------------------------------


def read_index(lane_root: Path) -> dict[str, dict[str, str]]:
    """Latest INDEX.csv row per path."""
    index = lane_root / "raw" / "INDEX.csv"
    latest: dict[str, dict[str, str]] = {}
    if not index.exists():
        return latest
    with index.open(newline="") as f:
        for row in csv.DictReader(f):
            latest[row["path"]] = row
    return latest


def captured_day(day: date, lane_root: Path, index: dict[str, dict[str, str]]) -> ParsedDay | None:
    """The parsed capture for ``day`` when its bytes match the INDEX, else None."""
    path = raw_path(day, lane_root)
    entry = index.get(str(path))
    if entry is None or not path.exists():
        return None
    body = path.read_bytes()
    if common.sha256_bytes(body) != entry["sha256"]:
        return None
    return parse_bhavcopy(body, day)


def days_between(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def fetch_range(
    client: McxClient,
    start: date,
    end: date,
    *,
    lane_root: Path | None = None,
    refetch_empty_within_days: int = 5,
    log=print,
) -> dict[str, int]:
    """Capture every calendar day in [start, end], skipping days already captured.

    A day is skipped when its raw file exists and its sha256 matches the latest
    INDEX row. Empty ("No data found.") captures within the last
    ``refetch_empty_within_days`` days are fetched again, since the exchange
    publishes the file in the evening and a same-day capture may predate it.
    """
    root = lane_root or common.lane_root(LANE)
    index = read_index(root)
    today = common.now_utc().date()
    counts = {"skipped": 0, "fetched": 0, "sessions": 0, "empty": 0, "errors": 0}
    for day in days_between(start, end):
        existing = captured_day(day, root, index)
        if existing is not None and (
            existing.has_data or (today - day).days > refetch_empty_within_days
        ):
            counts["skipped"] += 1
            continue
        try:
            capture = fetch_day(client, day)
            parsed = parse_bhavcopy(capture.body, day)
        except Exception as exc:  # noqa: BLE001 - logged and counted, the run continues
            counts["errors"] += 1
            log(f"{day} ERROR {type(exc).__name__}: {exc}")
            continue
        counts["fetched"] += 1
        if parsed.has_data:
            counts["sessions"] += 1
            log(
                f"{day} ok rows={len(parsed.rows)} bytes={len(capture.body)} sha256={capture.sha256[:12]}"
            )
        else:
            counts["empty"] += 1
            log(f"{day} empty ({parsed.message}) bytes={len(capture.body)}")
    return counts


# -- assembly ------------------------------------------------------------------


def load_captures(lane_root: Path | None = None) -> tuple[list[dict], list[ParsedDay]]:
    """Parse every indexed capture under raw/bhavcopy; returns (records, empty days)."""
    root = lane_root or common.lane_root(LANE)
    index = read_index(root)
    raw_dir = root / "raw" / RAW_SUBDIR
    records: list[dict] = []
    empties: list[ParsedDay] = []
    for path in sorted(raw_dir.glob("*.json")):
        day = date.fromisoformat(path.stem)
        entry = index.get(str(path))
        if entry is None:
            raise RuntimeError(f"{path} is not in INDEX.csv; refuse to use an unindexed capture")
        body = path.read_bytes()
        if common.sha256_bytes(body) != entry["sha256"]:
            raise RuntimeError(f"{path} sha256 differs from its INDEX row")
        parsed = parse_bhavcopy(body, day)
        if parsed.has_data:
            records.extend(normalize_day(parsed, f"{RAW_SUBDIR}/{path.name}"))
        else:
            empties.append(parsed)
    return records, empties


def add_open_interest_change(frame: pd.DataFrame) -> pd.DataFrame:
    """OI change against the immediately preceding session present in the frame."""
    if frame.empty:
        return frame
    sessions = sorted(frame["session_date"].unique())
    previous_session = {cur: prev for prev, cur in pairwise(sessions)}
    frame = frame.sort_values(["contract_id", "session_date"]).reset_index(drop=True)
    prev_date = frame.groupby("contract_id")["session_date"].shift(1)
    prev_oi = frame.groupby("contract_id")["open_interest"].shift(1)
    expected_prev = frame["session_date"].map(previous_session)
    contiguous = prev_date.notna() & (prev_date == expected_prev)
    change = (frame["open_interest"] - prev_oi).where(contiguous)
    frame["open_interest_change"] = change.astype("Int64")
    return frame.sort_values(["session_date", "symbol", "expiry_date"]).reset_index(drop=True)


def session_closes() -> dict[date, datetime]:
    """session_date -> close instant (UTC) for every calendar date with a session."""
    from .sessions import build_rows, session_close_utc

    closes = {}
    for row in build_rows():
        close = session_close_utc(row)
        if close is not None:
            closes[row["session_date"]] = close
    return closes


def add_availability(frame: pd.DataFrame, closes: dict[date, datetime]) -> pd.DataFrame:
    """available_utc = the session's close instant; a Bhav Copy date the calendar has no session for is an error."""
    if frame.empty:
        frame["available_utc"] = pd.Series(dtype="datetime64[us, UTC]")
        return frame
    missing = sorted(set(frame["session_date"].unique()) - set(closes))
    if missing:
        raise RuntimeError(
            f"Bhav Copy has rows on {[d.isoformat() for d in missing[:5]]} but mcx_sessions has no session close for them"
        )
    frame["available_utc"] = pd.to_datetime(frame["session_date"].map(closes), utc=True).astype(
        "datetime64[us, UTC]"
    )
    return frame


def to_table(records: list[dict], closes: dict[date, datetime] | None = None) -> pa.Table:
    frame = pd.DataFrame.from_records(
        records, columns=[f.name for f in DAILY_SCHEMA if f.name != "available_utc"]
    )
    frame = add_open_interest_change(frame)
    frame = add_availability(frame, closes if closes is not None else session_closes())
    for column in ("volume_lots", "open_interest"):
        frame[column] = frame[column].astype("int64")
    frame["open_interest_change"] = frame["open_interest_change"].astype("Int64")
    return pa.Table.from_pandas(frame, schema=DAILY_SCHEMA, preserve_index=False)


def build_command(argv: list[str]) -> str:
    return "python -m swing_dataset.mcx " + " ".join(argv)


def write_daily(
    table: pa.Table,
    empties: list[ParsedDay],
    *,
    lane_root: Path | None = None,
    command: str,
    notes: list[str] | None = None,
) -> Path:
    import pyarrow.parquet as pq

    root = lane_root or common.lane_root(LANE)
    out = root / "mcx_daily.parquet"
    pq.write_table(table, out, compression="zstd")
    index = read_index(root)
    frame = table.to_pandas()
    sources = []
    for source_file in sorted(frame["source_file"].unique()):
        entry = index[str(root / "raw" / source_file)]
        sources.append(
            {
                "url": entry["url"],
                "fetched_at_utc": entry["fetched_at_utc"],
                "sha256": entry["sha256"],
                "bytes": int(entry["bytes"]),
                "local_path": common.sidecar_path(entry["path"]),
            }
        )
    dates = sorted(frame["session_date"].unique())
    meta = {
        "name": "mcx_daily",
        "rows": table.num_rows,
        "columns": [
            {"name": f.name, "type": str(f.type), "meaning": COLUMN_MEANINGS[f.name]}
            for f in DAILY_SCHEMA
        ],
        "date_range": {
            "start": dates[0].isoformat() if dates else None,
            "end": dates[-1].isoformat() if dates else None,
            "sessions": len(dates),
            "calendar_days_answered_no_data": len(empties),
        },
        "sources": sources,
        "clock_quality": {
            "file": "published",
            "row_column": "clock_quality",
            "rows": {"published": table.num_rows},
            "clocks": {"published": "available_utc"},
            "rule": (
                "The exchange publishes the Bhav Copy after the session closes and we did not capture it live, "
                "so every row is published with unknown latency. available_utc is the session's close instant "
                "from mcx_sessions (evening close 23:30 or 23:55 IST per the hours circular in force, the 17:00 "
                "morning close on morning-only days, the Muhurat or budget-session close where one applies); "
                "for a session whose evening close no retained circular states, the later published close "
                "(23:55 IST) is used so the instant is never earlier than the true close."
            ),
        },
        "build": {**common.git_identity(Path(__file__).resolve().parent), "command": command},
        "notes": [
            "One row per (session_date, contract_id) for every FUTCOM contract the exchange listed that day, all commodities, whole curve.",
            "Units are native: INR per MCX quotation unit (GOLD per 10 g, SILVER per kg, COPPER per kg, CRUDEOIL per barrel; see mcx_contract_specs.json).",
            "The Bhav Copy has a single closing price column (`Close`); it is the daily settlement price. `settlement` carries it for every row, `close` only where the contract traded.",
            "Zero `Open`/`High`/`Low` in the exchange file mean no trade and are stored as null, never as 0.",
            "`open_interest_change` is derived here from consecutive sessions; the exchange file does not publish it.",
            *(notes or []),
        ],
    }
    common.write_meta(out, meta)
    return out
