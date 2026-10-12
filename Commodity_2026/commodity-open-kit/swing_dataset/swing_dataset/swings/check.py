"""Independent recomputation of the swing inventory, plus whole-file invariants.

The reference below re-derives bars and legs from ``designations.parquet``, ``mcx_daily.parquet``,
``mcx_sessions.parquet`` and ``mcx_quotes.parquet`` with pandas, without importing this lane's
``bars``, ``zigzag`` or ``build``. The walk is written in the sign form (``+1`` while an up leg
runs, ``-1`` while a down leg runs) rather than the branch form of ``zigzag.py``, so the two
share the rule and not the code. It then compares leg for leg with the written ``swings.parquet``.

The invariants: availability never before the extreme,
confirmation never before the extreme and its availability never before it, a leg never spanning
a roll without ``contract_changed``, the threshold respected to 1e-9, legs alternating and
chaining turn to extreme, and ``swing_sessions`` agreeing with the legs it counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..mcx.sessions import session_close_utc
from . import (
    ASSETS,
    BAR_DAILY,
    BAR_HOURLY,
    BAR_MIXED,
    DESIGNATION_HORIZON,
    DOWN,
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

TOL = 1e-9
SIDE_NAME = {1: UP, -1: DOWN}
#: Scalar columns of ``swings.parquet`` the reference reproduces and this module compares.
COMPARED = (
    "asset",
    "direction",
    "turn_price",
    "extreme_price",
    "move_pct",
    "sessions_spanned",
    "contract_id",
    "contract_changed",
    "bar",
    "threshold_pct",
    "clock_quality",
)
#: Timestamp columns, compared after converting the pandas value to a datetime.
COMPARED_INSTANTS = (
    "turn_utc",
    "extreme_utc",
    "available_utc",
    "confirm_utc",
    "confirm_available_utc",
)
#: Every column of ``swings.parquet`` is on one of the two lists.
FIELDS_COMPARED = len(COMPARED) + len(COMPARED_INSTANTS)


@dataclass
class CheckReport:
    recomputed: int = 0
    mismatches: list[dict] = field(default_factory=list)
    invariants: dict[str, dict] = field(default_factory=dict)
    fields_compared: int = FIELDS_COMPARED

    @property
    def ok(self) -> bool:
        return not self.mismatches and all(v.get("ok") for v in self.invariants.values())

    def render(self) -> str:
        head = f"swings check: recomputed={self.recomputed} fields={self.fields_compared}"
        lines = [f"{head} mismatches={len(self.mismatches)}"]
        for m in self.mismatches[:20]:
            lines.append(f"  MISMATCH {m}")
        for name, inv in sorted(self.invariants.items()):
            lines.append(f"  {'ok  ' if inv.get('ok') else 'FAIL'} {name}: {inv}")
        lines.append("  RESULT " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "name": "swings_check",
            "recomputed": self.recomputed,
            "fields_compared": self.fields_compared,
            "compared": list(COMPARED) + list(COMPARED_INSTANTS),
            "mismatches": self.mismatches,
            "invariants": self.invariants,
            "ok": self.ok,
        }


def _session_windows(mcx_dir: Path) -> tuple[dict, dict]:
    sessions = pd.read_parquet(mcx_dir / "mcx_sessions.parquet")
    trading = sessions[sessions["state"] != "closed"].sort_values("session_date")
    windows: dict = {}
    order: dict = {}
    for position, row in enumerate(trading.to_dict("records")):
        opening = row["morning_open_ist"] or row["evening_open_ist"]
        start = (
            None
            if opening is None
            else datetime.combine(row["session_date"], opening, tzinfo=IST).astimezone(UTC)
        )
        windows[row["session_date"]] = (start, session_close_utc(row))
        order[row["session_date"]] = position
    return windows, order


def reference_bars(labels_dir: Path, mcx_dir: Path, quotes_dir: Path, windows: dict) -> dict:
    """Per asset, the designated-contract bar list, rebuilt with pandas."""
    designations = pd.read_parquet(labels_dir / "designations.parquet")
    designations = designations[designations["horizon"] == DESIGNATION_HORIZON]
    daily = pd.read_parquet(mcx_dir / "mcx_daily.parquet")
    end = min(daily["session_date"].max(), designations["cutoff_date"].max())
    daily_key = {
        (r.asset, r.contract_id, r.session_date): (r.high, r.low) for r in daily.itertuples()
    }

    quotes = pd.read_parquet(
        quotes_dir / "mcx_quotes.parquet",
        columns=["available_utc", "asset", "contract_id", "ltp"],
    )
    ist = quotes["available_utc"].dt.tz_convert(IST)
    quotes = quotes.assign(day=ist.dt.date, hour=ist.dt.floor("h"))
    starts = quotes["day"].map(lambda d: windows.get(d, (None, None))[0])
    ends = quotes["day"].map(lambda d: windows.get(d, (None, None))[1])
    quotes = quotes[
        starts.notna() & (quotes["available_utc"] >= starts) & (quotes["available_utc"] <= ends)
    ]
    agg = (
        quotes.groupby(["asset", "contract_id", "day", "hour"])["ltp"]
        .agg(["max", "min"])
        .reset_index()
        .sort_values("hour")
    )
    hourly: dict = {}
    for key, group in agg.groupby(["asset", "contract_id", "day"]):
        hourly[key] = group

    out: dict = {}
    for asset in ASSETS:
        rows = designations[(designations["asset"] == asset) & (designations["cutoff_date"] <= end)]
        bars: list[dict] = []
        for row in rows.sort_values("cutoff_date").itertuples():
            day, contract = row.cutoff_date, row.contract_id
            if contract is None:
                continue
            group = hourly.get((asset, contract, day)) if day >= HOURLY_START else None
            if group is not None:
                bars.extend(
                    {
                        "day": day,
                        "contract": contract,
                        "start": item.hour.tz_convert(UTC).to_pydatetime(),
                        "high": float(item.max),
                        "low": float(item.min),
                        "kind": BAR_HOURLY,
                    }
                    for item in group.itertuples()
                )
                continue
            high, low = daily_key.get((asset, contract, day), (None, None))
            if high is None or pd.isna(high) or pd.isna(low):
                continue
            bars.append(
                {
                    "day": day,
                    "contract": contract,
                    "start": windows[day][0],
                    "high": float(high),
                    "low": float(low),
                    "kind": BAR_DAILY,
                }
            )
        out[asset] = bars
    return out


def reference_legs(
    asset: str, bars: list[dict], threshold_pct: float, windows: dict, order: dict
) -> list[dict]:
    """The rule in sign form: ``side`` is +1 while an up leg runs and -1 while a down leg runs."""
    step = threshold_pct / 100.0
    legs: list[dict] = []
    if not bars:
        return legs
    price = {1: "high", -1: "low"}
    extreme = {1: (0, bars[0]["high"]), -1: (0, bars[0]["low"])}
    side = 0
    turn = (0, bars[0]["high"])

    for i in range(1, len(bars)):
        bar = bars[i]
        if side == 0:
            hits = [
                candidate
                for candidate in (-1, 1)
                if -candidate * bar[price[candidate]]
                <= -candidate * extreme[-candidate][1] * (1 + candidate * step)
            ]
            if len(hits) == 2:
                hits = [-1] if extreme[1][0] <= extreme[-1][0] else [1]
            if hits:
                side = hits[0]
                turn = extreme[-side]
                extreme[side] = (i, bar[price[side]])
            else:
                for candidate in (-1, 1):
                    if candidate * bar[price[candidate]] > candidate * extreme[candidate][1]:
                        extreme[candidate] = (i, bar[price[candidate]])
            continue
        index, value = extreme[side]
        if side * bar[price[-side]] <= side * value * (1 - side * step):
            legs.append(
                _leg(asset, bars, side, turn, (index, value), i, threshold_pct, windows, order)
            )
            side, turn = -side, (index, value)
            extreme[side] = (i, bar[price[side]])
        elif side * bar[price[side]] > side * value:
            extreme[side] = (i, bar[price[side]])
    return legs


def _leg(asset, bars, side, turn, extreme, confirm_i, threshold_pct, windows, order) -> dict:
    turn_i, turn_p = turn
    ext_i, ext_p = extreme
    span = bars[turn_i : ext_i + 1]
    kinds = {b["kind"] for b in span}
    return {
        "asset": asset,
        "direction": SIDE_NAME[side],
        "turn_utc": bars[turn_i]["start"],
        "turn_price": turn_p,
        "extreme_utc": bars[ext_i]["start"],
        "extreme_price": ext_p,
        "move_pct": (ext_p - turn_p) / turn_p * 100,
        "sessions_spanned": order[bars[ext_i]["day"]] - order[bars[turn_i]["day"]] + 1,
        "contract_id": bars[ext_i]["contract"],
        "contract_changed": len({b["contract"] for b in span}) > 1,
        "bar": kinds.pop() if len(kinds) == 1 else BAR_MIXED,
        "threshold_pct": threshold_pct,
        "available_utc": windows[bars[ext_i]["day"]][1],
        "confirm_utc": bars[confirm_i]["start"],
        "confirm_available_utc": windows[bars[confirm_i]["day"]][1],
        "clock_quality": "published",
    }


def run_check(
    out: Path = LANE_BUILD,
    labels_dir: Path = LABELS_BUILD,
    mcx_dir: Path = MCX_BUILD,
    quotes_dir: Path = QUOTES_BUILD,
    report_path: Path | None = None,
) -> CheckReport:
    report = CheckReport()
    swings = pd.read_parquet(out / "swings.parquet")
    sessions = pd.read_parquet(out / "swing_sessions.parquet")

    windows, order = _session_windows(mcx_dir)
    bars = reference_bars(labels_dir, mcx_dir, quotes_dir, windows)
    for asset in ASSETS:
        theirs = reference_legs(asset, bars[asset], THRESHOLD_PCT[asset], windows, order)
        ours = swings[swings["asset"] == asset].sort_values("extreme_utc").reset_index(drop=True)
        report.recomputed += len(theirs)
        if len(ours) != len(theirs):
            report.mismatches.append(
                {"asset": asset, "field": "count", "ours": len(ours), "reference": len(theirs)}
            )
            continue
        for i, expected in enumerate(theirs):
            row = ours.iloc[i]
            for name in COMPARED:
                got, want = row[name], expected[name]
                if isinstance(want, float):
                    same = abs(float(got) - want) <= TOL * max(1.0, abs(want))
                else:
                    same = got == want
                if not same:
                    report.mismatches.append(
                        {"asset": asset, "leg": i, "field": name, "ours": got, "reference": want}
                    )
            for name in COMPARED_INSTANTS:
                if row[name].to_pydatetime() != expected[name]:
                    report.mismatches.append(
                        {
                            "asset": asset,
                            "leg": i,
                            "field": name,
                            "ours": str(row[name]),
                            "reference": str(expected[name]),
                        }
                    )

    report.invariants.update(invariants(swings, sessions))
    if report_path is None:
        report_path = out / "check.json"
    report_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str) + "\n"
    )
    return report


def invariants(swings: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    early = swings[swings["available_utc"] < swings["extreme_utc"]]
    out["availability_at_or_after_extreme"] = {"ok": early.empty, "rows": len(early)}
    backwards = swings[swings["extreme_utc"] <= swings["turn_utc"]]
    out["extreme_after_turn"] = {"ok": backwards.empty, "rows": len(backwards)}

    unconfirmed = swings[swings["confirm_utc"].isna() | swings["confirm_available_utc"].isna()]
    before_extreme = swings[swings["confirm_utc"] < swings["extreme_utc"]]
    out["confirmation_at_or_after_extreme"] = {
        "ok": unconfirmed.empty and before_extreme.empty,
        "null_rows": len(unconfirmed),
        "rows": len(before_extreme),
    }
    confirm_early = swings[swings["confirm_available_utc"] < swings["confirm_utc"]]
    lag = swings["confirm_available_utc"] - swings["available_utc"]
    out["confirm_availability_at_or_after_confirmation"] = {
        "ok": confirm_early.empty and bool((lag >= pd.Timedelta(0)).all()),
        "rows": len(confirm_early),
        "rows_available_before_extreme_availability": int((lag < pd.Timedelta(0)).sum()),
    }

    threshold = swings["asset"].map(THRESHOLD_PCT)
    short = swings[swings["move_pct"].abs() < threshold - TOL]
    out["threshold_respected"] = {
        "ok": short.empty,
        "rows": len(short),
        "min_abs_move_over_threshold": float((swings["move_pct"].abs() - threshold).min())
        if len(swings)
        else None,
    }

    sign_ok = (
        ((swings["direction"] == UP) & (swings["move_pct"] > 0))
        | ((swings["direction"] == DOWN) & (swings["move_pct"] < 0))
    ).all()
    out["direction_matches_sign"] = {"ok": bool(sign_ok)}

    chain: dict[str, object] = {"ok": True, "assets": {}}
    for asset in ASSETS:
        rows = swings[swings["asset"] == asset].sort_values("extreme_utc").reset_index(drop=True)
        alternating = all(
            rows["direction"].iloc[i] != rows["direction"].iloc[i + 1] for i in range(len(rows) - 1)
        )
        chained = all(
            rows["turn_utc"].iloc[i + 1] == rows["extreme_utc"].iloc[i]
            and abs(rows["turn_price"].iloc[i + 1] - rows["extreme_price"].iloc[i]) <= TOL
            for i in range(len(rows) - 1)
        )
        chain["assets"][asset] = {"alternating": bool(alternating), "chained": bool(chained)}
        chain["ok"] = chain["ok"] and alternating and chained
    out["legs_alternate_and_chain"] = chain
    out["rolled_legs_flagged"] = rolled_legs_flagged(swings, sessions)

    counted: dict[str, object] = {"ok": True, "assets": {}}
    for asset in ASSETS:
        legs = swings[swings["asset"] == asset]
        rows = sessions[sessions["asset"] == asset]
        same_turns = int(rows["turn_count"].sum()) == len(legs)
        same_extremes = int(rows["extreme_count"].sum()) == len(legs)
        counted["assets"][asset] = {"turns": same_turns, "extremes": same_extremes, "legs": len(legs)}
        counted["ok"] = counted["ok"] and same_turns and same_extremes
    out["session_counts_match_legs"] = counted

    out["one_session_row_per_asset_day"] = {
        "ok": not bool(sessions.duplicated(["asset", "session_date"]).any()),
        "rows": len(sessions),
    }
    cutoff_ist = sessions["cutoff_utc"].dt.tz_convert(IST)
    out["cutoff_is_0830_ist"] = {
        "ok": bool(
            (
                (cutoff_ist.dt.hour == PRE_OPEN_IST.hour)
                & (cutoff_ist.dt.minute == PRE_OPEN_IST.minute)
            ).all()
        )
    }
    out["clock_quality_published"] = {
        "ok": bool((swings["clock_quality"] == "published").all())
        and bool((sessions["clock_quality"] == "published").all())
    }
    return out


def rolled_legs_flagged(swings: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    """``contract_changed`` must equal "the designation rolled inside the leg's own span".

    The designated contract per session comes from ``swing_sessions.parquet``, not from the leg
    rows, so this tests the two files against each other.
    """
    wrong: list[dict] = []
    rolled = 0
    for asset in ASSETS:
        rows = sessions[(sessions["asset"] == asset) & (sessions["bar"] != "none")]
        rows = rows.sort_values("session_date").reset_index(drop=True)
        days = list(rows["session_date"])
        contracts = list(rows["contract_id"])
        position = {day: i for i, day in enumerate(days)}
        for leg in swings[swings["asset"] == asset].itertuples():
            first = position[leg.turn_utc.tz_convert(IST).date()]
            last = position[leg.extreme_utc.tz_convert(IST).date()]
            changed = len(set(contracts[first : last + 1])) > 1
            rolled += int(changed)
            if changed != bool(leg.contract_changed):
                wrong.append(
                    {
                        "asset": asset,
                        "extreme_utc": str(leg.extreme_utc),
                        "flag": bool(leg.contract_changed),
                        "sessions_say": changed,
                    }
                )
    return {"ok": not wrong, "rolled_legs": rolled, "wrong": wrong[:10]}
