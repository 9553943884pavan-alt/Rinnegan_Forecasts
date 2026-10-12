"""The price path a signal is graded against: one contract, tick level where we have ticks.

A signal names an asset, an instant and a direction; the scorer needs the price at that instant
and the order in which the price later touched the barriers. Both come from one contract: the one
designated for the signal's session in ``labels/designations.parquet`` at horizon 5, which is the
series the swing inventory is built on. The path follows that contract to the end of the window,
so a roll inside the window never injects a calendar spread.

Resolution. From 2026-06-22 the production feed's quotes give `ltp` on an exact capture clock, and
a session with at least ``MIN_TICKS_PER_SESSION`` of them is walked tick by tick, which settles the
order of two barriers touched on the same day. Before that there are only session bars, where the
order inside the bar is unknowable; the adverse barrier is taken first and the row is flagged
``ambiguous``. Availability is respected everywhere: a settlement enters at its publication
instant, a tick at its capture instant.

The same walk measures the move to the swing's end (rules 2.6 section 3): the favorable extreme
the price reached before it first turned back a full threshold from its running favorable
extreme, or before the window ended. It walks exactly the prices the barriers are checked on, and
takes a session bar favorable side first, since the order inside it is unknowable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import ASSETS, PCT_TOLERANCE

IST = ZoneInfo("Asia/Kolkata")
PREOPEN_IST = time(8, 30)
DEFAULT_OPEN_IST = time(9, 0)
DEFAULT_CLOSE_IST = time(23, 30)
MIN_TICKS_PER_SESSION = 5
DESIGNATION_HORIZON = 5
DAY_NS = 86_400_000_000_000


def ist_instant(session_date: date, clock: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(session_date, clock, tzinfo=IST)).tz_convert("UTC")


def _ns(values) -> np.ndarray:
    return (
        pd.to_datetime(pd.Series(values), utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
    )


def ns_of(value) -> int:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value)


def utc_of(ns: int | None) -> pd.Timestamp | None:
    return None if ns is None else pd.Timestamp(int(ns), tz="UTC")


@dataclass(frozen=True, slots=True)
class PathResult:
    """One signal's price path: the entry price, the barrier touches and the excursions."""

    p0: float
    price_source: str
    price_available_utc: pd.Timestamp
    contract_id: str
    window_start_utc: pd.Timestamp
    window_end_utc: pd.Timestamp
    window_complete: bool
    sessions_in_window: int
    bars: str
    mfe_pct: float
    mae_pct: float
    touches: dict[str, tuple[int, int] | None]
    ambiguous: bool
    sessions_without_data: int
    #: The move to the swing's end, in percent of the entry price; NaN unless asked for.
    swing_end_pct: float = float("nan")

    def touched(self, name: str) -> bool:
        return self.touches.get(name) is not None

    def first(self, name: str, before: str) -> bool:
        """``name`` was touched and ``before`` was not touched earlier."""
        a = self.touches.get(name)
        if a is None:
            return False
        b = self.touches.get(before)
        return b is None or a < b

    def decided(self, names: tuple[str, ...]) -> bool:
        return any(self.touches.get(n) is not None for n in names)


class PriceSeries:
    """Designated-contract prices for the four assets, with point-in-time lookups."""

    def __init__(
        self,
        daily: pd.DataFrame,
        designations: pd.DataFrame,
        sessions: pd.DataFrame | None = None,
        quotes: pd.DataFrame | None = None,
    ) -> None:
        self._sessions: dict[str, dict[str, np.ndarray]] = {}
        self._grid: dict[str, dict] = {}
        self._ticks: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._frames: dict[str, pd.DataFrame] = {}
        hours = _session_hours(sessions)
        design = _designation_map(designations)
        for asset, rows in daily.groupby("asset", sort=True):
            if asset not in ASSETS:
                continue
            self._build_asset(str(asset), rows, hours, design.get(str(asset), {}))
        if quotes is not None and len(quotes):
            self._build_ticks(quotes)

    # ------------------------------------------------------------------ construction

    def _build_asset(
        self, asset: str, rows: pd.DataFrame, hours: dict, design: dict[date, str]
    ) -> None:
        session_dates = np.array(sorted(rows["session_date"].unique()))
        contracts = np.array(sorted(rows["contract_id"].unique()))
        col = {c: i for i, c in enumerate(contracts)}

        def grid(column: str) -> np.ndarray:
            wide = rows.pivot_table(
                index="session_date", columns="contract_id", values=column, aggfunc="last"
            )
            return wide.reindex(index=session_dates, columns=contracts).to_numpy(dtype=float)

        avail = rows.pivot_table(
            index="session_date", columns="contract_id", values="available_utc", aggfunc="last"
        ).reindex(index=session_dates, columns=contracts)
        avail_ns = np.where(
            avail.notna().to_numpy(),
            avail.astype("datetime64[ns, UTC]").astype("int64").to_numpy(),
            -1,
        )
        preopen, open_, close = [], [], []
        for d in session_dates:
            h = hours.get(d)
            open_ist = h[0] if h else DEFAULT_OPEN_IST
            close_ist = h[1] if h else DEFAULT_CLOSE_IST
            preopen.append(ns_of(ist_instant(d, PREOPEN_IST)))
            open_.append(ns_of(ist_instant(d, open_ist)))
            close.append(ns_of(ist_instant(d, close_ist)))
        contract_of = np.array(
            [design.get(d, _fallback_contract(rows, d)) for d in session_dates], dtype=object
        )
        self._sessions[asset] = {
            "session_date": session_dates,
            "preopen_ns": np.array(preopen, dtype="int64"),
            "open_ns": np.array(open_, dtype="int64"),
            "close_ns": np.array(close, dtype="int64"),
            "contract_id": contract_of,
        }
        self._grid[asset] = {
            "contracts": contracts,
            "col": col,
            "high": grid("high"),
            "low": grid("low"),
            "settlement": grid("settlement"),
            "available_ns": avail_ns,
        }

    def _build_ticks(self, quotes: pd.DataFrame) -> None:
        frame = quotes[["asset", "contract_id", "available_utc", "ltp"]].dropna()
        frame = frame[frame["asset"].isin(ASSETS)]
        for (asset, contract), group in frame.groupby(["asset", "contract_id"], sort=False):
            g = group.sort_values("available_utc")
            self._ticks[(str(asset), str(contract))] = (
                _ns(g["available_utc"]),
                g["ltp"].to_numpy(dtype=float),
            )

    @classmethod
    def from_paths(
        cls,
        mcx_daily: Path,
        designations: Path,
        mcx_sessions: Path | None = None,
        quotes: Path | None = None,
    ) -> PriceSeries:
        daily = pq.read_table(
            mcx_daily,
            columns=[
                "session_date",
                "asset",
                "contract_id",
                "high",
                "low",
                "close",
                "settlement",
                "volume_lots",
                "available_utc",
            ],
        ).to_pandas()
        design = pq.read_table(
            designations, columns=["asset", "cutoff_date", "horizon", "contract_id"]
        ).to_pandas()
        sessions = (
            pq.read_table(
                mcx_sessions,
                columns=["session_date", "state", "morning_open_ist", "evening_close_ist"],
            ).to_pandas()
            if mcx_sessions is not None and Path(mcx_sessions).exists()
            else None
        )
        ticks = None
        if quotes is not None and Path(quotes).exists():
            ticks = pq.read_table(
                quotes, columns=["asset", "contract_id", "available_utc", "ltp"]
            ).to_pandas()
        return cls(daily, design, sessions, ticks)

    # ------------------------------------------------------------------ sessions

    def assets(self) -> list[str]:
        return sorted(self._sessions)

    def session_dates(self, asset: str) -> np.ndarray:
        return self._sessions[asset]["session_date"]

    def session_bounds(self, asset: str, pos: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        s = self._sessions[asset]
        return utc_of(s["preopen_ns"][pos]), utc_of(s["close_ns"][pos])

    def sessions_between(self, asset: str, start, end) -> np.ndarray:
        """Session dates of the asset inside the closed instant range [start, end]."""
        s = self._sessions[asset]
        lo, hi = ns_of(start), ns_of(end)
        keep = (s["preopen_ns"] >= lo) & (s["close_ns"] <= hi)
        return s["session_date"][keep]

    def session_pos(self, asset: str, when) -> int | None:
        """Position of the session whose pre-open to close range contains ``when``."""
        s = self._sessions.get(asset)
        if s is None:
            return None
        t = ns_of(when)
        i = int(np.searchsorted(s["preopen_ns"], t, side="right")) - 1
        if i < 0:
            return None
        return i if t <= s["close_ns"][i] else None

    def last_session_pos(self, asset: str, when) -> int:
        """Position of the last session whose pre-open is at or before ``when``; -1 if none."""
        s = self._sessions.get(asset)
        if s is None:
            return -1
        return int(np.searchsorted(s["preopen_ns"], ns_of(when), side="right")) - 1

    def designated_contract(self, asset: str, when) -> str | None:
        pos = self.session_pos(asset, when)
        if pos is None:
            pos = self.last_session_pos(asset, when)
        if pos < 0:
            return None
        value = self._sessions[asset]["contract_id"][pos]
        return None if value is None else str(value)

    def window(self, asset: str, when, horizon: int) -> tuple[pd.Timestamp, bool, int, int, int]:
        """The horizon window: the ``horizon`` sessions that close strictly after ``when``.

        Returns the end instant, whether all of them exist in the data, how many do, and the
        first and last session positions covered.
        """
        s = self._sessions[asset]
        t = ns_of(when)
        first = int(np.searchsorted(s["close_ns"], t, side="right"))
        last = first + horizon - 1
        n = len(s["close_ns"])
        complete = last < n
        capped = min(last, n - 1)
        if first >= n:
            return utc_of(int(s["close_ns"][-1])), False, 0, n, n - 1
        return (
            utc_of(int(s["close_ns"][capped])),
            complete,
            capped - first + 1,
            first,
            capped,
        )

    # ------------------------------------------------------------------ prices

    def price_at(self, asset: str, contract: str, when) -> tuple[float, str, pd.Timestamp] | None:
        """The last price of ``contract`` available at or before ``when``.

        A settlement enters at its publication instant and a tick at its capture instant; the
        later of the two wins, so a pre-open signal sees the previous settlement and an intraday
        signal sees the last trade.
        """
        t = ns_of(when)
        best: tuple[int, float, str] | None = None
        ticks = self._ticks.get((asset, contract))
        if ticks is not None:
            i = int(np.searchsorted(ticks[0], t, side="right")) - 1
            if i >= 0:
                best = (int(ticks[0][i]), float(ticks[1][i]), "tick")
        grid = self._grid.get(asset)
        if grid is not None and contract in grid["col"]:
            c = grid["col"][contract]
            avail = grid["available_ns"][:, c]
            known = np.flatnonzero((avail >= 0) & (avail <= t))
            for k in known[::-1]:
                price = grid["settlement"][k, c]
                if np.isfinite(price) and price > 0:
                    if best is None or int(avail[k]) > best[0]:
                        best = (int(avail[k]), float(price), "settlement")
                    break
        if best is None or not np.isfinite(best[1]) or best[1] <= 0:
            return None
        return best[1], best[2], utc_of(best[0])

    def _session_extremes(
        self, asset: str, contract: str, pos: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The session's ticks with their running maximum and minimum, cached."""
        key = (asset, contract, pos)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ticks = self._ticks.get((asset, contract))
        if ticks is None:
            return None
        s = self._sessions[asset]
        lo = int(s["close_ns"][pos - 1]) if pos > 0 else int(s["open_ns"][pos]) - DAY_NS
        hi = int(s["close_ns"][pos])
        a = int(np.searchsorted(ticks[0], lo, side="right"))
        b = int(np.searchsorted(ticks[0], hi, side="right"))
        if b - a < MIN_TICKS_PER_SESSION:
            return None
        prices = ticks[1][a:b]
        value = (np.maximum.accumulate(prices), np.minimum.accumulate(prices), prices)
        self._cache[key] = value
        return value

    def _tick_slice(
        self, asset: str, contract: str, pos: int, after_ns: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The ticks of session ``pos`` strictly after ``after_ns``, with their running maximum
        and minimum; None when fewer than ``MIN_TICKS_PER_SESSION`` of them exist."""
        s = self._sessions[asset]
        lo = int(s["close_ns"][pos - 1]) if pos > 0 else int(s["open_ns"][pos]) - DAY_NS
        if after_ns <= lo:
            return self._session_extremes(asset, contract, pos)
        ticks = self._ticks.get((asset, contract))
        if ticks is None:
            return None
        hi = int(s["close_ns"][pos])
        a = int(np.searchsorted(ticks[0], max(lo, after_ns), side="right"))
        b = int(np.searchsorted(ticks[0], hi, side="right"))
        if b - a < MIN_TICKS_PER_SESSION:
            return None
        prices = ticks[1][a:b]
        return np.maximum.accumulate(prices), np.minimum.accumulate(prices), prices

    # ------------------------------------------------------------------ paths

    def evaluate_path(
        self,
        asset: str,
        when,
        horizon: int,
        direction: str,
        favorable_pct: dict[str, float],
        adverse_pct: dict[str, float],
        contract: str | None = None,
        swing_threshold_pct: float | None = None,
    ) -> PathResult | None:
        """Walk the window and record where each named barrier was first touched.

        ``favorable_pct`` and ``adverse_pct`` are percentages of the entry price, by name. The
        touch key is ``(session position, position inside the session)``, so comparing two keys
        answers which barrier came first. Inside a session bar, where the order is unknowable, the
        adverse barriers rank first and the favorable ones by distance from the entry.

        With ``swing_threshold_pct`` the same walk also measures the move to the swing's end,
        ``PathResult.swing_end_pct``: see ``SwingEnd``.
        """
        contract = contract or self.designated_contract(asset, when)
        if contract is None:
            return None
        quoted = self.price_at(asset, contract, when)
        if quoted is None:
            return None
        p0, source, p_avail = quoted
        end_utc, complete, n_sessions, first, last = self.window(asset, when, horizon)
        up = direction == "up"
        sign = 1.0 if up else -1.0
        levels = {name: p0 * (1.0 + sign * pct / 100.0) for name, pct in favorable_pct.items()}
        levels.update({name: p0 * (1.0 - sign * pct / 100.0) for name, pct in adverse_pct.items()})
        adverse_names = set(adverse_pct)
        daily_rank = _bar_ranks(levels, adverse_names, p0)
        touches: dict[str, tuple[int, int] | None] = dict.fromkeys(levels)
        grid = self._grid[asset]
        column = grid["col"].get(contract)
        t_ns = ns_of(when)
        best_high, best_low = -np.inf, np.inf
        kinds: set[str] = set()
        ambiguous = False
        missing = 0
        swing = SwingEnd(up, swing_threshold_pct) if swing_threshold_pct is not None else None
        for pos in range(first, last + 1):
            extremes = self._tick_slice(asset, contract, pos, t_ns)
            if extremes is not None:
                runmax, runmin, ticks = extremes
                if swing is not None:
                    swing.walk(ticks, runmax if up else runmin)
                kinds.add("tick")
                best_high = max(best_high, float(runmax[-1]))
                best_low = min(best_low, float(runmin[-1]))
                for name, level in levels.items():
                    if touches[name] is not None:
                        continue
                    if (name in adverse_names) == up:
                        idx = int(np.searchsorted(-runmin, -level, side="left"))
                        hit = idx < len(runmin)
                    else:
                        idx = int(np.searchsorted(runmax, level, side="left"))
                        hit = idx < len(runmax)
                    if hit:
                        touches[name] = (pos, idx)
                continue
            if column is None:
                missing += 1
                continue
            high = grid["high"][pos, column]
            low = grid["low"][pos, column]
            if not (np.isfinite(high) and np.isfinite(low)):
                missing += 1
                continue
            kinds.add("daily")
            if swing is not None:
                bar = np.array([high, low] if up else [low, high], dtype=float)
                swing.walk(bar, np.maximum.accumulate(bar) if up else np.minimum.accumulate(bar))
            best_high = max(best_high, float(high))
            best_low = min(best_low, float(low))
            touched_here = []
            for name, level in levels.items():
                if touches[name] is not None:
                    continue
                hit = (high >= level) if ((name in adverse_names) != up) else (low <= level)
                if hit:
                    touches[name] = (pos, daily_rank[name])
                    touched_here.append(name)
            if any(n in adverse_names for n in touched_here) and any(
                n not in adverse_names for n in touched_here
            ):
                ambiguous = True
        mfe = (best_high - p0) if up else (p0 - best_low)
        mae = (p0 - best_low) if up else (best_high - p0)
        return PathResult(
            p0=p0,
            price_source=source,
            price_available_utc=p_avail,
            contract_id=contract,
            window_start_utc=utc_of(t_ns),
            window_end_utc=end_utc,
            window_complete=complete,
            sessions_in_window=n_sessions,
            bars="mixed" if len(kinds) > 1 else (kinds.pop() if kinds else "none"),
            mfe_pct=float(mfe / p0 * 100.0) if np.isfinite(mfe) else float("nan"),
            mae_pct=float(mae / p0 * 100.0) if np.isfinite(mae) else float("nan"),
            touches=touches,
            ambiguous=ambiguous,
            sessions_without_data=missing,
            swing_end_pct=swing.move_pct(p0) if swing is not None else float("nan"),
        )

    # ------------------------------------------------------------------ session bars

    def frame(self, asset: str) -> pd.DataFrame:
        """``session_frame`` for the asset, built once and kept."""
        hit = self._frames.get(asset)
        if hit is None:
            hit = self.session_frame(asset)
            self._frames[asset] = hit
        return hit

    def available_pos(self, asset: str, when) -> int:
        """Position of the last session whose settlement was published before ``when``; -1 if none."""
        frame = self.frame(asset)
        avail = frame["available_utc"]
        ns = np.where(
            avail.notna().to_numpy(),
            avail.astype("datetime64[ns, UTC]").astype("int64"),
            np.iinfo("int64").max,
        )
        return int(np.searchsorted(np.maximum.accumulate(ns), ns_of(when), side="left")) - 1

    def trailing_move_pct(self, asset: str, when, sessions: int = 5) -> float | None:
        """Percent move of the designated contract over the last ``sessions`` published sessions.

        The contract is held fixed at the one designated for the latest of them, so a roll inside
        the window does not enter as a calendar spread; when that contract has no settlement at the
        far end the designated series is used and the spread is accepted.
        """
        k = self.available_pos(asset, when)
        j = k - sessions
        if j < 0:
            return None
        frame = self.frame(asset)
        grid = self._grid[asset]
        contract = frame["contract_id"].iloc[k]
        column = grid["col"].get(contract) if contract is not None else None
        if column is not None:
            now = grid["settlement"][k, column]
            then = grid["settlement"][j, column]
            if np.isfinite(now) and np.isfinite(then) and then > 0:
                return float((now - then) / then * 100.0)
        now = frame["settlement"].iloc[k]
        then = frame["settlement"].iloc[j]
        if not (np.isfinite(now) and np.isfinite(then)) or then <= 0:
            return None
        return float((now - then) / then * 100.0)

    def session_frame(self, asset: str) -> pd.DataFrame:
        """Per session: the designated contract's high, low and settlement, with the instants."""
        s = self._sessions[asset]
        grid = self._grid[asset]
        cols = np.array(
            [grid["col"].get(c, -1) if c is not None else -1 for c in s["contract_id"]], dtype=int
        )
        rows = np.arange(len(cols))
        ok = cols >= 0

        def pick(name: str) -> np.ndarray:
            out = np.full(len(cols), np.nan)
            out[ok] = grid[name][rows[ok], cols[ok]]
            return out

        avail = np.full(len(cols), -1, dtype="int64")
        avail[ok] = grid["available_ns"][rows[ok], cols[ok]]
        return pd.DataFrame(
            {
                "session_date": s["session_date"],
                "contract_id": s["contract_id"],
                "preopen_utc": pd.to_datetime(s["preopen_ns"], utc=True),
                "close_utc": pd.to_datetime(s["close_ns"], utc=True),
                "high": pick("high"),
                "low": pick("low"),
                "settlement": pick("settlement"),
                "available_utc": pd.to_datetime(np.where(avail >= 0, avail, np.nan), utc=True),
            }
        )


class SwingEnd:
    """The move to the swing's end, fed one session of the window at a time, in order.

    A running favorable extreme is kept over the prices walked (the maximum for an up signal, the
    minimum for a down one), and the swing ends at the first price that turns back a full
    threshold from it: ``(extreme - price) / extreme`` for up, ``(price - extreme) / extreme`` for
    down, in percent, at least ``threshold_pct``. The swing's end is the running extreme at that
    price, or at the last price walked when nothing turned that far before the window ended. The
    walk starts at the first price after the signal, not at the entry price.
    """

    __slots__ = ("ended", "extreme", "threshold_pct", "up")

    def __init__(self, up: bool, threshold_pct: float) -> None:
        self.up = up
        self.threshold_pct = float(threshold_pct)
        self.extreme = float("nan")
        self.ended = False

    def walk(self, prices: np.ndarray, running: np.ndarray) -> None:
        """One session: its prices in order, and their running favorable extreme inside it."""
        if self.ended or not len(prices):
            return
        if np.isfinite(self.extreme):
            running = (
                np.maximum(running, self.extreme) if self.up else np.minimum(running, self.extreme)
            )
        turn = ((running - prices) if self.up else (prices - running)) / running * 100.0
        stop = np.flatnonzero(turn >= self.threshold_pct - PCT_TOLERANCE)
        self.ended = bool(stop.size)
        self.extreme = float(running[stop[0]] if stop.size else running[-1])

    def move_pct(self, p0: float) -> float:
        """From the entry price to the swing's end, in percent, positive in the signal's direction."""
        if not np.isfinite(self.extreme):
            return float("nan")
        move = (self.extreme - p0) if self.up else (p0 - self.extreme)
        return float(move / p0 * 100.0)


def _bar_ranks(levels: dict[str, float], adverse: set[str], p0: float) -> dict[str, int]:
    """Order inside one session bar: adverse barriers first, then favorable by distance."""
    ranks = {name: 0 for name in levels if name in adverse}
    rest = sorted((abs(levels[n] - p0), n) for n in levels if n not in adverse)
    for i, (_, name) in enumerate(rest):
        ranks[name] = i + 1
    return ranks


def _session_hours(sessions: pd.DataFrame | None) -> dict[date, tuple[time, time]]:
    if sessions is None or not len(sessions):
        return {}
    out: dict[date, tuple[time, time]] = {}
    for row in sessions.itertuples():
        if getattr(row, "state", None) == "closed":
            continue
        open_ist = getattr(row, "morning_open_ist", None) or DEFAULT_OPEN_IST
        close_ist = getattr(row, "evening_close_ist", None) or DEFAULT_CLOSE_IST
        out[row.session_date] = (open_ist, close_ist)
    return out


def _designation_map(designations: pd.DataFrame) -> dict[str, dict[date, str]]:
    frame = designations[designations["horizon"] == DESIGNATION_HORIZON]
    out: dict[str, dict[date, str]] = {}
    for asset, group in frame.groupby("asset", sort=False):
        out[str(asset)] = dict(
            zip(group["cutoff_date"], group["contract_id"].astype(object), strict=True)
        )
    return out


def _fallback_contract(rows: pd.DataFrame, session_date: date) -> str | None:
    """The most traded contract of a session, for sessions the designation file does not cover."""
    day = rows[rows["session_date"] == session_date]
    if day.empty or "volume_lots" not in day.columns:
        return None
    return str(day.loc[day["volume_lots"].idxmax(), "contract_id"])
