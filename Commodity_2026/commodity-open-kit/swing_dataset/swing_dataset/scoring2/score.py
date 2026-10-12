"""The three scores of each task, the coverage floor, the gates and hard conditions, the award booleans.

Task 1 (section 3, ``--challenge 1``): catch rate, correctness, depth, and their plain average,
with the ten large swings of the holdout, and the gates that cannot be averaged away: the success
rate, correctness and depth each at their own minimum, and seven of the ten large swings caught.
The large-swing catch rate at twice the threshold is reported as a diagnostic and never scored.
Task 2 (section 4, ``--challenge 2``): direction, depth, improvement, and their plain average,
with the five hard conditions that cannot be averaged away, two of them measured against the
prevailing-direction null.

Every score is a percentage and every score carries an interval from the shared episode draw of
``bootstrap.py``. A score with no units is reported as null rather than as zero, and the success
rate is null with it: an average of two numbers is not the average of three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import (
    BASELINE_HORIZONS,
    COVERAGE_FLOOR,
    HOLDOUT_START,
    IMPROVEMENT_MULTIPLIER,
    LARGE_SWING_MULTIPLE,
    MINIMUM_CORRECTNESS,
    MINIMUM_DEPTH,
    MINIMUM_DIRECTION_SCORE,
    MINIMUM_SUCCESS_RATE,
    PERIODS,
    THRESHOLD_PCT,
    TOP_SWINGS,
    TOP_SWINGS_REQUIRED,
    scored_horizons,
)
from . import outcomes as O
from . import swings as SW
from .bootstrap import EpisodeBootstrap, Replicates, episode_keys
from .prices import PriceSeries
from .signals import Decisions, in_window

NULL_LOOKBACK_SESSIONS = 5


@dataclass(frozen=True, slots=True)
class Window:
    """The scored window: which instants count. Which horizons it scores depends on the task as
    well as the period; how long an episode block is, on the entry too (``episode_horizon``)."""

    period: str
    start: pd.Timestamp
    end: pd.Timestamp

    def horizons(self, challenge: int) -> tuple[int, ...]:
        return scored_horizons(challenge, self.period)

    def to_dict(self, challenge: int | None = None) -> dict:
        out = {
            "period": self.period,
            "start_utc": self.start.isoformat(),
            "end_utc": self.end.isoformat(),
        }
        if challenge is not None:
            out["horizons_scored"] = list(self.horizons(challenge))
        return out


def resolve_window(period: str, prices: PriceSeries, start=None, end=None) -> Window:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    last = max(pd.Timestamp(prices.frame(a)["close_utc"].max()) for a in prices.assets())
    if period == "holdout":
        begin = pd.Timestamp(start) if start is not None else pd.Timestamp(HOLDOUT_START, tz="UTC")
    else:
        if start is None:
            raise ValueError(
                "the forward window has no fixed start in the data: pass --test-start, the first "
                "session after the submission close"
            )
        begin = pd.Timestamp(start)
    finish = pd.Timestamp(end) if end is not None else last
    if begin.tz is None or finish.tz is None:
        raise ValueError("window bounds must be timezone-aware")
    if finish < begin:
        raise ValueError("the window ends before it starts")
    return Window(period, begin, finish)


def episode_horizon(challenge: int, window: Window, scored: pd.DataFrame) -> int:
    """Section 5's episode block, in sessions: the unit the bootstrap resamples.

    Task 1: the longest horizon the period scores, 10 on the holdout and 5 on the forward window.
    Task 2: the longest horizon the entry's valid rows in the scored set use, so an entry at 5 and
    10 is drawn in blocks of 10, exactly as under 2.5.2, and one reaching 15 in blocks of 15,
    rather than every entry in blocks of the 20 sessions the holdout allows. An entry with no such
    row gets the longest horizon the shipped baselines are scored at in the period: 10 on the
    holdout, 5 on the forward window. An invalid row's horizon is not one the entry used.
    """
    allowed = window.horizons(challenge)
    if challenge == 1:
        return max(allowed)
    if len(scored):
        horizon = pd.to_numeric(scored["horizon"], errors="coerce")
        used = horizon[scored["valid"].astype(bool) & horizon.isin(allowed)]
        if len(used):
            return int(used.max())
    return max(h for h in BASELINE_HORIZONS if h in allowed)


def split_horizons(
    signals: pd.DataFrame, window: Window, challenge: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows this period scores, and valid rows of an unscored horizon, which are only recorded.

    A row whose horizon is not a horizon of the task at all stays in the scored set: it is
    invalid, and an invalid row counts as issued.
    """
    if signals.empty:
        return signals, signals
    horizon = pd.to_numeric(signals["horizon"], errors="coerce")
    recorded = signals["valid"].astype(bool) & ~horizon.isin(window.horizons(challenge))
    return signals[~recorded].reset_index(drop=True), signals[recorded].reset_index(drop=True)


def session_of(prices: PriceSeries, asset: str, instant) -> object | None:
    pos = prices.session_pos(asset, instant)
    if pos is None:
        return None
    return prices.session_dates(asset)[pos]


def coverage(prices: PriceSeries, decisions: Decisions, window: Window) -> dict:
    """Section 5: the share of the window's sessions carrying a decision record, per asset.

    A decision record is a valid signal row or a valid `no_signal` row. An invalid signal row is
    not a decision the rules recognise, and a record outside every session is counted apart.
    """
    signals = decisions.signals
    valid_signals = signals[signals["valid"].astype(bool)] if len(signals) else signals
    valid_signals = valid_signals[in_window(valid_signals, "signal_utc", window.start, window.end)]
    no_signals = decisions.no_signals
    if len(no_signals):
        no_signals = no_signals[no_signals["valid"].astype(bool)]
        no_signals = no_signals[in_window(no_signals, "decision_utc", window.start, window.end)]
    records = [
        (str(a), t)
        for a, t in zip(valid_signals["asset"], valid_signals["signal_utc"], strict=True)
    ] + [(str(a), t) for a, t in zip(no_signals["asset"], no_signals["decision_utc"], strict=True)]
    outside = 0
    covered: dict[str, set] = {}
    for asset, instant in records:
        if asset not in prices.assets():
            outside += 1
            continue
        session = session_of(prices, asset, instant)
        if session is None:
            outside += 1
            continue
        covered.setdefault(asset, set()).add(session)
    per_asset = {}
    total_sessions = total_covered = 0
    for asset in prices.assets():
        sessions = prices.sessions_between(asset, window.start, window.end)
        hit = len(covered.get(asset, set()) & set(sessions.tolist()))
        share = (hit / len(sessions)) if len(sessions) else None
        per_asset[asset] = {
            "sessions": len(sessions),
            "sessions_with_a_decision": int(hit),
            "share": share,
            "met": bool(share is not None and share >= COVERAGE_FLOOR),
        }
        total_sessions += len(sessions)
        total_covered += hit
    pooled = (total_covered / total_sessions) if total_sessions else None
    return {
        "floor": COVERAGE_FLOOR,
        "rule": "a valid signal row or a valid no_signal row on at least 90 percent of the asset's sessions in the window",
        "per_asset": per_asset,
        "pooled": {
            "sessions": int(total_sessions),
            "sessions_with_a_decision": int(total_covered),
            "share": pooled,
            "met": bool(pooled is not None and pooled >= COVERAGE_FLOOR),
        },
        "decision_records": len(records),
        "decision_records_outside_a_session": int(outside),
        "met": bool(per_asset) and all(v["met"] for v in per_asset.values()),
    }


def _halves(prices: PriceSeries, window: Window) -> tuple[set, set]:
    """The window's sessions cut in two by calendar date, for the second-half condition."""
    dates = sorted(
        {d for a in prices.assets() for d in prices.sessions_between(a, window.start, window.end)}
    )
    mid = len(dates) // 2
    return set(dates[:mid]), set(dates[mid:])


def _half_mask(prices: PriceSeries, rows: pd.DataFrame, half: set) -> np.ndarray:
    if rows.empty:
        return np.zeros(0, dtype=bool)
    return np.array(
        [
            session_of(prices, str(a), t) in half
            for a, t in zip(rows["asset"], rows["signal_utc"], strict=True)
        ],
        dtype=bool,
    )


def _share(values) -> float | None:
    values = np.asarray(values, dtype=float)
    return float(values.mean() * 100.0) if values.size else None


def _keys(
    prices: PriceSeries, window: Window, horizon: int, rows: pd.DataFrame, column: str
) -> np.ndarray:
    if rows.empty:
        return np.array([], dtype=object)
    return episode_keys(prices, window.start, horizon, rows["asset"], rows[column])


def _mean_of(parts: list[np.ndarray]) -> np.ndarray:
    stack = np.vstack(parts)
    return stack.mean(axis=0)


def _row_counts(graded: pd.DataFrame) -> dict:
    if graded.empty:
        return {
            "signals_in_window": 0,
            "invalid": 0,
            "unmatured": 0,
            "scored": 0,
            "no_price": 0,
            "ambiguous_bar": 0,
            "scored_on_incomplete_window": 0,
            "tick_priced": 0,
        }
    scorable = graded["scorable"].fillna(False).astype(bool)
    return {
        "signals_in_window": len(graded),
        "invalid": int((~graded["valid"].astype(bool)).sum()),
        "unmatured": int(graded["unmatured"].fillna(False).astype(bool).sum()),
        "scored": int(scorable.sum()),
        "no_price": int(graded["no_price"].fillna(False).astype(bool).sum()),
        "ambiguous_bar": int(graded["ambiguous"].fillna(False).astype(bool).sum()),
        "scored_on_incomplete_window": int(
            (scorable & ~graded["window_complete"].fillna(True).astype(bool)).sum()
        ),
        "tick_priced": int((graded["price_source"] == "tick").sum()),
    }


# --------------------------------------------------------------------------- task 1


def _decided_counts(graded: pd.DataFrame) -> dict:
    """Section 3's correctness denominator in counts.

    Scored rows split four ways: decided (a valid priced row whose price touched its threshold or
    its invalidation), undecided (its window completed with neither touched), invalid and
    unpriced. Correctness counts every scored row but the undecided ones; ``undecided_share`` is
    the undecided rows' share of the scored ones.
    """
    if graded.empty:
        return {"decided": 0, "undecided": 0, "undecided_share": None}
    scored = int(graded["scorable"].fillna(False).astype(bool).sum())
    undecided = int(graded["undecided"].fillna(False).astype(bool).sum())
    return {
        "decided": int(graded["decided"].fillna(False).astype(bool).sum()),
        "undecided": undecided,
        "undecided_share": undecided / scored if scored else None,
    }


def ten_large_swings(caught: pd.DataFrame) -> dict:
    """Section 3's ten large swings: the realized swings of the window with the largest move in
    asset thresholds, and whether and by which signal each was caught.

    Ranked by ``swings.largest``. A swing is caught exactly as for the catch rate. Whether the
    count clears the minimum is a gate of ``award_conditions``, which applies it on the holdout
    only; on the forward window the ten are listed and decide nothing.
    """
    top = SW.largest(caught, TOP_SWINGS)
    swings = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        by = row.caught_by_signal_utc
        swings.append(
            {
                "rank": rank,
                "asset": str(row.asset),
                "direction": str(row.direction),
                "turn_utc": row.turn_utc,
                "extreme_utc": row.extreme_utc,
                "move_pct": float(row.move_pct),
                "threshold_pct": float(row.threshold_pct),
                "threshold_multiple": float(row.threshold_multiple),
                "caught": bool(row.caught),
                "caught_by_signal_utc": None if by is None or pd.isna(by) else pd.Timestamp(by),
            }
        )
    return {
        "rule": (
            f"the {TOP_SWINGS} realized swings of the window with the largest move from turn to "
            "extreme divided by the asset's threshold, ties to the earlier turn and then the asset "
            "name, each caught or not exactly as for the catch rate; on the holdout at least "
            f"{TOP_SWINGS_REQUIRED} must be caught, on the forward window the check does not apply"
        ),
        "top": TOP_SWINGS,
        "required": TOP_SWINGS_REQUIRED,
        "swings_ranked": len(swings),
        "caught": sum(s["caught"] for s in swings),
        "swings": swings,
    }


def _large_swings(rep: Replicates, scores: dict, caught: pd.DataFrame, large: pd.DataFrame) -> dict:
    """Section 3's reported diagnostic: the catch rate over the legs a person would notice.

    It decides nothing. It is not averaged into the success rate, it is not in
    `award_conditions`, and it cannot move the three scored numbers: the episodes a large leg
    falls in are episodes a leg falls in, so the shared universe `bootstrap.run` draws from is
    the same set with the diagnostic added as without it, and every other score is recomputed on
    that same draw. The test suite asserts that rather than trusting it.
    """
    return {
        "decides": (
            "nothing on its own. It is reported for the one-to-one call of section 4a and is "
            "left out of the success rate and of award_conditions"
        ),
        "definition": (
            f"swings caught divided by realized swings in the window, over the legs whose move is "
            f"at least {LARGE_SWING_MULTIPLE} times the asset's threshold. Caught, the window and "
            "the episode blocks mean what they mean for the scored catch rate"
        ),
        "multiple": LARGE_SWING_MULTIPLE,
        "threshold_pct": dict(THRESHOLD_PCT),
        "large_move_pct": SW.large_move_pct(),
        "rate": rep.interval("large_catch_rate"),
        "large_swings": len(large),
        "large_swings_caught": int(large["caught"].sum()) if len(large) else 0,
        "swings_in_window": len(caught),
        "swings_caught": int(caught["caught"].sum()) if len(caught) else 0,
        "catch_rate_all_swings": scores["catch_rate"]["value"],
    }


def challenge1(
    decisions: Decisions,
    inventory: pd.DataFrame,
    prices: PriceSeries,
    window: Window,
    n_boot: int,
    seed: int,
) -> dict:
    signals = decisions.signals
    signals = signals[in_window(signals, "signal_utc", window.start, window.end)]
    scored_rows, recorded = split_horizons(signals, window, 1)
    graded = O.grade_c1(scored_rows, prices)
    scorable = graded[graded["scorable"].fillna(False).astype(bool)].reset_index(drop=True)
    # correctness is over decided signals: an undecided one leaves the denominator and is counted
    # apart, while an invalid or unpriced row stays in as incorrect
    counted = scorable[~scorable["undecided"].fillna(False).astype(bool)].reset_index(drop=True)
    correct_rows = counted[counted["correct"].fillna(False).astype(bool)].reset_index(drop=True)
    swings_in = SW.in_window(inventory, window.start, window.end)
    caught = O.catch(swings_in, graded)
    episode = episode_horizon(1, window, scored_rows)

    boot = EpisodeBootstrap(n_boot=n_boot, seed=seed)
    boot.add(
        "catch_rate",
        caught["caught"].to_numpy(dtype=float) if len(caught) else np.array([]),
        _keys(prices, window, episode, caught, "turn_utc"),
    )
    boot.add(
        "correctness",
        counted["correct"].fillna(False).to_numpy(dtype=float),
        _keys(prices, window, episode, counted, "signal_utc"),
    )
    boot.add(
        "depth",
        correct_rows["deep"].fillna(False).to_numpy(dtype=float),
        _keys(prices, window, episode, correct_rows, "signal_utc"),
    )
    large = caught[SW.is_large(caught)].reset_index(drop=True)
    boot.add(
        "large_catch_rate",
        large["caught"].to_numpy(dtype=float) if len(large) else np.array([]),
        _keys(prices, window, episode, large, "turn_utc"),
    )
    rep = boot.run()
    names = ("catch_rate", "correctness", "depth")
    scores = {name: rep.interval(name) for name in names}
    scores["success_rate"] = _success(rep, names, scores)
    return {
        "challenge": 1,
        "episode_horizon": episode,
        "scores": scores,
        "ten_large_swings": ten_large_swings(caught),
        "diagnostics": {"large_swing_catch_rate": _large_swings(rep, scores, caught, large)},
        "counts": {
            **_row_counts(graded),
            **_decided_counts(graded),
            "recorded_not_scored_other_horizon": len(recorded),
            "correct": len(correct_rows),
            "deep": int(correct_rows["deep"].fillna(False).astype(bool).sum())
            if len(correct_rows)
            else 0,
            "swings_in_window": len(caught),
            "swings_caught": int(caught["caught"].sum()) if len(caught) else 0,
            "swings_straddling_window_start": int(swings_in.attrs.get("straddles_start", 0)),
            "swings_straddling_window_end": int(swings_in.attrs.get("straddles_end", 0)),
            "swings_outside_window": int(swings_in.attrs.get("outside", 0)),
            **decisions.counts,
        },
        "graded": graded,
        "caught": caught,
    }


# --------------------------------------------------------------------------- task 2


def prevailing_null(signals: pd.DataFrame, prices: PriceSeries) -> pd.DataFrame:
    """Section 4's null: the sign of the last five completed sessions' move, same instants.

    The target is the asset's threshold and the invalidation is the threshold, as section 4 says.
    A move of exactly zero takes the up direction, so the null is a function of the data and not
    of a coin.
    """
    rows = []
    for row in signals.itertuples():
        move = prices.trailing_move_pct(str(row.asset), row.signal_utc, NULL_LOOKBACK_SESSIONS)
        if move is None:
            continue
        threshold = THRESHOLD_PCT[str(row.asset)]
        rows.append(
            {
                "asset": row.asset,
                "signal_utc": row.signal_utc,
                "direction": "up" if move >= 0 else "down",
                "horizon": row.horizon,
                "expected_pct": threshold,
                "p10_pct": 0.0,
                "p90_pct": float("inf"),
                "invalidation_pct": threshold,
                "evidence": f"prevailing direction: {NULL_LOOKBACK_SESSIONS}-session move {move:.3f} percent",
                "threshold_pct": threshold,
                "valid": True,
                "invalid_reasons": [],
                "signal_index": row.signal_index,
            }
        )
    columns = [
        "asset",
        "signal_utc",
        "direction",
        "horizon",
        "expected_pct",
        "p10_pct",
        "p90_pct",
        "invalidation_pct",
        "evidence",
        "threshold_pct",
        "valid",
        "invalid_reasons",
        "signal_index",
    ]
    return pd.DataFrame(rows, columns=columns)


def _direction_share(graded: pd.DataFrame) -> float | None:
    scorable = graded[graded["scorable"].fillna(False).astype(bool)] if len(graded) else graded
    if not len(scorable):
        return None
    return _share(scorable["realized"].fillna(False).to_numpy(dtype=float))


def _pair(live: pd.DataFrame, frozen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows both files issued at the same instant for the same asset and horizon.

    Duplicated keys keep their first row on each side, so a file that repeats a key cannot pair one
    of its rows against several of the other's.
    """
    key = ["asset", "signal_utc", "horizon"]
    if live.empty or frozen.empty:
        return live.iloc[:0], frozen.iloc[:0]
    a = live.drop_duplicates(key).set_index(key)
    b = frozen.drop_duplicates(key).set_index(key)
    shared = a.index.intersection(b.index)
    return a.loc[shared].reset_index(), b.loc[shared].reset_index()


def challenge2(
    decisions: Decisions,
    frozen: Decisions | None,
    prices: PriceSeries,
    window: Window,
    n_boot: int,
    seed: int,
) -> dict:
    signals = decisions.signals
    signals = signals[in_window(signals, "signal_utc", window.start, window.end)]
    scored_rows, recorded = split_horizons(signals, window, 2)
    graded = O.grade_c2(scored_rows, prices)
    scorable = graded[graded["scorable"].fillna(False).astype(bool)].reset_index(drop=True)
    realized_rows = scorable[scorable["realized"].fillna(False).astype(bool)].reset_index(drop=True)

    frozen_graded = None
    if frozen is not None:
        frozen_signals = frozen.signals
        frozen_signals = frozen_signals[
            in_window(frozen_signals, "signal_utc", window.start, window.end)
        ]
        frozen_scored, _ = split_horizons(frozen_signals, window, 2)
        frozen_graded = O.grade_c2(frozen_scored, prices)
    live_pair, frozen_pair = (
        _pair(scorable, frozen_graded[frozen_graded["scorable"].fillna(False).astype(bool)])
        if frozen_graded is not None
        else (scorable.iloc[:0], scorable.iloc[:0])
    )

    null_rows = prevailing_null(scored_rows[scored_rows["valid"].astype(bool)], prices)
    null_graded = O.grade_c2(null_rows, prices) if len(null_rows) else null_rows
    null_scorable = (
        null_graded[null_graded["scorable"].fillna(False).astype(bool)].reset_index(drop=True)
        if len(null_graded)
        else null_graded
    )
    frozen_null_scorable = null_scorable
    if frozen_graded is not None and len(frozen_graded):
        frozen_null = prevailing_null(frozen_graded[frozen_graded["valid"].astype(bool)], prices)
        frozen_null_graded = O.grade_c2(frozen_null, prices) if len(frozen_null) else frozen_null
        frozen_null_scorable = (
            frozen_null_graded[frozen_null_graded["scorable"].fillna(False).astype(bool)]
            if len(frozen_null_graded)
            else frozen_null_graded
        )

    episode = episode_horizon(2, window, scored_rows)
    boot = EpisodeBootstrap(n_boot=n_boot, seed=seed)
    boot.add(
        "direction",
        scorable["realized"].fillna(False).to_numpy(dtype=float),
        _keys(prices, window, episode, scorable, "signal_utc"),
    )
    boot.add(
        "depth",
        realized_rows["in_band"].fillna(False).to_numpy(dtype=float),
        _keys(prices, window, episode, realized_rows, "signal_utc"),
    )
    boot.add(
        "direction_paired_learning",
        live_pair["realized"].fillna(False).to_numpy(dtype=float)
        if len(live_pair)
        else np.array([]),
        _keys(prices, window, episode, live_pair, "signal_utc"),
    )
    boot.add(
        "direction_paired_frozen",
        frozen_pair["realized"].fillna(False).to_numpy(dtype=float)
        if len(frozen_pair)
        else np.array([]),
        _keys(prices, window, episode, frozen_pair, "signal_utc"),
    )
    rep = boot.run()
    improvement_draws = improvement_pct(
        rep.draws["direction_paired_learning"], rep.draws["direction_paired_frozen"]
    )
    improvement_value = improvement_pct(
        np.array([rep.observed["direction_paired_learning"]]),
        np.array([rep.observed["direction_paired_frozen"]]),
    )[0]
    scores = {
        "direction": rep.interval("direction"),
        "depth": rep.interval("depth"),
        "improvement": rep.of(
            improvement_value,
            improvement_draws,
            rep.n["direction_paired_learning"],
            rep.n_episodes["direction_paired_learning"],
        ),
    }
    success_draws = _mean_of(
        [rep.draws["direction"] * 100.0, rep.draws["depth"] * 100.0, improvement_draws]
    )
    values = [
        scores["direction"]["value"],
        scores["depth"]["value"],
        scores["improvement"]["value"],
    ]
    success_value = float(np.mean(values)) if all(v is not None for v in values) else float("nan")
    scores["success_rate"] = rep.of(
        success_value,
        success_draws if all(v is not None for v in values) else np.full(n_boot, np.nan),
        min(rep.n["direction"], rep.n["depth"]),
        rep.universe,
    )

    first, second = _halves(prices, window)
    first_share = _share(
        scorable.loc[_half_mask(prices, scorable, first), "realized"].fillna(False).to_numpy(float)
    )
    second_share = _share(
        scorable.loc[_half_mask(prices, scorable, second), "realized"].fillna(False).to_numpy(float)
    )
    null_share = _direction_share(null_scorable) if len(null_scorable) else None
    frozen_share = _direction_share(frozen_graded) if frozen_graded is not None else None
    frozen_null_share = (
        _direction_share(frozen_null_scorable) if len(frozen_null_scorable) else None
    )
    paired = {
        "n": len(live_pair),
        "learning": _share(live_pair["realized"].fillna(False).to_numpy(float))
        if len(live_pair)
        else None,
        "frozen": _share(frozen_pair["realized"].fillna(False).to_numpy(float))
        if len(frozen_pair)
        else None,
    }
    conditions = {
        "learning_beats_frozen": _gt(paired["learning"], paired["frozen"]),
        "second_half_at_least_first_half": _ge(second_share, first_share),
        "direction_at_least_minimum": bool(
            scores["direction"]["value"] is not None
            and scores["direction"]["value"] >= MINIMUM_DIRECTION_SCORE
        ),
        "frozen_beats_prevailing_null": _gt(frozen_share, frozen_null_share),
        "learning_beats_prevailing_null": _gt(scores["direction"]["value"], null_share),
    }
    return {
        "challenge": 2,
        "episode_horizon": episode,
        "scores": scores,
        "paired_direction": paired,
        "halves": {
            "first_half_direction": first_share,
            "second_half_direction": second_share,
            "first_half_sessions": len(first),
            "second_half_sessions": len(second),
        },
        "prevailing_direction_null": {
            "rule": (
                "the sign of the last five completed sessions' move, issued at the entry's "
                "instants with the entry's horizon, expected and invalidation at the asset threshold"
            ),
            "direction_at_entry_instants": null_share,
            "direction_at_frozen_instants": frozen_null_share,
            "n": len(null_scorable),
        },
        "frozen": {
            "direction": frozen_share,
            "signals_in_window": len(frozen_graded) if frozen_graded is not None else 0,
            "scored": int(frozen_graded["scorable"].fillna(False).sum())
            if frozen_graded is not None and len(frozen_graded)
            else 0,
        },
        "hard_conditions": conditions,
        "counts": {
            **_row_counts(graded),
            "recorded_not_scored_other_horizon": len(recorded),
            "directionally_realized": len(realized_rows),
            "in_band": int(realized_rows["in_band"].fillna(False).astype(bool).sum())
            if len(realized_rows)
            else 0,
            "invalidation_pct_median": float(
                np.nanmedian(scored_rows["invalidation_pct"].to_numpy(dtype=float))
            )
            if len(scored_rows)
            else None,
            "invalidation_pct_max": float(
                np.nanmax(scored_rows["invalidation_pct"].to_numpy(dtype=float))
            )
            if len(scored_rows)
            else None,
            **decisions.counts,
        },
        "graded": graded,
    }


def improvement_pct(learning: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """Section 4: ten times the direction gain in percentage points, floored at 0, capped at 100."""
    gain = (np.asarray(learning, dtype=float) - np.asarray(frozen, dtype=float)) * 100.0
    return np.clip(IMPROVEMENT_MULTIPLIER * gain, 0.0, 100.0)


def _success(rep: Replicates, names: tuple[str, ...], scores: dict) -> dict:
    values = [scores[n]["value"] for n in names]
    draws = _mean_of([rep.draws[n] * 100.0 for n in names])
    ok = all(v is not None for v in values)
    return rep.of(
        float(np.mean(values)) if ok else float("nan"),
        draws if ok else np.full(len(draws), np.nan),
        min(rep.n[n] for n in names),
        rep.universe,
    )


def _gt(a: float | None, b: float | None) -> bool:
    return bool(a is not None and b is not None and a > b)


def _ge(a: float | None, b: float | None) -> bool:
    return bool(a is not None and b is not None and a >= b)


# --------------------------------------------------------------------------- awards


def _gate(value, threshold, applicable: bool = True) -> dict:
    """One condition of the minimum: its value, the threshold it needs, and whether it is met.

    A value of None (a score with no units) never meets its threshold. A gate that does not apply
    to the period has ``met`` None and decides nothing.
    """
    met = bool(value is not None and value >= threshold) if applicable else None
    return {"value": value, "threshold": threshold, "met": met, "applicable": applicable}


def _coverage_value(coverage_block: dict) -> float | None:
    """The coverage floor binds per asset, so its value is the least covered asset's share."""
    shares = [v.get("share") for v in coverage_block.get("per_asset", {}).values()]
    if not shares or any(s is None for s in shares):
        return None
    return float(min(shares))


def award_conditions(report: dict, window: Window, coverage_block: dict) -> dict:
    """Section 3 and 4 minimums, as booleans, for the period this report covers.

    Task 1's minimum is five gates, each with its value, its threshold and whether it is met, and
    none of them averages with another: the success rate, correctness and depth at their own
    minimums, the coverage floor, and on the holdout seven of the ten large swings caught.
    """
    challenge = report["challenge"]
    success = report["scores"]["success_rate"]["value"]
    out = {
        "period": window.period,
        "horizons_scored": list(window.horizons(challenge)),
        "minimum_success_rate": MINIMUM_SUCCESS_RATE,
        "success_rate": success,
        "success_rate_at_least_minimum": bool(
            success is not None and success >= MINIMUM_SUCCESS_RATE
        ),
        "coverage_floor_met": bool(coverage_block["met"]),
        "minimum_met_this_period": bool(
            success is not None and success >= MINIMUM_SUCCESS_RATE and coverage_block["met"]
        ),
        "requires_both_periods": "the minimum must hold on the holdout and on the forward window; join two reports with `award`",
    }
    if challenge == 1:
        scores = report["scores"]
        block = report.get("ten_large_swings") or {}
        gates = {
            "success_rate": _gate(success, MINIMUM_SUCCESS_RATE),
            "correctness": _gate(scores["correctness"]["value"], MINIMUM_CORRECTNESS),
            "depth": _gate(scores["depth"]["value"], MINIMUM_DEPTH),
            "coverage_floor": {
                "value": _coverage_value(coverage_block),
                "threshold": COVERAGE_FLOOR,
                "met": bool(coverage_block["met"]),
                "applicable": True,
            },
            "ten_large_swings": _gate(
                block.get("caught"), TOP_SWINGS_REQUIRED, applicable=window.period == "holdout"
            ),
        }
        out["gates"] = gates
        out["minimum_met_this_period"] = all(g["met"] for g in gates.values() if g["applicable"])
    if challenge == 2:
        conditions = dict(report["hard_conditions"])
        if window.period != "holdout":
            conditions["frozen_beats_prevailing_null"] = None
        out["hard_conditions"] = conditions
        stated = [v for k, v in conditions.items() if v is not None]
        out["hard_conditions_met_this_period"] = bool(stated) and all(stated)
        out["minimum_met_this_period"] = bool(
            out["minimum_met_this_period"] and out["hard_conditions_met_this_period"]
        )
    return out


def combine(reports: list[dict]) -> dict:
    """Join a holdout report and a forward report into the both-periods booleans."""
    by_period = {r["award_conditions"]["period"]: r for r in reports}
    challenges = {r["challenge"] for r in reports}
    if len(challenges) != 1:
        raise ValueError("reports must be from one challenge")
    missing = [p for p in PERIODS if p not in by_period]
    per_period = {
        p: {
            "success_rate": by_period[p]["scores"]["success_rate"]["value"],
            "minimum_met": by_period[p]["award_conditions"]["minimum_met_this_period"],
            "coverage_floor_met": by_period[p]["award_conditions"]["coverage_floor_met"],
            "gates": by_period[p]["award_conditions"].get("gates"),
            "hard_conditions": by_period[p]["award_conditions"].get("hard_conditions"),
        }
        for p in by_period
    }
    paid = not missing and all(v["minimum_met"] for v in per_period.values())
    return {
        "challenge": challenges.pop(),
        "periods_present": sorted(by_period),
        "periods_missing": missing,
        "per_period": per_period,
        "minimum_met_on_both_periods": bool(paid),
        "ranking_key": {
            "forward_success_rate": per_period.get("forward", {}).get("success_rate"),
            "holdout_success_rate": per_period.get("holdout", {}).get("success_rate"),
            "rule": "ranked by the forward success rate, ties broken by the holdout success rate",
        },
    }
