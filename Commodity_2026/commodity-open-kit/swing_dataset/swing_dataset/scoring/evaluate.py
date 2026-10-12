"""Evaluation reports: metrics, baseline comparison, confidence intervals, award conditions.

``evaluate_challenge1`` and ``evaluate_challenge2`` take normalized prediction and outcome frames
and return the JSON-ready report dict; ``load_predictions`` and ``load_outcomes`` read the parquet
inputs described in FORMATS.md. The CLI in ``__main__`` wraps them.

Windows. The test window defaults to the span of cutoffs in the prediction file and can be
narrowed with ``test_start``/``test_end``. ``train_end`` (default: the first test cutoff) separates
the training slice: outcomes whose label was available strictly before it. The training slice
supplies the abstention prior, the empirical substitute distribution, and the false-alarm
threshold; the threshold needs predictions on training cutoffs, so when the prediction file has
none the threshold is fitted on the test rows and the report says so.

Eligibility. Every outcome row of the horizon (Challenge 1: barrier multiplier 1.0) is an eligible
cutoff. Rows labelled ``both`` or without a label are eligible but unscorable and are counted
separately. A prediction for an eligible row that abstains or is missing is scored with the
substitute and counts against coverage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import CLASSES, KEY, PROB_COLS, QUANTILE_COLS, SAMPLE_COUNT, SCORED_MULTIPLIER
from . import baselines as B
from . import metrics as M
from .chain import FeatureChains
from .folds import DEFAULT_BOOTSTRAP, episode_ids, session_calendar

COVERAGE_FLOOR = 0.90  # issued share of eligible cutoffs, CHALLENGE-SPEC section 5
AWARD_HORIZONS = (5, 10)
EPISODE_SCOPES = ("calendar", "asset")
DEFAULT_EPISODE_SCOPE = "asset"


def _check_scope(scope: str) -> str:
    if scope not in EPISODE_SCOPES:
        raise ValueError(f"episode_scope must be one of {EPISODE_SCOPES}")
    return scope


def _utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True)


def _aware_utc(values: pd.Series, name: str) -> np.ndarray:
    """Timezone-aware timestamps converted to UTC; naive values are rejected."""
    parsed = pd.to_datetime(pd.Series(values).reset_index(drop=True))
    if parsed.notna().any() and parsed.dt.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.dt.tz_convert("UTC").to_numpy()


def file_identity(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    data = path.read_bytes()
    rows = None
    if path.suffix == ".parquet":
        rows = pq.read_metadata(path).num_rows
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "rows": rows,
    }


# --------------------------------------------------------------------------- loading


def load_predictions(path: Path, challenge: int) -> pd.DataFrame:
    frame = pq.read_table(path).to_pandas()
    return normalize_predictions(frame, challenge)


def normalize_predictions(frame: pd.DataFrame, challenge: int) -> pd.DataFrame:
    missing = set(KEY) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file lacks columns {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    out["asset"] = out["asset"].astype(str)
    out["cutoff_utc"] = _aware_utc(out["cutoff_utc"], "cutoff_utc")
    out["horizon"] = out["horizon"].astype(int)
    if "abstain" not in out.columns:
        out["abstain"] = False
    out["abstain"] = out["abstain"].fillna(False).astype(bool)
    issued = ~out["abstain"]
    if challenge == 1:
        for c in PROB_COLS:
            if c not in out.columns:
                out[c] = np.nan
            out[c] = out[c].astype(float)
        if issued.any():
            M.check_probabilities(out.loc[issued, list(PROB_COLS)].to_numpy(dtype=float))
    else:
        has_q = all(c in out.columns for c in QUANTILE_COLS)
        has_s = "samples" in out.columns
        if has_q == has_s:
            raise ValueError(
                "Challenge 2 predictions need the nine quantile columns or a samples column, not both"
            )
        if has_q:
            for c in QUANTILE_COLS:
                out[c] = out[c].astype(float)
            Q = out.loc[issued, list(QUANTILE_COLS)].to_numpy(dtype=float)
            if not np.isfinite(Q).all():
                raise ValueError("issued quantile rows must be finite")
            if (np.diff(Q, axis=1) < -1e-12).any():
                raise ValueError("quantiles must be non-decreasing across the nine levels")
        else:
            out["samples"] = out["samples"].apply(
                lambda s: None if s is None else np.asarray(s, dtype=float)
            )
            bad = [len(s) for s in out["samples"] if s is not None and len(s) != SAMPLE_COUNT]
            if bad:
                raise ValueError(
                    f"samples rows must hold exactly {SAMPLE_COUNT} values; saw lengths {sorted(set(bad))[:5]}"
                )
            if issued.any():
                lengths = out.loc[issued, "samples"].map(lambda s: -1 if s is None else len(s))
                if (lengths != SAMPLE_COUNT).any():
                    raise ValueError("issued rows must carry samples")
        if "p_up" not in out.columns:
            out["p_up"] = np.nan
        out["p_up"] = out["p_up"].astype(float)
        p_up = out.loc[issued, "p_up"]
        if issued.any() and not (p_up.between(0.0, 1.0) & np.isfinite(p_up)).all():
            raise ValueError("p_up must lie in [0, 1] on issued rows")
    dupes = out.duplicated(KEY)
    if dupes.any():
        raise ValueError(
            f"{int(dupes.sum())} duplicated (asset, cutoff_utc, horizon) rows in predictions"
        )
    return out.sort_values(["cutoff_utc", "asset"]).reset_index(drop=True)


def load_outcomes(path: Path, challenge: int, horizon: int) -> pd.DataFrame:
    frame = pq.read_table(path).to_pandas()
    return normalize_outcomes(frame, challenge, horizon)


def normalize_outcomes(frame: pd.DataFrame, challenge: int, horizon: int) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["asset"] = out["asset"].astype(str)
    out["cutoff_utc"] = _aware_utc(out["cutoff_utc"], "cutoff_utc")
    out["horizon"] = out["horizon"].astype(int)
    out["available_utc"] = _aware_utc(out["available_utc"], "available_utc")
    out = out[out["horizon"] == horizon]
    if challenge == 1:
        if "k_multiplier" in out.columns:
            out = out[np.isclose(out["k_multiplier"].astype(float), SCORED_MULTIPLIER)]
        out = out[KEY + ["label", "available_utc"]]
        out["scorable"] = out["label"].isin(CLASSES) & out["available_utc"].notna()
    else:
        out = out[KEY + ["y_ret", "available_utc"]]
        out["y_ret"] = out["y_ret"].astype(float)
        out["scorable"] = np.isfinite(out["y_ret"]) & out["available_utc"].notna()
    dupes = out.duplicated(KEY)
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicated outcome rows for horizon {horizon}")
    return out.sort_values(["cutoff_utc", "asset"]).reset_index(drop=True)


# --------------------------------------------------------------------------- windows and joins


@dataclass(frozen=True, slots=True)
class Window:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "test_start_utc": self.test_start.isoformat(),
            "test_end_utc": self.test_end.isoformat(),
            "train_end_utc": self.train_end.isoformat(),
        }


def resolve_window(
    pred: pd.DataFrame,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
    train_end: pd.Timestamp | None = None,
) -> Window:
    if pred.empty:
        raise ValueError("no predictions")
    start = (
        pd.Timestamp(test_start).tz_convert("UTC")
        if test_start is not None
        else pred["cutoff_utc"].min()
    )
    end = (
        pd.Timestamp(test_end).tz_convert("UTC")
        if test_end is not None
        else pred["cutoff_utc"].max()
    )
    tr_end = pd.Timestamp(train_end).tz_convert("UTC") if train_end is not None else start
    if tr_end > start:
        raise ValueError("train_end must not be after the first test cutoff")
    return Window(pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(tr_end))


def split_frames(pred: pd.DataFrame, outcomes: pd.DataFrame, window: Window) -> dict:
    """Test-window eligible outcomes joined to predictions, the training slice, and counts."""
    in_test = (outcomes["cutoff_utc"] >= window.test_start) & (
        outcomes["cutoff_utc"] <= window.test_end
    )
    eligible = outcomes[in_test].copy()
    scorable = eligible[eligible["scorable"]].copy()
    test = scorable.merge(
        pred.drop(columns=["horizon"]), on=["asset", "cutoff_utc"], how="left", indicator=True
    )
    test["missing"] = test["_merge"] == "left_only"
    test = test.drop(columns=["_merge"])
    train = outcomes[outcomes["scorable"] & (outcomes["available_utc"] < window.train_end)].copy()
    train_pred = pred[pred["cutoff_utc"] < window.train_end]
    pred_test = pred[
        (pred["cutoff_utc"] >= window.test_start) & (pred["cutoff_utc"] <= window.test_end)
    ]
    matched_keys = set(zip(eligible["asset"], eligible["cutoff_utc"], strict=True))
    unmatched = sum(
        (a, c) not in matched_keys
        for a, c in zip(pred_test["asset"], pred_test["cutoff_utc"], strict=True)
    )
    issued_mask = eligible.merge(
        pred.drop(columns=["horizon"]), on=["asset", "cutoff_utc"], how="left"
    )
    issued = int((~issued_mask["abstain"].fillna(True).astype(bool)).sum())
    return {
        "eligible": eligible,
        "test": test,
        "train": train,
        "train_pred": train_pred,
        "counts": {
            "eligible": len(eligible),
            "scorable": len(scorable),
            "unscorable": int(len(eligible) - len(scorable)),
            "predicted_rows_in_window": len(pred_test),
            "predictions_for_ineligible_cutoffs": int(unmatched),
            "missing_predictions": int(test["missing"].sum()),
            "abstained": int(test["abstain"].fillna(False).astype(bool).sum()),
            "issued": issued,
            "training_outcomes": len(train),
            "training_predictions": len(train_pred),
        },
    }


def coverage_block(
    test: pd.DataFrame, counts: dict, episodes: dict[str, np.ndarray], scope: str
) -> dict:
    """Coverage for information: the issued share plus the episode and asset counts behind it."""
    issued_mask = (~test["abstain"].fillna(False).astype(bool) & ~test["missing"]).to_numpy()
    issued_episodes = {
        name: (int(pd.Series(np.asarray(ids)[issued_mask]).nunique()) if issued_mask.any() else 0)
        for name, ids in episodes.items()
    }
    return {
        "issued": counts["issued"],
        "eligible": counts["eligible"],
        "ratio": issued_share(counts),
        "scored_rows": len(test),
        "episode_scope": scope,
        "issued_scored_episodes": issued_episodes[scope],
        "issued_scored_episodes_by_scope": issued_episodes,
        "issued_scored_assets": int(test.loc[issued_mask, "asset"].nunique()),
    }


def issued_share(counts: dict) -> float | None:
    """Issued (non-abstaining) predictions over eligible cutoffs in the window; the coverage metric."""
    return counts["issued"] / counts["eligible"] if counts["eligible"] else None


def coverage_floor(counts: dict) -> dict:
    """CHALLENGE-SPEC section 5: a tier's coverage condition holds when the issued share is at
    least 90 percent of the eligible cutoffs in the scored window. Episode and asset counts are
    reported for information only and never gate."""
    share = issued_share(counts)
    return {
        "issued_share_at_least": COVERAGE_FLOOR,
        "issued_share": share,
        "met": bool(share is not None and share >= COVERAGE_FLOOR),
    }


def episodes_for(
    test: pd.DataFrame, outcomes: pd.DataFrame, horizon: int, window: Window
) -> dict[str, np.ndarray]:
    """Episode ids per row under both scopes, anchored at the first session of the window.

    ``asset`` (the default): every (asset, block of ``horizon`` sessions) pair is its own episode,
    which keeps enough units on a 19-session forward month and a 72-session holdout for the
    bootstrap. ``calendar``: one block is one episode for every asset, which keeps
    same-day forecasts of different assets together at the cost of a quarter of the units.
    """
    calendar = session_calendar(outcomes["cutoff_utc"])
    start_ns = int(pd.Timestamp(window.test_start).tz_convert("UTC").value)
    anchor = int(np.searchsorted(calendar, start_ns))
    blocks = episode_ids(test["cutoff_utc"], horizon, calendar, anchor=anchor)
    assets = sorted(outcomes["asset"].unique())
    asset_index = test["asset"].map({a: i for i, a in enumerate(assets)}).to_numpy()
    return {"calendar": blocks, "asset": blocks * len(assets) + asset_index}


def compare_rows(
    sub: pd.DataFrame,
    other: pd.DataFrame,
    episodes: np.ndarray,
    metrics: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Paired differences submission minus other, bootstrapped over episodes, per metric."""
    return {
        m: M.paired_comparison(
            sub[m].to_numpy(dtype=float),
            other[m].to_numpy(dtype=float),
            episodes,
            n_boot=n_boot,
            seed=seed,
        )
        for m in metrics
    }


# --------------------------------------------------------------------------- challenge 1


def score_c1(test: pd.DataFrame) -> pd.DataFrame:
    P = M.check_probabilities(test[list(PROB_COLS)].to_numpy(dtype=float))
    y = M.class_indices(test["label"])
    return pd.DataFrame(
        {"log_loss": M.log_loss_rows(P, y), "brier": M.brier_rows(P, y)}, index=test.index
    )


def c1_metrics(test: pd.DataFrame, train: pd.DataFrame, train_pred: pd.DataFrame) -> dict:
    P = M.check_probabilities(test[list(PROB_COLS)].to_numpy(dtype=float))
    y = M.class_indices(test["label"])
    rows = score_c1(test)
    ece, table = M.ece_classwise(P, y)
    top_ece, top_table = M.ece_top_label(P, y)
    joined = train.merge(
        train_pred.drop(columns=["horizon"]), on=["asset", "cutoff_utc"], how="inner"
    )
    joined = joined[
        ~joined["abstain"].fillna(False).astype(bool) & joined[list(PROB_COLS)].notna().all(axis=1)
    ]
    if len(joined):
        P_train = M.check_probabilities(joined[list(PROB_COLS)].to_numpy(dtype=float))
        y_train = M.class_indices(joined["label"])
        capture = M.swing_capture(P, y, P_train, y_train)
        capture["threshold_fitted_on"] = "training_predictions"
        capture["threshold_rows"] = len(joined)
    else:
        capture = M.swing_capture(P, y, P, y)
        capture["threshold_fitted_on"] = "test_predictions_no_training_predictions_supplied"
        capture["threshold_rows"] = len(test)
    per_asset = {}
    for asset, idx in test.groupby("asset").groups.items():
        per_asset[str(asset)] = {
            "n": len(idx),
            "log_loss": float(rows.loc[idx, "log_loss"].mean()),
            "brier": float(rows.loc[idx, "brier"].mean()),
        }
    realized = pd.Series(y).map(dict(enumerate(CLASSES))).value_counts()
    return {
        "n_scored": len(test),
        "log_loss": float(rows["log_loss"].mean()),
        "brier": float(rows["brier"].mean()),
        "ece": ece,
        "reliability": table,
        "top_label_ece": top_ece,
        "top_label_reliability": top_table,
        "swing_capture": capture,
        "realized_classes": {c: int(realized.get(c, 0)) for c in CLASSES},
        "substituted_rows": int(test["substituted"].sum()),
        "per_asset": per_asset,
    }


def evaluate_challenge1(
    pred: pd.DataFrame,
    outcomes: pd.DataFrame,
    horizon: int,
    *,
    window: Window | None = None,
    baseline_preds: dict[str, pd.DataFrame] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    episode_scope: str = DEFAULT_EPISODE_SCOPE,
) -> dict:
    window = window or resolve_window(pred)
    parts = split_frames(pred, outcomes, window)
    test, train, counts = parts["test"], parts["train"], parts["counts"]
    if train.empty:
        raise ValueError("no training outcomes before train_end; the abstention prior needs some")
    if test.empty:
        raise ValueError("no scorable outcomes in the test window")
    priors = M.priors_by_asset(train)
    test = M.substitute_abstentions_c1(test, priors)
    scopes = episodes_for(test, outcomes, horizon, window)
    episodes = scopes[_check_scope(episode_scope)]
    metrics = c1_metrics(test, train, parts["train_pred"])
    sub_rows = score_c1(test)
    report = {
        "challenge": 1,
        "horizon": horizon,
        "window": window.to_dict(),
        "counts": counts,
        "coverage": coverage_block(test, counts, scopes, episode_scope),
        "coverage_floor": coverage_floor(counts),
        "metrics": metrics,
        "abstention": {
            "rule": "abstained or missing rows scored as the training class frequencies of the asset",
            "priors": {
                k: dict(zip(CLASSES, map(float, v), strict=True)) for k, v in priors.items()
            },
            "substituted": int(test["substituted"].sum()),
        },
        "baselines": {},
    }
    if baseline_preds:
        for name, bp in baseline_preds.items():
            btest = test[KEY + ["label"]].merge(
                bp.drop(columns=["horizon"]), on=["asset", "cutoff_utc"], how="left"
            )
            if btest[list(PROB_COLS)].isna().any(axis=None):
                raise ValueError(f"baseline {name} lacks predictions for some scored rows")
            btest.index = test.index
            b_rows = score_c1(btest)
            report["baselines"][name] = {
                "metrics": {
                    "log_loss": float(b_rows["log_loss"].mean()),
                    "brier": float(b_rows["brier"].mean()),
                },
                "comparison": compare_rows(
                    sub_rows, b_rows, episodes, ["log_loss", "brier"], n_boot, seed
                ),
            }
    report["award_conditions"] = award_conditions(report, "log_loss")
    report["definitions"] = definitions(1, episode_scope)
    return report


# --------------------------------------------------------------------------- challenge 2


def score_c2(test: pd.DataFrame) -> tuple[pd.DataFrame, M.DistributionRows]:
    dist = M.distribution_rows(test)
    y = test["y_ret"].to_numpy(dtype=float)
    rows = pd.DataFrame(
        {
            "crps": dist.crps(y),
            "brier_p_up": M.brier_binary_rows(dist.p_up, (y > 0).astype(float)),
            "pit": dist.pit(y),
            "in_90": dist.interval_coverage(y),
        },
        index=test.index,
    )
    if dist.kind == "quantiles":
        rows["pinball_mean"] = dist.pinball_mean(y)
    return rows, dist


def c2_metrics(test: pd.DataFrame) -> dict:
    rows, dist = score_c2(test)
    per_asset = {}
    for asset, idx in test.groupby("asset").groups.items():
        per_asset[str(asset)] = {
            "n": len(idx),
            "crps": float(rows.loc[idx, "crps"].mean()),
            "brier_p_up": float(rows.loc[idx, "brier_p_up"].mean()),
            "coverage_90": float(rows.loc[idx, "in_90"].mean()),
        }
    out = {
        "n_scored": len(test),
        "representation": dist.kind,
        "crps": float(rows["crps"].mean()),
        "brier_p_up": float(rows["brier_p_up"].mean()),
        "pit_histogram": M.pit_histogram(rows["pit"].to_numpy()),
        "coverage_90": float(rows["in_90"].mean()),
        "coverage_90_nominal": 0.90,
        "share_up": float((test["y_ret"] > 0).mean()),
        "substituted_rows": int(test["substituted"].sum()),
        "per_asset": per_asset,
    }
    if "pinball_mean" in rows:
        out["pinball_mean"] = float(rows["pinball_mean"].mean())
    return out


def _join_other_c2(test: pd.DataFrame, other: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = [c for c in other.columns if c not in ("horizon",)]
    joined = test[KEY + ["y_ret"]].merge(other[cols], on=["asset", "cutoff_utc"], how="left")
    if "samples" in joined.columns:
        gap = joined["samples"].isna()
    else:
        gap = joined[list(QUANTILE_COLS)].isna().any(axis=1)
    if gap.any() or joined["p_up"].isna().any():
        raise ValueError(f"{name} lacks predictions for some scored rows")
    joined.index = test.index
    return joined


def evaluate_challenge2(
    pred: pd.DataFrame,
    outcomes: pd.DataFrame,
    horizon: int,
    *,
    window: Window | None = None,
    baseline_preds: dict[str, pd.DataFrame] | None = None,
    frozen_pred: pd.DataFrame | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    episode_scope: str = DEFAULT_EPISODE_SCOPE,
) -> dict:
    window = window or resolve_window(pred)
    parts = split_frames(pred, outcomes, window)
    test, train, counts = parts["test"], parts["train"], parts["counts"]
    if train.empty:
        raise ValueError(
            "no training outcomes before train_end; the empirical substitute needs some"
        )
    if test.empty:
        raise ValueError("no scorable outcomes in the test window")
    empirical = M.empirical_targets_by_asset(train)
    if "samples" not in test.columns:
        for c in QUANTILE_COLS:
            if c not in test.columns:
                test[c] = np.nan
    test = M.substitute_abstentions_c2(test, empirical, SAMPLE_COUNT)
    scopes = episodes_for(test, outcomes, horizon, window)
    episodes = scopes[_check_scope(episode_scope)]
    metrics = c2_metrics(test)
    sub_rows, _ = score_c2(test)
    report = {
        "challenge": 2,
        "horizon": horizon,
        "window": window.to_dict(),
        "counts": counts,
        "coverage": coverage_block(test, counts, scopes, episode_scope),
        "coverage_floor": coverage_floor(counts),
        "metrics": metrics,
        "abstention": {
            "rule": "abstained or missing rows scored as the empirical training distribution of y_ret for the asset",
            "training_rows": {k: int(v.size) for k, v in empirical.items()},
            "substituted": int(test["substituted"].sum()),
        },
        "baselines": {},
        "loop_gate": None,
    }
    if baseline_preds:
        for name, bp in baseline_preds.items():
            btest = _join_other_c2(test, bp, f"baseline {name}")
            b_rows, _ = score_c2(btest)
            b_metrics = {
                "crps": float(b_rows["crps"].mean()),
                "brier_p_up": float(b_rows["brier_p_up"].mean()),
                "coverage_90": float(b_rows["in_90"].mean()),
            }
            if "pinball_mean" in b_rows:
                b_metrics["pinball_mean"] = float(b_rows["pinball_mean"].mean())
            report["baselines"][name] = {
                "metrics": b_metrics,
                "comparison": compare_rows(
                    sub_rows, b_rows, episodes, ["crps", "brier_p_up"], n_boot, seed
                ),
            }
    if frozen_pred is not None:
        frozen = M.substitute_abstentions_c2(
            _join_other_c2_lenient(test, frozen_pred), empirical, SAMPLE_COUNT
        )
        f_rows, _ = score_c2(frozen)
        gate = M.loop_gate(
            sub_rows["crps"].to_numpy(dtype=float),
            f_rows["crps"].to_numpy(dtype=float),
            episodes,
            n_boot=n_boot,
            seed=seed,
        )
        gate["frozen_substituted_rows"] = int(frozen["substituted"].sum())
        report["loop_gate"] = gate
    report["award_conditions"] = award_conditions(report, "crps")
    report["definitions"] = definitions(2, episode_scope)
    return report


def _join_other_c2_lenient(test: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Join a second submission (the frozen copy); missing rows are left NaN for substitution."""
    live_rep = "samples" if "samples" in test.columns else "quantiles"
    other_rep = "samples" if "samples" in other.columns else "quantiles"
    if live_rep != other_rep:
        raise ValueError(
            f"the frozen copy uses {other_rep} while the live submission uses {live_rep}; "
            "both must use the same representation"
        )
    cols = [c for c in other.columns if c != "horizon"]
    joined = test[KEY + ["y_ret"]].merge(other[cols], on=["asset", "cutoff_utc"], how="left")
    if "abstain" not in joined.columns:
        joined["abstain"] = False
    joined["abstain"] = joined["abstain"].fillna(False).astype(bool)
    if live_rep == "quantiles":
        for c in QUANTILE_COLS:
            if c not in joined.columns:
                joined[c] = np.nan
    if "p_up" not in joined.columns:
        joined["p_up"] = np.nan
    joined.index = test.index
    return joined


# --------------------------------------------------------------------------- awards


def award_conditions(report: dict, metric: str) -> dict:
    """Section 7 conditions evaluable from one horizon report; the cross-horizon join is ``award``."""
    baselines = report.get("baselines") or {}
    beats = {name: b["comparison"][metric]["beats"] for name, b in baselines.items()}
    floor = report["coverage_floor"]
    out = {
        "metric": metric,
        "horizon": report["horizon"],
        "horizon_counts_for_forward_award": report["horizon"] in AWARD_HORIZONS,
        "baselines_compared": sorted(beats),
        "beats_each_baseline_with_ci": beats,
        "beats_every_baseline_with_ci": bool(beats) and all(beats.values()),
        "coverage_floor_met": floor["met"],
        "forward_award_condition_this_horizon": bool(beats)
        and all(beats.values())
        and floor["met"],
        "forward_award_requires": "the condition true at both h=5 and h=10; combine with `award`",
    }
    if report.get("loop_gate") is not None:
        out["loop_gate_passes"] = report["loop_gate"]["passes"]
        out["loop_gate_award_condition_this_horizon"] = bool(
            report["loop_gate"]["passes"] and floor["met"]
        )
    return out


def combine_awards(reports: list[dict]) -> dict:
    """Join per-horizon reports into the section 7 booleans for one submission."""
    by_h = {r["horizon"]: r for r in reports}
    challenge = {r["challenge"] for r in reports}
    if len(challenge) != 1:
        raise ValueError("reports must be from one challenge")
    challenge = challenge.pop()
    missing = [h for h in AWARD_HORIZONS if h not in by_h]
    ignored = sorted(h for h in by_h if h not in AWARD_HORIZONS)
    per_h = {
        str(h): {
            "forward_condition": by_h[h]["award_conditions"][
                "forward_award_condition_this_horizon"
            ],
            "beats_every_baseline_with_ci": by_h[h]["award_conditions"][
                "beats_every_baseline_with_ci"
            ],
            "coverage_floor_met": by_h[h]["award_conditions"]["coverage_floor_met"],
            "margins": {
                name: b["comparison"][by_h[h]["award_conditions"]["metric"]]["relative_margin"]
                for name, b in by_h[h]["baselines"].items()
            },
        }
        for h in AWARD_HORIZONS
        if h in by_h
    }
    forward = not missing and all(v["forward_condition"] for v in per_h.values())
    out = {
        "challenge": challenge,
        "horizons_present": sorted(by_h),
        "horizons_missing_for_award": missing,
        "horizons_ignored": ignored,
        "per_horizon": per_h,
        "forward_score_award": bool(forward),
        "forward_score_award_name": "C1 forward score" if challenge == 1 else "C2 forward score",
        "smallest_relative_margin_over_baselines": (
            min(m for v in per_h.values() for m in v["margins"].values() if m is not None)
            if forward and any(v["margins"] for v in per_h.values())
            else None
        ),
    }
    if challenge == 2:
        gates = {
            str(h): by_h[h]["award_conditions"].get("loop_gate_award_condition_this_horizon")
            for h in AWARD_HORIZONS
            if h in by_h
        }
        out["loop_gate_per_horizon"] = gates
        out["loop_gate_award"] = bool(not missing and all(bool(g) for g in gates.values()))
    return out


# --------------------------------------------------------------------------- definitions


def definitions(challenge: int, episode_scope: str = DEFAULT_EPISODE_SCOPE) -> dict:
    scope_text = {
        "calendar": "block of `horizon` consecutive MCX sessions counted from the first test cutoff; one episode for all assets on those sessions",
        "asset": "block of `horizon` consecutive MCX sessions counted from the first test cutoff, per asset; every (asset, block) pair is one episode",
    }[episode_scope]
    common = {
        "episode": scope_text,
        "episode_scope": episode_scope,
        "confidence_interval": f"percentile interval of the mean paired difference over {DEFAULT_BOOTSTRAP} episode-block bootstrap resamples, 90 percent, seeded",
        "beats": "mean difference (submission minus baseline) below zero with the interval excluding zero",
        "relative_margin": "(baseline - submission) / baseline; comparable across representations",
        "coverage": "issued (non-abstaining) predictions for eligible cutoffs divided by eligible cutoffs in the window",
        "coverage_floor": "issued share of at least 0.9 of the eligible cutoffs in the scored window; episode and asset counts are reported for information and do not gate",
    }
    if challenge == 1:
        common.update(
            {
                "log_loss": "mean of -log p(realized class), p clipped at 1e-15",
                "brier": "mean over rows of the summed squared error over the three class indicators, range [0, 2]",
                "ece": "classwise: all (row, class) pairs binned into ten equal-width bins on the predicted probability; weighted mean absolute gap between mean predicted probability and realized frequency",
                "top_label_ece": "ten-bin ECE of the argmax-class confidence against its accuracy",
                "swing_capture": "share of realized up/down swings whose class probability exceeded the 80th percentile of that probability over training rows of other classes (20 percent false-alarm threshold); pooled and per class",
                "abstention": "training class frequencies of the asset substituted and flagged",
            }
        )
    else:
        common.update(
            {
                "crps_samples": "exact empirical CRPS: mean |x_i - y| minus half the mean pairwise |x_i - x_j| over the 512 samples",
                "crps_quantiles": "twice the pinball losses over the nine levels summed with midpoint-rule weights over the level spacing; the discrete pinball integral, on the scale of the sample estimator",
                "pinball_mean": "twice the equal-weight mean pinball loss over the nine levels; reported for quantile submissions as a secondary number, it sits below the integral form because the levels crowd the tails",
                "brier_p_up": "mean (p_up - 1[y_ret > 0])^2",
                "pit": "samples: mid-rank empirical CDF at y_ret; quantiles: piecewise-linear CDF through the nine points, clamped to the outer levels in the tails",
                "coverage_90": "share of y_ret inside [q_050, q_950] (samples: the empirical 5 and 95 percent quantiles)",
                "loop_gate": "live minus frozen CRPS, same rows, same interval rule; passes when the live model is lower with the interval excluding zero",
                "abstention": "empirical training distribution of y_ret for the asset substituted (stratified 512-sample or its nine quantiles) and flagged; p_up becomes the training share of positive returns",
            }
        )
    return common


# --------------------------------------------------------------------------- baselines runner


def run_baselines_c1(
    labels: pd.DataFrame, horizon: int, chains: FeatureChains, targets: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    return {
        b.name: B.run_challenge1(b, targets)
        for b in B.challenge1_baselines(labels, horizon, chains)
    }


def run_baselines_c2(
    return_targets: pd.DataFrame,
    horizon: int,
    chains: FeatureChains,
    targets: pd.DataFrame,
    fmt: str,
) -> dict[str, pd.DataFrame]:
    return {
        b.name: B.run_challenge2(b, targets, fmt=fmt)
        for b in B.challenge2_baselines(return_targets, horizon, chains)
    }


def baseline_bar(
    preds: dict[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    horizon: int,
    challenge: int,
    window: Window,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    episode_scope: str = DEFAULT_EPISODE_SCOPE,
) -> dict:
    """Every baseline scored against every other on the same rows: the bar participants see."""
    parts = split_frames(next(iter(preds.values())), outcomes, window)
    test = parts["test"]
    scopes = episodes_for(test, outcomes, horizon, window)
    episodes = scopes[_check_scope(episode_scope)]
    rows: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict] = {}
    if challenge == 1:
        for name, bp in preds.items():
            joined = test[KEY + ["label"]].merge(
                bp.drop(columns=["horizon"]), on=["asset", "cutoff_utc"], how="left"
            )
            joined.index = test.index
            rows[name] = score_c1(joined)
            P = M.check_probabilities(joined[list(PROB_COLS)].to_numpy(dtype=float))
            y = M.class_indices(joined["label"])
            ece, table = M.ece_classwise(P, y)
            summary[name] = {
                "log_loss": float(rows[name]["log_loss"].mean()),
                "brier": float(rows[name]["brier"].mean()),
                "ece": ece,
                "reliability": table,
                "fallback_rows": int(bp["fallback"].sum()) if "fallback" in bp else None,
            }
        metrics = ["log_loss", "brier"]
    else:
        for name, bp in preds.items():
            joined = _join_other_c2(test, bp, name)
            r, _ = score_c2(joined)
            rows[name] = r
            summary[name] = {
                "crps": float(r["crps"].mean()),
                "brier_p_up": float(r["brier_p_up"].mean()),
                "coverage_90": float(r["in_90"].mean()),
                "pit_histogram": M.pit_histogram(r["pit"].to_numpy()),
                "fallback_rows": int(bp["fallback"].sum()) if "fallback" in bp else None,
            }
            if "pinball_mean" in r:
                summary[name]["pinball_mean"] = float(r["pinball_mean"].mean())
        metrics = ["crps", "brier_p_up"]
    pairwise = {
        a: {
            b: compare_rows(rows[a], rows[b], episodes, metrics, n_boot, seed)
            for b in preds
            if b != a
        }
        for a in preds
    }
    realized = None
    if challenge == 1:
        counts = test["label"].value_counts()
        realized = {c: int(counts.get(c, 0)) for c in CLASSES}
    return {
        "challenge": challenge,
        "horizon": horizon,
        "window": window.to_dict(),
        "counts": parts["counts"],
        "n_scored": len(test),
        "n_episodes": int(pd.Series(episodes).nunique()),
        "n_episodes_by_scope": {k: int(pd.Series(v).nunique()) for k, v in scopes.items()},
        "assets": sorted(test["asset"].unique()),
        "realized_classes": realized,
        "baselines": summary,
        "pairwise": pairwise,
        "definitions": definitions(challenge, episode_scope),
    }


def sanitize(value):
    """Make a report JSON-clean: numpy scalars to Python, NaN and infinities to None."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [sanitize(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def dump_json(report: dict, path: Path | None) -> str:
    text = json.dumps(sanitize(report), indent=2, sort_keys=False, default=_json_default) + "\n"
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value)}")
