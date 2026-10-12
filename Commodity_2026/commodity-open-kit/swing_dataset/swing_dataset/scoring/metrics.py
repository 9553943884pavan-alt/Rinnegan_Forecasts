"""Scoring metrics for both challenges, per CHALLENGE-SPEC sections 3 and 4.

Every function that scores rows returns one number per row so the callers can pool, split by
asset, and bootstrap over episodes. Aggregate helpers are the plain means of those rows.

Definitions that the specification leaves open are fixed here and repeated in FORMATS.md:

- Probabilities are clipped to ``[PROB_FLOOR, 1]`` inside the log before averaging.
- Brier for three classes is the sum of squared errors over the three indicators, so it lies
  in ``[0, 2]``.
- ECE is the classwise expected calibration error: every scored row contributes three
  (predicted probability, class realized) pairs, binned into ten equal-width bins on the
  predicted probability. The top-label variant (confidence of the argmax class against its
  accuracy) is reported alongside.
- The false-alarm threshold for a class is the ``1 - false_alarm`` quantile of that class's
  predicted probability over training rows whose realized class is different; capture is the
  share of realized outcomes of the class whose predicted probability exceeded the threshold.
- CRPS from samples is the exact empirical form. CRPS from quantiles is the pinball integral
  ``2 * integral of pinball over tau`` evaluated on the nine levels with midpoint-rule weights
  over the level spacing; it tracks the sample estimator to a few percent. The plain nine-level
  average (twice the mean pinball loss, ``pinball_mean``) sits about a third lower because the
  levels crowd the tails, and is kept as a secondary number.
- PIT from samples is the mid-rank empirical CDF at the outcome; from quantiles it is the
  piecewise-linear CDF through the nine (quantile, level) points, flat in the tails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import CLASSES, PROB_COLS, QUANTILE_COLS, QUANTILE_LEVELS
from .folds import DEFAULT_BOOTSTRAP, block_bootstrap_mean

PROB_FLOOR = 1e-15
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
SWING_CLASSES = ("up_swing", "down_swing")


# --------------------------------------------------------------------------- challenge 1


def class_indices(labels: pd.Series | np.ndarray) -> np.ndarray:
    """Map class names to 0..2; raises on anything outside the three scored classes."""
    values = np.asarray(labels, dtype=object)
    unknown = sorted({v for v in values if v not in CLASS_INDEX})
    if unknown:
        raise ValueError(f"unscorable labels: {unknown}")
    return np.array([CLASS_INDEX[v] for v in values], dtype=int)


def check_probabilities(P: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Validate an (n, 3) probability matrix: finite, non-negative, rows summing to one."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != len(CLASSES):
        raise ValueError(f"expected an (n, {len(CLASSES)}) matrix, got {P.shape}")
    if not np.isfinite(P).all() or (P < -atol).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = P.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=atol):
        bad = np.argwhere(~np.isclose(sums, 1.0, atol=atol)).ravel()[:5]
        raise ValueError(f"probability rows do not sum to one at positions {bad.tolist()}")
    return np.clip(P, 0.0, 1.0)


def log_loss_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return -np.log(np.clip(P[np.arange(len(y)), y], PROB_FLOOR, 1.0))


def brier_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    onehot = np.eye(P.shape[1])[y]
    return ((P - onehot) ** 2).sum(axis=1)


def reliability_table(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> list[dict]:
    """Bin (probability, indicator) pairs into equal-width bins; one dict per bin."""
    p = np.asarray(p, dtype=float)
    hit = np.asarray(hit, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    table = []
    for b in range(bins):
        mask = which == b
        n = int(mask.sum())
        table.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": n,
                "mean_predicted": float(p[mask].mean()) if n else None,
                "observed_frequency": float(hit[mask].mean()) if n else None,
                "gap": float(abs(p[mask].mean() - hit[mask].mean())) if n else None,
            }
        )
    return table


def ece_from_table(table: list[dict]) -> float:
    total = sum(row["count"] for row in table)
    if total == 0:
        return float("nan")
    return float(sum(row["count"] / total * row["gap"] for row in table if row["count"]))


def ece_classwise(P: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Classwise ECE: all (row, class) pairs, binned on the predicted probability of the class."""
    P = np.asarray(P, dtype=float)
    onehot = np.eye(P.shape[1])[y]
    table = reliability_table(P.ravel(), onehot.ravel(), bins)
    return ece_from_table(table), table


def ece_top_label(P: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Top-label ECE: confidence of the argmax class against whether it was realized."""
    P = np.asarray(P, dtype=float)
    top = P.argmax(axis=1)
    table = reliability_table(P[np.arange(len(y)), top], (top == y).astype(float), bins)
    return ece_from_table(table), table


def false_alarm_threshold(p_class: np.ndarray, is_class: np.ndarray, false_alarm: float) -> float:
    """Threshold above which exactly ``false_alarm`` of the non-realized outcomes fire.

    The ``1 - false_alarm`` quantile of the class probability over rows where the class did not
    realize (linear interpolation). NaN when no such rows exist.
    """
    negatives = np.asarray(p_class, dtype=float)[~np.asarray(is_class, dtype=bool)]
    if negatives.size == 0:
        return float("nan")
    return float(np.quantile(negatives, 1.0 - false_alarm))


def swing_capture(
    P_test: np.ndarray,
    y_test: np.ndarray,
    P_train: np.ndarray,
    y_train: np.ndarray,
    false_alarm: float = 0.20,
) -> dict:
    """Share of realized swings whose class probability exceeded the training false-alarm threshold.

    Per swing class and pooled over the two. ``realized_false_alarm`` reports the false-alarm
    rate the threshold actually produced on the test rows, for the reliability of the number.
    """
    P_test = np.asarray(P_test, dtype=float)
    P_train = np.asarray(P_train, dtype=float)
    out: dict = {"false_alarm_target": false_alarm, "classes": {}}
    captured_total = 0
    realized_total = 0
    for name in SWING_CLASSES:
        c = CLASS_INDEX[name]
        tau = false_alarm_threshold(P_train[:, c], y_train == c, false_alarm)
        realized = y_test == c
        fired = P_test[:, c] > tau if np.isfinite(tau) else np.zeros(len(y_test), dtype=bool)
        n_realized = int(realized.sum())
        n_captured = int((fired & realized).sum())
        n_neg = int((~realized).sum())
        out["classes"][name] = {
            "threshold": None if not np.isfinite(tau) else tau,
            "realized": n_realized,
            "captured": n_captured,
            "capture": (n_captured / n_realized) if n_realized else None,
            "realized_false_alarm": (int((fired & ~realized).sum()) / n_neg) if n_neg else None,
        }
        captured_total += n_captured
        realized_total += n_realized
    out["realized"] = realized_total
    out["captured"] = captured_total
    out["capture"] = (captured_total / realized_total) if realized_total else None
    return out


def class_prior(labels: pd.Series | np.ndarray) -> np.ndarray:
    """Training class frequencies over the three scored classes (unsmoothed)."""
    y = class_indices(labels)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    if counts.sum() == 0:
        raise ValueError("no training labels for the prior")
    return counts / counts.sum()


def priors_by_asset(train_labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-asset class frequencies, with the pooled prior under the key ``"*"``."""
    scored = train_labels[train_labels["label"].isin(CLASSES)]
    priors = {"*": class_prior(scored["label"])}
    for asset, group in scored.groupby("asset"):
        priors[str(asset)] = class_prior(group["label"])
    return priors


def substitute_abstentions_c1(pred: pd.DataFrame, priors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Replace abstained or missing probability rows with the training prior and flag them.

    The asset's own prior is used when the training slice has that asset, else the pooled prior.
    Adds ``substituted`` (bool) and ``substituted_with`` (``"asset_prior"``, ``"pooled_prior"``
    or None).
    """
    out = pred.copy()
    abstain = out["abstain"].fillna(False).astype(bool)
    missing = out[list(PROB_COLS)].isna().any(axis=1)
    target = abstain | missing
    out["substituted"] = target
    out["substituted_with"] = None
    for i in np.flatnonzero(target.to_numpy()):
        asset = out["asset"].iat[i]
        key = asset if asset in priors else "*"
        out.loc[out.index[i], list(PROB_COLS)] = priors[key]
        out.loc[out.index[i], "substituted_with"] = (
            "asset_prior" if key == asset else "pooled_prior"
        )
    return out


# --------------------------------------------------------------------------- challenge 2


def crps_samples_rows(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS per row: E|X - y| - 0.5 E|X - X'| over an (n, m) sample matrix."""
    X = np.sort(np.asarray(samples, dtype=float), axis=1)
    y = np.asarray(y, dtype=float)
    m = X.shape[1]
    term1 = np.abs(X - y[:, None]).mean(axis=1)
    weights = 2.0 * np.arange(1, m + 1) - m - 1  # sum_i (2i - m - 1) x_(i) = sum_{i<j} (x_j - x_i)
    term2 = (weights * X).sum(axis=1) / (m * m)
    return term1 - term2


def pinball_rows(q: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = y - q
    return np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)


def quadrature_weights(levels=QUANTILE_LEVELS) -> np.ndarray:
    """Midpoint-rule weights on [0, 1] for a sorted level grid; they sum to one."""
    levels = np.asarray(levels, dtype=float)
    mids = (levels[1:] + levels[:-1]) / 2.0
    edges = np.concatenate([[0.0], mids, [1.0]])
    return np.diff(edges)


def crps_quantiles_rows(
    Q: np.ndarray, y: np.ndarray, levels=QUANTILE_LEVELS, weights: np.ndarray | None = None
) -> np.ndarray:
    """CRPS from quantiles: twice the pinball losses summed with quadrature weights.

    With ``weights`` None the midpoint-rule weights over the level spacing are used, so the sum
    approximates ``2 * integral of pinball over tau``, the CRPS itself, on the scale of the
    sample estimator. Pass explicit weights (they should sum to one) for another rule.
    """
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=float)
    losses = np.stack([pinball_rows(Q[:, k], y, tau) for k, tau in enumerate(levels)], axis=1)
    if weights is None:
        weights = quadrature_weights(levels)
    weights = np.asarray(weights, dtype=float)
    return 2.0 * (losses * weights).sum(axis=1)


def pinball_mean_rows(Q: np.ndarray, y: np.ndarray, levels=QUANTILE_LEVELS) -> np.ndarray:
    """Twice the equal-weight mean pinball loss over the levels: the plain nine-level average.

    Reported next to ``crps`` for quantile submissions as a secondary number. It understates the
    integral form when the levels crowd the tails, which the nine levels do.
    """
    n = len(levels)
    return crps_quantiles_rows(Q, y, levels, weights=np.full(n, 1.0 / n))


def brier_binary_rows(p: np.ndarray, event: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(event, dtype=float)) ** 2


def pit_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)[:, None]
    return ((X < y).sum(axis=1) + 0.5 * (X == y).sum(axis=1)) / X.shape[1]


def pit_from_quantiles(Q: np.ndarray, y: np.ndarray, levels=QUANTILE_LEVELS) -> np.ndarray:
    """Piecewise-linear CDF through the nine quantiles; tails clamp to the outer levels."""
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=float)
    levels = np.asarray(levels, dtype=float)
    out = np.empty(len(y))
    for i in range(len(y)):
        q = np.maximum.accumulate(Q[i])  # enforce monotone quantiles for interpolation
        out[i] = np.interp(y[i], q, levels, left=levels[0], right=levels[-1])
    return out


def pit_histogram(pit: np.ndarray, bins: int = 10) -> dict:
    """Counts per equal-width PIT bin with the chi-square test against uniformity."""
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    counts, edges = np.histogram(np.clip(pit, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    n = int(counts.sum())
    expected = n / bins if n else 0.0
    if n:
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        p_value = float(stats.chi2.sf(chi2, df=bins - 1))
    else:
        chi2, p_value = float("nan"), float("nan")
    return {
        "bins": bins,
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "expected_per_bin": expected,
        "chi2": chi2,
        "p_value": p_value,
        "n": n,
    }


def interval_coverage_rows(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return ((y >= np.asarray(lo, dtype=float)) & (y <= np.asarray(hi, dtype=float))).astype(float)


@dataclass(frozen=True, slots=True)
class DistributionRows:
    """A batch of Challenge 2 forecasts in one representation: samples or nine quantiles."""

    kind: str  # "samples" | "quantiles"
    values: np.ndarray  # (n, m) samples or (n, 9) quantiles
    p_up: np.ndarray

    def crps(self, y: np.ndarray) -> np.ndarray:
        """Exact empirical CRPS from samples; the quadrature pinball integral from quantiles."""
        if self.kind == "samples":
            return crps_samples_rows(self.values, y)
        return crps_quantiles_rows(self.values, y)

    def pinball_mean(self, y: np.ndarray) -> np.ndarray:
        """The plain nine-level pinball average; quantile representation only."""
        if self.kind != "quantiles":
            raise ValueError("pinball_mean is defined for the quantile representation only")
        return pinball_mean_rows(self.values, y)

    def pit(self, y: np.ndarray) -> np.ndarray:
        if self.kind == "samples":
            return pit_from_samples(self.values, y)
        return pit_from_quantiles(self.values, y)

    def interval(
        self, lo_level: float = 0.05, hi_level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "samples":
            return (
                np.quantile(self.values, lo_level, axis=1),
                np.quantile(self.values, hi_level, axis=1),
            )
        lo_k = QUANTILE_LEVELS.index(lo_level)
        hi_k = QUANTILE_LEVELS.index(hi_level)
        return self.values[:, lo_k], self.values[:, hi_k]

    def interval_coverage(self, y: np.ndarray) -> np.ndarray:
        lo, hi = self.interval()
        return interval_coverage_rows(lo, hi, y)


def distribution_rows(pred: pd.DataFrame) -> DistributionRows:
    """Read a Challenge 2 prediction frame into the representation it uses."""
    has_q = all(c in pred.columns for c in QUANTILE_COLS)
    has_s = "samples" in pred.columns
    if has_q and not has_s:
        return DistributionRows(
            "quantiles",
            pred[list(QUANTILE_COLS)].to_numpy(dtype=float),
            pred["p_up"].to_numpy(dtype=float),
        )
    if has_s and not has_q:
        matrix = np.stack([np.asarray(s, dtype=float) for s in pred["samples"]])
        return DistributionRows("samples", matrix, pred["p_up"].to_numpy(dtype=float))
    raise ValueError("a Challenge 2 frame needs the nine quantile columns or a samples column")


def empirical_targets_by_asset(train_targets: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sorted training ``y_ret`` per asset, plus the pooled set under ``"*"``."""
    finite = train_targets[np.isfinite(train_targets["y_ret"].astype(float))]
    out = {"*": np.sort(finite["y_ret"].to_numpy(dtype=float))}
    for asset, group in finite.groupby("asset"):
        out[str(asset)] = np.sort(group["y_ret"].to_numpy(dtype=float))
    if out["*"].size == 0:
        raise ValueError("no training targets for the empirical substitute")
    return out


def empirical_quantiles(values: np.ndarray, levels=QUANTILE_LEVELS) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=float), np.asarray(levels))


def empirical_samples(values: np.ndarray, n: int) -> np.ndarray:
    """Stratified draw: the empirical quantile function at midpoints (i + 0.5) / n."""
    grid = (np.arange(n) + 0.5) / n
    return np.quantile(np.asarray(values, dtype=float), grid)


def substitute_abstentions_c2(
    pred: pd.DataFrame, empirical: dict[str, np.ndarray], n_samples: int
) -> pd.DataFrame:
    """Replace abstained or missing rows with the empirical training distribution of the asset.

    Works in whichever representation the frame uses. ``p_up`` becomes the training share of
    positive returns. Adds ``substituted`` and ``substituted_with``.
    """
    out = pred.copy()
    uses_samples = "samples" in out.columns
    abstain = out["abstain"].fillna(False).astype(bool)
    if uses_samples:
        missing = out["samples"].isna()
    else:
        missing = out[list(QUANTILE_COLS)].isna().any(axis=1)
    target = abstain | missing | out["p_up"].isna()
    out["substituted"] = target
    out["substituted_with"] = None
    if uses_samples:
        out["samples"] = out["samples"].astype(object)
    for i in np.flatnonzero(target.to_numpy()):
        asset = out["asset"].iat[i]
        key = asset if asset in empirical else "*"
        values = empirical[key]
        row = out.index[i]
        if uses_samples:
            out.at[row, "samples"] = empirical_samples(values, n_samples)
        else:
            out.loc[row, list(QUANTILE_COLS)] = empirical_quantiles(values)
        out.loc[row, "p_up"] = float((values > 0).mean())
        out.loc[row, "substituted_with"] = "asset_empirical" if key == asset else "pooled_empirical"
    return out


# --------------------------------------------------------------------------- paired comparisons


def paired_comparison(
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    episodes: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Mean of ``rows_a - rows_b`` with the episode-block interval; ``beats`` when a is lower.

    ``relative_margin`` is ``(mean_b - mean_a) / mean_b``, comparable across CRPS estimators.
    """
    a = np.asarray(rows_a, dtype=float)
    b = np.asarray(rows_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired rows must align")
    ci = block_bootstrap_mean(a - b, episodes, n_boot=n_boot, seed=seed)
    mean_b = float(np.nanmean(b)) if b.size else float("nan")
    return {
        "submission": float(np.nanmean(a)) if a.size else float("nan"),
        "other": mean_b,
        "difference": ci.to_dict(),
        "relative_margin": (-ci.mean / mean_b) if mean_b else None,
        "beats": bool(ci.mean < 0 and ci.excludes_zero),
    }


def loop_gate(
    crps_live: np.ndarray,
    crps_frozen: np.ndarray,
    episodes: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """CHALLENGE-SPEC section 4 loop gate: the live model against its day-zero frozen copy."""
    comparison = paired_comparison(crps_live, crps_frozen, episodes, n_boot=n_boot, seed=seed)
    return {
        "rule": (
            "live model must beat its day-zero frozen copy on CRPS with the 90 percent "
            "episode-block interval excluding zero"
        ),
        "crps_live": comparison["submission"],
        "crps_frozen": comparison["other"],
        "difference": comparison["difference"],
        "relative_margin": comparison["relative_margin"],
        "passes": comparison["beats"],
    }
