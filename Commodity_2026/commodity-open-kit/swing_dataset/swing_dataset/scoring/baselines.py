"""Predeclared baselines for both challenges (CHALLENGE-SPEC sections 3 and 4).

Every baseline is constructed with the full label (or target) history and the feature chains,
then driven as a loop: ``fit(before_cutoff)`` uses only rows whose label was available strictly
before that instant, ``predict(asset, as_of)`` uses only chain rows available strictly before
``as_of``. ``run_challenge1`` and ``run_challenge2`` call ``fit`` once per cutoff and ``predict``
for every asset row at it, in cutoff order.

Challenge 1 (class probabilities over up_swing, down_swing, no_swing):

- ``class_frequency``: training class frequencies per asset (pooled when the asset has none).
- ``persistence``: a first-order Markov forecast from the realized class of the latest cutoff of
  the same asset whose label had matured before ``as_of``; the 3x3 transition matrix is fitted on
  training rows with add-one smoothing; the prior is used when no matured label exists yet.
- ``price_only_logit``: multinomial logistic regression (L2, C=1) pooled over assets with asset
  indicators, on chain returns over 1, 5 and 20 sessions in units of the 20-session realized
  volatility, the log of that volatility, and the 20-session stretch, all standardized on the
  training rows. Falls back to the class frequencies when the chain has no features yet.

Challenge 2 (distribution of the horizon log return):

- ``empirical``: the unconditional training distribution of ``y_ret`` per asset.
- ``rv_gaussian``: N(0, (kappa * rv20 * sqrt(h))^2), kappa the training root mean square of
  ``y_ret / (rv20 * sqrt(h))``.
- ``har_rv``: log-HAR (Corsi 2009) of the forward mean squared chain return over the horizon on
  the daily, weekly (5) and monthly (22) realized-variance components, pooled over assets with
  asset intercepts; the scale is sqrt(h * exp(prediction + s^2 / 2)) and the shape the training
  distribution of ``y_ret`` standardized by the fitted scale.
- ``price_only_ls``: the price-only feature set as a location-scale model: least-squares mean,
  the rv_gaussian scale, and the training distribution of the standardized residuals.

Distributions are represented as location, scale and a standardized shape (standard normal or a
sorted sample); quantiles and stratified samples come from the shape's quantile function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from . import CLASSES, PROB_COLS, QUANTILE_COLS, QUANTILE_LEVELS, SAMPLE_COUNT
from .chain import FeatureChains, forward_realized_variance
from .metrics import CLASS_INDEX, class_prior

MIN_FIT_ROWS = 30
TRANSITION_SMOOTHING = 1.0
LOGIT_FEATURES = ["z1", "z5", "z20", "log_rv20", "stretch"]
HAR_FEATURES = ["log_rv_d", "log_rv_w", "log_rv_m"]
VARIANCE_FLOOR = 1e-10


def _utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True)


# --------------------------------------------------------------------------- feature frames


def label_features(rows: pd.DataFrame, chains: FeatureChains) -> pd.DataFrame:
    """Chain features at every row's cutoff (point in time by construction), joined by position."""
    parts = []
    for asset, group in rows.groupby("asset", sort=False):
        feats = chains.feature_frame(str(asset), group["cutoff_utc"])
        feats.index = group.index
        parts.append(feats.drop(columns=["cutoff_utc"]))
    feats = pd.concat(parts).reindex(rows.index) if parts else pd.DataFrame(index=rows.index)
    out = rows.join(feats)
    rv = out["rv20"]
    out["z1"] = out["ret1"] / rv
    out["z5"] = out["ret5"] / (rv * np.sqrt(5.0))
    out["z20"] = out["ret20"] / (rv * np.sqrt(20.0))
    for c in ("rv_d", "rv_w", "rv_m"):
        out[f"log_{c}"] = np.log(out[c].clip(lower=VARIANCE_FLOOR))
    return out


def design_matrix(frame: pd.DataFrame, columns: list[str], assets: list[str]) -> np.ndarray:
    """Feature columns followed by one indicator per asset."""
    X = frame[columns].to_numpy(dtype=float)
    dummies = np.stack([(frame["asset"] == a).to_numpy(dtype=float) for a in assets], axis=1)
    return np.hstack([X, dummies])


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, n_features: int) -> Standardizer:
        mean = np.zeros(X.shape[1])
        std = np.ones(X.shape[1])
        mean[:n_features] = X[:, :n_features].mean(axis=0)
        s = X[:, :n_features].std(axis=0)
        std[:n_features] = np.where(s > 0, s, 1.0)
        return cls(mean, std)

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


# --------------------------------------------------------------------------- challenge 1


class ClassBaseline(Protocol):
    name: str

    def fit(self, before_cutoff: pd.Timestamp) -> None: ...

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[np.ndarray, bool]: ...


def _scored_labels(labels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = labels[(labels["horizon"] == horizon) & labels["label"].isin(CLASSES)].copy()
    if "k_multiplier" in out.columns:
        out = out[np.isclose(out["k_multiplier"].astype(float), 1.0)]
    out["cutoff_utc"] = _utc(out["cutoff_utc"]).to_numpy()
    out["available_utc"] = _utc(out["available_utc"]).to_numpy()
    out["y"] = out["label"].map(CLASS_INDEX).astype(int)
    return out.sort_values(["cutoff_utc", "asset"]).reset_index(drop=True)


class ClassFrequency:
    name = "class_frequency"

    def __init__(self, labels: pd.DataFrame, horizon: int) -> None:
        self.rows = _scored_labels(labels, horizon)
        self.priors: dict[str, np.ndarray] = {}

    def fit(self, before_cutoff: pd.Timestamp) -> None:
        train = self.rows[self.rows["available_utc"] < before_cutoff]
        self.priors = {}
        if len(train):
            self.priors["*"] = class_prior(train["label"])
            for asset, group in train.groupby("asset"):
                self.priors[str(asset)] = class_prior(group["label"])

    def prior(self, asset: str) -> np.ndarray | None:
        return self.priors.get(asset, self.priors.get("*"))

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[np.ndarray, bool]:
        prior = self.prior(asset)
        if prior is None:
            return np.full(len(CLASSES), 1.0 / len(CLASSES)), True
        return prior, False


def latest_matured_state(rows: pd.DataFrame) -> pd.Series:
    """Per row, the class index of the latest cutoff of the same asset matured before its cutoff.

    -1 where nothing had matured yet.
    """
    state = pd.Series(-1, index=rows.index, dtype=int)
    for _, group in rows.groupby("asset", sort=False):
        by_avail = group.sort_values("available_utc")
        avail = by_avail["available_utc"].to_numpy()
        cut_sorted = by_avail["cutoff_utc"].to_numpy()
        y_sorted = by_avail["y"].to_numpy()
        # Running argmax of cutoff along the availability order.
        best_idx = np.empty(len(by_avail), dtype=int)
        best = 0
        for i in range(len(by_avail)):
            if cut_sorted[i] >= cut_sorted[best]:
                best = i
            best_idx[i] = best
        k = np.searchsorted(avail, group["cutoff_utc"].to_numpy(), side="left")
        have = k > 0
        picked = np.where(have, y_sorted[best_idx[np.clip(k - 1, 0, None)]], -1)
        state.loc[group.index] = picked
    return state


class Persistence:
    name = "persistence"

    def __init__(self, labels: pd.DataFrame, horizon: int) -> None:
        self.rows = _scored_labels(labels, horizon)
        self.rows["prev_state"] = latest_matured_state(self.rows)
        self.frequency = ClassFrequency(labels, horizon)
        self.transition = np.full((len(CLASSES), len(CLASSES)), 1.0 / len(CLASSES))

    def fit(self, before_cutoff: pd.Timestamp) -> None:
        self.frequency.fit(before_cutoff)
        train = self.rows[
            (self.rows["available_utc"] < before_cutoff) & (self.rows["prev_state"] >= 0)
        ]
        counts = np.full((len(CLASSES), len(CLASSES)), TRANSITION_SMOOTHING)
        np.add.at(counts, (train["prev_state"].to_numpy(), train["y"].to_numpy()), 1.0)
        self.transition = counts / counts.sum(axis=1, keepdims=True)

    def state_at(self, asset: str, as_of: pd.Timestamp) -> int:
        rows = self.rows[(self.rows["asset"] == asset) & (self.rows["available_utc"] < as_of)]
        if rows.empty:
            return -1
        return int(rows.loc[rows["cutoff_utc"].idxmax(), "y"])

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[np.ndarray, bool]:
        state = self.state_at(asset, as_of)
        if state < 0:
            return self.frequency.predict(asset, as_of)[0], True
        return self.transition[state], False


class PriceOnlyLogit:
    name = "price_only_logit"

    def __init__(
        self, labels: pd.DataFrame, horizon: int, chains: FeatureChains, C: float = 1.0
    ) -> None:
        self.chains = chains
        self.assets = chains.assets()
        rows = _scored_labels(labels, horizon)
        self.rows = label_features(rows, chains)
        self.frequency = ClassFrequency(labels, horizon)
        self.C = C
        self.model: LogisticRegression | None = None
        self.scaler: Standardizer | None = None
        self.n_train = 0

    def _usable(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[np.isfinite(frame[LOGIT_FEATURES].to_numpy(dtype=float)).all(axis=1)]

    def fit(self, before_cutoff: pd.Timestamp) -> None:
        self.frequency.fit(before_cutoff)
        train = self._usable(self.rows[self.rows["available_utc"] < before_cutoff])
        self.n_train = len(train)
        self.model = None
        if len(train) < MIN_FIT_ROWS or train["y"].nunique() < 2:
            return
        X = design_matrix(train, LOGIT_FEATURES, self.assets)
        self.scaler = Standardizer.fit(X, len(LOGIT_FEATURES))
        model = LogisticRegression(C=self.C, max_iter=2000)
        model.fit(self.scaler.apply(X), train["y"].to_numpy())
        self.model = model

    def _features(self, asset: str, as_of: pd.Timestamp) -> np.ndarray | None:
        feats = self.chains.features_at(asset, as_of)
        if feats is None or not feats["rv20"] > 0:
            return None
        rv = feats["rv20"]
        frame = pd.DataFrame(
            [
                {
                    "asset": asset,
                    "z1": feats["ret1"] / rv,
                    "z5": feats["ret5"] / (rv * np.sqrt(5.0)),
                    "z20": feats["ret20"] / (rv * np.sqrt(20.0)),
                    "log_rv20": feats["log_rv20"],
                    "stretch": feats["stretch"],
                }
            ]
        )
        return design_matrix(frame, LOGIT_FEATURES, self.assets)

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[np.ndarray, bool]:
        X = self._features(asset, as_of)
        if self.model is None or self.scaler is None or X is None:
            return self.frequency.predict(asset, as_of)[0], True
        proba = self.model.predict_proba(self.scaler.apply(X))[0]
        full = np.zeros(len(CLASSES))
        full[self.model.classes_] = proba
        return full, False

    def weights(self) -> dict | None:
        """Fitted coefficients, for the kit."""
        if self.model is None or self.scaler is None:
            return None
        names = LOGIT_FEATURES + [f"asset_{a}" for a in self.assets]
        classes = [CLASSES[i] for i in self.model.classes_]
        return {
            "features": names,
            "standardize_mean": self.scaler.mean.tolist(),
            "standardize_std": self.scaler.std.tolist(),
            "classes": classes,
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "C": self.C,
            "n_train": self.n_train,
        }


def challenge1_baselines(labels: pd.DataFrame, horizon: int, chains: FeatureChains) -> list:
    return [
        ClassFrequency(labels, horizon),
        Persistence(labels, horizon),
        PriceOnlyLogit(labels, horizon, chains),
    ]


def run_challenge1(baseline, targets: pd.DataFrame) -> pd.DataFrame:
    """Drive a Challenge 1 baseline over ``targets`` rows (asset, cutoff_utc, horizon).

    Fits once per distinct cutoff in order, then predicts every asset at it. Returns the
    prediction frame in the submission format plus a ``fallback`` flag.
    """
    rows = targets[["asset", "cutoff_utc", "horizon"]].copy()
    rows["cutoff_utc"] = _utc(rows["cutoff_utc"]).to_numpy()
    rows = rows.sort_values(["cutoff_utc", "asset"]).reset_index(drop=True)
    probs = np.zeros((len(rows), len(CLASSES)))
    fallback = np.zeros(len(rows), dtype=bool)
    for cutoff, idx in rows.groupby("cutoff_utc", sort=True).groups.items():
        baseline.fit(cutoff)
        for i in idx:
            p, fb = baseline.predict(rows.at[i, "asset"], cutoff)
            probs[i] = p
            fallback[i] = fb
    out = rows.copy()
    for k, col in enumerate(PROB_COLS):
        out[col] = probs[:, k]
    out["abstain"] = False
    out["fallback"] = fallback
    out["baseline"] = baseline.name
    return out


# --------------------------------------------------------------------------- challenge 2


@dataclass(frozen=True, slots=True)
class DistributionForecast:
    """Location-scale-shape distribution; ``shape`` None means standard normal."""

    mu: float
    sigma: float
    shape: np.ndarray | None = field(default=None, repr=False)

    def _q(self, levels: np.ndarray) -> np.ndarray:
        if self.shape is None:
            return stats.norm.ppf(levels)
        return np.quantile(self.shape, levels)

    def _cdf(self, z: float) -> float:
        if self.shape is None:
            return float(stats.norm.cdf(z))
        s = self.shape
        return float(((s < z).sum() + 0.5 * (s == z).sum()) / s.size)

    def quantiles(self, levels=QUANTILE_LEVELS) -> np.ndarray:
        return self.mu + self.sigma * self._q(np.asarray(levels, dtype=float))

    def samples(self, n: int = SAMPLE_COUNT) -> np.ndarray:
        grid = (np.arange(n) + 0.5) / n
        return self.mu + self.sigma * self._q(grid)

    def cdf(self, y: float) -> float:
        return self._cdf((y - self.mu) / self.sigma)

    @property
    def p_up(self) -> float:
        return 1.0 - self.cdf(0.0)


class DistributionBaseline(Protocol):
    name: str

    def fit(self, before_cutoff: pd.Timestamp) -> None: ...

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[DistributionForecast, bool]: ...


def _targets(targets: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = targets[(targets["horizon"] == horizon)].copy()
    out = out[np.isfinite(out["y_ret"].astype(float))]
    out["cutoff_utc"] = _utc(out["cutoff_utc"]).to_numpy()
    out["available_utc"] = _utc(out["available_utc"]).to_numpy()
    out["y_ret"] = out["y_ret"].astype(float)
    return out.sort_values(["cutoff_utc", "asset"]).reset_index(drop=True)


class Empirical:
    name = "empirical"

    def __init__(self, targets: pd.DataFrame, horizon: int) -> None:
        self.rows = _targets(targets, horizon)
        self.values: dict[str, np.ndarray] = {}

    def fit(self, before_cutoff: pd.Timestamp) -> None:
        train = self.rows[self.rows["available_utc"] < before_cutoff]
        self.values = {}
        if len(train):
            self.values["*"] = np.sort(train["y_ret"].to_numpy())
            for asset, group in train.groupby("asset"):
                self.values[str(asset)] = np.sort(group["y_ret"].to_numpy())

    def forecast(self, asset: str) -> DistributionForecast | None:
        values = self.values.get(asset, self.values.get("*"))
        if values is None or values.size < 2:
            return None
        return DistributionForecast(0.0, 1.0, values)

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[DistributionForecast, bool]:
        forecast = self.forecast(asset)
        if forecast is None:
            return DistributionForecast(0.0, 1.0, np.array([-1e-3, 1e-3])), True
        return forecast, False


class RvGaussian:
    name = "rv_gaussian"

    def __init__(self, targets: pd.DataFrame, horizon: int, chains: FeatureChains) -> None:
        self.horizon = horizon
        self.chains = chains
        self.rows = label_features(_targets(targets, horizon), chains)
        self.empirical = Empirical(targets, horizon)
        self.kappa = 1.0
        self.n_train = 0

    def fit(self, before_cutoff: pd.Timestamp) -> None:
        self.empirical.fit(before_cutoff)
        train = self.rows[(self.rows["available_utc"] < before_cutoff) & (self.rows["rv20"] > 0)]
        self.n_train = len(train)
        if len(train) < MIN_FIT_ROWS:
            self.kappa = 1.0
            return
        scaled = train["y_ret"] / (train["rv20"] * np.sqrt(self.horizon))
        self.kappa = float(np.sqrt(np.mean(scaled**2)))

    def scale(self, asset: str, as_of: pd.Timestamp) -> float | None:
        feats = self.chains.features_at(asset, as_of)
        if feats is None or not feats["rv20"] > 0:
            return None
        return self.kappa * feats["rv20"] * np.sqrt(self.horizon)

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[DistributionForecast, bool]:
        sigma = self.scale(asset, as_of)
        if sigma is None or self.n_train < MIN_FIT_ROWS:
            return self.empirical.predict(asset, as_of)[0], True
        return DistributionForecast(0.0, sigma, None), False


def _ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


class HarRv:
    name = "har_rv"

    def __init__(self, targets: pd.DataFrame, horizon: int, chains: FeatureChains) -> None:
        self.horizon = horizon
        self.chains = chains
        self.assets = chains.assets()
        rows = label_features(_targets(targets, horizon), chains)
        # HAR target from the chain: mean squared chain return over the horizon after the cutoff.
        parts = []
        for asset, chain in chains.chains.items():
            fwd = forward_realized_variance(chain, horizon).assign(asset=asset)
            fwd = fwd.rename(columns={"available_utc": "fwd_available_utc"})
            parts.append(fwd)
        fwd_all = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["cutoff_utc", "fwd_rv", "fwd_available_utc", "asset"])
        )
        self.rows = rows.merge(fwd_all, on=["asset", "cutoff_utc"], how="left")
        self.empirical = Empirical(targets, horizon)
        self.beta: np.ndarray | None = None
        self.s2 = 0.0
        self.shape: np.ndarray | None = None
        self.n_train = 0

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        X = design_matrix(frame, HAR_FEATURES, self.assets)
        return X

    def fit(self, before_cutoff: pd.Timestamp) -> None:
        self.empirical.fit(before_cutoff)
        rows = self.rows
        usable = np.isfinite(rows[HAR_FEATURES].to_numpy(dtype=float)).all(axis=1)
        matured = (rows["available_utc"] < before_cutoff) & (
            rows["fwd_available_utc"] < before_cutoff
        )
        train = rows[usable & matured.to_numpy() & (rows["fwd_rv"] > 0)]
        self.n_train = len(train)
        self.beta = None
        if len(train) < MIN_FIT_ROWS:
            return
        X = self._design(train)
        target = np.log(train["fwd_rv"].clip(lower=VARIANCE_FLOOR).to_numpy())
        beta = _ols(X, target)
        resid = target - X @ beta
        self.beta = beta
        self.s2 = float(resid.var(ddof=X.shape[1])) if len(train) > X.shape[1] else 0.0
        sigma_train = np.sqrt(self.horizon * np.exp(X @ beta + self.s2 / 2.0))
        self.shape = np.sort(train["y_ret"].to_numpy() / sigma_train)

    def scale(self, asset: str, as_of: pd.Timestamp) -> float | None:
        if self.beta is None:
            return None
        feats = self.chains.features_at(asset, as_of)
        if feats is None:
            return None
        frame = pd.DataFrame(
            [
                {
                    "asset": asset,
                    **{
                        f"log_{c}": np.log(max(feats[c], VARIANCE_FLOOR))
                        for c in ("rv_d", "rv_w", "rv_m")
                    },
                }
            ]
        )
        pred = float((self._design(frame) @ self.beta)[0])
        return float(np.sqrt(self.horizon * np.exp(pred + self.s2 / 2.0)))

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[DistributionForecast, bool]:
        sigma = self.scale(asset, as_of)
        if sigma is None or self.shape is None or self.shape.size < 2:
            return self.empirical.predict(asset, as_of)[0], True
        return DistributionForecast(0.0, sigma, self.shape), False

    def coefficients(self) -> dict | None:
        if self.beta is None:
            return None
        names = HAR_FEATURES + [f"asset_{a}" for a in self.assets]
        return {
            "features": names,
            "beta": self.beta.tolist(),
            "s2": self.s2,
            "n_train": self.n_train,
        }


class PriceOnlyLocationScale:
    name = "price_only_ls"

    def __init__(self, targets: pd.DataFrame, horizon: int, chains: FeatureChains) -> None:
        self.horizon = horizon
        self.chains = chains
        self.assets = chains.assets()
        self.rows = label_features(_targets(targets, horizon), chains)
        self.empirical = Empirical(targets, horizon)
        self.beta: np.ndarray | None = None
        self.scaler: Standardizer | None = None
        self.kappa = 1.0
        self.shape: np.ndarray | None = None
        self.n_train = 0

    def fit(self, before_cutoff: pd.Timestamp) -> None:
        self.empirical.fit(before_cutoff)
        rows = self.rows
        usable = (
            np.isfinite(rows[LOGIT_FEATURES].to_numpy(dtype=float)).all(axis=1)
            & (rows["rv20"] > 0).to_numpy()
        )
        train = rows[usable & (rows["available_utc"] < before_cutoff).to_numpy()]
        self.n_train = len(train)
        self.beta = None
        if len(train) < MIN_FIT_ROWS:
            return
        X = design_matrix(train, LOGIT_FEATURES, self.assets)
        self.scaler = Standardizer.fit(X, len(LOGIT_FEATURES))
        Xs = self.scaler.apply(X)
        y = train["y_ret"].to_numpy()
        self.beta = _ols(Xs, y, ridge=1e-6)
        resid = y - Xs @ self.beta
        base = train["rv20"].to_numpy() * np.sqrt(self.horizon)
        self.kappa = float(np.sqrt(np.mean((resid / base) ** 2)))
        self.shape = np.sort(resid / (self.kappa * base))

    def predict(self, asset: str, as_of: pd.Timestamp) -> tuple[DistributionForecast, bool]:
        feats = self.chains.features_at(asset, as_of)
        if self.beta is None or self.scaler is None or feats is None or not feats["rv20"] > 0:
            return self.empirical.predict(asset, as_of)[0], True
        rv = feats["rv20"]
        frame = pd.DataFrame(
            [
                {
                    "asset": asset,
                    "z1": feats["ret1"] / rv,
                    "z5": feats["ret5"] / (rv * np.sqrt(5.0)),
                    "z20": feats["ret20"] / (rv * np.sqrt(20.0)),
                    "log_rv20": feats["log_rv20"],
                    "stretch": feats["stretch"],
                }
            ]
        )
        Xs = self.scaler.apply(design_matrix(frame, LOGIT_FEATURES, self.assets))
        mu = float((Xs @ self.beta)[0])
        sigma = self.kappa * rv * np.sqrt(self.horizon)
        return DistributionForecast(mu, sigma, self.shape), False

    def coefficients(self) -> dict | None:
        if self.beta is None or self.scaler is None:
            return None
        names = LOGIT_FEATURES + [f"asset_{a}" for a in self.assets]
        return {
            "features": names,
            "standardize_mean": self.scaler.mean.tolist(),
            "standardize_std": self.scaler.std.tolist(),
            "beta": self.beta.tolist(),
            "kappa": self.kappa,
            "n_train": self.n_train,
        }


def challenge2_baselines(targets: pd.DataFrame, horizon: int, chains: FeatureChains) -> list:
    return [
        Empirical(targets, horizon),
        RvGaussian(targets, horizon, chains),
        HarRv(targets, horizon, chains),
        PriceOnlyLocationScale(targets, horizon, chains),
    ]


def run_challenge2(
    baseline, targets: pd.DataFrame, fmt: str = "quantiles", n_samples: int = SAMPLE_COUNT
) -> pd.DataFrame:
    """Drive a Challenge 2 baseline over ``targets`` rows; output in ``quantiles`` or ``samples``."""
    if fmt not in ("quantiles", "samples"):
        raise ValueError("fmt must be quantiles or samples")
    rows = targets[["asset", "cutoff_utc", "horizon"]].copy()
    rows["cutoff_utc"] = _utc(rows["cutoff_utc"]).to_numpy()
    rows = rows.sort_values(["cutoff_utc", "asset"]).reset_index(drop=True)
    q = np.zeros((len(rows), len(QUANTILE_LEVELS)))
    samples: list[np.ndarray | None] = [None] * len(rows)
    p_up = np.zeros(len(rows))
    fallback = np.zeros(len(rows), dtype=bool)
    for cutoff, idx in rows.groupby("cutoff_utc", sort=True).groups.items():
        baseline.fit(cutoff)
        for i in idx:
            forecast, fb = baseline.predict(rows.at[i, "asset"], cutoff)
            if fmt == "quantiles":
                q[i] = forecast.quantiles()
            else:
                samples[i] = forecast.samples(n_samples)
            p_up[i] = forecast.p_up
            fallback[i] = fb
    out = rows.copy()
    if fmt == "quantiles":
        for k, col in enumerate(QUANTILE_COLS):
            out[col] = q[:, k]
    else:
        out["samples"] = samples
    out["p_up"] = p_up
    out["abstain"] = False
    out["fallback"] = fallback
    out["baseline"] = baseline.name
    return out
