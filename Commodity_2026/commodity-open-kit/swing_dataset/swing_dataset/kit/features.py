"""Price-only features from the historical designated-contract chain (CHALLENGE-SPEC section 2).

The chain return of session ``s`` uses the contract designated at cutoff ``s`` for the same
horizon: ``log(settle_s / settle_{s-1})`` on that contract, null when the designation changed from
the previous session, is absent, or a settlement is missing. This is the series behind
``sigma_c`` in ``labels/designations``, so the 20-return sample standard deviation computed here
equals ``sigma_c`` wherever both exist.

Inputs are pandas frames of ``labels/designations`` and ``mcx/mcx_daily`` as returned by the
point-in-time client; every function filters by ``available_utc < cutoff_utc`` again, so a frame
collected at a later instant gives the same features as one collected at the cutoff itself.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import KEY_COLUMNS

WINDOW = 20
FEATURE_NAMES = ("z_1", "z_5", "z_20", "stretch_20", "log_rv_20")
"""Model inputs: vol-normalized 1, 5 and 20 session chain returns, stretch, log realized vol."""

DESIGNATION_COLUMNS = ["asset", "cutoff_utc", "horizon", "cutoff_date", "contract_id", "eligible", "p_ref", "sigma_c", "available_utc"]
DAILY_COLUMNS = ["session_date", "asset", "contract_id", "settlement", "available_utc"]


def chain_return_series(designations: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, horizon, session) with the chain's daily log settlement return.

    Columns: asset, horizon, session_date, contract_id, ret (null when undefined), available_utc
    (the session's close instant, when the settlement row exists).
    """
    settle = daily[["contract_id", "session_date", "settlement", "available_utc"]].drop_duplicates(["contract_id", "session_date"])
    settle = settle.set_index(["contract_id", "session_date"])
    parts = []
    for (asset, horizon), group in designations.groupby(["asset", "horizon"], sort=True, observed=True):
        g = group.sort_values("cutoff_utc").reset_index(drop=True)
        contract = g["contract_id"].astype(object)
        prev_contract = contract.shift(1)
        dates = g["cutoff_date"]
        prev_dates = dates.shift(1)
        cur_idx = pd.MultiIndex.from_arrays([contract, dates])
        prev_idx = pd.MultiIndex.from_arrays([contract, prev_dates])
        cur = settle.reindex(cur_idx)
        prev = settle.reindex(prev_idx)
        s_now = cur["settlement"].to_numpy(dtype=float)
        s_prev = prev["settlement"].to_numpy(dtype=float)
        same = (contract.to_numpy() == prev_contract.to_numpy()) & contract.notna().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(same & np.isfinite(s_now) & np.isfinite(s_prev) & (s_prev > 0) & (s_now > 0), np.log(s_now / s_prev), np.nan)
        parts.append(
            pd.DataFrame(
                {
                    "asset": asset,
                    "horizon": int(horizon),
                    "session_date": dates.to_numpy(),
                    "contract_id": contract.to_numpy(),
                    "ret": ret,
                    "available_utc": cur["available_utc"].to_numpy(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["asset", "horizon", "session_date", "contract_id", "ret", "available_utc"])
    out = pd.concat(parts, ignore_index=True)
    out["available_utc"] = pd.to_datetime(out["available_utc"], utc=True)
    return out


def features_from_returns(returns: np.ndarray) -> dict[str, float] | None:
    """Features from chain returns ordered oldest to newest; None with fewer than WINDOW returns."""
    r = np.asarray(returns, dtype=float)
    if len(r) < WINDOW:
        return None
    r = r[-WINDOW:]
    rv = float(np.std(r, ddof=1))
    if not (math.isfinite(rv) and rv > 0):
        return None
    level = np.cumsum(r)
    ret_1 = float(r[-1])
    ret_5 = float(r[-5:].sum())
    ret_20 = float(r.sum())
    return {
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_20": ret_20,
        "rv_20": rv,
        "stretch_20": float((level[-1] - level.mean()) / (rv * math.sqrt(WINDOW))),
        "z_1": ret_1 / rv,
        "z_5": ret_5 / (rv * math.sqrt(5)),
        "z_20": ret_20 / (rv * math.sqrt(WINDOW)),
        "log_rv_20": math.log(rv),
    }


def feature_table(
    designations: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    cutoffs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Features at cutoffs, each using only chain returns with ``available_utc < cutoff_utc``.

    ``cutoffs`` restricts the output to the given (asset, cutoff_utc, horizon) keys; by default
    every designation row gets a row. Output columns: the keys, cutoff_date, contract_id,
    eligible, p_ref, sigma_c, n_returns, the feature columns (null when fewer than 20 returns are
    available), and ``has_features``.
    """
    series = chain_return_series(designations, daily)
    des = designations.copy()
    des["cutoff_utc"] = pd.to_datetime(des["cutoff_utc"], utc=True)
    if cutoffs is not None:
        keys = cutoffs[list(KEY_COLUMNS)].copy()
        keys["cutoff_utc"] = pd.to_datetime(keys["cutoff_utc"], utc=True)
        des = des.merge(keys, on=list(KEY_COLUMNS), how="inner")
    rows: list[dict] = []
    feature_keys = ["ret_1", "ret_5", "ret_20", "rv_20", "stretch_20", *FEATURE_NAMES]
    for (asset, horizon), group in des.groupby(["asset", "horizon"], sort=True, observed=True):
        s = series[(series["asset"] == asset) & (series["horizon"] == int(horizon))]
        s = s[s["ret"].notna() & s["available_utc"].notna()].sort_values("available_utc")
        avail = s["available_utc"].to_numpy(dtype="datetime64[us]")
        rets = s["ret"].to_numpy(dtype=float)
        for rec in group.sort_values("cutoff_utc").to_dict(orient="records"):
            cutoff = np.datetime64(pd.Timestamp(rec["cutoff_utc"]).tz_convert("UTC").tz_localize(None), "us")
            n = int(np.searchsorted(avail, cutoff, side="left"))
            feats = features_from_returns(rets[:n]) if n else None
            row = {
                "asset": asset,
                "cutoff_utc": rec["cutoff_utc"],
                "horizon": int(horizon),
                "cutoff_date": rec.get("cutoff_date"),
                "contract_id": rec.get("contract_id"),
                "eligible": bool(rec.get("eligible", True)),
                "p_ref": rec.get("p_ref"),
                "sigma_c": rec.get("sigma_c"),
                "n_returns": min(n, WINDOW),
                "has_features": feats is not None,
            }
            for key in feature_keys:
                row[key] = feats[key] if feats else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["cutoff_utc"] = pd.to_datetime(out["cutoff_utc"], utc=True)
        out = out.sort_values(["cutoff_utc", "asset", "horizon"]).reset_index(drop=True)
    return out


def matrix(features: pd.DataFrame) -> np.ndarray:
    """The model input matrix in FEATURE_NAMES order."""
    return features[list(FEATURE_NAMES)].to_numpy(dtype=float)
