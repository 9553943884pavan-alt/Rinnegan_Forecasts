"""Backward-looking feature chain per asset from ``mcx_daily.parquet``.

The price-only baselines need a continuous daily series available before each cutoff. This
module builds it with the CHALLENGE-SPEC section 2 chain rule, horizon-free: on every session
the designated contract is the one with the highest volume over the two most recent completed
sessions, excluding contracts with fewer than ``min_sessions_to_expiry`` sessions left (the
staggered tender period is at most five trading days for the metals; crude is cash settled), ties
by earlier expiry. The chain return on a session is the same-contract log settlement return of
that session's designated contract, so the calendar spread never enters. Sessions where the
designation changes are flagged ``transition`` and, as in the sigma_c rule, are excluded from
the realized-volatility windows (the windows extend backwards to keep their length).

The label chain (barriers, roll plans, delivery-period exclusion per contract specification)
is the labels lane's; this chain only feeds features and its rules are stated in FORMATS.md.
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

IST = ZoneInfo("Asia/Kolkata")
CUTOFF_IST = time(8, 30)
ASSETS = ("gold", "silver", "copper", "crude")
DAILY_COLUMNS = [
    "session_date",
    "asset",
    "contract_id",
    "expiry_date",
    "open",
    "high",
    "low",
    "close",
    "settlement",
    "volume_lots",
    "open_interest",
    "available_utc",
]
FEATURE_COLUMNS = ["ret1", "ret5", "ret20", "log_rv20", "stretch"]
VARIANCE_FLOOR = 1e-10


def _utc_ns(values: pd.Series) -> np.ndarray:
    """tz-aware timestamps as int64 nanoseconds, so numpy can search them."""
    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def _utc_ns_scalar(value: pd.Timestamp) -> int:
    return int(pd.Timestamp(value).tz_convert("UTC").value)


def cutoff_instant(session_date: date) -> pd.Timestamp:
    """The MCX pre-open cutoff of a session: 08:30 Asia/Kolkata, as a UTC timestamp."""
    return pd.Timestamp(datetime.combine(session_date, CUTOFF_IST, tzinfo=IST)).tz_convert("UTC")


def cutoffs_for(session_dates: pd.Series) -> pd.Series:
    naive = pd.to_datetime(session_dates).dt.normalize() + pd.Timedelta(hours=8, minutes=30)
    return naive.dt.tz_localize(IST).dt.tz_convert("UTC").astype("datetime64[us, UTC]")


def load_mcx_daily(path: Path) -> pd.DataFrame:
    """Rows of the four challenge assets, sorted by asset, session, expiry.

    Only the columns present in the file are read; the chain needs session_date, asset,
    contract_id, expiry_date, settlement, volume_lots and available_utc, the rest is optional.
    """
    present = set(pq.read_schema(path).names)
    frame = pq.read_table(path, columns=[c for c in DAILY_COLUMNS if c in present]).to_pandas()
    frame = frame[frame["asset"].isin(ASSETS)].copy()
    frame["available_utc"] = pd.to_datetime(frame["available_utc"], utc=True)
    return frame.sort_values(["asset", "session_date", "expiry_date"]).reset_index(drop=True)


def _sessions_to_expiry(sessions: np.ndarray, expiry_dates: np.ndarray) -> np.ndarray:
    """Sessions in (session_k, expiry] as an offset from k=0, for each contract.

    Inside the data the count uses the session list; past the last session it continues in
    weekdays, so contracts expiring after the data end keep a finite, approximately right count.
    """
    last = sessions[-1]
    in_data = np.searchsorted(sessions, expiry_dates, side="right") - 1
    beyond = expiry_dates > last
    extra = np.zeros(len(expiry_dates), dtype=int)
    if beyond.any():
        start = np.datetime64(last, "D") + np.timedelta64(1, "D")
        ends = expiry_dates[beyond].astype("datetime64[D]") + np.timedelta64(1, "D")
        extra[beyond] = np.busday_count(start, ends)
    return in_data + extra


def designate(asset_rows: pd.DataFrame, min_sessions_to_expiry: int = 5) -> pd.DataFrame:
    """Per session of one asset: the designated contract and its same-contract settlement return.

    Returns a frame indexed 0..n-1 in session order with columns session_date, cutoff_utc,
    contract_id, expiry_date, settlement, prev_settlement, high, low, volume_lots, ret,
    transition, available_utc. The first two sessions have no designation (nothing completed
    before them) and are dropped, as is any session without an eligible traded contract.
    """
    rows = asset_rows.sort_values(["session_date", "expiry_date"])
    sessions = np.array(sorted(rows["session_date"].unique()))
    contracts = rows.drop_duplicates("contract_id").sort_values(["expiry_date", "contract_id"])
    contract_ids = contracts["contract_id"].to_numpy()
    expiry_dates = contracts["expiry_date"].to_numpy()
    expiry_pos = _sessions_to_expiry(sessions, expiry_dates)

    def grid(column: str, fill: float) -> np.ndarray:
        wide = rows.pivot(index="session_date", columns="contract_id", values=column)
        return wide.reindex(index=sessions, columns=contract_ids).to_numpy(
            dtype=float, na_value=fill
        )

    volume = grid("volume_lots", 0.0)
    settlement = grid("settlement", np.nan)
    high = grid("high", np.nan)
    low = grid("low", np.nan)
    present = ~np.isnan(settlement)
    available = rows.groupby("session_date")["available_utc"].first().reindex(sessions)

    out: list[dict] = []
    previous: int | None = None
    for k in range(2, len(sessions)):
        two = volume[k - 2] + volume[k - 1]
        eligible = (expiry_pos - k >= min_sessions_to_expiry) & (two > 0)
        if not eligible.any():
            previous = None
            continue
        two = np.where(eligible, two, -np.inf)
        chosen = int(
            np.argmax(two)
        )  # contracts are sorted by expiry, so argmax breaks ties earlier
        today = settlement[k, chosen] if present[k, chosen] else np.nan
        prev = settlement[k - 1, chosen] if present[k - 1, chosen] else np.nan
        ret = float(np.log(today / prev)) if today > 0 and prev > 0 else np.nan
        out.append(
            {
                "session_date": sessions[k],
                "cutoff_utc": cutoff_instant(sessions[k]),
                "contract_id": contract_ids[chosen],
                "expiry_date": expiry_dates[chosen],
                "settlement": today,
                "prev_settlement": prev,
                "high": high[k, chosen],
                "low": low[k, chosen],
                "volume_lots": volume[k, chosen],
                "ret": ret,
                "transition": previous is not None and chosen != previous,
                "available_utc": available.iloc[k],
            }
        )
        previous = chosen
    frame = pd.DataFrame(out)
    if frame.empty:
        return frame
    frame["cutoff_utc"] = pd.to_datetime(frame["cutoff_utc"], utc=True)
    frame["available_utc"] = pd.to_datetime(frame["available_utc"], utc=True)
    return frame.reset_index(drop=True)


def _compacted_rolling(values: pd.Series, mask: pd.Series, window: int, fn: str) -> pd.Series:
    """Rolling statistic over the last ``window`` values where ``mask`` holds, carried forward.

    The window skips masked-out rows and extends backwards to keep ``window`` observations,
    which is the sigma_c convention for roll transitions.
    """
    kept = values.where(mask)
    compact = kept.dropna()
    stat = getattr(compact.rolling(window), fn)()
    return stat.reindex(values.index).ffill()


def add_features(chain: pd.DataFrame) -> pd.DataFrame:
    """Row k carries the features known once session k has closed (usable at any later cutoff).

    ret1: chain log return of the session. ret5, ret20: log wealth change over the last 5 and
    20 sessions (transitions included; the wealth chain is continuous by construction).
    rv20: standard deviation (ddof=1) of the last 20 non-transition returns. log_rv20: its log.
    stretch: (log W - 20-session mean of log W) / 20-session std of log W, 0 when the std is 0.
    rv_d, rv_w, rv_m: HAR-RV components, squared non-transition returns averaged over 1, 5 and
    22 sessions. All windows are on the non-transition series and extend backwards.
    """
    frame = chain.copy()
    ret = frame["ret"].astype(float)
    frame["log_w"] = ret.fillna(0.0).cumsum()
    frame["ret1"] = ret
    frame["ret5"] = frame["log_w"] - frame["log_w"].shift(5)
    frame["ret20"] = frame["log_w"] - frame["log_w"].shift(20)
    ok = ~frame["transition"].astype(bool) & ret.notna()
    frame["rv20"] = _compacted_rolling(ret, ok, 20, "std")
    frame["log_rv20"] = np.log(frame["rv20"].clip(lower=np.sqrt(VARIANCE_FLOOR)))
    mean20 = frame["log_w"].rolling(20).mean()
    std20 = frame["log_w"].rolling(20).std()
    stretch = (frame["log_w"] - mean20) / std20
    frame["stretch"] = stretch.where(std20 > 0, 0.0)
    sq = ret.pow(2)
    frame["rv_d"] = _compacted_rolling(sq, ok, 1, "mean")
    frame["rv_w"] = _compacted_rolling(sq, ok, 5, "mean")
    frame["rv_m"] = _compacted_rolling(sq, ok, 22, "mean")
    return frame


def forward_realized_variance(chain: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """HAR target per cutoff: mean squared chain return over the ``horizon`` sessions after it.

    Row for cutoff at session t covers sessions t .. t+h-1 and becomes available when session
    t+h-1 closes. Only cutoffs with all h sessions present are returned.
    """
    ret = chain["ret"].astype(float).to_numpy()
    n = len(chain)
    out = []
    for i in range(n - horizon + 1):
        window = ret[i : i + horizon]
        if np.isnan(window).any():
            continue
        out.append(
            {
                "cutoff_utc": chain["cutoff_utc"].iloc[i],
                "fwd_rv": float(np.mean(window**2)),
                "available_utc": chain["available_utc"].iloc[i + horizon - 1],
            }
        )
    return pd.DataFrame(out, columns=["cutoff_utc", "fwd_rv", "available_utc"])


class FeatureChains:
    """Feature chains for every asset, with point-in-time lookups."""

    def __init__(self, daily: pd.DataFrame, min_sessions_to_expiry: int = 5) -> None:
        self.chains: dict[str, pd.DataFrame] = {}
        for asset, rows in daily.groupby("asset", sort=True):
            chain = designate(rows, min_sessions_to_expiry=min_sessions_to_expiry)
            if not chain.empty:
                self.chains[asset] = add_features(chain)

    @classmethod
    def from_parquet(cls, path: Path, **kwargs) -> FeatureChains:
        return cls(load_mcx_daily(path), **kwargs)

    def assets(self) -> list[str]:
        return sorted(self.chains)

    def latest_before(self, asset: str, as_of: pd.Timestamp) -> pd.Series | None:
        """The last chain row whose availability instant is strictly before ``as_of``."""
        chain = self.chains.get(asset)
        if chain is None:
            return None
        idx = int(np.searchsorted(_utc_ns(chain["available_utc"]), _utc_ns_scalar(as_of), "left"))
        if idx == 0:
            return None
        return chain.iloc[idx - 1]

    def features_at(self, asset: str, as_of: pd.Timestamp) -> dict[str, float] | None:
        """Baseline features at a cutoff, or None when the chain is too short or has a gap."""
        row = self.latest_before(asset, as_of)
        if row is None:
            return None
        values = {c: float(row[c]) for c in FEATURE_COLUMNS + ["rv20", "rv_d", "rv_w", "rv_m"]}
        if not all(np.isfinite(v) for v in values.values()):
            return None
        return values

    def feature_frame(self, asset: str, cutoffs: pd.Series) -> pd.DataFrame:
        """Vectorised ``features_at`` for many cutoffs of one asset; NaN rows where unavailable."""
        chain = self.chains.get(asset)
        cols = FEATURE_COLUMNS + ["rv20", "rv_d", "rv_w", "rv_m"]
        cut = pd.to_datetime(pd.Series(cutoffs), utc=True).reset_index(drop=True)
        if chain is None:
            return pd.DataFrame(np.nan, index=range(len(cut)), columns=cols).assign(cutoff_utc=cut)
        idx = np.searchsorted(_utc_ns(chain["available_utc"]), _utc_ns(cut), side="left") - 1
        picked = chain[cols].reindex(idx).reset_index(drop=True)
        picked.loc[idx < 0, :] = np.nan
        return picked.assign(cutoff_utc=cut)
