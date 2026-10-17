"""Point-in-Time Data Pipeline for Kronos Task 1.

Strictly mediates all data access through `PointInTime`.
Constructs roll-normalized continuous OHLCV series from the backward-looking wealth chain.
Ensures zero future information leakage: every query uses `asof = instant`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from swing_dataset.kit.loop import price_view
from swing_dataset.pack.pit import PointInTime
from swing_dataset.scoring.chain import FeatureChains, cutoff_instant
from swing_dataset.scoring2.prices import PriceSeries

IST = ZoneInfo("Asia/Kolkata")
CUTOFF_IST = time(8, 30)
PRICE_COLS = ["open", "high", "low", "close"]
ALL_COLS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass
class SeriesData:
    asset: str
    as_of: datetime
    p0: float
    contract_id: str
    last_close: float
    input_df: pd.DataFrame
    x_timestamp: pd.DatetimeIndex
    y_timestamp: pd.DatetimeIndex


class DataPipeline:
    """Caches point-in-time daily frames per session instant and builds roll-adjusted OHLCV."""

    def __init__(self, pack: PointInTime, max_context: int = 400) -> None:
        self.pack = pack
        self.max_context = max_context
        self._cached_asof: datetime | None = None
        self._cached_prices: PriceSeries | None = None
        self._cached_chains: FeatureChains | None = None
        self._cached_daily: pd.DataFrame | None = None
        self._cached_sessions: pd.DataFrame | None = None

    def _ensure_cache(self, as_of: datetime) -> None:
        if self._cached_asof == as_of and self._cached_prices is not None:
            return

        self._cached_asof = as_of
        self._cached_prices, self._cached_chains = price_view(
            self.pack, as_of, lookback_sessions=self.max_context + 20
        )
        self._cached_daily = self.pack.frame("mcx/mcx_daily", as_of).collect()
        self._cached_sessions = self.pack.frame("mcx/mcx_sessions", as_of).collect()

    def get_future_timestamps(self, as_of: datetime, pred_len: int = 10) -> pd.DatetimeIndex:
        """Derive the next `pred_len` trading session pre-open timestamps."""
        sess_df = self._cached_sessions
        current_date = as_of.astimezone(IST).date()
        future_dates: list[date] = []

        if sess_df is not None and len(sess_df):
            open_states = {"open", "evening_only", "morning_only"}
            matching = sess_df[
                (sess_df["session_date"] >= current_date)
                & (sess_df["state"].astype(str).str.lower().isin(open_states))
            ]["session_date"].tolist()
            future_dates.extend(matching[:pred_len])

        # Fallback if published sessions end
        if len(future_dates) < pred_len:
            last_date = future_dates[-1] if future_dates else current_date
            next_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=pred_len - len(future_dates))
            future_dates.extend([d.date() for d in next_dates])

        future_timestamps = [cutoff_instant(d) for d in future_dates[:pred_len]]
        return pd.DatetimeIndex(future_timestamps)

    def build_series(self, asset: str, as_of: datetime, pred_len: int = 10) -> SeriesData | None:
        """Construct the roll-normalized continuous OHLCV inputs for Kronos."""
        self._ensure_cache(as_of)
        prices = self._cached_prices
        chains = self._cached_chains
        daily = self._cached_daily

        if prices is None or chains is None or daily is None:
            return None

        contract_id = prices.designated_contract(asset, as_of)
        if contract_id is None:
            return None

        quoted = prices.price_at(asset, contract_id, as_of)
        if quoted is None or quoted[0] <= 0 or not np.isfinite(quoted[0]):
            return None
        p0 = float(quoted[0])

        if asset not in chains.chains:
            return None
        chain_df = chains.chains[asset]
        if chain_df.empty:
            return None

        p_frame = prices.frame(asset)
        if p_frame.empty:
            return None

        # Filter daily rows strictly for this asset
        daily_asset = daily[daily["asset"] == asset]
        if daily_asset.empty:
            return None

        # Merge designated contract per session with daily OHLCV and wealth chain log_w
        merged = p_frame[["session_date", "contract_id"]].merge(
            daily_asset, on=["session_date", "contract_id"], how="left"
        )
        merged = merged.merge(
            chain_df[["session_date", "log_w", "settlement"]],
            on="session_date",
            how="inner",
            suffixes=("", "_chain"),
        )

        if len(merged) == 0:
            return None

        # Only consider sessions published strictly before as_of
        # Note: p_frame & chains already enforce point-in-time available_utc <= as_of
        # Slice to max_context
        if len(merged) > self.max_context:
            merged = merged.iloc[-self.max_context :].reset_index(drop=True)

        # Fill any missing OHLC from settlement
        for col in PRICE_COLS:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged["settlement"])
            else:
                merged[col] = merged["settlement"]

        # Roll normalization via wealth index
        last_settle = float(merged["settlement"].iloc[-1])
        last_log_w = float(merged["log_w"].iloc[-1])

        # Wealth index scaled so last session wealth == last settlement
        # wealth_s = last_settle * exp(log_w_s - last_log_w)
        wealth = np.exp(merged["log_w"].to_numpy(dtype=float) - last_log_w) * last_settle
        settle_arr = merged["settlement"].to_numpy(dtype=float)
        
        # Avoid zero-division
        valid_settle = np.where(settle_arr > 0, settle_arr, 1.0)
        factor = wealth / valid_settle

        # Adjusted OHLC
        adj_open = merged["open"].to_numpy(dtype=float) * factor
        adj_high = merged["high"].to_numpy(dtype=float) * factor
        adj_low = merged["low"].to_numpy(dtype=float) * factor
        adj_close = merged["close"].to_numpy(dtype=float) * factor

        # Strictly enforce valid OHLC bounds
        adj_high = np.maximum(adj_high, np.maximum(adj_open, adj_close))
        adj_low = np.minimum(adj_low, np.minimum(adj_open, adj_close))

        # Volume and Amount
        vol = merged["volume_lots"].fillna(0.0).to_numpy(dtype=float) if "volume_lots" in merged else np.zeros(len(merged))
        amt = merged["value_inr"].fillna(0.0).to_numpy(dtype=float) if "value_inr" in merged else np.zeros(len(merged))

        # Construct timestamps
        session_dates = merged["session_date"].tolist()
        x_timestamps = pd.DatetimeIndex([cutoff_instant(d) for d in session_dates])
        y_timestamps = self.get_future_timestamps(as_of, pred_len=pred_len)

        input_df = pd.DataFrame(
            {
                "open": adj_open,
                "high": adj_high,
                "low": adj_low,
                "close": adj_close,
                "volume": vol,
                "amount": amt,
            },
            index=x_timestamps,
        )

        last_close = float(adj_close[-1])

        return SeriesData(
            asset=asset,
            as_of=as_of,
            p0=p0,
            contract_id=str(contract_id),
            last_close=last_close,
            input_df=input_df,
            x_timestamp=x_timestamps,
            y_timestamp=y_timestamps,
        )
