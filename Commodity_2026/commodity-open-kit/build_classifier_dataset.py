"""Build Production-Grade 8 Datasets for Task 1: Classifier & Regressor for 4 Commodities.

Generates 8 individual commodity datasets + 2 unified datasets:
- 4 Classifier Datasets: task1_classifier_{gold,silver,copper,crude}.parquet
- 4 Regressor Datasets: task1_regressor_{gold,silver,copper,crude}.parquet
- 2 Unified Datasets: task1_classifier_unified.parquet, task1_regressor_unified.parquet

Feature Count: 55-60 rich, leak-free input features per dataset.
Leakage Prevention (Rules Section 6 & 9):
- All features strictly available_utc < cutoff_utc (08:30 IST / 03:00 UTC pre-open).
- Contract roll normalization via same-contract returns.
- No lookahead Bhav Copy, no unvintaged consensus calendar values.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add kit to python path
KIT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\swing_dataset")
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from swing_dataset.kit.features import feature_table

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
OUT_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FINBERT_CACHE = CACHE_DIR / "finbert_sentiment_cache.parquet"


def build_task1_suite():
    print("================================================================================")
    print("   COMMODITY OPEN 2026: BUILDING 8 CLASSIFIER & REGRESSOR DATASETS")
    print("   (Targeting 55-60 Features per Dataset with Strict Point-in-Time Clocks)")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # STEP 1: Ground Truth Labels & Target Horizons
    # -------------------------------------------------------------------------
    print("\n[Step 1/11] Loading Ground Truth Swing Labels & Return Targets...")
    labels = pd.read_parquet(PACK_BASE / "labels" / "swing_labels.parquet")
    labels_5 = labels[(labels["horizon"] == 5) & (labels["k_multiplier"] == 1.0)].copy()
    labels_5["cutoff_utc"] = pd.to_datetime(labels_5["cutoff_utc"], utc=True).astype("datetime64[us, UTC]")
    print(f"  Loaded {len(labels_5)} swing label rows (horizon=5, k=1.0).")

    # Load return targets for Regressors
    rt = pd.read_parquet(PACK_BASE / "labels" / "return_targets.parquet")
    rt_5 = rt[rt["horizon"] == 5].copy()
    rt_5["cutoff_utc"] = pd.to_datetime(rt_5["cutoff_utc"], utc=True).astype("datetime64[us, UTC]")
    rt_cols = ["asset", "cutoff_utc", "horizon", "y_ret", "y_mfe", "y_mae", "y_range"]
    targets_merged = pd.merge(
        labels_5[["cutoff_utc", "asset", "horizon", "label"]],
        rt_5[rt_cols],
        on=["cutoff_utc", "asset", "horizon"],
        how="inner",
    )
    print(f"  Merged targets: {len(targets_merged)} rows with classification & regression labels.")

    # -------------------------------------------------------------------------
    # STEP 2: Designations & MCX Daily Bhav Copy (Core Roll-Normalized Features)
    # -------------------------------------------------------------------------
    print("\n[Step 2/11] Generating Core Kit Features (Roll-Normalized & Leak-Free)...")
    designations = pd.read_parquet(PACK_BASE / "labels" / "designations.parquet")
    daily = pd.read_parquet(PACK_BASE / "mcx" / "mcx_daily.parquet")

    X_core = feature_table(designations, daily, cutoffs=labels_5)
    X_core["cutoff_utc"] = pd.to_datetime(X_core["cutoff_utc"], utc=True).astype("datetime64[us, UTC]")

    df = pd.merge(
        X_core,
        targets_merged,
        on=["cutoff_utc", "asset", "horizon"],
        how="inner",
    )
    print(f"  Core feature table merged: {len(df)} rows.")

    # -------------------------------------------------------------------------
    # STEP 3: Advanced Contract Microstructure & Range Geometry Features
    # -------------------------------------------------------------------------
    print("\n[Step 3/11] Computing Contract-Level Microstructure & Price Geometry...")
    traded = daily[daily["asset"].notnull() & (daily["volume_lots"] > 0)].copy()
    traded["available_utc"] = pd.to_datetime(traded["available_utc"], utc=True).astype("datetime64[us, UTC]")
    traded = traded.sort_values(["asset", "contract_id", "available_utc"]).reset_index(drop=True)

    # 1. Day-over-day and 5-day Open Interest changes
    traded["oi_prev"] = traded.groupby(["asset", "contract_id"])["open_interest"].shift(1)
    traded["oi_5d_ago"] = traded.groupby(["asset", "contract_id"])["open_interest"].shift(5)
    traded["settle_prev"] = traded.groupby(["asset", "contract_id"])["settlement"].shift(1)

    traded["oi_1s_change"] = ((traded["open_interest"] - traded["oi_prev"]) / traded["oi_prev"].replace(0, np.nan)).fillna(0.0)
    traded["oi_5s_change"] = ((traded["open_interest"] - traded["oi_5d_ago"]) / traded["oi_5d_ago"].replace(0, np.nan)).fillna(0.0)

    # 2. Open Interest Price Classification (definitions.md)
    dp = traded["settlement"] - traded["settle_prev"]
    doi = traded["open_interest"] - traded["oi_prev"]
    
    traded["oi_class_longs"] = ((dp > 0) & (doi > 0)).astype(float)
    traded["oi_class_short_cov"] = ((dp > 0) & (doi < 0)).astype(float)
    traded["oi_class_shorts"] = ((dp < 0) & (doi > 0)).astype(float)
    traded["oi_class_liquid"] = ((dp < 0) & (doi < 0)).astype(float)

    # 3. Session True Range ratio and Range Expansion
    high_low_diff = traded["high"] - traded["low"]
    traded["range_ratio"] = (high_low_diff / traded["settlement"]).fillna(0.0)
    traded["range_ma10"] = traded.groupby(["asset", "contract_id"])["range_ratio"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    traded["range_expansion_ratio"] = (traded["range_ratio"] / traded["range_ma10"].replace(0, np.nan)).fillna(1.0)

    # 4. Range Geometry: Close Position in Session Range (bullish push vs bearish push)
    traded["close_pos_in_range"] = (
        (traded["settlement"] - traded["low"]) / high_low_diff.replace(0, np.nan)
    ).fillna(0.5)

    # 5. True Range Percentage & ATR (5-day rolling average)
    traded["tr_pct"] = (high_low_diff / traded["settlement"] * 100.0).fillna(0.0)
    traded["atr_5d"] = traded.groupby(["asset", "contract_id"])["tr_pct"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # 6. Volume Surge, 1-day Volume Change & Log Volume
    traded["vol_ma10"] = traded.groupby(["asset", "contract_id"])["volume_lots"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    traded["vol_surge_10d"] = (traded["volume_lots"] / traded["vol_ma10"].replace(0, np.nan)).fillna(1.0)
    traded["vol_chg_1d"] = traded.groupby(["asset", "contract_id"])["volume_lots"].pct_change(1).fillna(0.0)
    traded["log_volume"] = np.log1p(traded["volume_lots"].astype(float))

    # 7. Distance to Trailing 5-Day High and Low (Breakout Proximity)
    high_5d = traded.groupby(["asset", "contract_id"])["high"].transform(lambda s: s.rolling(5, min_periods=1).max())
    low_5d = traded.groupby(["asset", "contract_id"])["low"].transform(lambda s: s.rolling(5, min_periods=1).min())
    traded["dist_5d_high"] = ((high_5d - traded["settlement"]) / traded["settlement"] * 100.0).fillna(0.0)
    traded["dist_5d_low"] = ((traded["settlement"] - low_5d) / traded["settlement"] * 100.0).fillna(0.0)

    # Backward as-of merge (strictly available_utc < cutoff_utc)
    micro_cols = [
        "available_utc", "asset", "contract_id",
        "oi_1s_change", "oi_5s_change",
        "oi_class_longs", "oi_class_short_cov", "oi_class_shorts", "oi_class_liquid",
        "range_ratio", "range_expansion_ratio", "close_pos_in_range", "tr_pct", "atr_5d",
        "vol_surge_10d", "vol_chg_1d", "log_volume",
        "dist_5d_high", "dist_5d_low"
    ]
    traded_asof = traded[micro_cols].sort_values("available_utc").reset_index(drop=True)

    df = df.sort_values("cutoff_utc").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        traded_asof,
        left_on="cutoff_utc",
        right_on="available_utc",
        by=["asset", "contract_id"],
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.drop(columns=["available_utc"])
    print("  Contract microstructure & price geometry features attached.")

    # -------------------------------------------------------------------------
    # STEP 4: CFTC Commitments of Traders (COT Smart Money Positioning)
    # -------------------------------------------------------------------------
    print("\n[Step 4/11] Joining Macro CFTC Commitments of Traders (COT) Data...")
    cot = pd.read_parquet(PACK_BASE / "macro" / "cot.parquet")
    cot = cot[cot["asset"].notnull()].copy()
    cot["available_utc"] = pd.to_datetime(cot["available_utc"], utc=True).astype("datetime64[us, UTC]")
    cot = cot.sort_values("available_utc").reset_index(drop=True)

    cot["cot_net_smart_money"] = (
        cot["m_money_positions_long_all"] - cot["m_money_positions_short_all"]
    ) / cot["open_interest_all"].replace(0, np.nan)
    cot["cot_weekly_change"] = cot["change_in_m_money_long_all"] / cot["open_interest_all"].replace(0, np.nan)
    cot["cot_crowded_trade"] = cot["pct_of_oi_m_money_long_all"] / 100.0

    cot_asof = cot[["available_utc", "asset", "cot_net_smart_money", "cot_weekly_change", "cot_crowded_trade"]].copy()
    cot_asof = cot_asof.dropna(subset=["available_utc"]).sort_values("available_utc").reset_index(drop=True)

    df = pd.merge_asof(
        df,
        cot_asof,
        left_on="cutoff_utc",
        right_on="available_utc",
        by="asset",
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.drop(columns=["available_utc"])
    print("  COT institutional positioning attached.")

    # -------------------------------------------------------------------------
    # STEP 5: Macro Currency Returns (USD/INR FBIL Rates)
    # -------------------------------------------------------------------------
    print("\n[Step 5/11] Joining Macro Currency (USD/INR) Rates & Momentum...")
    fx = pd.read_parquet(PACK_BASE / "macro" / "usdinr.parquet").copy()
    fx["available_utc"] = pd.to_datetime(fx["available_utc"], utc=True).astype("datetime64[us, UTC]")
    fx = fx.sort_values("available_utc").reset_index(drop=True)

    fx["usdinr_ret_1d"] = fx["usdinr_fbil"].pct_change(1).fillna(0.0)
    fx["usdinr_ret_5d"] = fx["usdinr_fbil"].pct_change(5).fillna(0.0)

    fx_asof = fx[["available_utc", "usdinr_fbil", "usdinr_ret_1d", "usdinr_ret_5d"]].sort_values("available_utc").reset_index(drop=True)

    df = pd.merge_asof(
        df,
        fx_asof,
        left_on="cutoff_utc",
        right_on="available_utc",
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.drop(columns=["available_utc"])
    print("  USD/INR currency exchange features attached.")

    # -------------------------------------------------------------------------
    # STEP 6: Global Benchmark / Hyperliquid Perps (macro/hl_candles_1d & funding)
    # -------------------------------------------------------------------------
    print("\n[Step 6/11] Joining Hyperliquid Global Perpetuals & Funding Rates...")
    hl = pd.read_parquet(PACK_BASE / "macro" / "hl_candles_1d.parquet").copy()
    hl["available_utc"] = pd.to_datetime(hl["available_utc"], utc=True).astype("datetime64[us, UTC]")
    
    inst_map = {"xyz:GOLD": "gold", "xyz:SILVER": "silver", "xyz:COPPER": "copper", "xyz:CL": "crude"}
    hl["asset"] = hl["instrument"].map(inst_map)
    hl = hl[hl["asset"].notnull()].sort_values(["asset", "available_utc"]).reset_index(drop=True)

    hl["hl_ret_1d"] = hl.groupby("asset")["close"].pct_change(1).fillna(0.0)
    hl["hl_ret_5d"] = hl.groupby("asset")["close"].pct_change(5).fillna(0.0)

    # Load funding rates (column is funding_rate)
    hl_fund = pd.read_parquet(PACK_BASE / "macro" / "hl_funding.parquet").copy()
    hl_fund["available_utc"] = pd.to_datetime(hl_fund["available_utc"], utc=True).astype("datetime64[us, UTC]")
    hl_fund["asset"] = hl_fund["instrument"].map(inst_map)
    hl_fund = hl_fund[hl_fund["asset"].notnull()].sort_values("available_utc").reset_index(drop=True)

    hl_asof = hl[["available_utc", "asset", "hl_ret_1d", "hl_ret_5d"]].sort_values("available_utc").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        hl_asof,
        left_on="cutoff_utc",
        right_on="available_utc",
        by="asset",
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.drop(columns=["available_utc"])

    fund_asof = hl_fund[["available_utc", "asset", "funding_rate"]].rename(columns={"funding_rate": "hl_funding"}).sort_values("available_utc").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        fund_asof,
        left_on="cutoff_utc",
        right_on="available_utc",
        by="asset",
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.drop(columns=["available_utc"])
    print("  Hyperliquid perps & funding features attached.")

    # -------------------------------------------------------------------------
    # STEP 7: Macroeconomic Release Calendar (Leak-Free Schedule)
    # -------------------------------------------------------------------------
    print("\n[Step 7/11] Computing Scheduled Macro Release Counts (Next 24h & 48h)...")
    sched = pd.read_parquet(PACK_BASE / "calendar" / "release_schedule.parquet")
    sched_pub = sched[sched["release_utc"].notnull() & (sched["clock_quality"] == "published")].copy()
    sched_pub["release_utc"] = pd.to_datetime(sched_pub["release_utc"], utc=True).astype("datetime64[us, UTC]")

    cutoffs = df["cutoff_utc"].drop_duplicates().sort_values().reset_index(drop=True)
    calendar_records = []
    for c in cutoffs:
        end_24h = c + pd.Timedelta(hours=24)
        sub = sched_pub[(sched_pub["release_utc"] >= c) & (sched_pub["release_utc"] < end_24h)]
        calendar_records.append({
            "cutoff_utc": c,
            "macro_events_next_24h": len(sub),
            "energy_events_next_24h": len(sub[sub["event_family"] == "energy"]),
            "inflation_events_next_24h": len(sub[sub["event_family"] == "inflation"]),
            "central_bank_events_next_24h": len(sub[sub["event_family"] == "central_bank"]),
        })
    df_cal = pd.DataFrame(calendar_records)
    df = pd.merge(df, df_cal, on="cutoff_utc", how="left")
    print("  Calendar event counts attached without vintage leakage.")

    # -------------------------------------------------------------------------
    # STEP 8: Advanced Cross-Asset Interactions & Volatility Term Structure
    # -------------------------------------------------------------------------
    print("\n[Step 8/11] Generating Volatility Term Structure & Interaction Features...")
    df = df.sort_values(["asset", "cutoff_utc"]).reset_index(drop=True)

    # 1. Short-term Realized Volatility (5-session) & Volatility Term Structure (rv_5 / rv_20)
    df["rv_5"] = df.groupby("asset")["ret_1"].transform(lambda s: s.rolling(5, min_periods=2).std().fillna(0.0))
    df["vol_term_structure"] = (df["rv_5"] / df["rv_20"].replace(0, np.nan)).fillna(1.0)

    # 2. Volatility Regime: current 20d vol vs 60d baseline
    df["rv_60_baseline"] = df.groupby("asset")["rv_20"].transform(lambda s: s.rolling(60, min_periods=1).mean())
    df["volatility_regime"] = (df["rv_20"] / df["rv_60_baseline"].replace(0, np.nan)).fillna(1.0)
    df = df.drop(columns=["rv_60_baseline"])

    # 3. Volume * Volatility Interaction (Liquidity shock)
    df["vol_rv_interaction"] = df["vol_surge_10d"] * df["rv_20"]

    # 4. Return & Trend Agreement
    df["ret_trend_agreement"] = np.sign(df["ret_1"].fillna(0)) * np.sign(df["ret_20"].fillna(0))

    # 5. Open Interest Momentum Interaction
    df["oi_momentum_interaction"] = np.sign(df["ret_1"].fillna(0)) * df["oi_1s_change"].fillna(0)

    # 6. COT Divergence: Smart Money Direction vs 20-session Price Trend
    df["cot_price_divergence"] = np.sign(df["cot_weekly_change"].fillna(0)) * np.sign(df["ret_20"].fillna(0))

    # 7. FX Shock Interaction: Currency trend vs Asset trend
    df["fx_momentum_interaction"] = df["usdinr_ret_5d"].fillna(0) * df["ret_5"].fillna(0)

    # 8. Calendar Day of Week Encoding (Cyclical dynamics)
    dow = df["cutoff_utc"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

    # -------------------------------------------------------------------------
    # STEP 9: News Headline Sentiment via FinBERT (Trailing 24h)
    # -------------------------------------------------------------------------
    print("\n[Step 9/11] Processing News Sentiment (FinBERT on Trailing 24h Headlines)...")
    news = pd.read_parquet(PACK_BASE / "pack" / "news" / "news_rows.parquet")
    news = news[news["asset"].isin(["gold", "silver", "copper", "crude"])].copy()
    news["available_utc"] = pd.to_datetime(news["available_utc"], utc=True).astype("datetime64[us, UTC]")

    if FINBERT_CACHE.exists():
        print(f"  Loading pre-computed FinBERT sentiment cache from {FINBERT_CACHE}...")
        cache_df = pd.read_parquet(FINBERT_CACHE)
        title_to_score = dict(zip(cache_df["original_title"], cache_df["finbert_score"]))
    else:
        print("  Computing FinBERT sentiment across unique commodity news headlines...")
        selected_indices = set()
        for c in cutoffs:
            start_w = c - pd.Timedelta(hours=24)
            sub = news[(news["available_utc"] >= start_w) & (news["available_utc"] < c)]
            if len(sub) == 0:
                continue
            for a in ["gold", "silver", "copper", "crude"]:
                sub_a = sub[sub["asset"] == a].sort_values("trust_weight", ascending=False).head(20)
                selected_indices.update(sub_a.index)

        sub_top = news.loc[list(selected_indices)].copy()
        unique_titles = sub_top["original_title"].unique().tolist()
        print(f"  Scoring {len(unique_titles)} unique titles on FinBERT...")

        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device, batch_size=128)
        
        t0 = time.time()
        preds = pipe(unique_titles)
        print(f"  Inference finished in {time.time() - t0:.2f}s.")

        scores = []
        for p in preds:
            s = p["score"] if p["label"] == "positive" else (-p["score"] if p["label"] == "negative" else 0.0)
            scores.append(s)

        title_to_score = dict(zip(unique_titles, scores))
        cache_df = pd.DataFrame({"original_title": unique_titles, "finbert_score": scores})
        cache_df.to_parquet(FINBERT_CACHE, index=False)
        print(f"  Saved FinBERT sentiment cache to {FINBERT_CACHE}.")

    news["finbert_score"] = news["original_title"].map(title_to_score).fillna(0.0)

    news_records = []
    for c in cutoffs:
        start_w = c - pd.Timedelta(hours=24)
        sub_w = news[(news["available_utc"] >= start_w) & (news["available_utc"] < c)]
        for a in ["gold", "silver", "copper", "crude"]:
            sub_a = sub_w[sub_w["asset"] == a]
            if len(sub_a) > 0:
                scores_arr = sub_a["finbert_score"].to_numpy()
                news_records.append({
                    "cutoff_utc": c,
                    "asset": a,
                    "is_news_exist": 1,
                    "news_count_24h": len(sub_a),
                    "finbert_sentiment_mean_24h": float(np.mean(scores_arr)),
                    "finbert_sentiment_sum_24h": float(np.sum(scores_arr)),
                    "news_pos_share_24h": float(np.mean(scores_arr > 0)),
                    "news_neg_share_24h": float(np.mean(scores_arr < 0)),
                })
            else:
                news_records.append({
                    "cutoff_utc": c,
                    "asset": a,
                    "is_news_exist": 0,
                    "news_count_24h": 0,
                    "finbert_sentiment_mean_24h": 0.0,
                    "finbert_sentiment_sum_24h": 0.0,
                    "news_pos_share_24h": 0.0,
                    "news_neg_share_24h": 0.0,
                })

    df_news = pd.DataFrame(news_records)
    df = pd.merge(df, df_news, on=["cutoff_utc", "asset"], how="left")

    # 9. Sentiment * Price Agreement Interaction
    df["sentiment_trend_interaction"] = df["finbert_sentiment_mean_24h"] * np.sign(df["ret_1"].fillna(0))
    print("  News NLP sentiment & interaction features attached.")

    # -------------------------------------------------------------------------
    # STEP 10: Cleaning, Target Formulation & Feature Identification
    # -------------------------------------------------------------------------
    print("\n[Step 10/11] Cleaning Rows & Formulating Classifier and Regressor Targets...")
    
    # Calculate Favorable Excursion percentage for Employee Regressor
    # For an up swing, favorable run is e^(y_mfe) - 1. For a down swing, it is 1 - e^(y_mae).
    df["mfe_pct_up"] = (np.exp(df["y_mfe"].astype(float)) - 1.0) * 100.0
    df["mfe_pct_down"] = (1.0 - np.exp(df["y_mae"].astype(float))) * 100.0
    
    df["target_mfe_pct"] = np.where(
        df["label"] == "up_swing",
        df["mfe_pct_up"],
        np.where(df["label"] == "down_swing", df["mfe_pct_down"], 0.0)
    )
    df["is_swing"] = (df["label"].isin(["up_swing", "down_swing"])).astype(int)
    df["swing_direction"] = np.where(
        df["label"] == "up_swing", "up",
        np.where(df["label"] == "down_swing", "down", "none")
    )

    # Clean rows: eligible designations, features present, label in the 3 scored classes
    clean_df = df[
        df["has_features"].astype(bool)
        & df["eligible"].astype(bool)
        & df["label"].isin(["up_swing", "down_swing", "no_swing"])
    ].copy().reset_index(drop=True)

    non_feature_cols = {
        "asset", "cutoff_utc", "horizon", "cutoff_date", "contract_id",
        "label", "y_ret", "y_mfe", "y_mae", "y_range",
        "target_mfe_pct", "mfe_pct_up", "mfe_pct_down", "is_swing", "swing_direction"
    }
    feature_cols = [c for c in clean_df.columns if c not in non_feature_cols]

    print(f"  Final Clean Pool: {len(clean_df)} rows, Total Columns: {len(clean_df.columns)}")
    print(f"  Identified {len(feature_cols)} Pure Input Feature Columns! (Target range: 50-60)")

    # -------------------------------------------------------------------------
    # STEP 11: Exporting 8 Commodity Datasets + 2 Unified Datasets
    # -------------------------------------------------------------------------
    print("\n[Step 11/11] Splitting and Exporting 8 Specialized Parquet Files...")
    
    # 1. Base Column Sets
    meta_cols = ["asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon"]
    clf_cols = meta_cols + feature_cols + ["label"]
    reg_cols = meta_cols + feature_cols + ["target_mfe_pct", "y_mfe", "y_mae", "y_ret", "y_range", "is_swing", "swing_direction"]

    # Export Unified Datasets
    clf_unified_path = OUT_DIR / "task1_classifier_unified.parquet"
    reg_unified_path = OUT_DIR / "task1_regressor_unified.parquet"
    clean_df[clf_cols].to_parquet(clf_unified_path, index=False)
    clean_df[reg_cols].to_parquet(reg_unified_path, index=False)
    print(f"  [Unified Classifier] {clf_unified_path} ({len(clean_df)} rows, {len(clf_cols)} cols)")
    print(f"  [Unified Regressor]  {reg_unified_path} ({len(clean_df)} rows, {len(reg_cols)} cols)")

    # Export 8 Commodity Datasets (4 Classifier + 4 Regressor)
    manifest = {}
    for asset in ["gold", "silver", "copper", "crude"]:
        asset_sub = clean_df[clean_df["asset"] == asset].copy().reset_index(drop=True)
        
        # Classifier Dataset for Commodity
        clf_path = OUT_DIR / f"task1_classifier_{asset}.parquet"
        asset_sub[clf_cols].to_parquet(clf_path, index=False)

        # Regressor Dataset for Commodity
        reg_path = OUT_DIR / f"task1_regressor_{asset}.parquet"
        asset_sub[reg_cols].to_parquet(reg_path, index=False)

        manifest[asset] = {
            "rows": len(asset_sub),
            "classifier_file": str(clf_path.name),
            "regressor_file": str(reg_path.name),
            "feature_count": len(feature_cols),
            "class_distribution": asset_sub["label"].value_counts().to_dict(),
            "swing_count": int(asset_sub["is_swing"].sum()),
            "mean_favorable_mfe_pct": float(asset_sub[asset_sub["is_swing"] == 1]["target_mfe_pct"].mean()),
        }
        print(f"  [Done] {asset.upper():6s} -> Classifier: {clf_path.name} | Regressor: {reg_path.name} ({len(asset_sub)} rows, {len(feature_cols)} features)")

    # Save Complete 8-Dataset Summary Manifest
    summary_file = OUT_DIR / "task1_8datasets_summary.json"
    full_summary = {
        "built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_commodities": 4,
        "input_feature_count": len(feature_cols),
        "input_features": feature_cols,
        "commodities": manifest,
        "classifier_target": "label (up_swing, down_swing, no_swing)",
        "regressor_primary_target": "target_mfe_pct (favorable excursion %)",
        "regressor_auxiliary_targets": ["y_mfe", "y_mae", "y_ret", "y_range"]
    }
    with open(summary_file, "w") as f:
        json.dump(full_summary, f, indent=2)

    print("\n================================================================================")
    print(f"   SUCCESS! ALL 8 COMMODITY DATASETS CREATED WITH {len(feature_cols)} INPUT FEATURES!")
    print(f"   Manifest: {summary_file}")
    print("================================================================================")


if __name__ == "__main__":
    build_task1_suite()
