"""Diagnose label distributions and model predictions across train and holdout."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
ASSETS = ["gold", "silver", "copper", "crude"]

for asset in ASSETS:
    df = pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet")
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    tr = df[df["cutoff_utc"] < pd.Timestamp("2026-06-01", tz="UTC")]
    ho = df[df["cutoff_utc"] >= pd.Timestamp("2026-06-01", tz="UTC")]
    
    print("=" * 60)
    print(f"ASSET: {asset.upper()}")
    print("=" * 60)
    print("Train label distribution:")
    print(tr["label"].value_counts(dropna=False, normalize=True))
    print(tr["label"].value_counts(dropna=False))
    
    print("\nHoldout label distribution (actual 2.6 ground truth in dataset):")
    print(ho["label"].value_counts(dropna=False, normalize=True))
    print(ho["label"].value_counts(dropna=False))
    
    # Check if up_swing and down_swing are balanced in train
    n_up = (tr["label"] == "up_swing").sum()
    n_down = (tr["label"] == "down_swing").sum()
    n_none = (tr["label"] == "no_swing").sum()
    print(f"Train Counts: up={n_up}, down={n_down}, none={n_none} (down/up ratio = {n_down/max(1, n_up):.2f})")
