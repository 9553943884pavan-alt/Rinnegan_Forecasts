"""Test class_weight='balanced' on LightGBM Classifiers."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
CUTOFF_SPLIT = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
ASSETS = ["gold", "silver", "copper", "crude"]

for asset in ASSETS:
    df = pd.read_parquet(DATA_DIR / f"task1_classifier_{asset}.parquet")
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    meta = {"asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon", "label", "label_legacy_24"}
    feats = [c for c in df.columns if c not in meta]
    
    tr = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
    ho = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
    
    # 1. Standard
    clf_std = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=130, random_state=42, verbose=-1)
    clf_std.fit(tr[feats], tr["label"])
    preds_std = clf_std.predict(ho[feats])
    
    # 2. Balanced
    clf_bal = lgb.LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced", n_estimators=130, random_state=42, verbose=-1)
    clf_bal.fit(tr[feats], tr["label"])
    preds_bal = clf_bal.predict(ho[feats])
    
    acc_std = accuracy_score(ho["label"], preds_std)
    acc_bal = accuracy_score(ho["label"], preds_bal)
    
    print(f"Asset: {asset.upper():<6} | Standard Acc: {acc_std*100:.2f}% | Balanced Acc: {acc_bal*100:.2f}%")
    print("  Standard class distribution in holdout preds:", pd.Series(preds_std).value_counts().to_dict())
    print("  Balanced class distribution in holdout preds:", pd.Series(preds_bal).value_counts().to_dict())
