"""Train Employee LightGBM Regressors for Task 1 Across 4 Commodities.

Role:
- Boss (Classifier) decides direction and whether to issue a signal.
- Employee (Regressor) predicts expected swing depth (target_mfe_pct).
- target_pct = max(threshold_pct, predicted_mfe * 0.70)
- invalidation_pct = threshold_pct

Leakage Prevention:
- Train strictly on cutoff_utc < 2026-06-01.
- Zero future data leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PACK_BASE = Path(r"C:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-pack\v2.0.1")
DATA_DIR = Path(r"c:\Users\Pavan\Downloads\Rinnegan_Forecasts\Commodity_2026\commodity-open-kit\data")
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_SPLIT = pd.Timestamp("2026-06-01 03:00:00", tz="UTC")
ASSETS = ["gold", "silver", "copper", "crude"]
THRESHOLD_PCT = {"gold": 1.5, "silver": 2.0, "copper": 1.5, "crude": 3.0}

def train_employee_suite():
    print("================================================================================")
    print("   TRAINING EMPLOYEE REGRESSORS (LIGHTGBM) FOR 4 COMMODITIES")
    print("================================================================================")

    results = {}
    
    for asset in ASSETS:
        df = pd.read_parquet(DATA_DIR / f"task1_regressor_{asset}.parquet")
        df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
        
        meta = {
            "asset", "cutoff_utc", "cutoff_date", "contract_id", "horizon",
            "label", "label_legacy_24", "y_ret", "y_mfe", "y_mae", "y_range", "target_mfe_pct",
            "swing_direction", "target_is_swing"
        }
        feats = [c for c in df.columns if c not in meta]
        
        # Point-in-time train/holdout split
        tr = df[df["cutoff_utc"] < CUTOFF_SPLIT].copy().reset_index(drop=True)
        ho = df[df["cutoff_utc"] >= CUTOFF_SPLIT].copy().reset_index(drop=True)
        
        # Train strictly on actual swings (where a swing formed)
        tr_swings = tr[tr["label"] != "no_swing"].copy().reset_index(drop=True)
        
        X_tr = tr_swings[feats]
        y_tr = tr_swings["target_mfe_pct"]
        
        X_ho = ho[feats]
        y_ho = ho["target_mfe_pct"]
        
        # LightGBM Regressor with Huber loss
        reg = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=12,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1,
        )
        reg.fit(X_tr, y_tr)
        
        preds_ho = reg.predict(X_ho)
        
        # Evaluate on holdout swings
        ho_swings_mask = ho["label"] != "no_swing"
        mae = mean_absolute_error(y_ho[ho_swings_mask], preds_ho[ho_swings_mask])
        rmse = np.sqrt(mean_squared_error(y_ho[ho_swings_mask], preds_ho[ho_swings_mask]))
        r2 = r2_score(y_ho[ho_swings_mask], preds_ho[ho_swings_mask])
        
        print(f"\nAsset: {asset.upper():<6} (Threshold = {THRESHOLD_PCT[asset]}%)")
        print(f"  Training samples: {len(X_tr)} swings | Holdout samples: {len(X_ho)}")
        print(f"  Holdout MAE:  {mae:.4f}% | RMSE: {rmse:.4f}% | R2: {r2:.4f}")
        print(f"  Predicted MFE Range: [{preds_ho.min():.2f}%, {preds_ho.max():.2f}%] (Actual Mean: {y_ho[ho_swings_mask].mean():.2f}%)")
        
        # Save model
        model_path = MODELS_DIR / f"regressor_{asset}.txt"
        reg.booster_.save_model(str(model_path))
        
        # Save predictions to parquet for Boss-Employee pairing
        pred_df = ho[["cutoff_utc", "asset"]].copy()
        pred_df["predicted_mfe_pct"] = preds_ho
        pred_df["recommended_target_pct"] = np.maximum(THRESHOLD_PCT[asset], preds_ho * 0.70)
        pred_df.to_parquet(DATA_DIR / f"predictions_regressor_{asset}.parquet", index=False)
        
        results[asset] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "pred_mean": float(preds_ho.mean()),
            "actual_mean": float(y_ho[ho_swings_mask].mean()),
        }

    with open(DATA_DIR / "regressor_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRegressor training complete. Models and predictions saved.")

if __name__ == "__main__":
    train_employee_suite()
