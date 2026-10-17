"""Custom Raw Path Forecaster for Kronos.

Bypasses stock `predict()` path averaging to extract all N distinct sample paths:
    Z in R^{N x pred_len x 4}  (Open, High, Low, Close)
Preserves path-level distribution, excursion quantiles, and prevents variance collapse.
Implements per-session forecast caching to eliminate redundant compute between h=5 and h=10.
Enforces deterministic seeding for reproducible generation.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kronos_vendor.model.kronos import Kronos, KronosTokenizer, calc_time_stamps, sample_from_logits

MASTER_SEED = 20260919


def raw_path_inference(
    tokenizer: KronosTokenizer,
    model: Kronos,
    x: torch.Tensor,
    x_stamp: torch.Tensor,
    y_stamp: torch.Tensor,
    max_context: int,
    pred_len: int,
    clip: float = 5.0,
    T: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    sample_count: int = 30,
) -> np.ndarray:
    """Autoregressive rollouts returning un-averaged tensor of sample paths.

    Returns:
        np.ndarray of shape (sample_count, pred_len, num_features)
    """
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device

        # Replicate for sample_count parallel rollouts
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        # z shape: (sample_count, total_seq_len, num_features)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        # For batch size 1, extract future pred_len steps across sample paths
        # Shape: (sample_count, pred_len, num_features)
        return preds[0, :, -pred_len:, :]


class RawPathPredictor:
    """Predictor wrapping Kronos foundation model and returning un-averaged paths."""

    def __init__(
        self,
        weights_dir: Path | str | None = None,
        device: str | None = None,
        max_context: int = 512,
        clip: float = 5.0,
    ) -> None:
        if weights_dir is None:
            weights_dir = HERE.parent / "weights"
        weights_path = Path(weights_dir)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_context = max_context
        self.clip = clip

        self.price_cols = ["open", "high", "low", "close"]
        self.vol_col = "volume"
        self.amt_col = "amount"
        self.feature_cols = ["open", "high", "low", "close", "volume", "amount"]

        # Load tokenizer and model from local offline weights
        tok_dir = weights_path / "tokenizer"
        mod_dir = weights_path / "model"
        if not tok_dir.exists() or not mod_dir.exists():
            raise FileNotFoundError(f"Kronos weights not found in {weights_path}. Ensure Step 1 was executed.")

        self.tokenizer = KronosTokenizer.from_pretrained(str(tok_dir)).to(self.device).eval()
        self.model = Kronos.from_pretrained(str(mod_dir)).to(self.device).eval()

    def predict_raw(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.DatetimeIndex,
        y_timestamp: pd.DatetimeIndex,
        pred_len: int = 10,
        T: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 30,
        seed: int | None = None,
    ) -> np.ndarray:
        """Run autoregressive forecasting and return raw un-averaged OHLC paths.

        Returns:
            np.ndarray of shape (sample_count, pred_len, 4) in original price scale.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        if isinstance(x_timestamp, pd.DatetimeIndex):
            x_timestamp = pd.Series(x_timestamp)
        if isinstance(y_timestamp, pd.DatetimeIndex):
            y_timestamp = pd.Series(y_timestamp)

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp.iloc[:pred_len] if hasattr(y_timestamp, 'iloc') else y_timestamp[:pred_len])

        x_vals = df[self.feature_cols].values.astype(np.float32)
        x_mean = np.mean(x_vals, axis=0)
        x_std = np.std(x_vals, axis=0)

        # Standardize
        x_norm = (x_vals - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -self.clip, self.clip)

        x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(self.device)
        x_stamp_tensor = torch.from_numpy(x_time_df.values.astype(np.float32)[np.newaxis, :]).to(self.device)
        y_stamp_tensor = torch.from_numpy(y_time_df.values.astype(np.float32)[np.newaxis, :]).to(self.device)

        raw_preds = raw_path_inference(
            self.tokenizer,
            self.model,
            x_tensor,
            x_stamp_tensor,
            y_stamp_tensor,
            max_context=self.max_context,
            pred_len=pred_len,
            clip=self.clip,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
        )

        # raw_preds shape: (sample_count, pred_len, 6)
        # Unstandardize back to continuous price scale
        unnorm = raw_preds * (x_std + 1e-5) + x_mean
        # Extract OHLC: shape (sample_count, pred_len, 4)
        ohlc = unnorm[:, :, :4]
        return ohlc


class TrajectoryForecaster:
    """Manages raw-path prediction and session-level caching to cut compute in half."""

    def __init__(
        self,
        predictor: RawPathPredictor | None = None,
        max_pred_len: int = 10,
        sample_count: int = 30,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> None:
        self.predictor = predictor or RawPathPredictor()
        self.max_pred_len = max_pred_len
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p

        self._current_asof = None
        self._cache: Dict[str, np.ndarray] = {}

    def get_paths(
        self,
        asset: str,
        as_of,
        series_df: pd.DataFrame,
        x_timestamp: pd.DatetimeIndex,
        y_timestamp: pd.DatetimeIndex,
        horizon: int,
    ) -> np.ndarray:
        """Retrieve simulated paths for `asset` up to `horizon`. Uses cached 10-step trajectory if available."""
        # Reset cache on new session
        if self._current_asof != as_of:
            self._current_asof = as_of
            self._cache.clear()

        if asset not in self._cache:
            # Deterministic seed per forecast (cross-process stable)
            key = f"{as_of.isoformat()}_{asset}_{MASTER_SEED}"
            seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF

            full_paths = self.predictor.predict_raw(
                df=series_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=self.max_pred_len,
                T=self.temperature,
                top_k=0,
                top_p=self.top_p,
                sample_count=self.sample_count,
                seed=seed,
            )
            self._cache[asset] = full_paths

        cached_10 = self._cache[asset]
        # Slice steps 0:horizon
        return cached_10[:, :horizon, :]
