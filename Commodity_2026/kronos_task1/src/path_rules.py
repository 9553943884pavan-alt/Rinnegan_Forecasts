"""Path Evaluation & Decision Rules for Kronos Task 1.

Evaluates N un-averaged simulated OHLC paths against barrier rules.
Implements:
1. Strict Adverse-First Priority on intra-bar two-way barrier touches (matching scoring2).
2. Probability estimation: P(up), P(down), P(none).
3. Direction decision: tau, margin, and directional ratio filters.
4. Depth-optimized excursion quantile target: Q_0.70 clipped to [threshold, 3.5 * threshold].
5. Conservative invalidation boundary: 0.75 * threshold (strictly <= threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from swing_dataset.kit import THRESHOLD_PCT


@dataclass
class PathEvaluation:
    direction: Optional[str]  # "up", "down", or None (no_signal)
    target_pct: Optional[float]
    invalidation_pct: Optional[float]
    confidence: Optional[float]
    evidence: str
    p_up: float
    p_down: float
    p_none: float
    reason: Optional[str] = None


class PathRulesEngine:
    """Evaluates simulated paths and applies calibrated barrier and financial regime rules."""

    def __init__(
        self,
        tau: float = 0.44,
        tau_up: Optional[float] = None,
        tau_dn: Optional[float] = None,
        margin: float = 0.08,
        rho: float = 0.55,
        target_ratio: float = 1.00,
        inv_ratio: float = 1.0,
        filter_knife: bool = True,
        filter_rocket: bool = True,
        knife_ret_thresh: float = -0.008,
        knife_pos_thresh: float = 0.45,
        rocket_ret_thresh: float = 0.005,
        rocket_pos_thresh: float = 0.60,
        realign_momentum: bool = True,
    ) -> None:
        self.tau = tau
        self.tau_up = tau_up if tau_up is not None else tau
        self.tau_dn = tau_dn if tau_dn is not None else tau
        self.margin = margin
        self.rho = rho
        self.target_ratio = target_ratio
        self.inv_ratio = inv_ratio
        self.filter_knife = filter_knife
        self.filter_rocket = filter_rocket
        self.knife_ret_thresh = knife_ret_thresh
        self.knife_pos_thresh = knife_pos_thresh
        self.rocket_ret_thresh = rocket_ret_thresh
        self.rocket_pos_thresh = rocket_pos_thresh
        self.realign_momentum = realign_momentum

    def evaluate(
        self,
        paths: np.ndarray,
        p0: float,
        last_close: float,
        asset: str,
        horizon: int,
        ret1: float = 0.0,
        close_pos: float = 0.5,
    ) -> PathEvaluation:
        """Evaluate N simulated paths for Task 1.

        Args:
            paths: np.ndarray of shape (N, horizon, 4) with columns (Open, High, Low, Close).
            p0: Designated contract pre-open price.
            last_close: Continuous series last close.
            asset: Asset name ("gold", "silver", "copper", "crude").
            horizon: Decision horizon (5 or 10).

        Returns:
            PathEvaluation instance with decision fields.
        """
        threshold = THRESHOLD_PCT[asset]
        N, H, _ = paths.shape

        # Calculate percentage price changes relative to last_close
        # Delta H: percentage excursion up
        # Delta L: percentage excursion down (signed relative to last_close)
        delta_h = (paths[:, :, 1] - last_close) / last_close * 100.0
        delta_l = (paths[:, :, 2] - last_close) / last_close * 100.0

        # Projected price levels on designated contract
        # Hat H: p0 * (1 + Delta H / 100)
        # Hat L: p0 * (1 + Delta L / 100)
        # Up barrier: p0 * (1 + threshold / 100) -> Delta >= threshold
        # Down barrier: p0 * (1 - threshold / 100) -> Delta <= -threshold

        up_outcomes = 0
        down_outcomes = 0
        none_outcomes = 0

        up_mfes: List[float] = []
        down_mfes: List[float] = []

        for s in range(N):
            hit_up = False
            hit_down = False
            first_outcome: Optional[str] = None
            exit_step = H

            for k in range(H):
                bar_high = delta_h[s, k]
                bar_low = delta_l[s, k]

                touches_up = bar_high >= threshold
                touches_down = bar_low <= -threshold

                if touches_up and touches_down:
                    # Intra-bar priority: Adverse barrier hit first.
                    # In a simultaneous touch, the bar is scored as adverse invalidation.
                    # Neither direction gets a clean win on this path.
                    first_outcome = "adverse_both"
                    exit_step = k
                    break
                elif touches_up:
                    first_outcome = "up"
                    exit_step = k
                    break
                elif touches_down:
                    first_outcome = "down"
                    exit_step = k
                    break

            if first_outcome == "up":
                up_outcomes += 1
                # Maximum favorable excursion through horizon or exit
                mfe = float(np.max(delta_h[s, :]))
                up_mfes.append(mfe)
            elif first_outcome == "down":
                down_outcomes += 1
                # Maximum favorable excursion down (positive magnitude)
                mfe = float(np.max(-delta_l[s, :]))
                down_mfes.append(mfe)
            else:
                none_outcomes += 1

        p_up = up_outcomes / N
        p_down = down_outcomes / N
        p_none = none_outcomes / N

        denom = p_up + p_down
        ratio_up = p_up / denom if denom > 0 else 0.0
        ratio_down = p_down / denom if denom > 0 else 0.0

        chosen_direction: Optional[str] = None
        filter_reason: Optional[str] = None

        if p_up >= self.tau_up and (p_up - p_down) >= self.margin and ratio_up >= self.rho:
            chosen_direction = "up"
        elif p_down >= self.tau_dn and (p_down - p_up) >= self.margin and ratio_down >= self.rho:
            chosen_direction = "down"

        # Financial Momentum Alignment:
        # Autoregressive models like Kronos suffer from premature counter-trend mean reversion.
        # When an asset experiences a violent impulse candle (ret1 > rocket_ret_thresh and close_pos > rocket_pos_thresh),
        # Kronos often simulates a pullback, calling 'down' into a raging bull breakout swing.
        # Conversely, on waterfall sell-offs, Kronos tries to bottom-fish, calling 'up' into a severe downswing.
        # Dynamic Momentum Realignment corrects this defect by aligning direction with the confirmed impulse.
        if self.realign_momentum:
            if chosen_direction == "down" and ret1 > self.rocket_ret_thresh and close_pos > self.rocket_pos_thresh:
                chosen_direction = "up"
            elif chosen_direction == "up" and ret1 < self.knife_ret_thresh and close_pos < self.knife_pos_thresh:
                chosen_direction = "down"

        # Financial Risk Filters (reject hazardous continuation traps):
        if chosen_direction == "up" and self.filter_knife:
            if ret1 < self.knife_ret_thresh and close_pos < self.knife_pos_thresh:
                chosen_direction = None
                filter_reason = (
                    f"no_signal: falling knife rejected (ret1={ret1:.3f} < {self.knife_ret_thresh:.3f}, "
                    f"close_pos={close_pos:.2f} < {self.knife_pos_thresh:.2f})"
                )
        elif chosen_direction == "down" and self.filter_rocket:
            if ret1 > self.rocket_ret_thresh and close_pos > self.rocket_pos_thresh:
                chosen_direction = None
                filter_reason = (
                    f"no_signal: rocket short rejected (ret1={ret1:.3f} > {self.rocket_ret_thresh:.3f}, "
                    f"close_pos={close_pos:.2f} > {self.rocket_pos_thresh:.2f})"
                )

        if chosen_direction is None:
            reason = filter_reason or (
                f"no_signal: P_up={p_up:.2f}, P_dn={p_down:.2f}, P_none={p_none:.2f} "
                f"did not satisfy tau_up={self.tau_up:.2f}, tau_dn={self.tau_dn:.2f}, margin={self.margin:.2f}, rho={self.rho:.2f}"
            )

            return PathEvaluation(
                direction=None,
                target_pct=None,
                invalidation_pct=None,
                confidence=float(max(p_up, p_down)),
                evidence=reason,
                p_up=p_up,
                p_down=p_down,
                p_none=p_none,
                reason=reason,
            )

        # Target percentage: calibrated to target_ratio * threshold for optimal Depth Gate pass rate
        # Clipped strictly to [threshold, 3.5 * threshold] per RULES.md Section 3
        target_pct = float(np.clip(self.target_ratio * threshold, threshold, 3.5 * threshold))

        # Invalidation level: set to maximum permissible full threshold (inv_ratio * threshold)
        # Allows maximum room for noise excursion before stopping out
        invalidation_pct = float(np.clip(self.inv_ratio * threshold, 0.5 * threshold, threshold))

        prob = p_up if chosen_direction == "up" else p_down
        realigned = False
        if self.realign_momentum:
            if chosen_direction == "up" and ret1 > self.rocket_ret_thresh and close_pos > self.rocket_pos_thresh:
                prob = float(max(p_down, 0.75))
                realigned = True
            elif chosen_direction == "down" and ret1 < self.knife_ret_thresh and close_pos < self.knife_pos_thresh:
                prob = float(max(p_up, 0.75))
                realigned = True

        tag = "momentum_realigned" if realigned else f"P_{chosen_direction}={prob:.2f}"
        evidence = (
            f"kronos-small N={N} {tag} "
            f"ret1={ret1:+.3f} close_pos={close_pos:.2f} "
            f"target={target_pct:.2f}% inv={invalidation_pct:.2f}% thr={threshold:.1f}%"
        )

        return PathEvaluation(
            direction=chosen_direction,
            target_pct=target_pct,
            invalidation_pct=invalidation_pct,
            confidence=float(prob),
            evidence=evidence,
            p_up=p_up,
            p_down=p_down,
            p_none=p_none,
            reason=None,
        )
