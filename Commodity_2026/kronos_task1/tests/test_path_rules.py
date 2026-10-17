"""Unit tests for path_rules.py.

Verifies:
1. Strict adverse-first priority when high and low touch both barriers.
2. Depth target bounds (target_pct >= threshold and <= 3.5 * threshold).
3. Invalidation level bounds (invalidation_pct <= threshold).
4. Determinism of probability counts and direction selection.
"""

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.path_rules import PathRulesEngine
from swing_dataset.kit import THRESHOLD_PCT


def test_adverse_first_priority():
    """Verify simultaneous barrier touch is treated as adverse failure."""
    engine = PathRulesEngine(tau=0.40, margin=0.05, rho=0.50)
    asset = "gold"
    threshold = THRESHOLD_PCT[asset]  # 1.5%
    p0 = 100.0
    last_close = 100.0
    horizon = 5
    N = 10

    # Create paths where every path touches both +2.0% and -2.0% in bar 0
    paths = np.zeros((N, horizon, 4))
    paths[:, :, 0] = 100.0  # Open
    paths[:, :, 1] = 100.0 + threshold + 0.5  # High touches up barrier
    paths[:, :, 2] = 100.0 - threshold - 0.5  # Low touches down barrier
    paths[:, :, 3] = 100.0  # Close

    eval_res = engine.evaluate(paths, p0, last_close, asset, horizon)
    # Under adverse-first priority, simultaneous touch counts as adverse_both, neither up nor down wins
    assert eval_res.p_up == 0.0
    assert eval_res.p_down == 0.0
    assert eval_res.direction is None
    assert eval_res.reason is not None


def test_clean_up_swing():
    """Verify clean up moves register as up outcomes with valid target and invalidation."""
    engine = PathRulesEngine(tau=0.40, margin=0.05, rho=0.50, target_ratio=1.05)
    asset = "crude"
    threshold = THRESHOLD_PCT[asset]  # 3.0%
    p0 = 5000.0
    last_close = 5000.0
    horizon = 5
    N = 20

    paths = np.zeros((N, horizon, 4))
    paths[:, :, 0] = 5000.0
    # Paths exceed up barrier without touching down barrier
    paths[:, :, 1] = 5000.0 * (1 + (threshold + 1.0) / 100.0)
    paths[:, :, 2] = 5000.0 * (1 - 0.5 / 100.0)  # low only moves -0.5%
    paths[:, :, 3] = 5000.0 * (1 + threshold / 100.0)

    eval_res = engine.evaluate(paths, p0, last_close, asset, horizon)
    assert eval_res.direction == "up"
    assert eval_res.p_up == 1.0
    assert eval_res.p_down == 0.0
    assert eval_res.target_pct >= threshold
    assert eval_res.target_pct <= 3.5 * threshold
    assert eval_res.invalidation_pct <= threshold
    assert eval_res.invalidation_pct >= 0.5 * threshold
    assert "kronos-small" in eval_res.evidence


def test_clean_down_swing():
    """Verify clean down moves register as down outcomes with valid target and invalidation."""
    engine = PathRulesEngine(tau=0.40, margin=0.05, rho=0.50, target_ratio=1.05)
    asset = "silver"
    threshold = THRESHOLD_PCT[asset]  # 2.0%
    p0 = 70000.0
    last_close = 70000.0
    horizon = 10
    N = 20

    paths = np.zeros((N, horizon, 4))
    paths[:, :, 0] = 70000.0
    paths[:, :, 1] = 70000.0 * (1 + 0.5 / 100.0)  # high only moves +0.5%
    paths[:, :, 2] = 70000.0 * (1 - (threshold + 1.5) / 100.0)  # low exceeds down barrier
    paths[:, :, 3] = 70000.0 * (1 - threshold / 100.0)

    eval_res = engine.evaluate(paths, p0, last_close, asset, horizon)
    assert eval_res.direction == "down"
    assert eval_res.p_down == 1.0
    assert eval_res.p_up == 0.0
    assert eval_res.target_pct >= threshold
    assert eval_res.target_pct <= 3.5 * threshold
    assert eval_res.invalidation_pct <= threshold
    assert eval_res.invalidation_pct >= 0.5 * threshold


def test_falling_knife_rejection():
    """Verify that an otherwise valid UP call is rejected as no_signal if ret1 < -0.5% and close in bottom 45% (when realign=False)."""
    engine = PathRulesEngine(tau=0.40, margin=0.05, rho=0.50, filter_knife=True, realign_momentum=False)
    asset = "gold"
    threshold = THRESHOLD_PCT[asset]
    paths = np.zeros((20, 5, 4))
    paths[:, :, 1] = 100.0 * (1 + (threshold + 1.0) / 100.0)  # clean up path
    paths[:, :, 2] = 100.0 * (1 - 0.2 / 100.0)

    # Severe down day yesterday: ret1 = -1.2%, closed at bottom 20% of range
    eval_res = engine.evaluate(paths, 100.0, 100.0, asset, 5, ret1=-0.012, close_pos=0.20)
    assert eval_res.direction is None
    assert "falling knife rejected" in eval_res.reason


def test_rocket_short_rejection():
    """Verify that an otherwise valid DOWN call is rejected as no_signal if ret1 > +0.8% and close in top 65% (when realign=False)."""
    engine = PathRulesEngine(tau=0.40, margin=0.05, rho=0.50, filter_rocket=True, realign_momentum=False)
    asset = "copper"
    threshold = THRESHOLD_PCT[asset]
    paths = np.zeros((20, 5, 4))
    paths[:, :, 1] = 100.0 * (1 + 0.2 / 100.0)
    paths[:, :, 2] = 100.0 * (1 - (threshold + 1.0) / 100.0)  # clean down path

    # Huge bull day yesterday: ret1 = +1.5%, closed at top 90% of range
    eval_res = engine.evaluate(paths, 100.0, 100.0, asset, 5, ret1=0.015, close_pos=0.90)
    assert eval_res.direction is None
    assert "rocket short rejected" in eval_res.reason


def test_momentum_realignment():
    """Verify that when realign_momentum=True, counter-momentum calls are realigned with the impulse."""
    engine = PathRulesEngine(tau=0.40, margin=0.05, rho=0.50, realign_momentum=True)
    asset = "gold"
    threshold = THRESHOLD_PCT[asset]
    
    # Path simulates UP, but yesterday was a severe breakdown
    paths = np.zeros((20, 5, 4))
    paths[:, :, 1] = 100.0 * (1 + (threshold + 1.0) / 100.0)
    paths[:, :, 2] = 100.0 * (1 - 0.2 / 100.0)
    eval_res = engine.evaluate(paths, 100.0, 100.0, asset, 5, ret1=-0.012, close_pos=0.20)
    # Realigns to down!
    assert eval_res.direction == "down"

    # Path simulates DOWN, but yesterday was a huge rocket rally
    paths[:, :, 1] = 100.0 * (1 + 0.2 / 100.0)
    paths[:, :, 2] = 100.0 * (1 - (threshold + 1.0) / 100.0)
    eval_res = engine.evaluate(paths, 100.0, 100.0, asset, 5, ret1=0.015, close_pos=0.90)
    # Realigns to up!
    assert eval_res.direction == "up"


if __name__ == "__main__":
    test_adverse_first_priority()
    test_clean_up_swing()
    test_clean_down_swing()
    test_falling_knife_rejection()
    test_rocket_short_rejection()
    test_momentum_realignment()
    print("All test_path_rules passed!")
