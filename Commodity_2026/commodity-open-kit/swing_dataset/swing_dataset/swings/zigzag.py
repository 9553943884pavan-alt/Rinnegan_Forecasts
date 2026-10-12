"""The percent-threshold zigzag of CHALLENGE-SPEC.md section 2, as a pure function.

Bars carry a high and a low and nothing else; the caller owns instants, contracts and sessions
and maps them back through the returned indices. The rule:

- the running extreme of the current direction extends while the bars keep making it;
- a leg completes when the price reverses at least ``threshold_pct`` of the running extreme
  against it. The completed leg runs from the previous extreme (its turn) to the running
  extreme, and the running extreme becomes the next leg's turn.

Two conventions the rule leaves open, fixed here and stated in the sidecar:

- **Reversal before extension.** A bar that both reverses the threshold from the running
  extreme and makes a new extreme of its own completes the leg. Within a bar the order of the
  high and the low is unknown, so the extreme established in an earlier bar wins.
- **The first leg.** Before a direction exists the running high and the running low are both
  tracked from the first bar; whichever threshold is crossed first fixes the direction and its
  extreme is the first turn. If one bar crosses both, the earlier of the two running extremes
  is the turn, and a tie (both on the same bar) resolves to ``down``.

Every completed leg moves at least ``threshold_pct`` by construction: its turn is the extreme
that the previous leg's completing reversal was measured against, and its own extreme is at
least as far as that reversal reached.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import DOWN, UP


@dataclass(frozen=True, slots=True)
class Leg:
    """One completed swing, in bar indices."""

    direction: str
    turn_index: int
    turn_price: float
    extreme_index: int
    extreme_price: float
    #: Bar whose reversal completed the leg; always after ``extreme_index``.
    confirm_index: int

    @property
    def move_pct(self) -> float:
        return (self.extreme_price - self.turn_price) / self.turn_price * 100.0


def zigzag(highs: Sequence[float], lows: Sequence[float], threshold_pct: float) -> list[Leg]:
    """Completed legs over ``highs``/``lows``, newest last.

    The trailing leg in progress at the end of the series is not returned: it has not reversed,
    so it is not yet a swing.
    """
    if len(highs) != len(lows):
        raise ValueError(f"highs {len(highs)} and lows {len(lows)} differ in length")
    if threshold_pct <= 0:
        raise ValueError(f"threshold_pct {threshold_pct} must be positive")
    up_factor = 1.0 + threshold_pct / 100.0
    down_factor = 1.0 - threshold_pct / 100.0

    legs: list[Leg] = []
    if not highs:
        return legs

    high, high_i = float(highs[0]), 0
    low, low_i = float(lows[0]), 0
    direction: str | None = None
    turn_i, turn_p = 0, float(highs[0])

    for i in range(1, len(highs)):
        bar_high, bar_low = float(highs[i]), float(lows[i])
        if direction is None:
            falls = bar_low <= high * down_factor
            rises = bar_high >= low * up_factor
            if falls and rises:
                falls = high_i <= low_i
                rises = not falls
            if falls:
                direction, turn_i, turn_p = DOWN, high_i, high
                low, low_i = bar_low, i
            elif rises:
                direction, turn_i, turn_p = UP, low_i, low
                high, high_i = bar_high, i
            else:
                if bar_high > high:
                    high, high_i = bar_high, i
                if bar_low < low:
                    low, low_i = bar_low, i
            continue
        if direction == UP:
            if bar_low <= high * down_factor:
                legs.append(Leg(UP, turn_i, turn_p, high_i, high, i))
                direction, turn_i, turn_p = DOWN, high_i, high
                low, low_i = bar_low, i
            elif bar_high > high:
                high, high_i = bar_high, i
        elif bar_high >= low * up_factor:
            legs.append(Leg(DOWN, turn_i, turn_p, low_i, low, i))
            direction, turn_i, turn_p = UP, low_i, low
            high, high_i = bar_high, i
        elif bar_low < low:
            low, low_i = bar_low, i
    return legs
