# Commodity Open 2026 scoring, rules 2.6: file formats and score definitions

Companion to CHALLENGE-SPEC.md sections 2, 3, 4, 4a and 5. Where the specification leaves a
choice, the choice made here is binding for the scorer and written down below. Task 1 is
`--challenge 1` in every tool and Task 2 is `--challenge 2`. The 2.4 scorer and its `FORMATS.md`
next door still describe the probability and distribution files graded against the one-sigma
barrier labels; those labels ship for research and are no longer the scored task.

The counts of realized legs in this file, 2,089 legs and the 169 legs of the holdout to 14 September among them, describe the inventory as it stood on 14 September 2026, where the limits below were set, and are not revised. Pack v2.0.1 holds 2,107 legs to 24 September, 187 of them in the holdout to 25 September.

## What a submission is

A submission is a set of decision records over the scored window: signal rows and `no_signal`
rows. Parquet is the interchange format and timestamps must be timezone-aware; naive timestamps
are rejected.

The two kinds may live in one file or in two. In one file, a `no_signal` row leaves `signal_utc`
null and fills `decision_utc` and `reason`; in two, pass the second with `--no-signals`. Any
number of signals per asset per session is allowed; on the holdout every one of them is in the denominator, and in the forward window every one with a horizon of at most 5 sessions (`FORWARD_MAX_HORIZON`).

### Task 1 signal row

| column | type | meaning |
|---|---|---|
| `asset` | string | `gold`, `silver`, `copper`, `crude` |
| `signal_utc` | timestamp[us, UTC] | the instant the signal was issued, inside a published MCX session or at the 08:30 IST pre-open |
| `direction` | string | `up` or `down` |
| `horizon` | int | 5 or 10 completed MCX sessions |
| `target_pct` | double | the model's own expected move from the price at `signal_utc`, at least the asset's threshold |
| `invalidation_pct` | double | how far against the signal the price may go first, at most the asset's threshold |
| `confidence` | double | the model's own confidence; stored, never scored |
| `evidence` | string or list of strings | the reason for the signal; stored, never scored, required |

### Task 2 signal row

| column | type | meaning |
|---|---|---|
| `asset`, `signal_utc`, `direction` | as above | |
| `horizon` | int | any whole number of completed MCX sessions from 1 to 20: the signal's stated duration |
| `expected_pct` | double | the expected move in the signalled direction from the price at `signal_utc`; scored in depth, not in direction |
| `p10_pct`, `p90_pct` | double | bounds on that move, `p10_pct` at or below `p90_pct` and at most `BAND_WIDTH_CAP` asset thresholds apart |
| `invalidation_pct` | double | how far against the signal the model considers itself wrong; stored and reported, never scored |
| `evidence` | string or list of strings | required, never scored |

A Task 2 horizon is graded exactly like a Task 1 one: the window is the `h` sessions that close
after the signal, so a 1-session signal issued at the pre-open is decided by the close of that
session, and a 20-session one by the close of the twentieth. `scoring2.HORIZONS` holds the valid
horizons of each task.

Section 4 sets no ceiling on a Task 2 `invalidation_pct` and the scorer sets none either, because
since 2.5.2 the number decides nothing: direction is scored on the asset's threshold on both
sides. The scorer still reports the median and the maximum of the file's values under `counts`,
so a reader can see what a model claimed to treat as its own stop even though the score does not
use it.

### `no_signal` row

| column | type | meaning |
|---|---|---|
| `asset` | string | the asset the model looked at |
| `decision_utc` | timestamp[us, UTC] | the instant of the decision |
| `reason` | string | one line saying why no signal was issued; required |

### Thresholds

Percent of the price at the turn, CHALLENGE-SPEC section 2: gold 1.5, silver 2.0, copper 1.5,
crude 3.0.

### `BAND_WIDTH_CAP`

`scoring2.BAND_WIDTH_CAP = 12`: `p90_pct - p10_pct` may be at most twelve times the row's
threshold, so a Task 2 band runs at most 18 points on gold and copper, 24 on silver and 36 on
crude. Depth asks only whether the realized move landed inside the band, so `[-99, +99]` contains
every move and scores 100 with no model behind it; the cap is what stops that.

Twelve is set off our own baselines and the realized legs, not chosen for roundness. On the
holdout to 14 September, the window the cap was set on, the three shipped Task 2 baselines quote
band widths, in threshold units, with a median of 4, a p99 of 11 and a maximum of 22.26
(`prevailing_direction` 4.95 / 11.08 / 16.25, `rv_band` 4.03 / 11.31 / 12.06, `range_break_vol`
4.04 / 10.98 / 22.26). The 2,089 legs of the inventory of that date run 1.95 thresholds at the
median, 4.32 at the p90 and 8.48 at the p99. Twelve therefore sits above the p99 of what the
baselines quote and above the p99 of what the market delivers, and cost 0.33 to 0.57 percent of
baseline rows: six rows of 1,566, all at h=10 and all of them bands whose upper end came from a
trailing excursion quantile taken in June. The baseline files in the kit's `notebooks/output/` run
to 25 September and give medians of 4.78 / 3.95 / 3.88, p99s of 11.08 / 11.20 / 10.95 and maxima of
16.25 / 12.06 / 22.26; the cap costs 0.29, 0.29 and 0.51 percent of their rows, six rows of 1,750,
still all at h=10.

## Validation

A signal row is invalid when any of these holds, and `counts.invalid_reasons` names which:

| reason | rule |
|---|---|
| `unknown_asset` | the asset is not one of the four |
| `bad_direction` | the direction is not `up` or `down` |
| `bad_horizon` | Task 1: the horizon is not 5 or 10. Task 2: it is not a whole number from 1 to 20, so 0, 21 and 2.5 are all invalid |
| `missing_evidence` | `evidence` is empty after stripping whitespace |
| `bad_target_pct` / `bad_expected_pct` | not finite, or not positive |
| `target_below_threshold` | Task 1: `target_pct` below the asset's threshold |
| `bad_invalidation_pct` | not finite, or not positive |
| `invalidation_above_threshold` | Task 1: `invalidation_pct` above the asset's threshold |
| `bad_band` / `band_inverted` | Task 2: `p10_pct` or `p90_pct` not finite, or `p10_pct` above `p90_pct` |
| `band_too_wide` | Task 2: `p90_pct - p10_pct` above `BAND_WIDTH_CAP` times the asset's threshold |

A row the price series cannot price at its instant is not a validation failure but is treated the
same way: it is counted under `counts.no_price`, stays in the denominator, and is never correct.

**An invalid row counts as issued and is never correct.** Dropping it would pay a model for
issuing rows the rules reject: a target under the threshold would cost nothing, and a row with no
evidence would be free. It is not a decision record, so it does not help the coverage floor
either. A `no_signal` row with an empty reason is not a decision record for the same reason.

## The price path

Everything a score needs about what the price did comes from one contract and one walk over it.

- **Series.** The contract designated for the signal's session in
  `labels/designations.parquet` of the pack at horizon 5: the series the realized-swing inventory is
  built on. A signal's own horizon sets the length of the window, never the contract, and the walk
  holds that contract to the end of the window, so a roll inside the window never enters as a
  calendar spread.
- **Entry price.** The last price of that contract available at or before `signal_utc`: the last
  captured tick or the last published settlement, whichever became available later. A pre-open
  signal therefore enters on the previous settlement and an intraday signal on the last trade.
  The report carries `price_source` per row and `counts.tick_priced` over the file.
- **Resolution.** A session with at least five captured quotes after the signal is walked tick by
  tick, which settles the order of two barriers touched on the same day. The production feed's
  quotes begin on 2026-06-22; before that a session is one bar of its high and its low, where the
  order inside the bar is unknowable. There the adverse barrier is taken first and the row is
  flagged `ambiguous`, counted in `counts.ambiguous_bar`. A barrier the evidence cannot place
  before the invalidation has not been shown to come first.
- **Window.** From `signal_utc` to the close of the `h`-th session that closes strictly after it.
  A session already under way when the signal is issued is one of the `h`, and only the part of it
  after the signal is walked.
- **Excursions.** `mfe_pct` is the furthest the price went in the signalled direction over the
  whole window and `mae_pct` the furthest against it, both in percent of the entry price.
- **Move to the swing's end.** `swing_end_pct`, Task 1 rows only: the same walk keeps a running
  favorable extreme (the highest price so far for an up signal, the lowest for a down one) and
  the swing ends at the first price that turns back a full asset threshold from it, `(extreme -
  price) / extreme` for up and `(price - extreme) / extreme` for down, in percent. The move is
  from the entry price to the extreme at that point, or to the extreme of the whole walk when
  nothing turned that far before the window ended. The walk starts at the first price after the
  signal, and its prices are exactly the barrier walk's: the ticks after the signal in a session
  that has at least five of them, the session's high and low otherwise. The order inside a bar is
  unknowable, so a bar is walked favorable side first: high then low for up, low then high for
  down. That lets a bar whose own range turns a threshold end the swing at the bar's extreme, and
  lets a turn from an earlier extreme be measured from the bar's new one when the bar made one.
  A later move in the same direction, after a turn of a full threshold, belongs to the next swing,
  which is why `swing_end_pct` is never more than `mfe_pct`.
- **Maturity.** A window that runs past the end of the data is scored anyway when the outcome is
  already decided inside the part that exists, and `counts.scored_on_incomplete_window` says how
  many rows that was. Otherwise the row is unmatured: counted in `counts.unmatured`, in neither
  numerator nor denominator, and still a decision record for coverage.

## Task 1, section 3

| score | definition |
|---|---|
| catch rate | swings caught divided by realized swings in the window |
| correctness | correct signals divided by decided signals; undecided signals are left out and counted apart, invalid and unpriced rows stay in as incorrect |
| depth | among correct signals, the share that reached `target_pct` before `invalidation_pct` **and** whose `target_pct` was at least half of the move to the swing's end |
| success rate | the plain average of the three |

- **Correct**: from the entry price, the price moved at least the asset's threshold in the
  signalled direction inside the horizon before it moved `invalidation_pct` against it.
- **Decided**: the price touched the threshold level or the invalidation level inside the
  horizon. A scored signal whose window completed with neither touched is **undecided**: the
  market did not move far enough to settle the call either way. It is left out of correctness,
  and being neither correct nor deep it is in no score at all; `counts.undecided` holds how many
  there were, `counts.undecided_share` their share of the scored signals, and `counts.decided`
  the valid priced rows that were decided. Scored rows split exactly four ways: `decided +
  undecided + invalid + no_price = scored`, and the correctness denominator is every scored row
  but the undecided ones. A row whose window has not completed and is not decided is unmatured,
  not undecided, and stays out of everything as before.
- **Deep**: correct, `target_pct` reached before `invalidation_pct`, and `target_pct` at least
  `DEPTH_TARGET_SHARE = 0.5` of `swing_end_pct`. A target never reached fails whatever its size,
  and so does one under half the move. A target that is reached after the swing's end, in the next
  swing, still passes: the rule asks only that it be reached before the invalidation and be at
  least half the move. Until 2.6 the yardstick was `mfe_pct`, the furthest point of the whole
  window, which could include the next swing's extension.
- **Caught**: a correct signal in the swing's direction, issued strictly after the swing's turn
  instant and at or before its extreme instant, with at least `CATCH_ROOM_SHARE = 0.5` of the
  asset's threshold of the move still ahead of it: from the entry price to the swing's extreme
  price, in percent of the entry price and signed for the direction, at least half the threshold.
  The first such signal in issue order is recorded as the catcher, by `caught_by_signal_index`
  and `caught_by_signal_utc`. Until 2.6 a whole threshold of room was needed.
- **Which swings count**: those whose turn and extreme both lie inside the scored window, so the
  whole catch opportunity was inside it. The legs straddling either edge are counted separately
  (`swings_straddling_window_start`, `swings_straddling_window_end`) and never scored.

A comparison on one of these percentage boundaries (half a threshold of room, a turn of a full
threshold, half the move) allows `PCT_TOLERANCE = 1e-12`, so a value exactly on the boundary in
decimal that lands a few ulps under it in binary still counts as on it.

The inventory is `swings/swings.parquet` in the pack: `asset, direction,
turn_utc, turn_price, extreme_utc, extreme_price, move_pct, sessions_spanned, contract_id,
contract_changed, bar, threshold_pct, available_utc, confirm_utc, confirm_available_utc,
clock_quality`. The scorer needs the first six and the threshold; it carries the rest through.

`confirm_utc` is the instant of the bar whose move completed the leg, the first bar to reverse
`threshold_pct` from the extreme, and `confirm_available_utc` is the settlement publication of
that bar's session, the same convention `available_utc` uses. Both are non-null on every row,
because a leg is only written once it has completed; `confirm_utc` is at or after `extreme_utc`
and `confirm_available_utc` is at or after `confirm_utc`.

A point-in-time consumer must use `confirm_available_utc`. In pack v2.0.1 the sidecar of
`swings/swings.parquet` still names `available_utc` as the file's published clock, so `PointInTime`
returns a leg from the settlement of its extreme's session; filter the rows on `confirm_available_utc`
yourself. The second data file names `confirm_available_utc`, and the client then does it.
`available_utc` is when the extreme's
own settlement published, which is knowable, but the fact that the extreme was a swing's extreme
is not knowable until the leg confirms, which is a median of one session and up to eight sessions
later. `swings.normalize` keeps the columns listed above and drops the two, because catch,
correctness and depth are decided by the signal's own instant against the price path and never by
the inventory's availability; the two instants matter to a model that trains on the inventory, not
to a score.

A roll can confirm a leg on a different contract from the one the leg sits on: 90 of the 2,089
legs are confirmed by a bar of the next contract. This file reads `contract_changed` as a
statement about the leg's own turn-to-extreme span and nothing else, so it is false on all 90 of
those legs and true on each of their next legs, which are the ones whose own span crosses the
roll.

### The ten large swings

`scoring2.TOP_SWINGS = 10` and `TOP_SWINGS_REQUIRED = 7`. The swings of the window are ranked by
the move from turn to extreme divided by the asset's threshold (`move_pct` where the inventory
carries it, the turn-to-extreme move otherwise), largest first, ties to the earlier `turn_utc` and
then to the asset name, and the first ten are the ten large swings. Ranking by the threshold
multiple measures each commodity against its own normal range, so crude's larger percentage moves
do not fill the list. Each of the ten is caught or not exactly as for the catch rate.

The report prints them under `ten_large_swings`: `top`, `required`, `swings_ranked` (fewer than
ten only when the window holds fewer swings), `caught`, and per swing `rank`, `asset`,
`direction`, `turn_utc`, `extreme_utc`, `move_pct`, `threshold_pct`, `threshold_multiple`,
`caught` and `caught_by_signal_utc`. On the holdout at least seven must be caught, a gate of the
minimum. On the forward window the ten are listed and the check does not apply.

### The minimum

Five gates under `award_conditions.gates`, each with its `value`, its `threshold`, whether it is
`met`, and whether it is `applicable` to the period. None of them averages with another:

| gate | value | threshold |
|---|---|---|
| `success_rate` | the success rate | 70 |
| `correctness` | the correctness score | 60 |
| `depth` | the depth score | 60 |
| `coverage_floor` | the least covered asset's share of its sessions | 0.90, and every asset must reach it |
| `ten_large_swings` | how many of the ten large swings were caught | 7; applicable on the holdout only, `met` null on the forward window |

A score with no units has the value null and fails its gate. `minimum_met_this_period` is true when
every applicable gate is met, and the minimum must hold on both periods.

### `LARGE_SWING_MULTIPLE`

`scoring2.LARGE_SWING_MULTIPLE = 2`: a realized leg is large when its move is at least twice the
asset's threshold, so 3.0 percent on gold and copper, 4.0 on silver, 6.0 on crude. The catch rate
restricted to those legs is printed under `diagnostics.large_swing_catch_rate` with its own
episode-block interval, the two counts it divides, the multiple, the percentage per asset it
implies and the pooled catch rate beside it. Each shipped Task 1 baseline carries the same block
under `baselines.<name>.diagnostics`, so the number has something to sit next to.

It is reported and never scored. It is not one of the three scores of section 3, it is not
averaged into the success rate and it does not appear in `award_conditions`. The scored catch
rate is the pooled one over every leg above the threshold, and the ten large swings of the
minimum are a different set: the ten largest legs of the window by threshold multiple.
CHALLENGE-SPEC section 3 and step 4 of section 4a say what the diagnostic is for: the organizers'
own test of Task 1 is whether a model catches the swings a person would notice, and the holdout's
legs run from 1.52 to 18.93 percent, so one pooled number over all of them does not answer that
on its own.

Two is set off the holdout rather than chosen for roundness. On the window the scorer resolves,
2026-06-01 to 2026-09-14, the catch rate scores 169 legs and 78 of them are large, 46.2 percent:
gold 16 of 36, silver 24 of 51, copper 12 of 30, crude 26 of 52. The median leg of that window
runs 1.89 thresholds, so the line falls just above the middle of the inventory and each asset
keeps a denominator worth reading. Silver's 51 legs have a median turn price of 234,571, and four
percent of that is 9,383 points, which is where the owner independently drew the line when
describing a layman-visible silver move as up to ten thousand points. Three thresholds would cut
the set to 40 legs and copper to 3, which is too thin to report per asset; one restricts nothing.
Over the whole 2,089-leg inventory the multiple selects 998 legs, 47.8 percent, so the holdout is
not a window where the definition happens to behave.

## Task 2, section 4

| score | definition |
|---|---|
| direction | signals whose direction was realized divided by signals issued, realized on the asset's threshold in both directions |
| depth | among directionally right signals, the share whose realized move landed inside `[p10_pct, p90_pct]` |
| improvement | ten times the learning model's direction gain over its frozen copy in percentage points, floored at 0 and capped at 100 |
| success rate | the plain average of the three |

- **Realized**: the price moved at least the asset's threshold in the signalled direction inside
  the horizon before it moved the threshold against it. This is the same test `grade_c1` decides
  `correct` on and the same test the prevailing-direction null is measured by, deliberately so:
  the two tasks are then scored with one ruler and neither side of the Task 2 direction test is a
  number the entrant supplies. Before 2.5.2 the test ran on the entrant's own `expected_pct` and
  `invalidation_pct` with no bound on either, and `CHANGES.md` records what that bought a
  submission that set them to suit itself. `expected_pct` and the band are scored in depth
  instead, and `invalidation_pct` is carried and reported but decides nothing.
- **The realized move** is the favorable extreme of the window from the entry price, `mfe_pct`.
  The direction rule and the depth rule are then answers about the same excursion.
- **Horizons**: any whole number from 1 to 20, every one scored on the holdout, those of at most 5
  on the forward window. The shipped Task 2 baselines stay at 5 and 10.
- **Improvement** is measured on the rows the learning model and its frozen copy both issued at
  the same instant for the same asset and horizon; a repeated key keeps its first row on each side
  so that one row cannot be paired against several. `paired_direction` reports the two scores and
  the number of rows behind them.
- **The prevailing-direction null**: the sign of the last five completed sessions' move of the
  designated contract, issued at the entry's own instants with the entry's own horizon,
  `expected_pct` and `invalidation_pct` at the asset's threshold. The move is measured on one
  contract where the far end of the five sessions has a settlement for it. A move of exactly zero
  takes the up direction, so the null is a function of the data and not of a coin. Since 2.5.2
  scores direction on the threshold on both sides, the null is graded by the same rule as every
  other row, and an entry issuing the null's directions at the null's instants scores the null's
  direction number exactly.

The five hard conditions of section 4 are reported as booleans under `hard_conditions`:

| condition | rule |
|---|---|
| `learning_beats_frozen` | the paired direction score of the learning model exceeds the frozen copy's |
| `second_half_at_least_first_half` | the direction score over the window's later sessions is at least the score over its earlier ones, splitting the window's session calendar in two |
| `direction_at_least_minimum` | the direction score alone is at least 60 percent |
| `frozen_beats_prevailing_null` | the frozen copy's direction score exceeds the null's at the frozen copy's instants; stated on the holdout, null elsewhere |
| `learning_beats_prevailing_null` | the learning model's direction score exceeds the null's at its own instants, on both periods: section 4's eligibility sentence on the direction score |

Without a frozen copy there is no improvement score and therefore no success rate: section 4's
third score is not optional. A score whose denominator is empty is reported as null rather than as
zero, and the success rate is null with it, because an average of two numbers is not the average
of three.

## Coverage floor, section 5

A decision record on at least 90 percent of the asset's sessions in the scored window, per asset
and pooled. A decision record is a valid signal row or a valid `no_signal` row, mapped to the
session whose pre-open to close range contains its instant; records outside every session are
counted under `decision_records_outside_a_session` and cover nothing. The report carries
`coverage.per_asset`, `coverage.pooled` and `coverage.met`, which is true only when every asset
meets it.

## Periods, episodes and intervals

| period | window |
|---|---|
| holdout | 2026-06-01 to the pack's last session; every valid horizon is scored, 5 and 10 on Task 1 and 1 to 20 on Task 2 |
| forward | after the submission close, `--test-start` required; only horizons of at most `FORWARD_MAX_HORIZON = 5` sessions are scored, 5 on Task 1 and 1 to 5 on Task 2, and a valid row with a longer horizon is counted under `recorded_not_scored_other_horizon` and never scored |

An episode is a block of `h` sessions per asset counted from the window start. For Task 1, `h`
is the longest horizon the period scores: 10 on the holdout, 5 on the forward window. For Task 2
it is the longest horizon the entry's valid rows in the scored set use, so an entry at 5 and 10
is drawn in blocks of 10, exactly as under 2.5.2, and one reaching 15 in blocks of 15. Blocks of
the 20 sessions the holdout allows would leave about four per asset on the 85 holdout sessions of
pack v2.0, too few for an interval to mean anything. An entry with no such row gets the longest
horizon the shipped baselines are scored at, 10 on the holdout and 5 on the forward window; an
invalid row's horizon is not one the entry used. The report states the `h` its intervals were
drawn with as `window.episode_horizon`, and each baseline's as `baselines.<name>.episode_horizon`.
The episode of a signal is the block holding its session; the episode of a swing is the block
holding its turn.

The three scores of a task live on three different unit sets, so the episodes are drawn once for
all of them. The universe is every `(asset, block)` pair any unit falls in; a replicate draws that
many episodes with replacement and recomputes every score, the success rate and the improvement
included, on the drawn episodes. Two thousand seeded resamples, the 5th and 95th percentiles,
printed for every score as `ci_lo` and `ci_hi`. For a single unit set covering the whole universe
this is exactly `scoring.folds.block_bootstrap_mean` with the same seed.

## The report

`evaluate` writes one JSON per task and period, with `rules_version` `2.6`:

| key | contents |
|---|---|
| `window` | the period, its bounds, `horizons_scored` for the task and `episode_horizon`, the block length the intervals were drawn with |
| `scores` | the three scores and the success rate, each with `value`, `ci_lo`, `ci_hi`, `level`, `n`, `n_episodes`, `n_boot`, `seed` |
| `counts` | rows in the window, invalid, unmatured, scored, recorded but not scored, correct, deep, swings and catches, no_signal rows, ambiguous bars, tick-priced rows; Task 1 also `decided`, `undecided` and `undecided_share` |
| `coverage` | the floor, per asset and pooled |
| `ten_large_swings` | Task 1 only: the ten largest swings of the window by threshold multiple, whether and by which signal each was caught, and how many were |
| `diagnostics` | Task 1 only: `large_swing_catch_rate`, reported and never scored |
| `hard_conditions`, `paired_direction`, `halves`, `prevailing_direction_null`, `frozen` | Task 2 only |
| `baselines` | each baseline's episode block, scores, counts, coverage and, on Task 1, diagnostics on the same window, with `--baselines` |
| `award_conditions` | the minimum of sections 3 and 4 for this period: `horizons_scored`, the success rate and coverage booleans, Task 1's `gates` and Task 2's `hard_conditions`, and `minimum_met_this_period` |
| `inputs` | every input by path, sha256, bytes and row count |
| `build` | git branch, commit and the exact command |
| `definitions` | one line for every number in the report |

A graded Task 1 row, as `update` receives it through the kit's loop, carries `p0`,
`price_source`, `contract_id`, `window_end_utc`, `window_complete`, `bars`, `ambiguous`,
`mfe_pct`, `mae_pct`, `swing_end_pct`, `threshold_hit`, `correct`, `decided`, `undecided`,
`target_first`, `deep`, `scorable`, `unmatured` and `no_price`.

`award` joins a holdout report and a forward report into `minimum_met_on_both_periods` and the
ranking key: the forward success rate, ties broken by the holdout success rate.

## Baselines

All of them are loops. Each issues a decision record for every asset and every session of the
window at that session's pre-open, from rows published before it, and refits its own quantities at
every cutoff: nothing is fitted once and reused. The trailing window for a refit is the last 60
matured sessions, 250 for the stretch cut; a quantity with fewer than 10 usable rows falls back to
its unconditional form and the row is flagged `fallback`. Every baseline issues at horizons 5 and
10 (`BASELINE_HORIZONS`) where the period scores them: both on the holdout, 5 on the forward
window, on both tasks.

Task 1:

| name | rule |
|---|---|
| `range_break` | the prior session settled above the high or below the low of the session before it. `target_pct` is the larger of the threshold and the trailing median favorable excursion over the horizon; `invalidation_pct` is the threshold |
| `stretch_reversal` | the chain's 20-session stretch, `(log W - 20-session mean) / 20-session standard deviation`, is beyond the fitted 80th percentile of its own absolute value, and the signal leans the other way. Same target and invalidation |

Task 2:

| name | rule |
|---|---|
| `prevailing_direction` | section 4's null: the sign of the last five completed sessions' move, `expected_pct` and `invalidation_pct` at the threshold, the band from the trailing 10th and 90th percentiles of the favorable excursion |
| `rv_band` | the sign of the 20-session trend, `expected_pct` the 20-session realized volatility scaled to the horizon times the trailing median of that ratio, the band from its trailing 10th and 90th percentiles, invalidation at the threshold |
| `range_break_vol` | the range break of Task 1 with the volatility-scaled target and band of `rv_band` |

They show the mechanics and give the report a floor. They clear no prize. `baselines` writes each
one's decision records to `baseline_signals_{challenge}_{name}_{period}.parquet` in `--out-dir`,
with a sidecar, and the scores to `baseline_report_{challenge}_{period}.json` there. A Task 2 baseline
scored against itself has an improvement of zero by construction, and the report says so.

## Commands

Teams score a signal file with `tools/score_holdout.py`, which points the scorer at the pack and prints every gate:

```bash
python tools/score_holdout.py --pack $SWING_PACK_ROOT --predictions predictions/
```

The same code runs from the command line. Outside the organizers' build tree every subcommand needs the four data flags, so the organizers' re-run passes them:

```bash
PACK=$SWING_PACK_ROOT
DATA="--mcx-daily $PACK/mcx/mcx_daily.parquet --designations $PACK/labels/designations.parquet \
      --mcx-sessions $PACK/mcx/mcx_sessions.parquet --quotes $PACK/pack/quotes/mcx_quotes.parquet"
python -m swing_dataset.scoring2 evaluate --challenge 1 --period holdout $DATA \
    --signals submission.parquet --swings $PACK/swings/swings.parquet \
    --baselines --out report_1_holdout.json
python -m swing_dataset.scoring2 evaluate --challenge 2 --period holdout $DATA \
    --signals live.parquet --frozen frozen.parquet --swings $PACK/swings/swings.parquet \
    --baselines --out report_2_holdout.json
python -m swing_dataset.scoring2 baselines --challenge 1 --period holdout $DATA \
    --swings $PACK/swings/swings.parquet --out-dir baselines/
python -m swing_dataset.scoring2 award --reports report_1_holdout.json report_1_forward.json
```

`--test-start` and `--test-end` narrow the window and are required for `--period forward`.
`--n-boot` and `--seed` control the bootstrap. `--swings` is needed for Task 2 too: its scores do not
use the inventory, but the report records the file among its inputs.
