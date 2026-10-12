# Commodity Open 2026, the superseded 2.4 scorer: file formats and metric definitions

This documents the superseded 2.4 scorer. `FORMATS2.md` documents the 2.6 scorer,
which is the one in force and the one the prizes are decided by.

Companion to RULES.md sections 3, 4 and 7. Where the specification leaves a choice, the
choice made here is binding for the scorer and written down below.

## Keys and types

Every prediction and outcome row is keyed by `(asset, cutoff_utc, horizon)`.

| column | type | meaning |
|---|---|---|
| `asset` | string | `gold`, `silver`, `copper`, `crude` |
| `cutoff_utc` | timestamp[us, UTC] | the MCX pre-open cutoff, 08:30 Asia/Kolkata expressed in UTC (03:00Z) |
| `horizon` | int | 5, 10 or 20 sessions |

Parquet is the interchange format. Timestamps must be timezone-aware; naive timestamps are
rejected. One row per key; duplicates are rejected.

## Outcome files (labels lane)

`swing_labels.parquet`: `asset, cutoff_utc, horizon, k_multiplier, label, available_utc`.
The scorer keeps `k_multiplier == 1.0`. `label` is one of `up_swing`, `down_swing`, `no_swing`,
`both`, or null (ungradable). `available_utc` is the settlement publication of the session that
decided the label.

`return_targets.parquet`: `asset, cutoff_utc, horizon, y_ret, y_mfe, y_mae, available_utc`. The
scorer uses `y_ret` and `available_utc`; a null `y_ret` is ungradable.

Eligible cutoffs are the rows of the outcome file for the horizon. Rows labelled `both` or
ungradable are eligible (they count in the coverage denominator) but are never scored.

## Challenge 1 prediction file

`asset, cutoff_utc, horizon, p_up, p_down, p_none, abstain`

- `p_up, p_down, p_none` are the probabilities of `up_swing`, `down_swing`, `no_swing`. They must
  be finite, non-negative and sum to one within 1e-6 on every row that does not abstain.
- `abstain` is boolean. When true the probabilities are ignored. A missing `abstain` column means
  false everywhere.
- A row that abstains, or an eligible cutoff with no row at all, is scored as the training class
  frequencies of its asset (the pooled frequencies when the asset has no training rows) and is
  flagged in the report (`substituted`, `substituted_with`). Both count as not issued.
- Rows for cutoffs that are not eligible for the horizon are ignored and counted
  (`predictions_for_ineligible_cutoffs`).

## Challenge 2 prediction file

`asset, cutoff_utc, horizon, <distribution>, p_up, abstain`

`<distribution>` is exactly one of:

- nine quantile columns `q_025, q_050, q_100, q_250, q_500, q_750, q_900, q_950, q_975` holding
  the quantiles of `y_ret` (the horizon log return of the chain) at levels
  `.025, .05, .10, .25, .50, .75, .90, .95, .975`; or
- a `samples` column holding a list of exactly 512 doubles per row.

`p_up = P(y_ret > 0)`. A file must not carry both representations. Abstained or missing rows are
scored as the empirical training distribution of `y_ret` for the asset (its nine quantiles, or a
stratified 512-sample at the midpoints `(i + 0.5) / 512`), with `p_up` the training share of
positive returns, and are flagged.

The frozen day-zero copy for the loop gate is a second file in the same format and the same
representation as the live submission.

## Metric definitions

Challenge 1, pooled across assets per horizon:

| metric | definition |
|---|---|
| `log_loss` | mean of `-log p(realized class)`, `p` clipped below at 1e-15 |
| `brier` | mean over rows of the summed squared error over the three class indicators, range 0 to 2 |
| `ece` | classwise expected calibration error: every scored row contributes three (predicted probability, class realized) pairs; pairs are binned into ten equal-width bins on the predicted probability; ECE is the count-weighted mean absolute gap between mean predicted probability and realized frequency. The `reliability` table lists each bin. `top_label_ece` (argmax confidence against accuracy) is reported alongside |
| `swing_capture` | for each swing class the threshold is the 80th percentile of that class's predicted probability over training rows whose realized class is different (a 20 percent false-alarm rate); capture is the share of realized outcomes of the class whose predicted probability exceeded the threshold; reported per class and pooled, with the false-alarm rate the threshold produced on the test rows. The threshold is fitted on the prediction file's rows before `train_end`; when there are none it is fitted on the test rows and the report says so |
| `coverage` | issued (non-abstaining) predictions for eligible cutoffs in the window divided by eligible cutoffs |

Challenge 2, pooled across assets per horizon:

| metric | definition |
|---|---|
| `crps` | samples: the exact empirical CRPS, `mean_i |x_i - y| - (1/2) mean_{i,j} |x_i - x_j|`. Quantiles: the pinball integral `2 * integral over tau of pinball_tau(q_tau, y)` evaluated on the nine levels with midpoint-rule weights over the level spacing (`0.0375, 0.0375, 0.1, 0.2, 0.25, 0.2, 0.1, 0.0375, 0.0375`). The two estimators agree within a few percent on the same forecast, but baselines are still scored in the submission's own representation and margins are also reported relative to the baseline |
| `pinball_mean` | quantile submissions only: twice the equal-weight mean pinball loss over the nine levels (the specification's plain pinball average). A secondary number; it sits about a third below `crps` because the levels crowd the tails |
| `brier_p_up` | mean `(p_up - 1[y_ret > 0])^2` |
| `pit_histogram` | ten equal bins of the probability integral transform. Samples: mid-rank empirical CDF at `y_ret`. Quantiles: piecewise-linear CDF through the nine (quantile, level) points, clamped to `.025` and `.975` outside them. A chi-square statistic against uniformity is attached |
| `coverage_90` | share of `y_ret` inside `[q_050, q_950]` (samples: the empirical 5 and 95 percent quantiles); nominal 0.90 |
| `loop_gate` | live minus frozen CRPS on the same rows, with the same interval rule; passes when the live model is lower and the interval excludes zero |

## Episodes, confidence intervals, awards

- Session calendar: the sorted distinct `cutoff_utc` values of the outcome file for the horizon.
- Episode: block of `horizon` consecutive sessions counted from the first session at or after the
  test start, per asset. Two scopes exist and the report names the one used (`--episode-scope`):
  `asset` (default) makes every (asset, block) pair its own episode; `calendar` treats all
  assets on the same block as one episode, which keeps same-day forecasts of correlated assets
  together in the bootstrap at the cost of a quarter of the units. Both counts are reported
  (`issued_scored_episodes_by_scope`). A partial tail block is its own episode, so the 72-session
  holdout gives 15 blocks at h=5 and 8 at h=10, which is 60 and 32 asset episodes against 15 and
  8 calendar ones; the 19-session forward month gives 4 and 2 blocks, which is 16 and 8 asset
  episodes against 4 and 2 calendar ones.
- Confidence interval: paired per-row differences (submission minus baseline) are grouped by
  episode; episodes are resampled with replacement 2000 times (seeded), the row-weighted mean
  difference recomputed, and the 5th and 95th percentiles taken. `beats` means the mean
  difference is below zero and the interval excludes zero.
- Coverage floor (RULES.md section 5): issued (non-abstaining) predictions for at least 90
  percent of the eligible cutoffs in the scored window. The report carries it as
  `coverage_floor: {issued_share_at_least: 0.9, issued_share, met}`, where `issued_share` is the
  `coverage` metric; the issued, episode and asset counts under `coverage` are information and
  never gate.
- `award_conditions` in a report evaluates one horizon. `python -m swing_dataset.scoring award`
  joins the h=5 and h=10 reports into the section 7 booleans, and names any other horizon it was
  handed under `horizons_ignored`: the forward-score award needs the submission to beat every
  baseline with the interval rule at both horizons and to meet the coverage floor at both; the
  loop-gate award needs the gate to pass at both.

## Baselines

All baselines are loops: at every cutoff they are refitted on outcomes whose `available_utc` is
strictly before the cutoff and predict from chain rows available strictly before it.

Shipped weights. The kit's `build/scoring/` carries each baseline's prediction file over the
scored window and, in `baseline_report_{challenge}_{horizon}.json` under `fitted_at_last_cutoff`,
the coefficients of the fitted baselines (`price_only_logit`, `har_rv`, `price_only_ls`) as they
stood at the labels' last cutoff for that horizon, fitted on the outcomes that had matured before
it like at every other cutoff. They are there to read and to reproduce, nothing more:
`evaluate --baselines` refits every baseline at every cutoff from the labels and price history it
is given, never loading those weights, so a participant's holdout score does not depend on them
and the organizers' re-run cannot diverge from it on their account.

Feature chain (`chain.py`): per asset, on every session the designated contract is the one with
the highest volume over the two most recent completed sessions among contracts with at least five
sessions to expiry, ties by earlier expiry. The chain return is the same-contract log settlement
return of that session's designated contract. Designation changes are flagged as transitions and
excluded from the volatility windows, which extend backwards to keep their length. Features known
after session `k` closes: `ret1` (chain return), `ret5`/`ret20` (log wealth change over 5 and 20
sessions), `rv20` (standard deviation of the last 20 non-transition returns), `stretch`
(`(log W - 20-session mean of log W) / 20-session std of log W`), and the HAR components `rv_d`,
`rv_w`, `rv_m` (squared non-transition returns averaged over 1, 5 and 22 sessions). This chain
feeds features only; labels come from the labels lane's chain.

Challenge 1:

| name | forecast |
|---|---|
| `class_frequency` | training class frequencies of the asset (pooled when none) |
| `persistence` | first-order Markov forecast from the realized class of the latest cutoff of the same asset whose label had matured before the cutoff; 3x3 transition matrix with add-one smoothing; the prior before anything has matured |
| `price_only_logit` | multinomial logistic regression (L2, C=1, lbfgs) pooled over assets with asset indicators on `ret1/rv20`, `ret5/(rv20 sqrt 5)`, `ret20/(rv20 sqrt 20)`, `log rv20`, `stretch`, standardized on the training rows; the class frequencies until 30 training rows exist |

Challenge 2 (each a location, a scale and a standardized shape):

| name | location | scale | shape |
|---|---|---|---|
| `empirical` | 0 | 1 | training `y_ret` of the asset |
| `rv_gaussian` | 0 | `kappa * rv20 * sqrt(h)`, `kappa` the training RMS of `y_ret / (rv20 sqrt h)` | standard normal |
| `har_rv` | 0 | `sqrt(h * exp(pred + s^2/2))`, `pred` the log-HAR (Corsi 2009) regression of the forward mean squared chain return over the horizon on `log rv_d, log rv_w, log rv_m` with asset intercepts, pooled over assets | training `y_ret` standardized by the fitted scale |
| `price_only_ls` | least squares of `y_ret` on the price-only features (standardized, asset indicators) | `kappa * rv20 * sqrt(h)` on the residuals | training standardized residuals |

Quantiles are the shape's quantile function at the nine levels; samples are the quantile function
at the 512 midpoints; `p_up` is one minus the shape CDF at `-mu/sigma`. Baseline prediction files
carry a `fallback` flag for cutoffs where the unconditional form was used.

## Commands

These commands belong to the superseded 2.4 scorer and read the organizers' build tree. Teams score with `tools/score_holdout.py`.

```bash
python -m swing_dataset.scoring evaluate --challenge 1 --horizon 5 \
    --predictions sub.parquet --labels $SWING_BUILD_ROOT/build/labels/swing_labels.parquet \
    --baselines --out report_1_5.json
python -m swing_dataset.scoring evaluate --challenge 2 --horizon 5 \
    --predictions live.parquet --frozen-predictions frozen.parquet \
    --labels $SWING_BUILD_ROOT/build/labels/return_targets.parquet --baselines --out report_2_5.json
python -m swing_dataset.scoring baselines --challenge 1 --horizon 5 \
    --labels $SWING_BUILD_ROOT/build/labels/swing_labels.parquet --test-start 2024-01-01
python -m swing_dataset.scoring award --reports report_1_5.json report_1_10.json
```

`--test-start/--test-end` narrow the scored window (default: the prediction file's span);
`--train-end` (default: the first test cutoff) separates the training slice used for the
abstention substitute and the false-alarm threshold. `--mcx-daily` points the baselines at a
different price history (default `build/mcx/mcx_daily.parquet`). `--episode-scope asset|calendar`
picks the episode unit (default `asset`); `--n-boot` and `--seed` control the bootstrap.
