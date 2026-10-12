# Submission template

Copy this directory to a new private repository together with the kit's `swing_dataset/` directory, replace `model.py` with your model, run `run.py` over the pack, commit the produced files, and submit the repository and commit hash through the submission link emailed to your team on 24 October. The organizers run the same `run.py` in a clean environment; what it produces must match what you committed.

## Files

| file | what it is |
|---|---|
| `run.py` | drives the loop: `fit` once, then at every session `update` with the outcomes of your own signals that settled since the previous one, and `predict` for every asset and horizon. Reads the pack only through `swing_dataset.pack.pit.PointInTime` and writes the read log next to the signals. Keep it as it is unless you know why you are changing it. |
| `model.py` | the model behind `fit(pack, train_end)`, `predict(as_of, asset, horizon)` and `update(matured_outcomes)`, one class per task, built by `build(challenge, horizons)`. Replace it. |
| `requirements.txt` | every dependency pinned to an exact version, the five packages the template names and the ones they pull in. The `swing-dataset` package installs from the copy of the kit's `swing_dataset/` in your repository, so the organizers' re-run installs exactly the code you ran. Pin any package you add with `==` too. |
| `memo_template.md` | the memo structure from RULES.md section 7. Rename it to `memo.md` and fill it in. |
| `predictions/` | the files `run.py` produced, committed. |
| `LICENSE` | Apache-2.0, as required for prize eligibility (MIT is also accepted). |

## Produce the files

Run these in your repository, where the kit's `swing_dataset/` directory sits next to `requirements.txt`: the last line of `requirements.txt` installs it. Inside the kit's own `template/` directory that line has nothing to install, and `pip install -r requirements.txt` stops on it; there, install from the kit's root, as the kit README does.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SWING_PACK_ROOT=/path/to/pack          # the directory with MANIFEST.json

python run.py --pack $SWING_PACK_ROOT --challenge 1 --period holdout
python run.py --pack $SWING_PACK_ROOT --challenge 2 --period holdout
python run.py --pack $SWING_PACK_ROOT --challenge 2 --period holdout --frozen
```

Each run writes three files under `predictions/`: `<stem>.parquet` (the signal file), `<stem>.readlog.jsonl` (every pack read, with the instant it served) and `<stem>.run.json` (arguments, pack identity, counts, timings). The stems are `c1_holdout`, `c2_holdout` and `c2_holdout_frozen`. Commit all of them. The template's own files, produced on pack v2.0.1, are already there as an example; your run replaces them.

The holdout starts on 2026-06-01 and runs to the close of 2026-10-23. Pack v2.0.1 reaches 2026-09-25, 85 MCX sessions, and the second data file adds the rest. `fit` receives `train_end = 2026-06-01 03:00 UTC` and must not use anything available at or after it. The run stops by itself at the first session the pack does not reach, so the same command over the second data file reproduces these rows and appends the new ones. With `--frozen`, `update` is never called: that file is the copy the improvement score is measured against, and it only matters for Task 2.

`--from` and `--to` narrow the sessions for a quick check (`--from 2026-07-01 --to 2026-07-14`); the organizers run without them.

`--period forward` is the run the organizers make over days the pack does not reach yet. On the pack as shipped it has no session to visit: it writes an empty file, `run.py` reports it INVALID and exits 1. To try that path now, give it days the pack holds, for example `python run.py --pack $SWING_PACK_ROOT --challenge 1 --period forward --from 2026-09-14 --to 2026-09-25`: `fit` receives the first of those sessions as its `train_end`, and the run visits the others.

## What a signal file holds

One row per decision. A signal row carries `asset, signal_utc, direction, horizon` and then, for Task 1, `target_pct, invalidation_pct, confidence, evidence`, or for Task 2, `expected_pct, p10_pct, p90_pct, invalidation_pct, evidence`. A `no_signal` row leaves `signal_utc` null and fills `decision_utc` and a one-line `reason`. Both kinds live in the same file; `FORMATS2.md` has the columns and the bounds, and `tools/validate_submission.py` checks a file against the scorer itself.

Two things are easy to get wrong. `evidence` is required: a signal without it is invalid, stays in the denominator and can never be correct. And the coverage floor of section 5 wants a decision record on at least 90 percent of each asset's sessions, where a decision record is a signal **or** a `no_signal` row. A model is never asked to signal when nothing is happening, only to say that it looked.

## Signalling, and when

Rules 2.6 lets a signal fire at any instant inside a published MCX session or at the 08:30 IST pre-open. **This template signals at the pre-open**, once per asset and session, because that is what the five shipped baselines do, it is the cadence the coverage floor is written for, and it keeps the worked example readable. Intra-session is allowed and the loop supports it two ways:

- return a list of rows from `predict`, each with its own `signal_utc` later in the same session; or
- define `instants(session_date, asset)` on your model, returning the instants inside that session you want to be called at, and the loop calls `predict` at each of them.

Either way the instant must lie between that session's pre-open and its close, and must be at or after the instant `predict` was called at, so the read log still proves that nothing was read past a row's own instant. If you want to decide at 14:00 on what the market did that morning, use `instants`: reading at the pre-open and dating the row at 14:00 gives you a row your reads could not have supported.

`instants` is given the date and the asset, not the session's bounds, and an instant outside them stops the whole run with a `LoopError`. The bounds are the 08:30 IST pre-open and the close, which is `evening_close_ist` in `mcx/mcx_sessions` (23:30 IST where the row has none). `swing_dataset.kit.loop.session_bounds(pack, asof, start, end)` returns them as `{session_date: (pre_open, close)}` with timezone-aware datetimes. The loop reads the calendar at `train_end`, so read it there once, in `fit`:

```python
from datetime import date

from swing_dataset.kit.loop import session_bounds


def fit(self, pack, train_end):
    self.bounds = session_bounds(pack, train_end, train_end.date(), date(2027, 12, 31))


def instants(self, session_date, asset):
    pre_open, close = self.bounds[session_date]
    return [pre_open, pre_open + (close - pre_open) / 2]  # the pre-open and the middle of the session
```

## What `predict` may return

| return | what the loop writes |
|---|---|
| a dict with `direction` and the challenge's fields | one signal row at the instant it was called with |
| a list of such dicts | one row each; any of them may carry its own `signal_utc` |
| `{"reason": "..."}` | a `no_signal` row with that reason |
| `None` or `{}` | a `no_signal` row with the loop's default reason |

One `no_signal` row is written per asset and session, and only when no horizon signalled there.

## What `update` receives

A frame of this run's own signals whose outcome has settled on rows available before the current session: the signal's own columns plus `p0` (the entry price), `mfe_pct` and `mae_pct` (how far the price ran in the signalled direction and against it), `correct` or `realized`, `deep` or `in_band`, and the window it was graded over. The grading is `swing_dataset.scoring2.outcomes`, the code the scorer runs, so what the model learns from is what the score will say. A row whose outcome is not settled yet comes back at a later session instead. `update` is never called in a frozen run.

## Check and score

```bash
python ../tools/validate_submission.py predictions/c1_holdout.parquet --pack $SWING_PACK_ROOT
python ../tools/score_holdout.py --pack $SWING_PACK_ROOT --predictions predictions/
```

These commands call the tools as `../tools/...`, which is where they are when you work inside the kit's `template/`. A repository built as the first paragraph says has no `tools/`: keep the kit unpacked and call its tools by path, for example `python /path/to/commodity-open-kit/tools/score_holdout.py ...`. A team that entered one task passes `--challenge 1` or `--challenge 2` to `score_holdout.py`; without it both tasks are scored and both files have to be there.

`score_holdout.py` runs the shipped scorer with the baselines and prints the three scores, the success rate, and every gate of the section 3 and 4 minimum for the holdout with its value and threshold, including Task 1's ten largest swings. Task 1's catch rate is counted against the realized-swing inventory; pass `--swings` if it is not inside your pack.

The template's own scores on pack v2.0.1 are in the table below, and your run of the template prints the same. They are far under the 70 percent minimum, which is the point: this is a worked example of the mechanics, not an entry.

| | scores on the holdout to 25 September | minimum not met |
|---|---|---|
| Task 1 | catch rate 12.3, correctness 52.3, depth 65.2, success rate 43.3 | success rate (70), correctness (60), ten large swings (6 caught, 7 needed) |
| Task 2 | direction 51.6, depth 69.7, improvement 22.6, success rate 47.9 | success rate (70), direction (60) |

## What the organizers do with it

1. `python tools/reproduce.py --repo <your URL> --commit <hash> --pack <pack> --workdir <scratch directory>`: clone, fresh venv, `pip install -r requirements.txt`, run the three commands above, compare the produced files with the committed ones byte for byte (or within 1e-9 per value), check both read logs against the instants they served, and check for a license, a memo and pinned dependencies. Direct parquet reads outside the client in your code are listed for review; a read log with a row past its instant disqualifies the entry. You can run the same check on your own commit, from the kit's `tools/`: `python /path/to/commodity-open-kit/tools/reproduce.py --repo . --commit $(git rev-parse HEAD) --pack $SWING_PACK_ROOT --workdir /tmp/repro` re-runs the whole holdout on what you committed, which takes a few minutes; add `--challenge 1` or `--challenge 2` if you entered only one task.
2. `tools/forward_run.py`: during the test week the same commit is run with `--period forward` over a pack that grows by one session a day. Rows computed from what was published before a run's pack was cut, 08:00 IST on its day, must not change from one day to the next. A row at a later instant, in-session or at the 08:30 pre-open, can: the next run's pack holds what was published in between. The file from the last run, made after the window, is the one that is scored. `fit` then receives the first forward session as its `train_end`, so anything available before it, including the holdout outcomes, is training data for the forward window.
3. The call (RULES.md section 4a): you walk the organizers through the code and the memo.

## Rules for `model.py`

- Read the pack only through the `pack` argument of `fit` and the client it gives you; never open a pack file by path. `predict` may read at `as_of` or earlier, never later.
- `fit` may load weights you shipped in the repository. Declare their training-data cutoff in the memo and in `TRAINING_DATA_CUTOFF`.
- No randomness without a fixed seed; set `SEED`. Two runs on the same machine must give identical files, and the organizers' machine must match within 1e-9.
- Say `no_signal` rather than leaving a session out. A missing row is not a decision record and does not count for the coverage floor.
- External data is allowed with declared clocks (RULES.md section 6). A live fetch during the forward window runs inside your code when the organizers run it; write the fetched bytes and the fetch time into the output directory next to the signals, and cache by content hash so a re-run reads the cache. A source that disappears is your risk.
