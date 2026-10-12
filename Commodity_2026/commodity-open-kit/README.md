# Commodity Open 2026: participant kit

Commodity Open 2026 is a forecasting challenge on four MCX commodity futures: gold, silver, copper and crude oil. A model reads the data pack point in time and says either that a swing is starting now, which way and how far (Task 1), or how far the price will move and in which direction over a stated number of sessions, from a model that keeps learning from its own results (Task 2). Everything runs on your own machine. You submit a private repository; the organizers re-run it, run it on sessions no team has seen, and talk to you about it. Prizes total USD 1,600.

The rules are `RULES.md` (version 2.6). Where this page and `RULES.md` differ, `RULES.md` wins. Signal file formats, report fields and score definitions are in `FORMATS2.md`, the scorer in force and the one the prizes are decided by; `FORMATS.md` documents the superseded 2.4 scorer and is kept for the research labels that still ship. What changed and why is in `CHANGES.md`. The tools and the package call the two tasks `--challenge 1` and `--challenge 2`.

## The two tasks

A swing is a move of at least a fixed share of the price: gold 1.5 percent, silver 2 percent, copper 1.5 percent, crude 3 percent. That is the minimum. How far it runs is for the model to judge. Signals may be issued at any moment inside a session or at the 08:30 IST pre-open; a model that sees nothing says so with a `no_signal` row, and a decision record on 90 percent of each asset's sessions is what section 5 asks for. `no_signal` rows count for coverage and never for correctness.

**Task 1, spot the swing (USD 600).** Say that a swing is starting, in which direction, over 5 or 10 sessions, with your own target, your own invalidation level and your reason. Scored on three numbers: of the real swings, how many you caught; of your decided signals, how many were right; of your right signals, how often the price reached your target with the target at least half of the move to the swing's end. The three average into a success rate. To be paid, on the holdout and on the forward window: success rate at least 70, correctness at least 60, depth at least 60, the coverage floor, and on the holdout seven of its ten largest swings caught. None of these averages with another. A Task 1 model may refit at any instant on anything published before it. First place USD 400, second USD 200.

**Task 2, the learning loop (USD 1,000).** The same markets, from a model that is fitted once and then keeps learning after every outcome: `fit(pack, train_end)`, `predict(as_of, asset, horizon)`, `update(matured_outcomes)`. Output a direction, an expected move with a band, a horizon of 1 to 20 sessions and an invalidation level. Scored on direction, depth, and improvement over an identical copy with learning switched off. Minimum 70 percent, plus hard conditions: the learning model must beat its frozen copy, its second half must be at least as good as its first, its direction score alone must be at least 60 percent, the frozen copy must itself beat the trend-following null on the holdout, and the learning model must beat that null on both periods. First place USD 700, second USD 300.

Nothing is paid for effort. A prize needs a model that reproduces byte for byte on the organizers' machine, clears the minimum on both periods, and is explained by its team in a one-to-one call.

## Calendar

| milestone | when |
|---|---|
| kit, pack with data to the close of Fri 25 Sep, rules 2.6 | from Thu 1 Oct 2026, as teams register |
| questions | any time, by email, quoting your ticket number; answered within two days |
| registration closes | Wed 7 Oct 2026, 23:59 IST |
| second data file, 25 Sep to 23 Oct | Fri 23 Oct 2026, after the close |
| submission window | Sat 24 Oct to Mon 26 Oct 2026, 5 pm IST |
| review, and the forward window of 27 Oct to 2 Nov | 27 Oct to 2 Nov 2026 |
| one-to-one calls with prize contenders, then verdicts | mid November 2026 |

## Get the pack

The pack arrives in the Google Drive folder shared with your team's registered address, next to this kit, as three files: the pack tarball, `MANIFEST.json` and `SHA256SUMS`. It replaces the pack of 1 October: the swing inventory in `swings/` is rebuilt from the quotes to 25 September and no other table changes. The pack's `README.md` has the detail, and the `README.txt` in the folder's `update-from-v2.0` subfolder says how to bring a pack and a kit you already hold up to the pack and the kit in the Drive folder.

```bash
sha256sum -c SHA256SUMS
tar --zstd -xf commodity-open-pack-v2.0.1.tar.zst
cd v2.0.1 && sha256sum MANIFEST.json
```

`sha256sum -c` prints OK for every file `SHA256SUMS` lists. The last command prints the hash of the manifest, which is the `MANIFEST.json` line of `SHA256SUMS`; the manifest lists the hash of every table. `tar --zstd` runs a `zstd` program: if it stops with `zstd: Cannot exec: No such file or directory` (GNU tar) or `unable to run program "zstd -d -qq"` (macOS), install one with `brew install zstd` on macOS or `sudo apt install zstd unzip` on Ubuntu and WSL.

The pack is a directory with `MANIFEST.json`, `SCHEMA.md`, `README.md` and one parquet file per table with a `.meta.json` sidecar; unpacked it is about 520 MB over 32 tables. Tables: MCX daily settlements and sessions from 2021 (`mcx/`), designations, wealth chain, swing labels and return targets (`labels/`), the realized swing inventory entries are scored against (`swings/`), release schedules, vintages and surprises from 2021 (`calendar/`), macro vintages, positioning, Hyperliquid candles and funding, USD/INR (`macro/`), and the production feed's captured quotes with five-level depth, open interest and volume, every headline since 16 July with story clusters, receipts and hourly market state (`pack/`). `SCHEMA.md` documents every column.

Score yourself against `swings/swings.parquet` rather than against your own re-derivation of it: that file is the inventory the organizers score Task 1's catch rate on, and a rebuild that differs by a single leg gives a number that will not match the official one. The baselines are not in the pack; they are code in this kit and `notebooks/02_baseline_submission.ipynb` rebuilds them.

Every row carries `clock_quality` (`exact`, `published`, `unvintaged`) and an availability column. `swing_dataset.pack.pit.PointInTime` returns only rows available strictly before the instant you ask for and logs every read. Pack timestamps are microsecond precision: compare them as timestamps or through `swing_dataset.scoring2.prices.ns_of`, never `astype("int64")` against `pd.Timestamp.value`, which is in nanoseconds.

One exception holds until the second data file: pack v2.0.1 marks a swing leg available when the session of its extreme is published, but a leg is only known once the price has reversed, a median of one session later and up to eight. Keep only the rows of `swings/swings.parquet` whose `confirm_available_utc` is before your instant, and do not read `turn_count`, `extreme_count`, `has_turn` or `has_extreme` of `swings/swing_sessions.parquet` at prediction time. The second data file corrects both tables.

The second data file adds the sessions from 25 Sep to 23 Oct in the same tables, removes the four `swing_sessions` columns named above, and its README lists the other rows of v2.0.1 that it corrects. The organizers' own news scores and news index are not in the pack and will not be, because reading the news is part of the task. Read the calendar caveat in `RULES.md` section 6 before using the surprise columns.

## Install

Python 3.13. The commands below are for a Linux or macOS shell; on Windows use WSL, because `tools/reproduce.py` builds its environment at `.venv/bin/python`, a path native Windows does not have. The package declares `>=3.13,<3.14`: pip refuses 3.12 and 3.14, and uv refuses 3.12 but ignores the upper bound, so make the environment with `-p 3.13` and check `python --version` in it. The `swing-dataset` package in this kit holds the point-in-time client, the validator, the chain features, the baselines, the loop, and the scorer. From the kit's root:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SWING_PACK_ROOT=/path/to/v2.0.1
```

With uv, whose environments have no `pip` of their own: `uv venv -p 3.13 && source .venv/bin/activate && uv pip install -r requirements.txt`.

## Run the notebooks, produce signals, score

`notebooks/01_explore_pack.ipynb` opens the manifest, reads the pack through `PointInTime` at one instant, and shows where a signal may fire. `notebooks/02_baseline_submission.ipynb` runs the five shipped baselines over the holdout, writes their signal files, validates them and scores them. `template/` is a complete submission: `run.py` drives the loop through the point-in-time client, `model.py` holds a worked model for each task, `predictions/` holds the files the template produced. Copy it into a new private repository and replace `model.py`.

```bash
cd template
python run.py --pack $SWING_PACK_ROOT --challenge 1 --period holdout
python run.py --pack $SWING_PACK_ROOT --challenge 2 --period holdout
python run.py --pack $SWING_PACK_ROOT --challenge 2 --period holdout --frozen
python ../tools/validate_submission.py predictions/c1_holdout.parquet --pack $SWING_PACK_ROOT
python ../tools/score_holdout.py --pack $SWING_PACK_ROOT --predictions predictions/
```

These commands call the tools as `../tools/...` from `template/` inside the kit. A repository built the way "Submit" describes has no `tools/`: keep the kit unpacked beside it and call its tools by path, for example `python /path/to/commodity-open-kit/tools/score_holdout.py ...`, with the interpreter of the environment the repository installs into.

The scorer prints, for Task 1, catch rate, correctness, depth and success rate, the undecided signals it left out, the ten largest swings with whether each was caught, and every gate with its value, its threshold and whether it is met; for Task 2, direction, depth, improvement and success rate with the hard conditions. The baselines, the prevailing-direction null and the coverage floor print beside them. The organizers run the same code.

Task 1's catch rate is counted against the realized-swing inventory, `swings/swings.parquet`. `score_holdout.py` looks for it inside the pack; if your copy does not carry it, pass `--swings /path/to/swings.parquet`. Task 2's scores do not use it, but the scorer records the file among the report's inputs, so it has to exist for both tasks. Without `--challenge` the tool scores both tasks and needs both files; a team that entered one task passes `--challenge 1` or `--challenge 2`.

## Submit

The submission link is emailed to your team's registered contact address and is open from Sat 24 Oct to Mon 26 Oct 2026, 5 pm IST. Through it you give your ticket number, the task or tasks you entered, your repository and the commit hash; the email names the GitHub account to give read access to. That commit is what the organizers run.

The repository stays private until the verdicts are sent, under Apache-2.0 or MIT, and contains `run.py`, pinned dependencies, fixed seeds, the signal files it produced, the read logs, the numeric results (`holdout_scores.json` and `report_<challenge>_holdout.json` from `tools/score_holdout.py`, with the same numbers in your README), and a memo of at most four pages (`template/memo_template.md`). `tools/reproduce.py` is the organizers' re-run, and you can run it on your own commit (`template/README.md` has the command); it re-runs `run.py`, compares the produced signal files and read logs with the committed ones, and lists a missing licence, memo or pinned requirement and any direct read of pack files for review, while the numeric results, the numbers in your README and the memo's length and sections are not read by it and are checked by a person, so `REPRODUCED` does not mean everything above is in order.

## Rules in brief

1. Research only. Nothing here is investment advice and no submission places orders.
2. Every input shows its availability clock. Publisher time, occurrence time and provider labels are not availability.
3. The submitted commit is final. Later commits are not run.
4. Code under Apache-2.0 or MIT for prize eligibility; you keep copyright; the organizers may integrate accepted components with attribution.
5. One submission per team per task; a team may enter both.
6. Sharing code, signals or the data with another team, collusion, sharing forward outputs, or manual editing of signal files disqualifies the entry.
7. Verdicts are written and sent to every team; one round of appeal within seven days.
8. Nothing is paid for effort. Only a model that reproduces and clears the minimum is paid.
9. The pack and the second data file are for your team's use in this challenge. Until the verdicts are sent they may not be shared outside the team or redistributed in any form.

## Contact

contact@commodity-open.xyz. Quote your ticket number in every message. Questions are answered privately; an answer that clarifies a rule is also sent to every team, without naming who asked.

## Layout

```text
README.md                       this page
RULES.md                        the task specification, version 2.6
CHANGES.md                      what changed in each version and why
FORMATS.md                      signal file formats and score definitions, superseded 2.4 scorer
FORMATS2.md                     the same for the 2.6 scorer, the one in force
CITATION.cff                    how to cite the pack and the kit
requirements.txt                pinned dependencies plus the swing-dataset package, installed from ./swing_dataset
swing_dataset/                  the swing-dataset package
notebooks/                      explore the pack, run and score the shipped baselines
template/                       a complete submission: run.py, model.py, predictions/, memo template
tools/                          validate, score, reproduce, forward_run
```

In code comments and docstrings, `CHALLENGE-SPEC` is the organizers' working name for `RULES.md`. The copies of `FORMATS.md` and `FORMATS2.md` inside `swing_dataset/` keep that name.

## License

Apache-2.0 for the kit and the `swing-dataset` package. The pack and the second data file are licensed to registered teams for use in this challenge only, as rule 9 above and `RULES.md` rule 13 state. The pack contains data captured from the sources named in its `README.md` and `SCHEMA.md`; those sources keep their own terms.
