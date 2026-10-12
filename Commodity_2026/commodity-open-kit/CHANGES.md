# Changes to the rules

## Kit 2.6.2, 6 October 2026

Every score, every score definition and the pack are unchanged. The rules now say when a swing is available (below). One check is corrected, the text about the forward window says what the organizers' tool does, and the tools, the package's installation and the documents are corrected where the audits of the whole flow found them wrong.

The read-log check reported "latest read returning rows ... is after the last instant in the file" for a model that defines `instants()` and signals at only some of them. The loop calls such a model at every instant and writes a row only when it signals, so a call that issued nothing left no row, and the check read its read as one past the last row. `tools/reproduce.py` printed NOT REPRODUCED for an honest model, and the organizers' re-run would have counted it. The check now takes the later of two instants as its bound: the last instant any row carries and the latest instant a `predict` call was made for, counted only on the Indian day of the file's last row. A read after that bound still fails, and so does a read past the instant it served, in `predict`, `update` or `fit`. A template and a pre-open-only model are judged as before. The code is in `swing_dataset/kit/repro.py`, which now calls a method of `swing_dataset/pack/pit.py` that only kit 2.6.1 and later have. A repository that vendors the package therefore replaces the whole `swing_dataset/` directory with the one in kit 2.6.2, never a single file of it, and runs `pip install --force-reinstall --no-deps ./swing_dataset`.

`tools/forward_run.py` is the organizers' tool, and its unchanged-rows check now treats a row issued after 08:00 IST on its run's day as provisional. A day's pack is cut at 08:00 IST, so a row at a later instant, in-session or at the 08:30 pre-open, is computed without what was published in between, and the next run can give another row for the same instant. Those rows are compared on the next run, and every other earlier row must still reappear unchanged. The file that is scored is the one from the last run, made after the window, in which every row is computed from everything published before its instant. The sentence in `template/README.md` that said rows already produced must not change now says this.

A swing is known only once the price has reversed the threshold from its extreme, a median of one session after the extreme and up to eight sessions. `RULES.md` and `FORMATS2.md` now say that its availability instant is the settlement of the session of the reversing bar, `confirm_available_utc`. Pack v2.0.1 still names `available_utc` as the clock of `swings/swings.parquet`, so its point-in-time client can return a leg before it is knowable: filter that file on `confirm_available_utc` yourself, and do not read `turn_count`, `extreme_count`, `has_turn` or `has_extreme` of `swings/swing_sessions.parquet` at prediction time. The second data file corrects both tables, adds the sessions to 23 October and removes those four columns; its README lists every other row of v2.0.1 that it changes. This replaces the sentence in the 2.6.1 entry below that said it keeps every row of v2.0.1, and the table in `RULES.md` section 5 no longer calls the second data file strictly additive.

Corrections you may notice. `pip install ./swing_dataset` no longer leaves `swing_dataset/pack/` out of the installed package when your repository's `.gitignore` lists `pack/`. `tools/reproduce.py` and `tools/forward_run.py` stop with one sentence when the installed `swing_dataset` is older than they are, saying what to replace. `tools/validate_submission.py` reports a CSV or JSON-lines file, which the scorer cannot open, and counts rows whose timestamps have no offset as signals. `tools/reproduce.py` no longer lists the template's own `./swing_dataset` line for review. `tools/score_holdout.py` says to pass `--challenge` when only one task's file is there, and the scorer no longer prints git's "not a git repository". The template's `requirements.txt` pins every package it installs, the kit's pins `argon2-cffi-bindings<26`, and the notebooks pass `nbformat.validate`. In the documents: section 3 says that in the forward window a signal counts when its horizon is at most 5 sessions, as section 5 already did; section 4a says that `tools/reproduce.py` only lists a direct read of a pack file for review and the organizers decide it from the code; `predict(as_of, asset, horizon)` is the signature, Task 2's minimum lists its fifth hard condition (the learning model beats the null on both periods, which the scorer already required), the Task 2 `evaluate` command in `FORMATS2.md` carries `--swings`, turns are located on daily bars where the pack holds no quotes, USD/INR ends on Fri 18 September, and `template/README.md` gives the bounds of an `instants()` instant, the template's scores on v2.0.1 and a complete `reproduce.py` command.

## Kit 2.6.1 and pack v2.0.1, 5 October 2026

Rules 2.6, the scorer, the thresholds and every score definition are unchanged. The pack has one correction and the kit three.

Pack v2.0.1 rebuilds the swing inventory, `swings/swings.parquet` and `swings/swing_sessions.parquet`. Pack v2.0 built it from quotes that ended on 15 September, so its sessions from 16 September had daily bars where the pack holds hourly quotes. The rebuild uses every quote in the pack and gives 2,107 legs instead of 2,105: silver 569 instead of 571, crude 597 instead of 593, gold and copper unchanged. 2,092 legs are identical. The 13 legs of v2.0 that were replaced and the 15 that replace them all have their turn and extreme between 14 and 24 September. The holdout to 25 September holds 187 legs instead of 185. No other table changed, and the second data file keeps every row of v2.0.1.

Kit 2.6.1 changes how the read log measures a read. `clock_max_utc` was the latest of every timestamp column among the rows returned. For `calendar/release_vintages` that includes the day a vintage published years ago was captured, so a default read at any as-of in the training period logged a clock after its as-of and `tools/reproduce.py` printed NOT REPRODUCED. It is now the latest clock that admitted a returned row: the fetch clock of an exact row, the availability instant of a published row, `available_utc` of an unvintaged row that an opt-in returned. `check_read_log` takes the pack and repeats a logged read whose clock fails the test, so a log written by kit 2.6 passes when the read was sound and a read that looked past its as-of still fails. The code is in `swing_dataset/pack/pit.py`, `swing_dataset/kit/repro.py`, `tools/reproduce.py` and `tools/forward_run.py`; a repository that vendors the package needs those four files. The template's predictions and the notebooks' outputs were regenerated on v2.0.1.

`tools/reproduce.py` ran pip from the directory it was started in, so the `./swing_dataset` line of a repository's `requirements.txt` installed the kit's own copy when the tool ran from the kit root and failed to install from anywhere else. It now installs from the cloned repository, which is what the organizers' re-run does, so a self-check tests the `swing_dataset/` your repository holds. A relative `--workdir` also works now.

`template/.gitignore` listed `pack/`, which also matched `swing_dataset/swing_dataset/pack/` once the template and the package sit in one repository, so `git add -A` left the point-in-time client out of the commit. It now lists `/pack/`. A repository made from the 2.6 template needs that line changed and the directory added: `git add swing_dataset/swing_dataset/pack`.

## Rules 2.6, 24 September 2026

Rules 2.6 follows the organizers' write-up of 23 September. The scores of 2.5.2 stand. What changed
is the name, who may enter, Task 1's minimum, Task 2's horizons, how teams submit and the calendar.

| topic | 2.5.2 | 2.6 |
|---|---|---|
| name | Swing Open, Season 1; Challenge 1 and Challenge 2 | Commodity Open 2026; Task 1 and Task 2. The tools keep `--challenge 1` and `--challenge 2` |
| who may enter | students and researchers at listed institutions, institutional address required | students and researchers anywhere; an institutional email ID is asked for, and an entry without one may be asked for it or cancelled |
| Task 1 minimum | success rate at or above 70 percent | the same, plus three checks that cannot be averaged away: at least seven of the ten largest swings of the holdout caught, correctness at or above 60 percent, depth at or above 60 percent |
| Task 2 horizons | 5 or 10 sessions | any whole number from 1 to 20 |
| holdout | 1 June to the pack's last session, then the extension | 1 June to the close of Fri 23 Oct: the pack to 25 Sep, the second data file after it |
| the extension | pack v1.2, two to three days before the close | the second data file, 25 Sep to 23 Oct, released after the close of 23 Oct |
| forward window | the week after the close, horizon 5 only | the sessions of 27 Oct to 2 Nov, horizons of at most 5 |
| submissions | public repository, URL and commit emailed | private repository through the submission link, 24 to 26 Oct, 5 pm IST, with the numeric results |
| questions | GitHub Discussions or email | email only, answered privately; a clarification goes to every team |
| the data | CC BY 4.0 | for registered teams in this challenge only; no sharing or redistribution until the verdicts |
| payment | by a method agreed with the team | in US dollars, by a method agreed with the team, never in cryptocurrency |
| the meeting | one-to-one meeting | one-to-one call, same content |
| calendar | start 22 Sep, close about 23 Oct, verdicts by 20 Nov | registration to 7 Oct, everything from 1 Oct as teams register, submissions 24 to 26 Oct, calls and verdicts in mid November |
| Task 1 catch | a correct signal with at least a whole threshold of the move still ahead | at least half a threshold still ahead |
| Task 1 correctness | correct signals over every signal issued | correct signals over decided signals; a signal whose horizon ends with neither its threshold nor its invalidation touched is left out and printed |
| Task 1 depth | target at least half of the furthest point over the whole horizon | target at least half of the move to the swing's end, the furthest point before the price turned back by a full threshold; a never-reached target still fails |
| Task 1 learning | not stated | a Task 1 model may refit at any instant on anything published before it, as the template's model does |
| timestamps | not stated | pack timestamps are microseconds; section 6 says how to compare them |

The three Task 1 checks come from the write-up: a model has to catch the moves a person would notice,
call them the right way and judge their size, and each of those is now tested on its own. Under 2.5.2
a high catch rate could lift the average over 70 percent for a model whose targets were badly sized.

The three corrected definitions follow a full test entry built and scored before circulation. Under
2.5.2 it was marked down for calls the market never settled either way, for real catches made after
the first half-threshold of a move, and for targets judged against the next swing's extension rather
than the swing it called. None of the gates moved: 70 overall, 60 correctness, 60 depth and seven of
the ten large swings.

### What changed in this kit

- `RULES.md` is version 2.6, `README.md` describes Commodity Open 2026, pack v2.0 and private submissions, and the contact is `contact@commodity-open.xyz`.
- The scorer and the kit implement 2.6: a report carries `rules_version` 2.6, and the printed result lists every gate with its value and threshold and the ten largest swings with the signal that caught each.
- The kit ships the point-in-time client, both scorers, the kit helpers and the builder of the swing list. The organizers' data-build code is not part of it.
- `SCHEMA.md` and the pack's sidecars cite `RULES.md`, and the pack README states the terms of use.
- The template's predictions and the notebooks' saved outputs were regenerated on pack v2.0 with the 2.6 scorer.

## Rules 2.4 to 2.5.2

Rules 2.4 was the first draft, written before the organizers had run the swing definition over live
data. Rules 2.5 replaced it on 19 September after a review of the pack, of MCX production data since
24 June, and of a three-week swing analysis covering 28 August to 18 September. Rules 2.5.1 followed
the same day, closing seven holes found on a second read of 2.5, and 2.5.2 followed 2.5.1, closing a
Task 2 scoring hole that read did not find.

These versions were internal drafts and were never circulated to teams. They stay here because section 4
records the evidence behind the swing thresholds.

### 1. What changed in the rules

| topic | 2.4 | 2.5 |
|---|---|---|
| swing definition | 1-sigma barrier scaled by the horizon; fires on about two days in three | a move of at least gold 1.5, silver 2.0, copper 1.5, crude 3.0 percent; a minimum only, no maximum stated |
| when a signal may fire | once a day at 08:30 IST | any instant inside a session, or at the pre-open |
| Task 1 output | class probabilities up, down, none | direction, horizon, own target, own invalidation, confidence |
| Task 2 output | nine quantiles or 512 samples plus p_up | direction, expected move with p10 and p90, horizon, invalidation, evidence |
| Task 1 scores | log loss, Brier, ECE, swing capture, coverage; three stacked tiers | catch rate, correctness, depth; averaged into a success rate |
| Task 2 scores | CRPS, Brier, PIT, coverage, loop gate; four stacked tiers | direction against the prevailing-trend null, depth, improvement over the frozen copy; averaged |
| minimum to be paid | beat a baseline with a confidence interval | success rate at or above 70 percent on holdout and forward, plus for Task 2 the hard learning conditions |
| prizes | seven stacked tiers | Task 1: 400 and 200. Task 2: 700 and 300. Unclaimed money stays with the organizers |
| after the season | none | the organizers intend to offer a paid continuation to the strongest one or two teams |
| calendar | launch 18 Sep, close 2 Oct, forward month October, verdicts 20 Nov, second call December | start about 1 Oct, three weeks, extension pack two to three days before the close, one test week, verdicts within a month |
| the extension | none | pack v1.2 with the days collected since the cut, strictly additive, scored |
| the meeting | after the forward month | one of four confirmation steps, with a backtest walk-through added |
| rule 8 | none | nothing is paid for effort |

### 2. What 2.5.1 fixed in 2.5, the same day

| hole in 2.5 | fix in 2.5.1 |
|---|---|
| depth equalled correctness for a team that set every target at the threshold | depth now also requires the target to be at least half of the move that actually happened |
| improvement could carry a weak learning model over 70 | direction alone must be at least 60 percent on both periods |
| a deliberately weak frozen copy inflated improvement | the frozen copy must beat the prevailing-direction null on the holdout |
| the coverage floor forced a signal on 90 percent of sessions, against the intra-session design | the floor is now a decision record (signal or explicit no-signal with a reason) on 90 percent of sessions |
| one test week cannot mature 10-session signals | the forward window is scored on the 5-session horizon only; 10-session signals there are reported, not scored |
| byte-for-byte reproduction was impossible for a model calling an external LLM, which the rules allow | every external call is cached in the repository with request hash, response and time; re-run reads the cache; the cache is part of the read log |
| Task 1 signals carried no reason, although a swing with its reason is the point | `evidence` is a required field on Task 1 signals, stored for the meeting, never scored |

### 3. What 2.5.2 fixed in 2.5.1, the same day

The seven holes above were found by reading 2.5. This one was not: it was found by writing the
worst submission we could think of, scoring it with the shipped scorer, and looking at the number
that came back.

Section 4 scored Task 2's direction on `expected_pct` and `invalidation_pct`, both supplied by
the entrant, and bounded neither. Section 3 never worked that way: Task 1 scores correctness on
the fixed asset threshold, and separately requires `target_pct` to be at least the threshold and
`invalidation_pct` to be at most it. The asymmetry was the hole.

Measured on the holdout. Take the directions from the organizers' own `prevailing_direction`
baseline, which scores a success rate of 44.19 honestly, and change three numbers: `expected_pct`
0.01, `invalidation_pct` 99 and the band `[-99, +99]` on the learning copy, and 2.00 / 99 /
`[-99, +99]` on the frozen copy. Direction 92.88, depth 100.00, improvement 100.00, success rate
**97.63**, coverage 100 percent, all four hard conditions passing, minimum met. There is no model in
it, and the 2.5.1 fix requiring the frozen copy to beat the null does not catch it, because the
frozen copy is the same trick tuned to 62.48 against the null's 48.02.

| hole in 2.5.1 | fix in 2.5.2 |
|---|---|
| Task 2 direction was scored on the entrant's own `expected_pct` and `invalidation_pct`, neither bounded | direction is scored on the asset's threshold in both directions: the ruler section 3 already scores correctness on, and the one the prevailing-direction null was already measured by. `expected_pct` and the band are scored in depth; `invalidation_pct` is stored and reported, never scored |
| a band of `[-99, +99]` contains every move, so depth was free | `p90_pct - p10_pct` may be at most twelve times the asset's threshold. A wider band is an invalid row: it stays in the denominator and is never realized |

Twelve is measured rather than chosen. The three Task 2 baselines quote band widths with a
median of 4 thresholds, a p99 of 11 and a maximum of 22.26, the 2,089 realized legs in the pack run
1.95 thresholds at the median and 8.48 at the p99, and the cap costs six of 1,566 baseline rows.

A third fix was considered and rejected: `expected_pct` at least the threshold and `invalidation_pct`
at most it, the literal Task 1 parallel. It closes the same hole, but it invalidates 33 percent
of `range_break_vol` rows and 29 percent of `rv_band` rows, which legitimately expect a sub-threshold
move over five or ten sessions. The yardstick and the cap cost the baselines six rows between them.

If you built against 2.5.1, what changes for you: set the band to what you believe, not to something
that cannot be wrong, and do not tune `expected_pct` or `invalidation_pct` for the direction score,
because neither enters it any more.

### 4. Why the swing definition changed

The 1-sigma barrier of rules 2.4 fired on roughly two sessions in three. A label that common is not a
swing; predicting it well says little about whether a model can find a move worth trading. The
percent thresholds of 2.5 fire far less often. Measured over the full history in the pack, 2021 to
2026, they give one leg per seven to ten sessions per asset on daily bars, and on the hourly bars the
scored windows use, 54 to 70 legs across the four assets in a 30-session window.

Two observations from the three-week analysis shaped the rest. Almost every move with a clean cause
began between 17:45 and 20:00 IST, well inside the session, which is why a signal may now fire at any
instant rather than once at the pre-open. And the mechanism driving the four assets changed three
times in as many weeks, ending with every metal move running through oil and yields; that is what
Task 2's loop gate is there to test.
