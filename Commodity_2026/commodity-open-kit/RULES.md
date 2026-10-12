# Commodity Open 2026: task specification

Version 2.6, 24 September 2026. This is the rules document for participants and the build contract for the labels and scoring code. It replaces 2.5.2 and follows the organizers' write-up of 23 September: the challenge is now Commodity Open 2026, the two challenges are called Task 1 and Task 2 (`--challenge 1` and `--challenge 2` in the tools), Task 1 gains three checks that cannot be averaged away and three of its definitions are corrected (catch room, correctness over decided signals, depth against the swing's end), Task 1 models may refit as the season runs, Task 2 takes horizons of 1 to 20 sessions, submissions are private, and the calendar is new. What changed and why is in `CHANGES.md`. Section numbers are kept from 2.4 so that references in the code and in `FORMATS.md` still point at the right place.

## 1. The task in one paragraph

Four MCX commodity futures: gold, silver, copper, crude oil. A model reads everything in the data pack that became available before the moment it speaks, and says one of two things: that a swing is starting now, which way and how far (Task 1), or that the price will move in a direction by a stated amount over a stated number of sessions, from a model that keeps learning from its own results (Task 2). A swing is a move of at least a fixed percentage of the price. Signals may be issued at any moment inside an MCX session or at the pre-open, not only once a day. Everything runs on the team's own machine from the data pack and must be reproducible from the team's repository. Entries are scored by the shipped scorer on an announced holdout, re-run by the organizers, then run on sessions no team has seen.

## 2. Instruments, the chain and the swing threshold

- Assets `gold`, `silver`, `copper`, `crude`; venue MCX; dated futures only. All four assets are required in every submission. The pack's `mcx/mcx_daily.parquet` holds every contract-day with settlement, OHLC, volume, and open interest from 2021-01-01; `pack/quotes/mcx_quotes.parquet` holds the production feed's intraday quotes with five-level depth, open interest and volume from 2026-06-22.
- Designated contract, roll plan and roll-normalized wealth chain: the designation is the dated contract with the highest volume over the two most recent completed sessions, excluding contracts in delivery or expiring within the horizon; the chain `W` removes the calendar spread at a roll. The pack ships designations, chain and labels; participants do not recompute them.
- **Swing threshold, per asset, in percent of the price at the turn:** gold 1.5, silver 2.0, copper 1.5, crude 3.0. A swing is a move of at least the threshold in one direction before a move of at least the threshold in the other. The threshold is a minimum only. How far a swing runs is for the model to judge; the organizers never state a maximum.
- Horizons, in completed MCX sessions: Task 1 `h ∈ {5, 10}`; Task 2 any whole number from 1 to 20. A signal's horizon is the window in which its move must happen; for Task 2 it is the signal's stated duration.
- Session numbering, the volatility scale `σ_c` and the `k = 1` convention belong to the labels in `labels/`, documented in their sidecars. Those 1-sigma barrier labels ship for research only and are not the scored task.

## 3. Task 1: spot the swing (USD 600)

**What the model outputs.** At any instant `t` inside a published MCX session, or at the 08:30 IST pre-open, a signal row: `asset, signal_utc, direction (up|down), horizon (5|10), target_pct, invalidation_pct, confidence, evidence`. `target_pct` is the model's own expected move from the price at `signal_utc`, and must be at least the asset's threshold. `invalidation_pct` is how far against the signal the price may go before the model considers itself wrong; it must be at most the threshold. `evidence` is a short text or a list of pack row identifiers giving the reason for the signal; it is stored for the call of section 4a and never scored, and a signal without it is not valid. Any number of signals per asset per session is allowed; on the holdout every one is in the denominator, and in the forward window every one with a horizon of at most 5 sessions (section 5).

**Realized swings.** The labels lane lists, per asset, every swing in the scored window by the threshold rule of section 2, with its turn instant and price, extreme instant and price, and direction. Turns are located on hourly highs and lows of the designated contract where the pack holds its quotes, from 22 June 2026, and on the session's daily high and low where it does not, which is before that date and on the few later sessions where the feed was not yet polling the designated contract (the `bar` column of `swings/swings.parquet` says which each leg used); a swing is known only once the price has reversed the threshold from its extreme, which can happen sessions after the extreme, so a swing's availability instant is the settlement publication of the session in which that reversing bar printed (`confirm_available_utc`), not the session of its extreme. The sidecar of `swings/swings.parquet` in pack v2.0.1 still names `available_utc` as its clock, so until the second data file `PointInTime` returns a swing from the session of its extreme, before it is knowable: filter the file on `confirm_available_utc` yourself. The second data file names `confirm_available_utc`, and the client then does it.

**A signal is correct** when, from the price at `signal_utc`, the designated contract moves at least the threshold in the signalled direction within the horizon before it moves `invalidation_pct` against it.

**A swing is caught** when at least one correct signal in its direction was issued after its turn instant, at or before its extreme, and while at least half the threshold of the move was still ahead (from the signal price to the swing's extreme is at least half the threshold).

**A signal is decided** when, inside its horizon, the price touched either its threshold level or its invalidation level. A signal whose horizon ends with neither touched is undecided: the market did not move far enough to settle the call either way.

**The move to the swing's end** is measured from the signal price to the furthest point the price reached in the signalled direction before it turned back by a full threshold, or before the horizon ended, whichever came first. A later move in the same direction, after a turn of a full threshold, belongs to the next swing and does not count.

**Three scores**, each a percentage, each pooled across the four assets and both horizons:

| score | definition |
|---|---|
| catch rate | swings caught divided by realized swings |
| correctness | correct signals divided by decided signals. Undecided signals are left out and printed separately; invalid rows stay in as incorrect |
| depth | among correct signals, the share where the price reached `target_pct` before `invalidation_pct` **and** `target_pct` was at least half of the move to the swing's end. A target that is never reached fails, and so does a target less than half of the move; the model is asked to judge how deep, not only which way |

**Success rate** = the plain average of the three. The report prints all four numbers and the same three for the shipped baselines.

**The ten large swings.** The ten largest realized swings of the holdout across all four commodities, ranked by the move from turn to extreme divided by the asset's threshold. Ranking by the threshold multiple measures each commodity against its own normal range, so crude's larger percentage moves do not fill the list. The report names the ten and says, for each one, whether it was caught and by which signal.

**Also reported.** Beside the catch rate the organizers print the same rate restricted to the swings that ran at least twice the asset's threshold, meaning 3.0 percent on gold and copper, 4.0 on silver and 6.0 on crude. It is a diagnostic and decides nothing by itself; the call of section 4a goes through it.

**Minimum to be paid.** All four of these, and none of them can be averaged away:

1. Success rate at or above 70 percent on the holdout and on the forward window.
2. Catch: at least seven of the ten large swings caught.
3. Direction: correctness at or above 60 percent on the holdout and on the forward window.
4. Magnitude: depth at or above 60 percent on the holdout and on the forward window. Depth fails a size call in both directions. Under-calling fails: silver runs 7,000 points before turning and the target was 3,000, less than half the move. Over-calling fails: silver turns at 5,000 and the target was 7,000, which is never hit.

The coverage floor of section 5 must also be met and the confirmation of section 4a passed. Above the minimum, entries are ranked by success rate on the forward window, ties broken by success rate on the holdout.

| place | prize |
|---|---|
| first | USD 400 |
| second | USD 200 |

Baselines, shipped with code and refit at every cutoff on data before it: a range-break rule on the prior session's high and low; a stretch-reversal rule on the 20-session mean and realized volatility. They exist to show the mechanics and to give the report a floor; they clear no prize.

## 4. Task 2: build a self-improving learning loop (USD 1,000)

**What the model is.** A model that is fitted once, then keeps learning: `fit(pack, train_end)` once on data available before `train_end`; `predict(as_of, asset, horizon)`, once for each horizon, at any instant inside a session or at the pre-open; `update(matured_outcomes)` after each signal matures, with outcomes whose availability instant is before the next call. A second run with `update` disabled produces the frozen copy. Any architecture is allowed: sequence models, gradient boosting, Bayesian updating, an LLM reading the news rows at inference, ensembles. What matters is that the pack is read only through the point-in-time client, that pretrained weights declare their training-data cutoff, and that the learning is the model's own.

**What the model outputs.** A signal row: `asset, signal_utc, direction (up|down), horizon (1 to 20), expected_pct, p10_pct, p90_pct, invalidation_pct, evidence`. `expected_pct` is the expected move in the signalled direction from the price at `signal_utc`; `p10_pct` and `p90_pct` bound it; `invalidation_pct` is how far against the signal the model considers itself wrong, and since 2.5.2 scores direction on the asset's threshold on both sides it is recorded and reported but never scored; `evidence` is a short text or a list of pack row identifiers the signal rests on, stored for the call of section 4a and never scored.

**Three scores:**

| score | definition |
|---|---|
| direction | signals whose direction was realized, divided by signals issued. Realized means the price moved at least the **asset's threshold** in the signalled direction within the horizon before it moved the **threshold** against it. That is deliberately the same ruler section 3 scores correctness on, and the same test the prevailing-direction null is measured by, so the two tasks are comparable and neither side of the test is a number the entrant chooses. `expected_pct` and the band keep their meaning and are scored in depth. The report also prints the always-prevailing-direction null (the direction of the last five sessions' move, issued at the same instants and horizons); an entry whose direction score does not exceed the null on both periods is not eligible |
| depth | among directionally right signals, the share where the realized move landed inside `[p10_pct, p90_pct]`. The band may be at most twelve times the asset's threshold wide, because one wide enough to contain every move would score full marks without saying anything; a wider band is an invalid row, which stays in the denominator and is never realized |
| improvement | ten times the gain, in percentage points, of the learning model's direction score over its frozen copy's on the same rows, capped at 100 and floored at 0 |

**Success rate** = the plain average of the three.

**Minimum to be paid:** success rate at or above 70 percent on the holdout and on the forward window, plus five hard conditions that cannot be averaged away: the learning model's direction score exceeds the frozen copy's on both periods; its direction score in the second half of each period is at least its score in the first half; its direction score alone is at least 60 percent on both periods, so that the improvement score cannot carry a weak model over the line; the frozen copy's direction score exceeds the prevailing-direction null on the holdout, so that `fit` produces a working model and the improvement is not measured against a copy built to fail; and the learning model's own direction score exceeds the same null on both periods, which is the eligibility sentence of the direction row above. Coverage floor and confirmation as in Task 1. Ranking as in Task 1.

| place | prize |
|---|---|
| first | USD 700 |
| second | USD 300 |

Baselines, all loops, refit at every cutoff: the prevailing-direction null; a realized-volatility band with the trend sign; the range-break rule of Task 1 with a volatility-scaled target.

## 4a. How prizes are paid, and what comes after

No prize is paid on a score alone. Every prize is confirmed in four steps, in this order:

1. Reproduction: the organizers run the submitted commit in a clean environment and the produced files match the submitted ones byte for byte, or within 1e-9 per value. A repository that reads pack files outside the point-in-time client, or whose read log shows rows past a signal instant, is out. `tools/reproduce.py` checks the produced files and the read log; a direct read of a pack file in a team's own code is not in the log, so the tool only lists such reads for review and the organizers decide them from the code. A model that calls an external language model or any other external service must commit a cache of every call (a hash of the request, the full response, the time it was made) and its `run.py` must read from that cache on re-run; a live call during the organizers' run is a mismatch. Local models with fixed seeds need no cache. The cache is part of the read log: a cached response made after a signal's instant, used for that signal, is a leak.
2. Forward window: the same commit is run over the sessions of 27 October to 2 November, which no team has seen, and the minimum holds there too.
3. Backtest walk-through: the organizers replay the model over the holdout with the shipped scorer and compare every score with the memo.
4. A one-to-one call, online, about an hour: the team walks the organizers through the implementation and the memo, the data and clocks, the training cutoff, the reported diagnostics including Task 1's catch rate on large swings, and what was tried and dropped. A team that cannot explain its own submission, or whose explanation contradicts the code, receives nothing.

Prizes are paid in US dollars within one week of the call, by a method agreed with the team, and never in cryptocurrency. Prizes are stated gross and are paid subject to any tax the organizers are required to withhold; a winner gives a PAN before payment, and receives a statement of anything withheld. If only one entry clears the minimum in a task, it takes first place and the second-place money stays with the organizers. If none does, nothing is paid.

**After the season.** This is research work. Teams that qualify or deliver strong work may be selected for paid contribution and further work with the organizers. No amount is promised and entering creates no obligation on either side.

## 5. Periods and how scoring works

| period | use |
|---|---|
| training | 2021-01-01 to 2026-05-31: anything |
| holdout | 2026-06-01 to the close of Fri 23 Oct 2026: scored by the team with the shipped scorer and by the organizers on re-run; `fit` may use only data available before 2026-06-01 03:00 UTC. After that a Task 1 model may refit at any instant on anything published before that instant, read through the point-in-time client, as the template's own model does; a Task 2 model learns only through `update`, so that its frozen copy stays frozen. The pack covers the holdout to the close of Fri 25 Sep and the second data file covers the rest |
| second data file | released after the close of Fri 23 Oct: the sessions from 25 Sep to 23 Oct in the same tables as the pack, additive except for the corrections its README lists row by row. Teams re-run on it before submitting, and its swings count in the holdout score |
| forward window | the MCX sessions of Tue 27 Oct to Mon 2 Nov 2026, run by the organizers only |

The holdout is public history. The scorer, the read log, the re-run, the second data file and the forward window are the safeguards. Coverage floor: a decision record on at least 90 percent of the sessions in the scored window, per asset. A decision record is either a signal or an explicit `no_signal` row with a one-line reason; a model is never required to signal when nothing is happening, only to say so. The forward window scores only signals with a horizon of at most 5 sessions, because a longer one issued there has not matured by the calls: a 5-session signal issued on Mon 2 Nov resolves by Mon 9 Nov. Longer signals issued in the forward window are recorded and reported, not scored. Confidence intervals use episode blocks of `h` sessions per asset, 2000 seeded resamples; the report prints them for every score.

## 6. Data and clocks

- The dataset is the Commodity Open pack (`MANIFEST.json` lists every file with hash, rows and clock quality). Every row carries `clock_quality` (`exact`, `published`, `unvintaged`) and an availability column; `swing_dataset.pack.pit.PointInTime` returns only rows available strictly before the instant asked for and logs every read. Unvintaged rows are excluded unless the participant opts in and declares it.
- Timestamps in the pack's parquet files are microsecond precision (`datetime64[us, UTC]`). Compare them as timestamps or through `swing_dataset.scoring2.prices.ns_of`; `astype("int64")` gives microseconds, and comparing that with `pd.Timestamp.value`, which is in nanoseconds, silently puts every row after every instant.
- The pack ships the production quotes with depth, open interest and volume, every headline since 16 July with story clusters, hourly market state, MCX daily history from 2021, the release calendar with schedule, vintages and surprises from 2021, macro vintages, COT, Hyperliquid candles and funding, and USD/INR, all up to the close of Fri 25 Sep 2026, except USD/INR, which ends on Fri 18 Sep because the source publishes it with a lag (the table's sidecar says so). The second data file carries the same tables from 25 Sep to 23 Oct. Neither contains the organizers' own news scores or news index, because reading the news is part of the task.
- Calendar caveat, stated rather than hidden: in `calendar/release_surprises.parquet` almost every consensus figure was captured after the release, and is marked `unvintaged`. Consensus figures captured live before a release carry an `exact` clock; that capture began on 2026-09-15 and ran only while the organizers' capture was switched on. A participant who uses the surprise columns must say so in the memo. The open fixes are tracked by the organizers.
- External data is allowed with declared clocks: historical data with a stated publication or capture time, and live fetches during the forward window performed by the participant's own code with a recorded fetch time. Those fetches are the participant's risk.
- Pretrained models declare model id and training-data cutoff, which must precede 2026-06-01.

## 7. Submissions

- A repository containing: `run.py` that regenerates every signal file from the pack (`--challenge 1|2` for the task, `--period holdout|forward`, `--frozen` for the copy without `update`), pinned dependencies, fixed seeds, the signal files it produced, the point-in-time read log, the numeric results, and a memo of at most four pages: problem framing, method, data and clocks used, training cutoff, ablation, failure modes, compute used. `template/memo_template.md` gives the structure. The numeric results are the `holdout_scores.json` and `report_<challenge>_holdout.json` files that `tools/score_holdout.py` writes, and the README states the same numbers.
- Format of the signal files and of the report: `FORMATS2.md` (`FORMATS.md` is the superseded 2.4 format); `tools/validate_submission.py` checks a file before submission.
- The submission link is emailed to each team's registered contact address and is open from Sat 24 Oct to Mon 26 Oct 2026, 5 pm IST. Through it the team gives its ticket number, the task or tasks it entered, its repository and the commit hash. The repository is private, with read access for the organizers as the email explains, and the commit is what the organizers run.
- The organizers re-run and consider for prizes the entries whose reported holdout results clear the minimum. Every other entry receives its verdict on the numbers it reported.
- One submission per team per task; a team may enter both.

## 8. Calendar

Registration closes on 7 October, and every genuine team with a sensible description is accepted. Selected teams receive the kit, the pack and these rules from 1 October, as they register, and build until the submission window closes on 26 October: about three and a half weeks with the pack for a team that registers on 1 October and about two and a half for one that registers on 7 October, then a weekend to re-run on the second data file.

| milestone | when |
|---|---|
| registration closes | Wed 7 October 2026, 23:59 IST |
| confirmation, kit, pack with data to the close of Fri 25 Sep, rules 2.6 | from Thu 1 October 2026, as teams register |
| questions | any time, by email to the contact address, quoting the ticket number; answered within two days |
| second data file, 25 Sep to 23 Oct | Fri 23 October 2026, after the close |
| submission window | Sat 24 October to Mon 26 October 2026, 5 pm IST |
| organizer review: reproduction and backtest walk-through | 27 October to 2 November 2026 |
| forward window: every entry run on sessions no team has seen | Tue 27 October to Mon 2 November 2026 |
| forward-window signals resolve | by Mon 9 November 2026 |
| one-to-one calls with prize contenders, then verdicts | mid November 2026 |
| payment | within one week of each team's call |

The calls wait for 9 November because of the forward window. It is scored on horizons of at most five sessions, so a signal issued on its last day, Mon 2 Nov, resolves by Mon 9 Nov, and clearing the minimum on the forward window is one of the conditions for payment.

## 9. Rules

1. Research only. Nothing here is investment advice and no submission places orders.
2. Every input shows its availability clock. Publisher time, occurrence time and provider labels are not availability.
3. The submitted commit is final. Later commits are not run.
4. Code under Apache-2.0 or MIT for prize eligibility; participants keep copyright; the organizers may integrate accepted components with attribution. The repository stays private until the verdicts are sent.
5. One submission per team per task; a team may enter both tasks.
6. Sharing code, signals or the data with another team, collusion between teams, sharing forward-window outputs, or manual editing of signal files disqualifies the entry.
7. Verdicts are written and sent to every team, with the reproduction and forward reports; one round of appeal within seven days.
8. Nothing is paid for effort. Only a model that reproduces and clears the minimum is paid.
9. This is a competition of skill. Entries are ranked on measured forecasting accuracy under a published scorer, and no prize turns on chance. There is no entry fee and no stake of any kind.
10. Prizes are stated gross. Where the organizers are required to withhold tax on a prize, they withhold it and give the winner a statement; a winner gives a PAN before payment. Each participant is responsible for their own tax position.
11. Open to students and researchers (BTech, MTech, MSc, PhD and similar), alone or in teams of up to four. Registration asks for an institutional email ID; an entry registered with another address may be asked for one, or cancelled, at the organizers' judgement. Eligibility is confirmed again at the call of section 4a with a current student ID, enrolment letter or equivalent; a team that cannot show it receives nothing, whatever it scored.
12. The registration form collects the team name, the number of members, each member's full name, a contact person with their email address, and a short description of the team. That data is used to run this season and nothing else, is not shared with anyone outside the organizers, and is deleted within thirty days of the final verdict. Ask at the contact address to see or delete yours sooner.
13. The pack and the second data file are for registered teams to use in this challenge. Until the verdicts are sent they may not be shared outside the team or redistributed in any form.
14. Questions are answered privately by email. An answer that clarifies a rule is also sent to every team, without naming who asked.

## 10. Where things are

Everything arrives by email at the registered contact address: from 1 October, as you register, the kit (notebooks, scoring code, formats, submission template, these rules) and the pack as a shared Google Drive folder, on 23 October the second data file, and on 24 October the submission link. Organizer contact: contact@commodity-open.xyz. Quote the ticket number in every message. The decisions behind this version and the evidence for the thresholds are held by the organizers and summarised in `CHANGES.md`.
