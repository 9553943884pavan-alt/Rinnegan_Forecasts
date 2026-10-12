# Memo: <team name>, Commodity Open 2026

At most four pages. Replace every angle-bracket placeholder. The organizers read this before the
one-to-one call (RULES.md section 4a) and check it against the code.

Repository: <URL>; the commit is the one you give in the submission form. Challenges entered: <1, 2 or both>.

## 1. Problem framing

What you decided to signal and why, in your own words: what makes your model say a swing is
starting, what `target_pct` or `expected_pct` means to it, how it sets `invalidation_pct`, which
horizons it uses, when it says `no_signal`, and how it treats the four assets (pooled, per asset,
hierarchical).

## 2. Method

The model, its inputs and its outputs. Name the architecture, the loss, the regularization, the
seeds. For Task 2 say what `update` changes between sessions and what the frozen copy
therefore does differently: the improvement score is measured on the rows the two copies issued at
the same instant, so say where they diverge and why. A diagram or a table beats a paragraph.

## 3. Signal instants

Whether you signal at the pre-open, inside the session, or both, and what decides the instant. If
you signal more than once per asset per session, say what stops the extra rows being noise: every
one of them is in the denominator of correctness and direction (in the forward window, every one
with a horizon of at most 5 sessions).

## 4. Data and clocks

Which pack tables you read and at which as-of instants. Whether you opted in to unvintaged rows,
and where. Any external data: source, how the availability clock was established, the fetch code
and its recorded fetch times; if you called an external model, where the cache lives and how
`run.py` reads from it. Pretrained models: model id and training-data cutoff (before 2026-06-01).

## 5. Training cutoff

The `train_end` your `fit` used for the holdout (2026-06-01 03:00 UTC) and the forward window, and
anything fitted before it outside `run.py` (weights shipped in the repository, how they were
produced, when).

## 6. Ablation

What you tried and dropped, with numbers from the shipped scorer on the holdout: the full model,
the model without its main ingredient, and the two closest baselines. One table, with catch rate,
correctness and depth (Task 1) or direction, depth and improvement (Task 2).

## 7. Failure modes

Where the model is wrong in a predictable way: regimes, assets, horizons, days after a roll, data
releases. Which of your signals were invalid or unmatured and why. What the forward window could
do to it.

## 8. Compute

Hardware, wall time of `fit`, wall time of a full holdout run, memory. Whether the organizers need
a GPU to reproduce (they run on CPU unless told otherwise).

## 9. Reproduction notes

The exact commands that regenerate `predictions/` from the pack, and anything the organizers must
know to get byte-identical files (BLAS, thread counts, environment variables).
