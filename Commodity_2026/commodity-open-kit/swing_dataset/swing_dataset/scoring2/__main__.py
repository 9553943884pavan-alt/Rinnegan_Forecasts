"""Commodity Open 2026 scoring CLI, rules 2.6. ``--challenge 1`` is Task 1, ``--challenge 2`` Task 2.

    python -m swing_dataset.scoring2 evaluate --challenge 1|2 --signals <parquet> \\
        [--no-signals <parquet>] [--frozen <parquet>] --swings <parquet> \\
        --period holdout|forward [--baselines] [--out report.json]

    python -m swing_dataset.scoring2 baselines --challenge 1|2 --period holdout|forward \\
        [--swings <parquet>] --out-dir <dir>

    python -m swing_dataset.scoring2 award --reports holdout.json forward.json [--out award.json]

The data flags (`--mcx-daily`, `--designations`, `--mcx-sessions`, `--quotes`) default to the
build tree and the pack; `--test-start` is required for the forward window, which has no fixed
start in the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .. import common
from ..scoring.chain import FeatureChains
from ..scoring.folds import DEFAULT_BOOTSTRAP
from . import (
    DESIGNATIONS_PATH,
    MCX_DAILY_PATH,
    MCX_SESSIONS_PATH,
    OUT_DIR,
    QUOTES_PATH,
    RULES_VERSION,
    SWINGS_PATH,
)
from . import baselines as B
from . import report as R
from . import score as S
from . import signals as SIG
from . import swings as SW
from .prices import PriceSeries


def _ts(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _prices(args: argparse.Namespace) -> PriceSeries:
    return PriceSeries.from_paths(
        Path(args.mcx_daily),
        Path(args.designations),
        Path(args.mcx_sessions),
        Path(args.quotes) if args.quotes else None,
    )


def _score(
    challenge: int,
    decisions: SIG.Decisions,
    frozen: SIG.Decisions | None,
    inventory,
    prices: PriceSeries,
    window: S.Window,
    n_boot: int,
    seed: int,
) -> tuple[dict, dict]:
    if challenge == 1:
        result = S.challenge1(decisions, inventory, prices, window, n_boot, seed)
    else:
        result = S.challenge2(decisions, frozen, prices, window, n_boot, seed)
    return result, S.coverage(prices, decisions, window)


def _baseline_block(
    args: argparse.Namespace,
    prices: PriceSeries,
    window: S.Window,
    inventory,
) -> dict:
    chains = FeatureChains.from_parquet(Path(args.mcx_daily))
    out = {}
    for baseline in B.baselines_for(args.challenge):
        frame = B.run(baseline, prices, chains, window)
        decisions = SIG.normalize(frame, args.challenge)
        result, cover = _score(
            args.challenge,
            decisions,
            decisions if args.challenge == 2 else None,
            inventory,
            prices,
            window,
            args.n_boot,
            args.seed,
        )
        out[baseline.name] = {
            "episode_horizon": result["episode_horizon"],
            "scores": result["scores"],
            "counts": result["counts"],
            "coverage": {
                "pooled": cover["pooled"],
                "met": cover["met"],
            },
        }
        if "diagnostics" in result:
            out[baseline.name]["diagnostics"] = result["diagnostics"]
        if args.challenge == 2:
            out[baseline.name]["hard_conditions"] = result["hard_conditions"]
            out[baseline.name]["note"] = (
                "the frozen copy of a baseline is the baseline itself, so its improvement score is "
                "zero by construction"
            )
    return out


def cmd_evaluate(args: argparse.Namespace) -> int:
    prices = _prices(args)
    window = S.resolve_window(args.period, prices, _ts(args.test_start), _ts(args.test_end))
    decisions = SIG.load(
        Path(args.signals), args.challenge, Path(args.no_signals) if args.no_signals else None
    )
    frozen = SIG.load(Path(args.frozen), args.challenge) if args.frozen else None
    inventory = None
    if args.challenge == 1:
        if not args.swings:
            raise SystemExit("Task 1 (--challenge 1) needs --swings, the realized-swing inventory")
        inventory = SW.load(Path(args.swings))
    result, cover = _score(
        args.challenge, decisions, frozen, inventory, prices, window, args.n_boot, args.seed
    )
    baselines = _baseline_block(args, prices, window, inventory) if args.baselines else None
    inputs = R.identities(
        {
            "signals": Path(args.signals),
            "no_signals": Path(args.no_signals) if args.no_signals else None,
            "frozen": Path(args.frozen) if args.frozen else None,
            "swings": Path(args.swings) if args.swings else None,
            "mcx_daily": Path(args.mcx_daily),
            "designations": Path(args.designations),
            "mcx_sessions": Path(args.mcx_sessions),
            "quotes": Path(args.quotes) if args.quotes and Path(args.quotes).exists() else None,
        }
    )
    report = R.assemble(
        R.strip_frames(result), cover, window, inputs, " ".join(sys.argv), baselines
    )
    text = R.write(report, Path(args.out) if args.out else None)
    if args.out:
        print(args.out)
    else:
        sys.stdout.write(text)
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    prices = _prices(args)
    window = S.resolve_window(args.period, prices, _ts(args.test_start), _ts(args.test_end))
    inventory = SW.load(Path(args.swings)) if args.swings else None
    if args.challenge == 1 and inventory is None:
        raise SystemExit("Task 1 (--challenge 1) baselines need --swings to score the catch rate")
    chains = FeatureChains.from_parquet(Path(args.mcx_daily))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        {
            "path": str(p),
            **{k: v for k, v in (R.identities({"x": p})["x"] or {}).items() if k != "path"},
        }
        for p in (Path(args.mcx_daily), Path(args.designations))
    ]
    written, blocks = [], {}
    for baseline in B.baselines_for(args.challenge):
        frame = B.run(baseline, prices, chains, window)
        path = out_dir / f"baseline_signals_{args.challenge}_{baseline.name}_{args.period}.parquet"
        R.write_signals(
            frame,
            path,
            path.stem,
            [
                (
                    f"{baseline.name} baseline, refit at every cutoff on rows published before "
                    "it; see FORMATS2.md"
                )
            ],
            sources,
        )
        written.append(str(path))
        decisions = SIG.normalize(frame, args.challenge)
        result, cover = _score(
            args.challenge,
            decisions,
            decisions if args.challenge == 2 else None,
            inventory,
            prices,
            window,
            args.n_boot,
            args.seed,
        )
        blocks[baseline.name] = {
            "episode_horizon": result["episode_horizon"],
            "scores": result["scores"],
            "counts": result["counts"],
            "coverage": cover,
            "signals": str(path),
        }
        if "diagnostics" in result:
            blocks[baseline.name]["diagnostics"] = result["diagnostics"]
        if args.challenge == 2:
            blocks[baseline.name]["hard_conditions"] = result["hard_conditions"]
    report = {
        "challenge": args.challenge,
        "rules_version": RULES_VERSION,
        "window": window.to_dict(args.challenge),
        "baselines": blocks,
        "signal_files": written,
        "inputs": R.identities(
            {
                "swings": Path(args.swings) if args.swings else None,
                "mcx_daily": Path(args.mcx_daily),
                "designations": Path(args.designations),
                "mcx_sessions": Path(args.mcx_sessions),
                "quotes": Path(args.quotes) if args.quotes and Path(args.quotes).exists() else None,
            }
        ),
        "build": {**common.git_identity(), "command": " ".join(sys.argv)},
        "definitions": R.definitions(args.challenge),
    }
    out = out_dir / f"baseline_report_{args.challenge}_{args.period}.json"
    R.write(report, out)
    print(out)
    return 0


def cmd_award(args: argparse.Namespace) -> int:
    reports = [json.loads(Path(p).read_text()) for p in args.reports]
    result = S.combine(reports)
    result["reports"] = [str(p) for p in args.reports]
    text = R.write(result, Path(args.out) if args.out else None)
    if args.out:
        print(args.out)
    else:
        sys.stdout.write(text)
    return 0


def _data_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mcx-daily", default=str(MCX_DAILY_PATH))
    parser.add_argument("--designations", default=str(DESIGNATIONS_PATH))
    parser.add_argument("--mcx-sessions", default=str(MCX_SESSIONS_PATH))
    parser.add_argument("--quotes", default=str(QUOTES_PATH))
    parser.add_argument("--test-start", help="required for --period forward")
    parser.add_argument("--test-end")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m swing_dataset.scoring2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="score one submission over one period")
    ev.add_argument("--challenge", type=int, choices=(1, 2), required=True)
    ev.add_argument("--signals", required=True)
    ev.add_argument("--no-signals", help="a separate file of no_signal rows, when not inline")
    ev.add_argument("--frozen", help="Task 2: the frozen copy's signals")
    ev.add_argument("--swings", default=str(SWINGS_PATH))
    ev.add_argument("--period", choices=("holdout", "forward"), required=True)
    ev.add_argument("--baselines", action="store_true", help="score the predeclared baselines too")
    ev.add_argument("--out")
    _data_flags(ev)
    ev.set_defaults(func=cmd_evaluate)

    bl = sub.add_parser("baselines", help="run the predeclared baselines and score them")
    bl.add_argument("--challenge", type=int, choices=(1, 2), required=True)
    bl.add_argument("--period", choices=("holdout", "forward"), required=True)
    bl.add_argument("--swings", default=str(SWINGS_PATH))
    bl.add_argument("--out-dir", default=str(OUT_DIR))
    _data_flags(bl)
    bl.set_defaults(func=cmd_baselines)

    aw = sub.add_parser("award", help="join a holdout report and a forward report")
    aw.add_argument("--reports", nargs="+", required=True)
    aw.add_argument("--out")
    aw.set_defaults(func=cmd_award)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
