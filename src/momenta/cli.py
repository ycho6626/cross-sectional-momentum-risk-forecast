"""Command-line entry points for the public package.

``momenta verify`` checks the released evidence and needs no data. The other
commands run the pipeline in its fixed order on licensed Sharadar inputs and
free French, JKP and FRED series: data, portfolios, features, design,
freeze-check, holdout, report and robustness. ``data pull`` reads
SHARADAR_API_KEY and FRED_API_KEY from .env; every command writes under the
data_root set in config/settings.toml. ``reproduce`` runs the whole order from
pinned downloads in a fresh root and compares the results with recorded digests.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROBUSTNESS = ("all", "decile", "universe_2000", "performance_zero", "performance_minus_one",
              "unexplained_minus_30", "no_cap_repairs", "daily", "costs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momenta", description="Momentum risk research: evidence checks and pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Verify released evidence and aggregate arithmetic")
    verify.add_argument(
        "--root", type=Path, help="Source checkout containing results/manifest.json",
    )
    data = commands.add_parser("data", help="Download, curate and check the data layer")
    steps = data.add_subparsers(dest="step", required=True)
    steps.add_parser("pull", help="Fetch or verify pinned full-history raw snapshots")
    curate = steps.add_parser(
        "curate", help="Build permanent-ID prices, returns, event audits and monthly universes",
    )
    curate.add_argument("--through", type=date.fromisoformat, help="Thin-path cutoff (YYYY-MM-DD)")
    steps.add_parser("check", help="Run the data-quality and acceptance gates")
    portfolios = commands.add_parser(
        "portfolios", help="Form momentum books and reconcile them with published factors",
    )
    portfolios.add_argument(
        "--through", type=date.fromisoformat, help="Thin-path cutoff (YYYY-MM-DD)",
    )
    features = commands.add_parser(
        "features", help="Build targets, L1, L2, copula and warmup inputs for the design period",
    )
    features.add_argument(
        "--through", type=date.fromisoformat, help="Last design origin (YYYY-MM-DD)",
    )
    design = commands.add_parser(
        "design", help="Fit and evaluate the baseline, L2 and L3 design-period studies",
    )
    design.add_argument(
        "--study", choices=("all", "baseline", "l2", "l3"), default="all",
        help="Run one study; each needs the ones before it",
    )
    commands.add_parser(
        "freeze-check", help="Check the frozen configuration, data manifest and settings",
    )
    holdout = commands.add_parser(
        "holdout", help="Run the frozen models and accounts on the holdout origins",
    )
    holdout.add_argument(
        "--mock", action="store_true", help="Use the mock boundary inside the design period",
    )
    commands.add_parser("report", help="Regenerate the released tables and figures")
    robustness = commands.add_parser(
        "robustness", help="Run the pre-specified sensitivities on the frozen holdout",
    )
    robustness.add_argument("variant", nargs="?", default="all", choices=ROBUSTNESS)
    reproduce = commands.add_parser(
        "reproduce", help="Rebuild everything from pinned downloads in a fresh data root",
    )
    reproduce.add_argument(
        "--source", type=Path, required=True, help="Data root whose raw/ holds pinned downloads",
    )
    reproduce.add_argument(
        "--output", type=Path, required=True, help="New directory for the rebuilt data root",
    )
    reproduce.add_argument(
        "--robustness", action="store_true", help="Also run the robustness grid",
    )
    return parser


def _run(arguments: argparse.Namespace) -> int:
    from momenta.settings import load_settings

    if arguments.command == "reproduce":
        from momenta.reproduce import run_reproduction

        receipt = run_reproduction(arguments.source, arguments.output,
                                   robustness=arguments.robustness)
        print(json.dumps({key: receipt[key] for key in (
            "matches", "expected", "holdout_map_matches", "published_matches",
            "missing_published", "passed", "elapsed_seconds")}, indent=2))
        return 0 if receipt["passed"] else 1
    settings = load_settings()
    if arguments.command == "data" and arguments.step == "pull":
        from momenta.data import pull

        pull(settings)
        return 0
    if arguments.command == "data" and arguments.step == "curate":
        from momenta.curate import curate

        curate(settings, through=arguments.through)
        return 0
    if arguments.command == "data":
        from momenta.checks import check

        return 0 if check(settings) else 1
    if arguments.command == "portfolios":
        from momenta.portfolios import build

        # A thin build is a smoke test; only a full build must pass the acceptance gates.
        return 0 if build(settings, through=arguments.through) or arguments.through else 1
    if arguments.command == "features":
        from momenta.pipeline import build_features

        build_features(settings, arguments.through)
        return 0
    if arguments.command == "design":
        from momenta.pipeline import run_design

        run_design(settings, arguments.study)
        return 0
    if arguments.command == "freeze-check":
        from momenta.frozen_holdout import check_freeze

        frozen = check_freeze(settings)
        print(f"Freeze check passed: best models {frozen['best_models']}")
        return 0
    if arguments.command == "holdout":
        from momenta.frozen_holdout import run_holdout

        result = run_holdout(settings, mock=arguments.mock)
        execution = result["execution"]
        print(f"Holdout stage {execution['stage']}: {len(execution['artifact_hashes'])} "
              f"artifacts; reproduced the first run: {execution['reproduced']}")
        return 0
    if arguments.command == "report":
        from momenta.final_report import build_final_report

        published = build_final_report(settings)["published"]
        print(f"Wrote {len(published['written'])} released files; "
              f"not yet available: {published['missing'] or 'none'}")
        return 0
    from momenta.robustness import run_robustness

    run_robustness(settings, arguments.variant)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "verify":
        from momenta.evidence import main as verify

        return verify(["verify", *(["--root", str(arguments.root)] if arguments.root else [])])
    try:
        return _run(arguments)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"momenta: {error}", file=sys.stderr)
        return 1
