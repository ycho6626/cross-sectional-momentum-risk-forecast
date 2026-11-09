"""The fixed stage order of the public pipeline (Methodology).

data → portfolios → features → design → freeze check → holdout → report, with
the robustness grid optional after the holdout. Each stage validates the
outputs of the stages before it and writes under the configured data root.
"""

from __future__ import annotations

from datetime import date
from typing import Any

STAGES = ("data", "portfolios", "features", "design", "freeze-check", "holdout", "report",
          "robustness")
DESIGN_STUDIES = ("baseline", "l2", "l3")


def build_features(settings: dict[str, Any], through: date | None = None) -> None:
    """Targets, L1, BDP decision, L2, copula clouds and HMM warmup through the design end."""
    from momenta.baseline_study import build_baseline
    from momenta.l2 import build_l2
    from momenta.l3_inputs import prepare_inputs

    bound = through or date.fromisoformat(settings["walkforward"]["design_last_origin"])
    build_baseline(settings, through=bound)
    build_l2(settings, through=bound)
    prepare_inputs(settings, through=bound)


def run_design(settings: dict[str, Any], study: str = "all") -> None:
    """Fit and evaluate the baseline, L2 and L3 design-period studies in order."""
    if study not in {"all", *DESIGN_STUDIES}:
        raise ValueError(f"Unknown design study: {study}")
    if study in {"all", "baseline"}:
        from momenta.baseline_study import run_study

        run_study(settings, "baseline")
    if study in {"all", "l2"}:
        from momenta.l2_study import run_l2_study

        run_l2_study(settings)
    if study in {"all", "l3"}:
        from momenta.l3_study import run_design as run_l3_design

        run_l3_design(settings)
