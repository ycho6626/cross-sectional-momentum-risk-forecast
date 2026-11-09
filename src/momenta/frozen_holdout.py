"""Freeze check and the guarded holdout run (Methodology).

``config/frozen.toml`` records the design-period choices: the selected feature
prefixes, the forecast, strategy and table registries, the origin bounds, the
trading population and the complete study settings. Before any holdout-period
data is read, ``check_freeze`` requires that

1. the frozen file has the SHA-256 recorded for it in ``docs/code_provenance.json``;
2. the data manifest has the frozen ``data_manifest_hash``; and
3. the study settings equal the frozen settings, apart from ``EXEMPT_SETTINGS``.

The research history records when the design was frozen. No git tag is used.
"""

from __future__ import annotations

import json
import tomllib
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from momenta.experiments import trial
from momenta.freeze import _settings, _validate_registry
from momenta.holdout import (
    _artifact_hashes,
    _json,
    _record_reproduction,
    _stage_settings,
    origin_bounds,
)
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO

FROZEN_PATH = REPO / "config/frozen.toml"
PROVENANCE_PATH = REPO / "docs/code_provenance.json"
# Settings a public checkout cannot share with the frozen study: the location of
# the user's data. Every other frozen setting must match exactly.
EXEMPT_SETTINGS = (("data", "data_root"),)


def recorded_frozen_hash(path: Path | None = None) -> str:
    """Return the published SHA-256 of config/frozen.toml from the provenance record."""
    record = json.loads((path or PROVENANCE_PATH).read_text())
    matches = [item for item in record.get("configuration", [])
               if item.get("path") == "config/frozen.toml"]
    if len(matches) != 1:
        raise ValueError("The frozen configuration hash is not recorded")
    return matches[0]["exported_source_sha256"]


def comparable_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Settings without the configuration path or the documented exempt fields."""
    value = deepcopy(_settings(settings))
    for section, key in EXEMPT_SETTINGS:
        if isinstance(value.get(section), dict):
            value[section].pop(key, None)
    return value


def check_freeze(
    settings: dict[str, Any],
    *,
    frozen_path: Path | None = None,
    recorded_hash: str | None = None,
) -> dict[str, Any]:
    """Verify the frozen file hash, the data manifest and the settings; return the freeze.

    Reads only the frozen file, the settings file and the raw manifest.
    """
    path = frozen_path or FROZEN_PATH
    expected = recorded_hash or recorded_frozen_hash()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError("Frozen configuration differs from its recorded hash")
    frozen = tomllib.loads(path.read_text())
    _validate_registry(frozen)
    manifest = Path(settings["data"]["data_root"]) / "raw/manifest.json"
    if not manifest.is_file() or sha256_file(manifest) != frozen["freeze"]["data_manifest_hash"]:
        raise ValueError("Data manifest differs from the frozen design")
    config = Path(settings["config_path"])
    if _settings(settings) != tomllib.loads(config.read_text()):
        raise ValueError("In-memory settings differ from their configuration file")
    if comparable_settings(settings) != comparable_settings(frozen["settings"]):
        raise ValueError("Study settings differ from the frozen design")
    return frozen


def run_holdout(
    settings: dict[str, Any],
    *,
    mock: bool = False,
    frozen_path: Path | None = None,
) -> dict[str, Any]:
    """Guard before building, then run all frozen models/tables in an isolated stage.

    Mock changes only the evaluation bounds, which stay inside the design period.
    Its features, fits, strategies, evaluation and reproduction checks are shared
    with the production path. Normal command settings are never mutated.
    """
    root = Path(settings["data"]["data_root"]).resolve()
    frozen_path = frozen_path or FROZEN_PATH
    initial = provenance(REPO, Path(settings["config_path"]), Manifest(root))
    with trial(
        settings,
        experiment="mock-holdout" if mock else "holdout",
        kind="diagnostic",
        series="RAW/RES",
        sample="mock design boundary" if mock else "locked holdout",
        specification={"execution": "frozen_pipeline", "mock": mock},
    ) as attempt:
        # No feature/target builder, fitted-model loader or post-design reader is
        # imported or called until the freeze check passes.
        frozen = check_freeze(settings, frozen_path=frozen_path)
        from momenta.l3_report import load_design_summary

        design = load_design_summary(settings)
        if design is None or design["best_models"] != frozen["best_models"]:
            raise ValueError(
                "The design run must precede the holdout and select the frozen prefixes"
            )
        freeze_hash = sha256_file(frozen_path)
        if mock:
            first = date.fromisoformat(settings["holdout"]["mock_first_origin"])
            last = date.fromisoformat(settings["holdout"]["mock_last_origin"])
            design_last = date.fromisoformat(settings["walkforward"]["design_last_origin"])
            if first > last or last > design_last:
                raise ValueError("Mock holdout origins must stay inside the design period")
        else:
            first = date.fromisoformat(frozen["bounds"]["holdout_first_origin"])
            last = date.fromisoformat(frozen["bounds"]["holdout_last_origin"])
        calendar = pl.read_parquet(root / "curated/trading_calendar.parquet")["date"].to_list()
        bounds = origin_bounds(
            calendar,
            first_origin=first,
            last_origin=last,
            common_end=date.fromisoformat(frozen["bounds"]["common_end"]),
            horizon=settings["targets"]["horizon"],
            lag=settings["targets"]["lag"],
        )
        if date.fromisoformat(bounds["last_origin"]) != last:
            raise ValueError("Frozen last origin is not a complete trading month-end")
        if not mock and bounds["final_target_end"] != frozen["bounds"]["final_target_end"]:
            raise ValueError("Frozen target end disagrees with the pinned trading calendar")
        directory = root / "derived/final" / ("mock_holdout" if mock else "holdout")
        if mock:
            directory /= freeze_hash
        stage = directory / "runs" / attempt.run_id
        initial = {**initial, "freeze_hash": freeze_hash, "mock_holdout": mock}
        scoped = _stage_settings(settings, stage, first, last, initial, frozen_path)
        scoped["_design_root"] = str(root)
        from momenta.l3_study import execute_frozen, prepare_inputs

        prepare_inputs(scoped, through=last, rebuild=True)
        summary = execute_frozen(
            scoped,
            frozen=frozen,
            output=stage / "derived/final",
            first_origin=first,
            last_origin=last,
            mock=mock,
        )
        required = set(frozen["registry"]["tables"])
        if not required.issubset(set(summary["produced_tables"])):
            raise ValueError("Holdout did not produce every frozen table")
        hashes = _artifact_hashes(stage, summary["deterministic_artifacts"])
        identity = {"freeze_hash": freeze_hash, "mock": mock, "bounds": bounds}
        reproduction = _record_reproduction(directory, stage, identity=identity, hashes=hashes)
        receipt = {
            **reproduction,
            "stage": str(stage),
            "identity": identity,
            "provenance": initial,
            "produced_tables": summary["produced_tables"],
        }
        _json(stage / "execution.json", receipt)
        attempt.metrics.update(
            reproduced=reproduction["reproduced"],
            deterministic_artifacts=len(hashes),
            first_origin=bounds["first_origin"],
            last_origin=bounds["last_origin"],
            stage=str(stage),
        )
        return {**summary, "execution": receipt}
