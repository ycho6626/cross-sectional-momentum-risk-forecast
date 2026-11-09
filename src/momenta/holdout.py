"""Holdout bounds, isolated stages and rerun comparison (Methodology)."""


from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from momenta.freeze import write_toml
from momenta.manifest import sha256_file


def origin_bounds(
    calendar: Sequence[date],
    *,
    first_origin: date,
    common_end: date,
    horizon: int,
    lag: int,
    last_origin: date | None = None,
) -> dict[str, str]:
    """Find complete calendar-month origins with t+lag+horizon≤common_end (Methodology).

    This inspects dates only. A nominal month-end on a nontrading day maps to
    the last trading day of that calendar month, as in the forecast engine.
    """
    days = list(calendar)
    if days != sorted(set(days)) or horizon < 1 or lag < 0:
        raise ValueError("A sorted unique calendar and valid target window are required")
    first_month = (first_origin.year, first_origin.month)
    last = last_origin or common_end
    last_month = (last.year, last.month)
    eligible = []
    for index, day in enumerate(days[:-1]):
        month = day.year, day.month
        end = index + lag + horizon
        if (
            month != (days[index + 1].year, days[index + 1].month)
            and first_month <= month <= last_month
            and day <= last
            and end < len(days)
            and days[end] <= common_end
        ):
            eligible.append((day, days[end]))
    if not eligible:
        raise ValueError("No complete target windows within the requested holdout bounds")
    return {
        "first_origin": str(eligible[0][0]),
        "last_origin": str(eligible[-1][0]),
        "final_target_end": str(eligible[-1][1]),
        "common_end": str(common_end),
    }


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _stage_settings(
    settings: dict[str, Any],
    stage: Path,
    first: date,
    last: date,
    initial_provenance: dict[str, Any],
    frozen_path: Path,
) -> dict[str, Any]:
    """Link approved inputs and give all generated outputs an isolated root."""
    source = Path(settings["data"]["data_root"])
    stage.mkdir(parents=True, exist_ok=False)
    for relative in ("raw", "curated", 'derived/portfolios'):
        target = source / relative
        if not target.is_dir():
            raise ValueError(f"Missing approved holdout input directory: {relative}")
        link = stage / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target.resolve(), target_is_directory=True)
    for relative in ('derived/baseline', 'derived/l2', 'derived/final'):
        (stage / relative).mkdir(parents=True, exist_ok=True)
    scoped = deepcopy(settings)
    scoped.pop("config_path")
    scoped["data"]["data_root"] = str(stage)
    scoped["walkforward"]["design_first_origin"] = str(first)
    scoped["walkforward"]["design_last_origin"] = str(last)
    runtime = stage / "settings.toml"
    write_toml(runtime, scoped)
    scoped["config_path"] = str(runtime)
    scoped["_execution_provenance"] = initial_provenance
    scoped["_registered_design_first_origin"] = settings["walkforward"]["design_first_origin"]
    (stage / "frozen.toml").write_bytes(frozen_path.read_bytes())
    return scoped


def _artifact_hashes(stage: Path, names: Sequence[str]) -> dict[str, str]:
    if not names or isinstance(names, str) or len(set(names)) != len(names):
        raise ValueError("Holdout must return unique deterministic artifact paths")
    hashes = {}
    for name in sorted(names):
        path = stage / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.is_file()
            or not path.resolve().is_relative_to(stage.resolve())
        ):
            raise ValueError(f"Invalid deterministic holdout artifact: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def _record_reproduction(
    directory: Path,
    stage: Path,
    *,
    identity: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    """Keep the first completed numerical outputs immutable and compare every rerun."""
    path = directory / "completed.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Holdout execution identity differs from the first completed run")
        original = Path(previous["stage"])
        if (
            _artifact_hashes(original, list(previous["artifact_hashes"]))
            != previous["artifact_hashes"]
        ):
            raise ValueError("Original holdout output changed after its first completed run")
        if hashes != previous["artifact_hashes"]:
            raise ValueError("Frozen holdout rerun did not reproduce the first outputs exactly")
        return {"reproduced": True, "first_stage": str(original), "artifact_hashes": hashes}
    _json(path, {"identity": identity, "stage": str(stage), "artifact_hashes": hashes})
    return {"reproduced": False, "first_stage": str(stage), "artifact_hashes": hashes}
