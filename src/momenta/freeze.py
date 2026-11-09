"""Content hashes and deterministic TOML for the design freeze (Methodology)."""


from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from momenta.manifest import sha256_file
from momenta.settings import REPO

_ROOT_INPUTS = ("pyproject.toml", "uv.lock", "uv.toml")


_REGISTRIES = ("forecasts", "strategies", "tables")


def _in_scope(name: str) -> bool:
    path = Path(name)
    return name in _ROOT_INPUTS or (
        path.parts[0] in {"src", "config"}
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and name != "config/frozen.toml"
    )


def source_config_files(repo: Path | None = None) -> dict[str, str]:
    """Hash execution source/config, including untracked inputs, without self-reference."""
    repo = repo or REPO
    paths = [repo / name for name in _ROOT_INPUTS if (repo / name).is_file()]
    for directory in (repo / "src", repo / "config"):
        if directory.is_symlink():
            raise ValueError("Freeze source/config directories cannot be symbolic links")
        for path in directory.rglob("*"):
            relative = path.relative_to(repo).as_posix()
            if path.is_symlink() and _in_scope(relative):
                raise ValueError(f"Freeze source/config must be regular files: {relative}")
            if path.is_file():
                paths.append(path)
    result = {}
    for path in sorted(paths):
        relative = path.relative_to(repo).as_posix()
        if _in_scope(relative):
            if path.is_symlink():
                raise ValueError(f"Freeze source/config must be regular files: {relative}")
            result[relative] = sha256_file(path)
    if not result:
        raise ValueError("Cannot freeze an empty execution tree")
    return result


def _content_hash(files: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(files), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def source_config_hash(repo: Path | None = None) -> str:
    """Hash relative file names and SHA-256 values; exclude config/frozen.toml (Methodology)."""
    return _content_hash(source_config_files(repo))


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (str, Path)):
        return json.dumps(str(value), ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ValueError(f"Unsupported frozen TOML value: {type(value).__name__}")


def write_toml(path: Path, document: Mapping[str, Any]) -> None:
    """Write deterministic scalar/list/table TOML, including every selected setting."""
    lines: list[str] = []

    def table(keys: tuple[str, ...], values: Mapping[str, Any]) -> None:
        if keys:
            lines.append("[" + ".".join(json.dumps(key) for key in keys) + "]")
        for key, value in sorted(values.items()):
            if not isinstance(value, Mapping):
                lines.append(f"{json.dumps(key)} = {_toml_value(value)}")
        lines.append("")
        for key, value in sorted(values.items()):
            if isinstance(value, Mapping):
                table((*keys, key), value)

    table((), document)
    encoded = "\n".join(lines)
    # Parse before writing, so malformed values cannot leave a partial freeze.
    tomllib.loads(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def _settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in settings.items() if key != "config_path"}


def _validate_registry(frozen: Mapping[str, Any]) -> None:
    if set(frozen["best_models"]) != {"RAW", "RES"}:
        raise ValueError("Freeze requires a selected feature prefix for RAW and RES")
    registry = frozen["registry"]
    if set(registry) != set(_REGISTRIES):
        raise ValueError("Freeze requires explicit forecast, strategy and table registries")
    for key in _REGISTRIES:
        names = registry[key]
        if (
            not names
            or not isinstance(names, list)
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"Invalid frozen {key} registry")
    if any(model not in registry["forecasts"] for model in frozen["best_models"].values()):
        raise ValueError("Selected feature models must appear in the forecast registry")
    bounds = frozen["bounds"]
    required = {
        "design_first_origin",
        "design_last_origin",
        "holdout_first_origin",
        "holdout_last_origin",
        "common_end",
        "final_target_end",
    }
    if set(bounds) != required:
        raise ValueError("Freeze requires complete design, holdout and target bounds")
    days = {key: date.fromisoformat(value) for key, value in bounds.items()}
    if not (
        days["design_first_origin"]
        <= days["design_last_origin"]
        < days["holdout_first_origin"]
        <= days["holdout_last_origin"]
        < days["final_target_end"]
        <= days["common_end"]
    ):
        raise ValueError("Frozen origin/target bounds are inconsistent")
