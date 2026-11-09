"""Persistent experiment attempts and distinct specifications (Methodology).

The ID describes the economic/model specification, not the execution or sample.
Every attempt gets its own row before any fitting starts, including failures.
"""


from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from momenta.manifest import Manifest, provenance
from momenta.settings import REPO

_KINDS = {"diagnostic", "forecast", "strategy"}


_EXECUTION_KEYS = {
    "timestamp", "timestamps", "started_at", "finished_at", "run_id", "results", "result",
    "metrics", "metrics_json", "sample", "sample_window", "sample_start", "sample_end",
    "design_first_origin", "design_last_origin", "git_hash", "git_dirty", "config_hash",
    "data_manifest_hash",
}


_SECRET = re.compile(r"api[_-]?key|password|secret|token|credential", re.IGNORECASE)


_COLUMNS = [
    "timestamp", "finished_at", "run_id", "status", "kind", "experiment", "series", "sample",
    "strategy_id", "specification_json", "metrics_json", "error_type", "git_hash", "git_dirty",
    "config_hash", "data_manifest_hash", "data_vintage",
]


def _json_value(value: Any, *, specification: bool = False) -> Any:
    """Normalize JSON values; specification identity excludes execution metadata."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Experiment metadata keys must be strings")
            if _SECRET.search(key):
                raise ValueError("Credentials must not appear in experiment metadata")
            if specification and key in _EXECUTION_KEYS:
                continue
            result[key] = _json_value(item, specification=specification)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, specification=specification) for item in value]
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if specification:
            raise ValueError("Specification values must be finite")
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported experiment metadata type: {type(value).__name__}")


def _json(value: Any, *, specification: bool = False) -> str:
    return json.dumps(
        _json_value(value, specification=specification), sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def strategy_id(kind: str, specification: Mapping[str, Any]) -> str:
    """Hash canonical specification JSON, excluding run/sample metadata (Methodology)."""
    if kind not in _KINDS:
        raise ValueError(f"Experiment kind must be one of {sorted(_KINDS)}")
    canonical = _json(specification, specification=True)
    if canonical == "{}":
        raise ValueError("An experiment needs a nonempty specification")
    return f"{kind}:" + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class Trial:
    """One execution; mutate metrics during the run, including partial failures."""

    run_id: str
    strategy_id: str
    metrics: dict[str, Any] = field(default_factory=dict)


def read_trials(path: Path | None = None) -> list[dict[str, str]]:
    """Read all attempts, retaining failed and interrupted/running rows."""
    path = path or REPO / "docs/TRIALS.csv"
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _COLUMNS:
            raise ValueError("Unrecognized TRIALS.csv schema; refusing to overwrite history")
        return list(reader)


def _write_row(path: Path, lock: Path, row: dict[str, Any], *, update: bool) -> None:
    """Serialize small atomic CSV updates without losing simultaneous attempts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        rows: list[dict[str, Any]] = read_trials(path)
        matches = [i for i, item in enumerate(rows) if item["run_id"] == row["run_id"]]
        if update:
            if len(matches) != 1:
                raise ValueError("Trial row vanished or duplicated while running")
            rows[matches[0]] = row
        else:
            if matches:
                raise ValueError("Duplicate experiment run ID")
            rows.append(row)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", dir=path.parent, prefix=".trials-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@contextmanager
def trial(
    settings: dict[str, Any], *, experiment: str, kind: str, series: str,
    sample: str, specification: Mapping[str, Any],
) -> Iterator[Trial]:
    """Log before execution and finalize even on exceptions (Methodology).

Series enters the stable ID; experiment label, sample and execution metadata do
not. Exception *types* are recorded without messages, which may contain secrets.
"""
    if not all(str(value).strip() for value in (experiment, series, sample)):
        raise ValueError("Experiment, series and sample must be nonempty")
    canonical = {"series": series, "specification": dict(specification)}
    identity = strategy_id(kind, canonical)
    context = Trial(run_id=uuid4().hex, strategy_id=identity)
    root = Path(settings["data"]["data_root"])
    path = REPO / "docs/TRIALS.csv"
    lock = root / "tmp/trials.lock"
    meta = provenance(REPO, Path(settings["config_path"]), Manifest(root))
    row = {
        "timestamp": datetime.now(UTC).isoformat(), "finished_at": "",
        "run_id": context.run_id, "status": "running", "kind": kind,
        "experiment": experiment, "series": series, "sample": sample,
        "strategy_id": identity, "specification_json": _json(canonical, specification=True),
        "metrics_json": "{}", "error_type": "", **meta,
    }
    row["data_vintage"] = _json(meta["data_vintage"])
    _write_row(path, lock, row, update=False)
    try:
        yield context
    except BaseException as error:
        row["status"] = "failed"
        row["error_type"] = type(error).__name__
        raise
    else:
        row["status"] = "completed"
    finally:
        row["finished_at"] = datetime.now(UTC).isoformat()
        row["metrics_json"] = _json(context.metrics)
        _write_row(path, lock, row, update=True)
