"""Rebuild the published results from pinned raw downloads in a fresh data root (Methodology).

The rebuild links the pinned downloads read-only, then runs every stage in the
public order: data, momentum books, features, design-period fits, freeze check,
holdout, strategy accounts and report. It compares the portfolio, design and
holdout artifacts with ``docs/reproduction_hashes.json`` and the regenerated
tables and figures with ``results/manifest.json``. Downloads are never repeated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO, load_settings

REFERENCE_PATH = REPO / "docs/reproduction_hashes.json"
EVIDENCE_PATH = REPO / "results/manifest.json"
# Wall-clock timings recorded per robustness variant cannot repeat.
TIMING_FIELDS = {"results/robustness.csv": ("elapsed_seconds",)}
# Released tables that only the robustness grid regenerates.
ROBUSTNESS_RELEASES = ("results/robustness.csv", "results/cost_sensitivity.csv")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str,
                                  allow_nan=False) + "\n")
    partial.replace(path)


def artifact_map_digest(hashes: dict[str, str]) -> str:
    """SHA-256 of the canonical sorted artifact-name-to-digest map."""
    return hashlib.sha256(
        json.dumps(dict(sorted(hashes.items())), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_artifacts(root: Path, expected: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Hash every registered relative artifact; report each match without stopping."""
    if not expected:
        raise ValueError("An empty reproduction registry cannot pass")
    result = {}
    for name, digest in sorted(expected.items()):
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f"Invalid reproduction artifact: {name}")
        actual = sha256_file(path) if path.is_file() else None
        result[name] = {"sha256": actual, "match": actual == digest}
    return result


def stage_raw_downloads(source: Path, destination: Path) -> None:
    """Link pinned raw downloads only; extraction, year staging and results are new.

    The manifest bytes are copied unchanged, including their recorded source
    paths, which are read but never written.
    """
    if source.resolve() == destination.resolve() or (destination / "raw").exists():
        raise ValueError("Raw reproduction requires a fresh isolated destination")
    manifest = Manifest(source)
    raw = destination / "raw"
    raw.mkdir(parents=True)
    for name, entry in manifest.entries.items():
        path = Path(entry["path"])
        if not path.resolve().is_relative_to((source / "raw").resolve()):
            raise ValueError(f"Pinned download escapes the source raw directory: {name}")
        if not manifest.verify(name):
            raise ValueError(f"Missing pinned reproduction download: {name}")
        relative = path.relative_to(source / "raw")
        if relative.parts[0] in {"extracted", "staged"}:
            raise ValueError("Reproduction must start from downloads, not generated raw staging")
        target = raw / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path.resolve())
    (raw / "manifest.json").write_bytes(manifest.path.read_bytes())


def _rows(path: Path, ignored: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return [{key: value for key, value in row.items() if key not in ignored}
                for row in csv.DictReader(stream)]


def verify_published(written: list[str], evidence: Path | None = None) -> dict[str, Any]:
    """Compare regenerated tables and figures with the released evidence digests."""
    manifest = json.loads((evidence or EVIDENCE_PATH).read_text())
    expected = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    result = {}
    for name in written:
        path = REPO / name
        digest = sha256_file(path)
        record = {"sha256": digest, "match": digest == expected.get(name)}
        if not record["match"] and name in TIMING_FIELDS:
            # Every other cell must be identical; only the recorded timings may differ.
            published = _published_bytes(name)
            ignored = TIMING_FIELDS[name]
            record["match_except_timing"] = published is not None and (
                _rows(path, ignored) == _rows(published, ignored)
            )
        result[name] = record
    return result


def _published_bytes(name: str) -> Path | None:
    """Return the committed version of a released file, for a column comparison."""
    import subprocess
    import tempfile

    content = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{name}"],
                             capture_output=True, check=False)
    if content.returncode:
        return None
    path = Path(tempfile.mkdtemp()) / Path(name).name
    path.write_bytes(content.stdout)
    return path


def reproduction_passed(receipt: dict[str, Any], *, robustness: bool) -> bool:
    """Every reference artifact, the holdout map and every released file must match.

    Without the robustness grid only its released tables may be missing; any
    other missing released file fails the reproduction.
    """
    allowed = set() if robustness else set(ROBUSTNESS_RELEASES)
    return (
        receipt["matches"] == receipt["expected"]
        and receipt["holdout_map_matches"]
        and receipt["published_matches"] == len(receipt["published"])
        and set(receipt["missing_published"]) <= allowed
    )


def run_reproduction(
    source: Path, output: Path, *, robustness: bool = False,
) -> dict[str, Any]:
    """Rebuild into output/data from the pinned downloads in source/raw; never download.

    The command runs from ``output`` with the published settings, whose relative
    data_root then names ``output/data``. After a failure, the remaining stages
    can be resumed from that directory with the ordinary commands.
    """
    import os
    from datetime import date

    from momenta.checks import check
    from momenta.curate import curate
    from momenta.data import stage_years
    from momenta.final_report import build_final_report
    from momenta.frozen_holdout import run_holdout
    from momenta.pipeline import build_features, run_design
    from momenta.portfolios import build

    started = time.perf_counter()
    reference = json.loads(REFERENCE_PATH.read_text())
    settings = load_settings()
    if Path(settings["data"]["data_root"]).is_absolute():
        raise ValueError("Reproduction needs the default relative data_root")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Reproduction requires a new or empty output directory")
    root = output / settings["data"]["data_root"]
    stage_raw_downloads(source.resolve(), root)
    if sha256_file(root / "raw/manifest.json") != reference["data_manifest_sha256"]:
        raise ValueError("The pinned downloads are a different data vintage")
    os.chdir(output)
    design_last = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    timings: dict[str, float] = {}
    receipt: dict[str, Any] = {"output": str(output), "source": str(source),
                               "data_manifest_sha256": reference["data_manifest_sha256"]}

    def timed(name: str, function: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        print(f"Reproduction: {name}", flush=True)
        result = function()
        timings[name] = time.perf_counter() - start
        _json(output / "progress.json", {"completed_phases": timings})
        return result

    timed("raw_year_staging", lambda: stage_years(settings, Manifest(root), "stocks"))
    timed("curation", lambda: curate(settings))
    if not timed("data_checks", lambda: check(settings)):
        raise ValueError("Rebuilt raw data did not pass the data checks")
    if not timed("portfolios", lambda: build(settings)):
        raise ValueError("Rebuilt portfolios did not pass reconciliation")
    receipt["portfolio_artifacts"] = verify_artifacts(root, reference["portfolio_artifacts"])
    timed("features", lambda: build_features(settings, design_last))
    timed("design", lambda: run_design(settings))
    receipt["design_artifacts"] = verify_artifacts(root, reference["design_artifacts"])
    holdout = timed("holdout", lambda: run_holdout(settings))
    stage = Path(holdout["execution"]["stage"])
    receipt["holdout_stage"] = str(stage)
    receipt["holdout_artifacts"] = verify_artifacts(stage, reference["holdout_artifacts"])
    hashes = holdout["execution"]["artifact_hashes"]
    receipt["holdout_artifact_map_sha256"] = artifact_map_digest(hashes)
    if robustness:
        from momenta.robustness import run_robustness

        timed("robustness", lambda: run_robustness(settings))
    report = timed("report", lambda: build_final_report(settings))
    receipt["published"] = verify_published(report["published"]["written"])
    receipt["missing_published"] = report["published"]["missing"]
    receipt["fixed_records"] = report["published"]["fixed_records"]
    groups = ("portfolio_artifacts", "design_artifacts", "holdout_artifacts")
    receipt["matches"] = {
        name: sum(item["match"] for item in receipt[name].values()) for name in groups
    }
    receipt["expected"] = {name: len(reference[name]) for name in groups}
    receipt["published_matches"] = sum(
        item["match"] or item.get("match_except_timing", False)
        for item in receipt["published"].values()
    )
    receipt["holdout_map_matches"] = (
        receipt["holdout_artifact_map_sha256"] == reference["holdout_artifact_map_sha256"]
    )
    receipt["passed"] = reproduction_passed(receipt, robustness=robustness)
    receipt.update(phase_seconds=timings, elapsed_seconds=time.perf_counter() - started,
                   robustness_included=robustness,
                   provenance=provenance(REPO, Path(settings["config_path"]), Manifest(root)))
    _json(output / "reproduction.json", receipt)
    return receipt
