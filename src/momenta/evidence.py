"""Verify the released aggregate evidence without accessing source datasets.

``momenta verify`` checks every artifact against results/manifest.json, then
checks unique result rows, required models, paired RAW/RES samples, MSE/R²
arithmetic, Holm adjustment and the primary table's headline consistency.
It verifies stored evidence; it does not rebuild portfolios, re-estimate
forecasts, recalculate Clark–West statistics or recompute bootstrap intervals.
The manifest detects changes relative to this release, not authenticity of an
independently modified manifest. Run from a source checkout, or pass --root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

FORECASTS = "results/forecast_results.csv"
STRATEGIES = "results/strategy_results.csv"
PRIMARY = "results/primary_hypothesis.csv"
SERIES = {"RAW", "RES"}
MODELS = {"L1-own", "L1", "L1+L2", "L1+L2+CM", "L1+L2+CM+OT", "best+HMMF2"}
RULES = {"Static", "BSC", "DM-transfer", "BDP-transfer", "L1", "L1+L2",
         "L1+L2+CM+OT", "best+HMMF2"}
FORECAST_SAMPLES = {
    "design": (120, "2005-12-30", "2015-11-30"),
    "holdout": (127, "2015-12-31", "2026-06-30"),
}
STRATEGY_SAMPLES = {
    "design": (120, "2006-01-31", "2015-12-31"),
    "holdout": (127, "2016-01-31", "2026-07-31"),
}


class EvidenceError(ValueError):
    """A release artifact or a stated result fails its integrity check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"Missing or invalid numeric column: {field}") from error
    _require(math.isfinite(value), f"Nonfinite value in column: {field}")
    return value


def _equal(actual: float, expected: float, label: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
             f"Inconsistent {label}")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        _require(bool(fields) and len(fields) == len(set(fields)),
                 f"Missing or duplicate CSV column names: {path.name}")
        rows = list(reader)
    _require(bool(rows), f"Empty evidence table: {path.name}")
    _require(all(None not in row and all(value is not None for value in row.values())
                 for row in rows), f"Malformed CSV row: {path.name}")
    keys = [tuple(row[field] for field in fields) for row in rows]
    _require(len(keys) == len(set(keys)), f"Duplicate result rows: {path.name}")
    return rows


def _index(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[tuple, dict]:
    try:
        keys = [tuple(row[field] for field in fields) for row in rows]
    except KeyError as error:
        raise EvidenceError(f"Missing result key: {error.args[0]}") from error
    _require(len(keys) == len(set(keys)), f"Duplicate result keys: {', '.join(fields)}")
    return dict(zip(keys, rows, strict=True))


def _sample(row: dict[str, str], expected: tuple[int, str, str], *, strategy=False) -> None:
    count, first, last = ("months", "first_month", "last_month") if strategy else (
        "n", "first_origin", "last_origin",
    )
    _require((_number(row, count), row.get(first), row.get(last)) == expected,
             "Result sample differs from the stated study period")


def _probabilities_and_intervals(row: dict[str, str]) -> None:
    for field, value in row.items():
        if value and (field.endswith("pvalue") or field == "dsr"):
            _require(0 <= _number(row, field) <= 1, f"Invalid probability: {field}")
        if field.endswith("_interval_lower"):
            upper = field.removesuffix("_lower") + "_upper"
            _require(bool(value) == bool(row.get(upper)), f"Incomplete interval: {field}")
            if value:
                _require(_number(row, field) <= _number(row, upper),
                         f"Reversed interval: {field}")


def _holm_pair(rows: list[dict[str, str]]) -> None:
    _require(len(rows) == 2 and {row["series"] for row in rows} == SERIES,
             "Holm family must contain RAW and RES exactly once")
    ordered = sorted(rows, key=lambda row: _number(row, "cw_pvalue"))
    first = min(1., 2 * _number(ordered[0], "cw_pvalue"))
    second = min(1., max(first, _number(ordered[1], "cw_pvalue")))
    for row, expected in zip(ordered, (first, second), strict=True):
        _equal(_number(row, "cw_holm_pvalue"), expected, "RAW/RES Holm adjustment")


def _verify_forecasts(rows: list[dict[str, str]]) -> dict[tuple, dict]:
    indexed = _index(rows, ("period", "series", "model"))
    expected = {(period, series, model) for period in FORECAST_SAMPLES
                for series in SERIES for model in MODELS}
    _require(set(indexed) == expected, "Forecast table omits or adds a series/model/period")
    for (period, series, model), row in indexed.items():
        _sample(row, FORECAST_SAMPLES[period])
        _probabilities_and_intervals(row)
        mse, baseline = _number(row, "mse"), _number(row, "baseline_mse")
        _require(mse >= 0 and baseline > 0, "Forecast losses must be nonnegative")
        benchmark = row.get("benchmark")
        if model == "L1-own":
            _require(not benchmark, "Own-risk benchmark cannot name another benchmark")
            _equal(mse, baseline, "own-risk baseline MSE")
            continue
        _require((period, series, benchmark) in indexed, "Unknown forecast benchmark")
        _equal(baseline, _number(indexed[period, series, benchmark], "mse"), "baseline MSE")
        _equal(_number(row, "mse_improvement"), baseline - mse, "MSE improvement")
        _equal(_number(row, "r2_oos"), 1 - mse / baseline, "out-of-sample R²")
    for period in FORECAST_SAMPLES:
        for model in MODELS:
            pair = [indexed[period, series, model] for series in sorted(SERIES)]
            if model not in {"L1-own", "L1+L2+CM"}:
                _holm_pair(pair)
    return indexed


def _verify_primary(rows: list[dict[str, str]], forecasts: dict[tuple, dict]) -> None:
    indexed = _index(rows, ("series",))
    _require(set(indexed) == {(series,) for series in SERIES},
             "Primary table must contain RAW and RES exactly once")
    _holm_pair(rows)
    for row in rows:
        _require((row.get("hypothesis"), row.get("model"), row.get("benchmark"))
                 == ("H3", "L1+L2", "L1"), "Unexpected primary comparison")
        _sample(row, FORECAST_SAMPLES["holdout"])
        _probabilities_and_intervals(row)
        for field in ("cw_n", "dm_n"):
            _equal(_number(row, field), 127, "primary inference sample count")
        headline = forecasts["holdout", row["series"], "L1+L2"]
        for field in ("mse", "baseline_mse", "mse_improvement", "r2_oos", "cw_pvalue",
                      "cw_holm_pvalue", "mse_improvement_interval_lower",
                      "mse_improvement_interval_upper"):
            _equal(_number(row, field), _number(headline, field), f"primary/headline {field}")


def _verify_strategies(rows: list[dict[str, str]]) -> None:
    indexed = _index(rows, ("period", "series", "strategy"))
    expected = {(period, series, rule) for period in STRATEGY_SAMPLES
                for series in SERIES for rule in RULES}
    _require(set(indexed) == expected, "Strategy table omits or adds a series/rule/period")
    for (period, _, _), row in indexed.items():
        _sample(row, STRATEGY_SAMPLES[period], strategy=True)
        _probabilities_and_intervals(row)
        volatility = _number(row, "annual_volatility")
        _require(volatility > 0, "Strategy volatility must be positive")
        _equal(_number(row, "sharpe"), _number(row, "annual_mean") / volatility,
               "ordinary strategy Sharpe ratio")
        _equal(_number(row, "cost_bp"), 10, "headline transaction cost")
        _require(_number(row, "annual_turnover") >= 0, "Turnover cannot be negative")
        _require(-1 <= _number(row, "max_drawdown") <= 0, "Invalid strategy drawdown")


def verify_release(root: Path) -> dict[str, Any]:
    """Check manifest bytes and stated aggregate relationships; perform no statistical refits."""
    root = root.resolve()
    try:
        manifest = json.loads((root / "results/manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise EvidenceError("Cannot read results/manifest.json") from error
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
             "Unsupported evidence manifest")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and bool(artifacts), "Empty artifact manifest")
    paths = set()
    tables = {}
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "Malformed artifact record")
        name, digest = artifact.get("path"), artifact.get("sha256")
        _require(isinstance(name, str) and bool(name), "Missing artifact path")
        relative = PurePosixPath(name)
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and str(relative) == name and "\\" not in name, "Invalid artifact path")
        _require(name not in paths, "Duplicate manifest artifact")
        paths.add(name)
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                 "Invalid artifact digest")
        target = root / name
        _require(target.resolve().is_relative_to(root) and target.is_file(),
                 f"Missing or invalid artifact: {name}")
        _require(_digest(target) == digest, f"Artifact changed: {name}")
        if target.suffix == ".csv":
            tables[name] = _rows(target)
    _require({FORECASTS, STRATEGIES, PRIMARY} <= tables.keys(),
             "Manifest omits a required headline evidence table")
    forecasts = _verify_forecasts(tables[FORECASTS])
    _verify_primary(tables[PRIMARY], forecasts)
    _verify_strategies(tables[STRATEGIES])
    return {"artifacts_verified": len(paths), "tables_verified": len(tables),
            "forecast_rows": len(forecasts), "strategy_rows": len(tables[STRATEGIES]),
            "primary_rows": len(tables[PRIMARY]), "statistical_refits": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Verify released evidence and aggregate arithmetic")
    verify.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Source checkout containing results/manifest.json")
    arguments = parser.parse_args(argv)
    try:
        summary = verify_release(arguments.root)
    except (EvidenceError, OSError, csv.Error) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified {summary['artifacts_verified']} artifacts and "
          f"{summary['tables_verified']} aggregate tables.")
    print("Checks cover file integrity and stated result arithmetic; "
          "forecast tests and bootstrap intervals were not recomputed.")
    return 0
