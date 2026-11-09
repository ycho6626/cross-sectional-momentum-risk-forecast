"""L2 design-period evidence: paired H3 tests and the redundancy audit (Methodology)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.manifest import sha256_file
from momenta.settings import REPO
from momenta.study_statistics import forecast_metrics, holm_adjust
from momenta.walkforward import assert_design_origins


def code_hash() -> str:
    """Pin the L2 code and its unchanged baseline numerical dependencies."""
    from momenta.baseline_report import code_hash as baseline_code_hash

    digest = hashlib.sha256(baseline_code_hash().encode())
    for name in ("l2.py", "l2_study.py", "l2_report.py", "settings.py"):
        path = REPO / "src/momenta" / name
        if path.exists():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def h3_table(
    forecasts: pl.DataFrame,
    placebo: pl.DataFrame,
    baseline_summary: dict[str, Any],
    settings: dict[str, Any],
    *,
    full: bool,
) -> list[dict[str, Any]]:
    """Paired L2|L1 loss tests and P(placebo R²≥observed R²)."""
    config = settings["evaluation"]
    assert_design_origins(forecasts["origin"].to_list(), settings)
    if forecasts.select(pl.struct("series", "origin").n_unique()).item() != forecasts.height:
        raise ValueError("Duplicate H3 forecast origins")
    if forecasts.filter(pl.col("refit") > pl.col("origin")).height:
        raise ValueError("H3 forecast fitted after its origin")
    records = []
    for (series,), group in forecasts.sort("origin").partition_by("series", as_dict=True).items():
        metrics = forecast_metrics(
            group,
            target="y1",
            reps=config["bootstrap_reps"],
            block_months=config["block_months"],
            seed=config["seed"],
            nw_lags=config["nw_lags_monthly"],
            confidence=config["confidence"],
        )
        row: dict[str, Any] = {
            "series": series,
            "target": "y1",
            "model": "L1+L2",
            "benchmark": "L1",
        }
        for name, value in metrics.items():
            if isinstance(value, dict):
                row.update({f"{name}_{key}": item for key, item in value.items()})
            else:
                row[name] = value
        if full:
            first = date.fromisoformat(baseline_summary["design_first_origin"])
            last = date.fromisoformat(baseline_summary["design_last_origin"])
            expected = (last.year - first.year) * 12 + last.month - first.month + 1
            if (
                row["n"] != expected
                or row["first_origin"] != str(first)
                or row["last_origin"] != str(last)
            ):
                raise ValueError(
                    f"H3 must preserve the complete pinned month-end sample for {series}"
                )
        identical = group["prediction"].equals(group["base_prediction"])
        row["identical_forecasts"] = identical
        if identical:
            row.update(
                cw_pvalue=1.0,
                dm_pvalue=1.0,
                cw_status="identical_nonrejection",
                dm_status="identical_nonrejection",
            )
        size = [
            item for item in baseline_summary["tables"]["size_check"] if item["series"] == series
        ]
        complete = {item["k"] for item in size} == set(config["size_check_blocks"]) and all(
            item["complete"] and item["bias_passed"] for item in size
        )
        inflated = any(item["cw_rejection_rate"] > config["size_check_max"] for item in size)
        row["cw_inference"] = (
            "descriptive_thin_scope"
            if not full
            else "size_check_unavailable"
            if not complete
            else "descriptive_size_failure"
            if inflated
            else "confirmatory_size_supported"
        )
        row["dm_inference"] = "descriptive"
        row["cw_holm_pvalue"] = None
        reference = placebo.filter(
            pl.col("series") == series,
            pl.col("k") == settings["l2"]["placebo_k"],
        )
        row["placebo_replications"] = reference.height
        row["placebo_position"] = None
        row["placebo_position_status"] = "not_comparable_thin_sample"
        if full:
            if (
                reference.height != config["size_check_reps"]
                or reference["rep"].n_unique() != reference.height
                or reference["error"].is_not_null().any()
                or not reference["r2"].is_finite().all()
            ):
                raise ValueError(
                    f"Incomplete k={settings['l2']['placebo_k']} placebos for {series}"
                )
            if row["r2_oos"] is None:
                raise ValueError("H3 baseline loss must be positive")
            row["placebo_position"] = float((reference["r2"] >= row["r2_oos"]).mean())
            row["placebo_position_status"] = "descriptive_same_series_null_position"
        records.append(row)
    family = {row["series"]: row for row in records}
    if full and set(family) == {"RAW", "RES"}:
        adjusted = holm_adjust([family[series]["cw_pvalue"] for series in ("RAW", "RES")])
        for series, pvalue in zip(("RAW", "RES"), adjusted, strict=True):
            family[series]["cw_holm_pvalue"] = pvalue
    return sorted(records, key=lambda row: row["series"])


def redundancy_audit(
    design: pl.DataFrame,
    l1_columns: Sequence[str],
    l2_columns: Sequence[str],
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, tuple[list[str], np.ndarray]]]:
    """OLS L2 on L1+intercept: R²=1−SSE/SST, VIF=1/(1−R²) (Methodology).

    This descriptive audit uses unclipped features. Standardizing regressors
    conditions the OLS system without changing its column space or fitted values.
    """
    rows, correlations = [], {}
    columns = [*l1_columns, *l2_columns]
    for (series,), frame in design.sort("date").partition_by("series", as_dict=True).items():
        raw = frame.select(columns).to_numpy().astype(float)
        complete = np.isfinite(raw).all(axis=1)
        values = raw[complete]
        if len(values) < 2:
            raise ValueError(f"Insufficient complete redundancy origins for {series}")
        dates = frame.filter(pl.Series(complete))["date"]
        scale = values.std(axis=0)
        standardized = np.divide(
            values - values.mean(axis=0), scale, out=np.zeros_like(values), where=scale > 0
        )
        correlation = standardized.T @ standardized / len(values)
        correlation[scale == 0, :] = np.nan
        correlation[:, scale == 0] = np.nan
        correlations[series] = (columns, correlation)
        x = np.column_stack([np.ones(len(values)), standardized[:, : len(l1_columns)]])
        rank = int(np.linalg.matrix_rank(x))
        for index, name in enumerate(l2_columns, start=len(l1_columns)):
            y = values[:, index]
            centered = y - y.mean()
            sst = float(centered @ centered)
            row = {
                "series": series,
                "feature": name,
                "origins": len(y),
                "first_origin": str(dates.min()),
                "last_origin": str(dates.max()),
                "regressor_rank": rank,
                "r2": None,
                "vif": None,
                "status": "ok",
            }
            if sst == 0:
                row["status"] = "constant_feature"
            elif len(y) <= rank:
                row["status"] = "insufficient_residual_degrees_of_freedom"
            else:
                residual = centered - x @ np.linalg.lstsq(x, centered, rcond=None)[0]
                fraction = float(residual @ residual) / sst
                row["r2"] = float(np.clip(1 - fraction, 0, 1))
                if fraction <= tolerance:
                    row["r2"] = 1.0
                    row["status"] = "infinite_vif"
                else:
                    row["vif"] = 1 / fraction
            rows.append(row)
    return rows, correlations


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )


def _table_file(name: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[Path]:
    path = REPO / "results/tables" / f"l2_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    sidecar = path.with_suffix(".provenance.json")
    _json(sidecar, metadata)
    return [path, sidecar]


def _heatmaps(
    correlations: dict[str, tuple[list[str], np.ndarray]],
    settings: dict[str, Any],
    metadata: dict[str, Any],
) -> list[Path]:
    cache = Path(settings["data"]["data_root"]) / "tmp/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for series, (columns, matrix) in correlations.items():
        figure, axes = plt.subplots(figsize=settings["l2"]["heatmap_size_inches"])
        plotted = axes.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap="RdBu_r")
        axes.set_xticks(range(len(columns)), columns, rotation=90, fontsize=8)
        axes.set_yticks(range(len(columns)), columns, fontsize=8)
        axes.set_title(f"{series}: L1 and L2 design month-end correlations")
        figure.colorbar(plotted, ax=axes, label="Pearson correlation", shrink=0.75)
        figure.tight_layout()
        path = REPO / "results/figures" / f"l2_redundancy_{series}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=settings["l2"]["heatmap_dpi"])
        plt.close(figure)
        sidecar = path.with_suffix(".provenance.json")
        _json(sidecar, {**metadata, "artifact_hash": sha256_file(path)})
        paths.extend([path, sidecar])
    return paths


def evaluate_l2(
    settings: dict[str, Any],
    output: Path,
    metadata: dict[str, Any],
    audit: dict[str, Any],
    baseline_summary: dict[str, Any],
    *,
    l1_columns: Sequence[str],
    l2_columns: Sequence[str],
) -> dict[str, Any]:
    """Evaluate saved H3 predictions and redundancy diagnostics, without refitting."""
    root = Path(settings["data"]["data_root"])
    forecasts = pl.read_parquet(output / "forecasts.parquet")
    design = pl.read_parquet(output / "redundancy_design.parquet")
    full = metadata["mode"] == "full"
    h3 = h3_table(
        forecasts,
        pl.read_parquet(root / "derived/baseline/size_check.parquet"),
        baseline_summary,
        settings,
        full=full,
    )
    redundancy, correlations = redundancy_audit(
        design,
        l1_columns,
        l2_columns,
        settings["l2"]["numerical_tolerance"],
    )
    tables = {
        "h3": h3,
        "redundancy": redundancy,
        "prefix_audit": audit["prefix"],
        "feature_availability": audit["feature_availability"],
        "feature_coverage": audit["feature_coverage"],
        "breadth": audit["breadth"],
    }
    inputs = dict(metadata["input_hashes"])
    for name in (
        "forecasts.parquet",
        "fits.parquet",
        "redundancy_design.parquet",
        "_metadata.json",
        "_audit.json",
        "trial_snapshot.json",
    ):
        path = output / name
        inputs[str(path)] = sha256_file(path)
    metadata = {**metadata, "input_hashes": inputs, "l2_code_hash": code_hash()}
    artifacts = []
    for name, rows in tables.items():
        artifacts.extend(_table_file(name, rows, metadata))
    artifacts.extend(_heatmaps(correlations, settings, metadata))
    summary = {
        "provenance": metadata,
        "tables": tables,
        "audit": audit,
        "first_origin": str(forecasts["origin"].min()),
        "last_origin": str(forecasts["origin"].max()),
        "forecast_rows": forecasts.height,
        "complete": full and {row["series"] for row in h3} == {"RAW", "RES"},
        "evaluation_settings": settings["evaluation"],
        "artifacts": {str(path): sha256_file(path) for path in artifacts},
    }
    _json(output / "summary.json", summary)
    return summary


def load_summary(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Validate current L2 inputs, code and outputs before they are reused."""
    path = Path(settings["data"]["data_root"]) / "derived/l2/summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    metadata = summary["provenance"]
    if metadata["config_hash"] != sha256_file(Path(settings["config_path"])):
        raise ValueError("Stale L2 configuration; rerun momenta design")
    if metadata["l2_code_hash"] != code_hash():
        raise ValueError("Stale L2 computation source; rerun momenta design")
    for source, digest in {**metadata["input_hashes"], **summary["artifacts"]}.items():
        if not Path(source).is_file() or sha256_file(Path(source)) != digest:
            raise ValueError(f"Stale L2 input or artifact: {source}")
    return summary
