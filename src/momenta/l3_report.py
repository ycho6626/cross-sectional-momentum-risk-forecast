"""Paired L3 evidence, causal regime diagnostics and deterministic tables (Methodology).

The evaluation reads saved forecasts, fits, accounts and audits only; it never
refits a model or reselects a prefix. ``numeric_results.json`` omits execution
provenance, so a rerun on the same inputs reproduces it byte for byte.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.freeze import source_config_hash
from momenta.manifest import sha256_file
from momenta.settings import REPO
from momenta.study_statistics import (
    diebold_mariano,
    forecast_metrics,
    holm_adjust,
    interval,
    qlike,
    stationary_indices,
)

FEATURE_PREFIXES = ("L1", "L1+L2", "L1+L2+CM", "L1+L2+CM+OT")
BASELINE_SETS = ("L1-own", "L1_without_M", "L1", "L1m")


def paired_predictions(
    forecasts: pl.DataFrame, series: str, model: str, benchmark: str,
) -> pl.DataFrame:
    """Require identical ordered origins, targets and refits for a paired loss test."""
    selected = forecasts.filter(pl.col("series") == series, pl.col("target") == "y1")
    candidate = selected.filter(pl.col("feature_set") == model).sort("origin")
    base = selected.filter(pl.col("feature_set") == benchmark).sort("origin")
    if candidate.is_empty() or base.is_empty():
        raise ValueError(f"Missing paired predictions for {series}: {model} vs {benchmark}")
    if not candidate.select("origin", "actual", "refit").equals(
        base.select("origin", "actual", "refit")
    ):
        raise ValueError(f"Paired model support or targets differ: {series}, {model}")
    if candidate["origin"].n_unique() != candidate.height:
        raise ValueError("Duplicate paired model origins")
    return candidate.with_columns(base["prediction"].alias("base_prediction"))


def _flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{key}_{subkey}": subvalue for key, value in metrics.items()
            if isinstance(value, dict) for subkey, subvalue in value.items()} | {
                key: value for key, value in metrics.items() if not isinstance(value, dict)}


def _non_nested_metrics(frame: pl.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    """H5b uses unadjusted paired losses, stationary intervals and two-sided DM, never CW."""
    cfg = settings["evaluation"]
    y, model, base = (
        frame[name].to_numpy() for name in ("actual", "prediction", "base_prediction")
    )
    model_loss, base_loss = (y-model)**2, (y-base)**2
    difference = base_loss-model_loss
    draws = stationary_indices(frame["origin"].to_list(), cfg["bootstrap_reps"],
                               cfg["block_months"], cfg["seed"])
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_draws = 1-model_loss[draws].sum(1)/base_loss[draws].sum(1)
    return {"n": len(y), "first_origin": str(frame["origin"].min()),
            "last_origin": str(frame["origin"].max()), "mse": float(model_loss.mean()),
            "baseline_mse": float(base_loss.mean()),
            "mse_improvement": float(difference.mean()),
            "r2_oos": float(1-model_loss.sum()/base_loss.sum()) if base_loss.sum() > 0 else None,
            "mse_improvement_interval": interval(difference[draws].mean(1), cfg["confidence"]),
            "r2_oos_interval": interval(r2_draws, cfg["confidence"]),
            "dm": diebold_mariano(y, base, model, cfg["nw_lags_monthly"])}


def comparison_table(
    forecasts: pl.DataFrame, settings: dict[str, Any], best_models: dict[str, str],
    size_check: list[dict[str, Any]], *, mode: str,
) -> list[dict[str, Any]]:
    """H3/H4/H4Q/H5a nested tests; H5b non-nested persistence test."""
    cfg = settings["evaluation"]
    records = []
    for series in ("RAW", "RES"):
        available = set(forecasts.filter(pl.col("series") == series,
                                        pl.col("target") == "y1")["feature_set"])
        pairs = [("H3", None, "L1+L2", "L1"),
                 ("H4", None, "L1+L2+CM+OT", "L1+L2+CM"),
                 ("H4Q", None, "L1+L2+CM+Q+OT", "L1+L2+CM+Q")]
        for k in (settings["hmm"]["states"], settings["hmm"]["secondary_states"]):
            pairs.extend([("H5a", k, f"best+HMMF{k}", best_models.get(series, "")),
                          ("H5b", k, f"best+HMMF{k}", f"best+HMME{k}")])
        for hypothesis, k, model, benchmark in pairs:
            if model not in available:
                continue
            frame = paired_predictions(forecasts, series, model, benchmark)
            metrics = (_non_nested_metrics(frame, settings) if hypothesis == "H5b" else
                       forecast_metrics(frame, target="y1", reps=cfg["bootstrap_reps"],
                           block_months=cfg["block_months"], seed=cfg["seed"],
                           nw_lags=cfg["nw_lags_monthly"], confidence=cfg["confidence"]))
            row = {"hypothesis": hypothesis, "states": k, "series": series,
                   "model": model, "benchmark": benchmark, **_flatten(metrics),
                   "dm_inference": "descriptive", "identical_forecasts":
                   frame["prediction"].equals(frame["base_prediction"])}
            if row["identical_forecasts"]:
                row.update(dm_pvalue=1., dm_status="identical_nonrejection")
            if hypothesis != "H5b":
                own = [r for r in size_check if r["series"] == series]
                supported = ({r["k"] for r in own} == set(cfg["size_check_blocks"])
                    and all(r.get("complete") and r.get("bias_passed") for r in own))
                inflated = any(r.get("cw_descriptive", False) or (
                    r.get("cw_rejection_rate") is not None
                    and r["cw_rejection_rate"] > cfg["size_check_max"]) for r in own)
                bias_failed = any(r.get("bias_exceeded", False) for r in own)
                row["cw_inference"] = ("descriptive_thin_scope" if mode == "thin" else
                    "invalid_placebo_bias" if bias_failed else
                    "size_check_unavailable" if not supported else
                    "descriptive_size_failure" if inflated else "size_supported")
                row["cw_holm_pvalue"] = None
                if row["identical_forecasts"]:
                    row.update(cw_pvalue=1., cw_status="identical_nonrejection")
            records.append(row)
    for family in {(r["hypothesis"], r["states"]) for r in records if r["hypothesis"] != "H5b"}:
        members = sorted([r for r in records if (r["hypothesis"], r["states"]) == family],
                         key=lambda r: r["series"])
        if mode != "thin" and {r["series"] for r in members} == {"RAW", "RES"}:
            for row, adjusted in zip(members, holm_adjust([r["cw_pvalue"] for r in members]),
                                     strict=True):
                row["cw_holm_pvalue"] = adjusted
    return sorted(records, key=lambda r: (r["hypothesis"], r["states"] or 0, r["series"]))


def prefix_table(forecasts: pl.DataFrame, best_models: dict[str, str]) -> list[dict[str, Any]]:
    """Report y1 prefix loss and the already-selected/frozen best label; never reselect."""
    rows = []
    chosen = forecasts.filter(pl.col("target") == "y1", pl.col("series").is_in(["RAW", "RES"]))
    groups = chosen.partition_by(["series", "feature_set"], as_dict=True)
    for (series, model), frame in groups.items():
        if model in {"L1-own", "L1_without_M", "L1m"}:
            continue
        error = frame["actual"].to_numpy()-frame["prediction"].to_numpy()
        rows.append({"series": series, "model": model, "n": frame.height,
                     "first_origin": str(frame["origin"].min()),
                     "last_origin": str(frame["origin"].max()), "mse": float(np.mean(error**2)),
                     "qlike": float(qlike(frame["actual"], frame["prediction"],
                                          frame["residual_variance"]).mean()),
                     "selection_candidate": model in FEATURE_PREFIXES,
                     "selected_best": best_models.get(series) == model})
    return sorted(rows, key=lambda row: (row["series"], row["model"]))


def regime_tables(
    records: list[dict[str, Any]], forecasts: pl.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Training diagnostics for every fit; full-refit stability and subsequent y1."""
    unique = {}
    for record in records:
        key = record["lineage"], record["series"], record["states"], record["fit_time"]
        if key in unique and unique[key]["fit_hash"] != record["fit_hash"]:
            raise ValueError("Conflicting regime fits for one lineage/series/state count/cutoff")
        unique[key] = record
    occupancy, transitions, stability, risk, restarts = [], [], [], [], []
    preceding, risk_values = {}, {}
    covered = set()
    for (lineage, series, k, cutoff), record in sorted(unique.items()):
        common = {"lineage": lineage, "series": series, "states": k, "refit": cutoff,
                  "training_months": record["training_months"]}
        p = np.asarray(record["hmm"]["transition"])
        restarts.extend({**common, **attempt} for attempt in record["hmm"]["restarts"])
        for state in range(k):
            occupancy.append({**common, "state": state,
                "occupancy": record["occupancy"][state],
                "posterior_occupancy": record["posterior_occupancy"][state],
                "below_five_percent": record["low_occupancy"][state],
                "mean_duration_months": record["mean_duration_months"][state],
                "infinite_duration": record["infinite_duration"][state]})
            for other in range(k) if lineage == "full" else ():
                transitions.append({**common, "from_state": state, "to_state": other,
                                    "probability": float(p[state, other])})
        if lineage != "full":
            continue
        earlier = preceding.get((series, k))
        stability.append({**common, "previous_refit": earlier[0] if earlier else None,
            "frobenius_change": float(np.linalg.norm(p-earlier[1])) if earlier else None,
            "max_change": float(np.abs(p-earlier[1]).max()) if earlier else None})
        preceding[series, k] = cutoff, p
        paths = {r["date"]: r for r in record["monthly_probabilities"]}
        scored = forecasts.filter(pl.col("series") == series, pl.col("target") == "y1",
            pl.col("feature_set") == f"best+HMMF{k}", pl.col("refit") == date.fromisoformat(cutoff))
        for origin in scored.iter_rows(named=True):
            if origin["origin"] < date.fromisoformat(cutoff):
                raise ValueError("Regime risk labels use a future fit")
            point = paths[str(origin["origin"])]
            if point.get("y1") is not None and point["y1"] != origin["actual"]:
                raise ValueError("Regime paths disagree with the scored subsequent risk")
            covered.add((series, k, origin["origin"]))
            risk_values.setdefault((series, k, point["filtered_state"]), []).append(
                origin["actual"])
    expected = set()
    for row in forecasts.filter(pl.col("target") == "y1").iter_rows(named=True):
        if row["feature_set"].startswith("best+HMMF"):
            expected.add((row["series"], int(row["feature_set"].removeprefix("best+HMMF")),
                          row["origin"]))
    if covered != expected:
        raise ValueError("Regime records do not cover every scored filtered-state origin")
    for (series, k, state), values in sorted(risk_values.items()):
        risk.append({"series": series, "states": k, "state": state, "origins": len(values),
                     "mean_log_realized_variance": float(np.mean(values)),
                     "mean_realized_variance": float(np.exp(values).mean())})
    return {"regime_occupancy": occupancy, "regime_transitions": transitions,
            "regime_stability": stability, "regime_subsequent_risk": risk,
            "regime_restarts": restarts}


def _numeric(value: Any) -> Any:
    """Remove execution provenance from the deterministic numerical result."""
    if isinstance(value, dict):
        return {key: _numeric(item) for key, item in value.items()
                if key not in {"provenance", "run_id", "run_ids", "parent_run_id", "timestamp",
                               "started_at", "finished_at", "elapsed_seconds", "runtime_seconds",
                               "artifacts", "source_hashes", "input_hashes"}
                and not key.endswith(("_path", "_paths", "_hash"))}
    if isinstance(value, (list, tuple)):
        return [_numeric(item) for item in value]
    return value


def diagnostic_summary(value: Any) -> Any:
    """Keep reported recovery rates/thresholds; full simulation draws stay external."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"noise_distances", "replications", "settings"}:
                continue
            if key == "fit" and isinstance(item, dict) and "restarts" in item:
                attempts = item["restarts"]
                statuses = sorted({attempt["status"] for attempt in attempts})
                result[key] = {"log_likelihood": item["log_likelihood"],
                    "restarts": len(attempts), "restart_statuses": {
                        status: sum(attempt["status"] == status for attempt in attempts)
                        for status in statuses}}
            else:
                result[key] = diagnostic_summary(item)
        return result
    if isinstance(value, (tuple, list)):
        return [diagnostic_summary(item) for item in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)+"\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_l3(
    settings: dict[str, Any], output: Path, metadata: dict[str, Any], audit: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate saved comparisons; design and thin modes also copy their tables to results."""
    from momenta.baseline_report import (
        forecast_table,
        h2_table,
        sharpe_difference_table,
        strategy_table,
    )

    mode = metadata["mode"]
    if mode not in {"design", "thin", "mock_holdout", "holdout"}:
        raise ValueError("Unknown L3 evaluation mode")
    frame = pl.read_parquet(output/"forecasts.parquet")
    if frame.is_empty() or frame.filter(pl.col("refit") > pl.col("origin")).height:
        raise ValueError("Forecasts are empty or use a future fit")
    keys = ["series", "target", "feature_set", "origin"]
    if frame.select(pl.struct(keys).n_unique()).item() != frame.height:
        raise ValueError("Duplicate L3 forecast keys")
    bound = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    if mode in {"design", "thin"} and frame["origin"].max() > bound:
        raise ValueError("Holdout origins cannot enter a design report")
    evaluation_settings = deepcopy(settings)
    if mode in {"mock_holdout", "holdout"}:
        # The guarded runner owns authorization; this copy adjusts only the shared
        # report helpers' assertion bound, never any fitted model or frozen config.
        evaluation_settings["walkforward"]["design_last_origin"] = str(frame["origin"].max())
    best = audit["best_models"]
    if any(value not in FEATURE_PREFIXES for value in best.values()):
        raise ValueError("Frozen best models must be one of the four registered prefixes")
    size = audit["baseline_summary"]["tables"]["size_check"]
    comparisons = comparison_table(frame, settings, best, size, mode=mode)
    tables = {"prefixes": prefix_table(frame, best), "size_check": size}
    tables.update({hypothesis.lower(): [r for r in comparisons if r["hypothesis"] == hypothesis]
                   for hypothesis in ("H3", "H4", "H4Q", "H5a", "H5b")})
    baseline = frame.filter(pl.col("feature_set").is_in(BASELINE_SETS))
    tables["baseline_forecasts"] = forecast_table(baseline, evaluation_settings, size_check=size)
    tables["h2"] = (h2_table(baseline, evaluation_settings)
                    if {"RAW", "RAW_c", "RES"}.issubset(set(baseline["series"])) else [])
    tables.update(regime_tables(audit.get("regime_records", []), frame))
    tables["prefix_checks"] = audit.get("prefix_checks", [])
    if (output/"strategy_monthly.parquet").exists():
        monthly = pl.read_parquet(output/"strategy_monthly.parquet")
        tables["strategies"] = strategy_table(monthly,
            pl.read_parquet(output/"strategy_daily.parquet"),
            pl.read_parquet(output/"strategy_rebalances.parquet"), settings,
            audit["strategy_population"])
        tables["break_even"] = audit.get("strategy_audit", {}).get("break_even", [])
        tables["sharpe_differences"] = sharpe_difference_table(monthly, settings)
    tables = _numeric(tables)
    numerical = {"best_models": best, "tables": tables,
                 "diagnostics": _numeric(diagnostic_summary(audit.get("diagnostics", {})))}
    output.mkdir(parents=True, exist_ok=True)
    table_directory = output/"tables"
    table_directory.mkdir(exist_ok=True)
    numeric_path = output/"numeric_results.json"
    _json(numeric_path, numerical)
    numeric_sidecar = output/"numeric_results.provenance.json"
    _json(numeric_sidecar, {**metadata, "artifact_hash": sha256_file(numeric_path)})
    files, deterministic, produced = [numeric_path, numeric_sidecar], [numeric_path], {}
    for name, rows in tables.items():
        path = table_directory/f"{name}.csv"
        _csv(path, rows)
        sidecar = path.with_suffix(".provenance.json")
        _json(sidecar, {**metadata, "artifact_hash": sha256_file(path)})
        files.extend([path, sidecar])
        deterministic.append(path)
        produced[name] = str(path)
        if mode in {"design", "thin"}:
            exported = REPO/"results/tables"/f"l3_design_{name}.csv"
            exported.parent.mkdir(parents=True, exist_ok=True)
            exported.write_bytes(path.read_bytes())
            exported_meta = exported.with_suffix(".provenance.json")
            exported_meta.write_bytes(sidecar.read_bytes())
            files.extend([exported, exported_meta])
    summary = {"provenance": metadata, **numerical, "forecast_rows": frame.height,
               "first_origin": str(frame["origin"].min()),
               "last_origin": str(frame["origin"].max()),
               "produced_tables": produced,
               "artifacts": {str(path): sha256_file(path) for path in files},
               "deterministic_artifacts": [str(path.relative_to(settings["data"]["data_root"]))
                                           for path in deterministic]}
    _json(output/"summary.json", summary)
    return summary


def load_design_summary(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Require unchanged producer source/config, manifest, inputs and result files."""
    root = Path(settings["data"]["data_root"])
    path = root/"derived/final/summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    metadata = summary["provenance"]
    if metadata["mode"] not in {"design", "thin"}:
        raise ValueError("The L3 design summary cannot load a holdout result")
    if summary.get("mode", metadata["mode"]) != metadata["mode"]:
        raise ValueError("L3 summary and producer modes disagree")
    if metadata["config_hash"] != sha256_file(Path(settings["config_path"])):
        raise ValueError("L3 configuration changed")
    if metadata["data_manifest_hash"] != sha256_file(root/"raw/manifest.json"):
        raise ValueError("L3 data manifest changed")
    if metadata["source_config_hash"] != source_config_hash():
        raise ValueError("L3 source/config tree changed")
    if summary.get("complete", False) and (
        metadata["mode"] != "design" or metadata["git_dirty"]
    ):
        raise ValueError("Complete L3 design results require a clean initial producer tree")
    for registry in (metadata["input_hashes"], summary["artifacts"]):
        for name, digest in registry.items():
            artifact = Path(name)
            if not artifact.is_file() or sha256_file(artifact) != digest:
                raise ValueError(f"Changed L3 input or result: {artifact.name}")
    return summary
