"""Baseline (L1) inputs and the design-period baseline study (Methodology).

``build_baseline`` writes the dated targets, the L1 feature panel and the BDP
recovery decision. ``run_study`` fits the registered L1 models in
``config/experiments.toml`` through annual walk-forward refits, runs the
design-period strategy accounts and the placebo size check, then evaluates the
saved outputs. No step reads or scores an origin after the design boundary.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.experiments import trial
from momenta.features import FEATURE_SETS, build_features
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO
from momenta.targets import build_targets, overlap_counts
from momenta.walkforward import assert_design_origins, walk_forward


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _base_paths(settings: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(settings["data"]["data_root"])
    return root / "curated", root / "derived/portfolios", root / "derived/baseline"


def checked_portfolio_summary(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Return the portfolio summary only if its config, code, data and outputs are current."""
    from momenta.portfolio_report import load_summary

    return load_summary(settings)


def build_baseline(settings: dict[str, Any], through: date | None = None) -> None:
    """Construct targets and L1 after checked portfolios and the logged BDP decision."""
    from momenta.bdp import ensure_bdp_decision

    summary = checked_portfolio_summary(settings)
    if summary is None or not summary["passed"]:
        raise ValueError("Current, reconciled portfolios are required for the baseline build")
    curated, books_root, output = _base_paths(settings)
    output.mkdir(parents=True, exist_ok=True)
    bound = through or date.fromisoformat(settings["walkforward"]["design_last_origin"])
    assert_design_origins([bound], settings)
    ensure_bdp_decision(settings)
    calendar = pl.read_parquet(curated / "trading_calendar.parquet")["date"].to_list()
    books = pl.read_parquet(books_root / "book_daily.parquet").filter(
        pl.col("construction") == settings["portfolios"]["primary"],
    )
    targets = build_targets(books, calendar, settings, last_origin=bound)
    targets.write_parquet(output / "targets.parquet")
    features_path = build_features(settings, through=bound)
    metadata = {
        **provenance(REPO, Path(settings["config_path"]), Manifest(curated.parent)),
        "through": str(bound), "target_rows": targets.height,
        "target_hash": sha256_file(output / "targets.parquet"),
        "target_source_hash": sha256_file(REPO / "src/momenta/targets.py"),
        "feature_metadata_hash": sha256_file(features_path.with_suffix(".metadata.json")),
        "input_hashes": {
            str(books_root / "book_daily.parquet"): sha256_file(books_root / "book_daily.parquet"),
            str(curated / "trading_calendar.parquet"):
                sha256_file(curated / "trading_calendar.parquet"),
        },
    }
    _write_json(output / "build_metadata.json", metadata)
    _write_json((output / "targets.parquet").with_suffix(".metadata.json"), metadata)
    print(f"Baseline inputs built through {bound}: {targets.height} dated targets", flush=True)


def _validate_build(settings: dict[str, Any], through: date) -> dict[str, Any]:
    """Check the pinned feature/target inputs once before a registered run."""
    from momenta.bdp import ensure_bdp_decision

    summary = checked_portfolio_summary(settings)
    if summary is None or not summary["passed"]:
        raise ValueError("Current, reconciled portfolios are required for the baseline run")
    ensure_bdp_decision(settings)
    curated, _, output = _base_paths(settings)
    built = json.loads((output / "build_metadata.json").read_text())
    metadata = json.loads((output / "features.metadata.json").read_text())
    for artifact in (built, metadata):
        for key, path in (("config_hash", Path(settings["config_path"])),
                          ("data_manifest_hash", curated.parent / "raw/manifest.json")):
            if artifact.get(key) != sha256_file(path):
                raise ValueError(f"Stale baseline build: {key}; rerun momenta features")
    if date.fromisoformat(built["through"]) < through:
        raise ValueError("Baseline inputs end before this run; rebuild through the requested date")
    expected = {
        **built["input_hashes"], **metadata["input_hashes"],
        str(output / "targets.parquet"): built["target_hash"],
        str(output / "features.parquet"): metadata["artifact_hash"],
        str(output / "features.metadata.json"): built["feature_metadata_hash"],
        str(REPO / "src/momenta/targets.py"): built["target_source_hash"],
        str(REPO / "src/momenta/features.py"): metadata["source_hash"],
    }
    for name, digest in expected.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Changed baseline input {name}; rerun momenta features")
    if "pending" in metadata["bdp_method"]:
        raise ValueError("The BDP recovery decision must precede forecasting")
    return metadata


def _bdp_paths(settings: dict[str, Any], cutoff: date) -> tuple[pl.DataFrame, Any, dict]:
    """Persist the expensive French fit and three causal paths once per cutoff.

    The identity pins the cutoff, the BDP settings, the French and book inputs
    and the producing source files; a cached path is reused only on an exact match.
    """
    from momenta.bdp import BDPFit, filter_bdp, fit_bdp

    curated, books_root, output = _base_paths(settings)
    directory = output / "bdp_paths"
    directory.mkdir(exist_ok=True)
    french_path, book_path = (curated / "french/momentum_deciles_daily.parquet",
                              books_root / "book_daily.parquet")
    identity = {
        "cutoff": str(cutoff), "settings": settings["bdp"],
        "french_hash": sha256_file(french_path), "book_hash": sha256_file(book_path),
        "source_hash": sha256_file(REPO / "src/momenta/bdp.py"),
        "path_source_hash": sha256_file(REPO / "src/momenta/baseline_study.py"),
        "primary_construction": settings["portfolios"]["primary"],
        "design_last_origin": settings["walkforward"]["design_last_origin"],
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = directory / f"{cutoff}.parquet"
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["identity_hash"] == fingerprint and record["artifact_hash"] == sha256_file(path):
            return pl.read_parquet(path), BDPFit(**record["fit"]), record
    french = pl.read_parquet(french_path).filter(pl.col("date") < cutoff).sort("date")
    fit = fit_bdp(100 * (french["d10"] - french["d1"]).to_numpy(), settings)
    if not fit.converged:
        raise ValueError(f"BDP-transfer French fit did not converge before {cutoff}")
    last = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    books = pl.read_parquet(book_path).filter(
        pl.col("construction") == settings["portfolios"]["primary"], pl.col("date") <= last,
    )
    frames = []
    for group in books.sort("date").partition_by("series"):
        states = filter_bdp(100 * group["daily_return"].to_numpy(), fit)
        frames.append(group.select("date", "series").with_columns(
            pl.Series("bdp_m", states[:, 0]), pl.Series("bdp_gamma", states[:, 1]),
            pl.Series("bdp_delta", states[:, 2]),
            pl.Series("bdp_ci", np.exp(states[:, 1]) * np.tanh(states[:, 2]) / 100),
        ))
    paths = pl.concat(frames).sort("date", "series")
    paths.write_parquet(path)
    record = {**identity, "identity_hash": fingerprint, "fit": asdict(fit),
              "fit_time": str(cutoff), "french_last_observation": str(french["date"].max()),
              "artifact_hash": sha256_file(path)}
    _write_json(sidecar, record)
    return paths, fit, record


def _structural_transform(settings: dict[str, Any], enabled: bool):
    """Return a cutoff-specific BDP transform; unrelated placebo columns are preserved."""
    if not enabled:
        return None
    paths_by_cutoff = {}

    def transform(frame: pl.DataFrame, cutoff: date):
        if cutoff not in paths_by_cutoff:
            paths_by_cutoff[cutoff] = _bdp_paths(settings, cutoff)
        paths, _, metadata = paths_by_cutoff[cutoff]
        transformed = frame.drop("bdp_ci").join(
            paths.select("date", "series", "bdp_ci"), on=["date", "series"],
            how="left", maintain_order="left",
        )
        return transformed.select(frame.columns), metadata

    return transform


def _registry() -> dict[str, Any]:
    with (REPO / "config/experiments.toml").open("rb") as stream:
        return tomllib.load(stream)


def _forecast_specification(settings: dict[str, Any], experiment: dict) -> dict:
    """Identify the fixed forecasting rule separately from its evaluated sample."""
    return {
        **experiment, "features": list(FEATURE_SETS[experiment["feature_set"]]),
        "walkforward": settings["walkforward"], "targets": settings["targets"],
        "feature_settings": settings["features"], "bdp": settings["bdp"],
    }


def _benchmark_forecasts(forecasts: pl.DataFrame) -> pl.DataFrame:
    """Use the matched L1-own forecast for y1; other targets retain fitted nulls."""
    benchmarks = forecasts.filter(
        (pl.col("target") == "y1") & (pl.col("feature_set") == "L1-own"),
    ).select("series", "origin", pl.col("prediction").alias("own_prediction"))
    result = forecasts.join(benchmarks, on=["series", "origin"], how="left")
    if result.filter((pl.col("target") == "y1") & pl.col("own_prediction").is_null()).height:
        raise ValueError("The y1 benchmark is absent on a scored origin")
    return result.with_columns(
        pl.when(pl.col("target") == "y1").then(pl.col("own_prediction"))
        .otherwise(pl.col("base_prediction")).alias("base_prediction"),
    ).drop("own_prediction")


def run_study(
    settings: dict[str, Any], name: str, *, through: date | None = None,
) -> dict[str, Any]:
    """Run a registered model or suite, logging successes and failures before work.

    The full ``baseline`` suite through the design boundary also runs the
    placebo size check; a thin sample never carries a full-study size result.
    """
    registry = _registry()
    if through is None and name in registry.get("samples", {}):
        through = date.fromisoformat(registry["samples"][name]["through"])
    bound = through or date.fromisoformat(settings["walkforward"]["design_last_origin"])
    curated, books_root, output = _base_paths(settings)
    suite = name in registry.get("suites", {})
    full_run = (name == "baseline"
                and bound == date.fromisoformat(settings["walkforward"]["design_last_origin"]))
    requested = registry.get("suites", {}).get(name, [name])
    with trial(
        settings, experiment=name, kind="diagnostic", series="requested",
        sample=f"design through {bound}", specification={
            "registered_request": name, "walkforward": settings["walkforward"],
            "amendment": settings.get("amendment", {}),
        },
    ) as outer:
        # This guard precedes reading model inputs or fitting any learned step.
        assert_design_origins([bound], settings)
        if any(item not in registry["experiments"] for item in requested):
            raise ValueError(f"Unknown registered experiment or suite: {name}")
        feature_metadata = _validate_build(settings, bound)
        features = pl.read_parquet(output / "features.parquet").filter(pl.col("date") <= bound)
        targets = pl.read_parquet(output / "targets.parquet").filter(pl.col("origin") <= bound)
        calendar = pl.read_parquet(curated / "trading_calendar.parquet")["date"].to_list()
        decision = json.loads((output / "bdp_decision.json").read_text())
        structural_transform = _structural_transform(settings, decision["full_model"])
        names = list(requested)
        for item in requested:
            definition = registry["experiments"][item]
            if definition["target"] == "y1":
                benchmark = f"{definition['series']}_y1_L1_own"
                if benchmark not in names:
                    names.append(benchmark)
        forecast_frames, fit_frames, training_frames = [], [], []
        run_directory = output / "runs" / outer.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        for experiment_name in names:
            definition = registry["experiments"][experiment_name]
            print(f"Forecast {experiment_name} through {bound}", flush=True)
            with trial(
                settings, experiment=experiment_name, kind="forecast",
                series=definition["series"], sample=f"month-end origins through {bound}",
                specification=_forecast_specification(settings, definition),
            ) as attempt:
                transformed = (structural_transform
                               if "bdp_ci" in FEATURE_SETS[definition["feature_set"]] else None)
                result = walk_forward(
                    features, targets, calendar, settings,
                    series=definition["series"], target=definition["target"],
                    feature_set=definition["feature_set"],
                    columns=FEATURE_SETS[definition["feature_set"]], through=bound,
                    transform=transformed,
                )
                frame = result.forecasts.with_columns(
                    pl.lit(attempt.strategy_id).alias("strategy_id"),
                    pl.lit(attempt.run_id).alias("run_id"),
                    pl.lit(decision["method"]).alias("bdp_method"),
                )
                frame.write_parquet(run_directory / f"{experiment_name}.parquet")
                forecast_frames.append(frame)
                fit_frames.append(result.fits)
                if not result.training_predictions.is_empty():
                    training_frames.append(result.training_predictions)
                errors = frame["actual"].to_numpy() - frame["prediction"].to_numpy()
                attempt.metrics.update({
                    "origins": frame.height, "first_origin": str(frame["origin"].min()),
                    "last_origin": str(frame["origin"].max()), "mean_squared_error":
                    float(np.mean(errors**2)), "annual_fits": result.fits.height,
                    "bdp_method": decision["method"], "parent_run_id": outer.run_id,
                })
        forecasts = _benchmark_forecasts(pl.concat(forecast_frames, how="diagonal_relaxed"))
        fits = pl.concat(fit_frames, how="diagonal_relaxed")
        training = (pl.concat(training_frames, how="diagonal_relaxed") if training_frames
                    else pl.DataFrame(schema={"series": pl.String, "refit": pl.Date,
                                              "origin": pl.Date, "target_end": pl.Date,
                                              "variance_prediction": pl.Float64,
                                              "bear_x_var": pl.Float64}))
        metadata = {
            **provenance(REPO, Path(settings["config_path"]), Manifest(curated.parent)),
            "mode": "full" if full_run else "thin",
            "experiment": name, "run_id": outer.run_id, "through": str(bound),
            "input_hashes": {
                str(REPO / "config/experiments.toml"):
                sha256_file(REPO / "config/experiments.toml"),
                str(output / "build_metadata.json"): sha256_file(output / "build_metadata.json"),
                str(output / "features.metadata.json"):
                sha256_file(output / "features.metadata.json"),
                str(books_root / "m2_summary.json"): sha256_file(books_root / "m2_summary.json"),
            },
        }
        for artifact, frame in (("forecasts", forecasts), ("fits", fits),
                                ("training_predictions", training)):
            frame.write_parquet(run_directory / f"{artifact}.parquet")
            _write_json(run_directory / f"{artifact}.metadata.json", metadata)
        if not suite:
            outer.metrics.update({"forecast_rows": forecasts.height, "output": str(run_directory)})
            return {"provenance": metadata, "forecast_rows": forecasts.height}
        # A suite atomically replaces the current scoreboard after its models finish.
        for artifact, frame in (("forecasts", forecasts), ("fits", fits),
                                ("training_predictions", training)):
            frame.write_parquet(output / f"{artifact}.parquet")
            _write_json(output / f"{artifact}.metadata.json", metadata)
        scored = forecasts.select("series", "origin").unique()
        overlap_counts(targets.join(scored, on=["series", "origin"], how="inner"), calendar
                       ).write_parquet(output / "target_overlaps.parquet")
        from momenta.strategy_study import run_strategies

        strategy_audit = run_strategies(
            settings, features, forecasts, training, last_origin=forecasts["origin"].max(),
            metadata=metadata, output=output,
            bdp_paths=(lambda cutoff: _bdp_paths(settings, cutoff))
            if decision["full_model"] else None,
        )
        audit = {
            **strategy_audit, "feature_availability": feature_metadata["availability"],
            "bdp_status": decision["method"], "bdp_explanation": decision.get("reason", ""),
            "model_count": len(names), "fit_count": fits.height,
            "training_target_overlaps_refit": fits.filter(
                pl.col("training_target_end_max") >= pl.col("refit"),
            ).height,
        }
        _write_json(output / "_metadata.json", metadata)
        _write_json(output / "_audit.json", audit)
        if full_run:
            from momenta.size_check import run_size_check

            run_size_check(features, targets, calendar, settings, output=output,
                           transform=structural_transform)
        else:
            # A previous full-study size result must never be attached to a thin study.
            (output / "size_check.parquet").unlink(missing_ok=True)
            (output / "size_check.metadata.json").unlink(missing_ok=True)
        from momenta.baseline_report import evaluate_study

        summary = evaluate_study(settings, output, metadata, audit)
        outer.metrics.update({"forecast_rows": forecasts.height, "models": len(names),
                              "size_check_complete": summary["size_check_complete"],
                              "bdp_method": decision["method"]})
        print(f"Baseline {metadata['mode']} run complete: {forecasts.height} forecasts", flush=True)
        return summary
