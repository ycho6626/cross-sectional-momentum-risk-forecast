"""Registered real-target placebo size experiment (Methodology).

The real RAW and RES y1/L1 baselines are compared with independent Gaussian
AR(1) blocks. Every replication uses the ordinary annual walk-forward pipeline.
"""


from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from momenta.experiments import trial
from momenta.features import FEATURE_BLOCKS, FEATURE_SETS
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO
from momenta.study_statistics import clark_west, diebold_mariano
from momenta.walkforward import FeatureTransform, month_end_origins, refit_for_origin, walk_forward


def ar1_predictors(n: int, k: int, persistence: float, seed: int) -> NDArray[np.float64]:
    """Independent unit-variance AR(1): X[t]=ρX[t−1]+sqrt(1−ρ²)ε[t] (Methodology)."""
    if n < 1 or k < 1 or not np.isfinite(persistence) or abs(persistence) >= 1:
        raise ValueError("Stationary AR(1) requires positive dimensions and |rho|<1")
    random = np.random.default_rng(seed)
    innovations = random.standard_normal((n, k))
    result = np.empty_like(innovations)
    result[0] = innovations[0]
    scale = np.sqrt(1 - persistence**2)
    for index in range(1, n):
        result[index] = persistence * result[index - 1] + scale * innovations[index]
    return result


def training_persistence(
    features: pl.DataFrame,
    targets: pl.DataFrame,
    calendar: Sequence[date],
    settings: dict[str, Any],
    transform: FeatureTransform | None = None,
    *,
    series: str = "RAW",
) -> tuple[float, dict[str, Any]]:
    """Median L1 lag-one correlation in completed first-refit training (Methodology).

    Only adjacent sessions whose two rows are complete enter a correlation;
    missing dates never become artificial neighbours. Constant columns are
    disclosed and excluded because their autocorrelation is undefined.
    """
    config = settings["walkforward"]
    first = date.fromisoformat(config["design_first_origin"])
    last = date.fromisoformat(config["design_last_origin"])
    origins = [
        day
        for day in month_end_origins(calendar)
        if (first.year, first.month) <= (day.year, day.month) and day <= last
    ]
    if not origins:
        raise ValueError("No size-check design origins")
    refit = refit_for_origin(origins[0], calendar)
    selected = features.filter((pl.col("series") == series) & (pl.col("date") <= last))
    target = targets.filter(pl.col("series") == series)
    if "construction" in target.columns:
        target = target.filter(pl.col("construction") == settings["portfolios"]["primary"])
    joined = selected.join(target.drop("series"), left_on="date", right_on="origin").sort("date")
    if joined["date"].n_unique() != joined.height:
        raise ValueError("Duplicate size-check training origins")
    if transform:
        transformed, _ = transform(joined, refit)
        if transformed["date"].to_list() != joined["date"].to_list():
            raise ValueError("Structural transform changed persistence origin keys")
        joined = transformed
    columns = list(FEATURE_SETS["L1"])
    values = joined.select(columns).to_numpy().astype(float)
    eligible = np.isfinite(values).all(axis=1) & np.isfinite(joined["y1"].to_numpy())
    eligible &= (joined["target_end"] < refit).to_numpy()
    eligible &= (joined["date"] >= date.fromisoformat(config["feature_start"])).to_numpy()
    indices = {day: index for index, day in enumerate(calendar)}
    positions = np.array([indices[day] for day in joined["date"].to_list()])
    adjacent = eligible[1:] & eligible[:-1] & (np.diff(positions) == 1)
    correlations, unavailable = {}, []
    for index, name in enumerate(columns):
        before, after = values[:-1, index][adjacent], values[1:, index][adjacent]
        if len(before) < 3 or np.ptp(before) == 0 or np.ptp(after) == 0:
            unavailable.append(name)
            continue
        correlations[name] = float(np.corrcoef(before, after)[0, 1])
    if not correlations:
        raise ValueError("No estimable first-refit L1 autocorrelations")
    persistence = float(np.median(list(correlations.values())))
    if not np.isfinite(persistence) or abs(persistence) >= 1:
        raise ValueError("First-refit persistence does not define a stationary AR(1)")
    return persistence, {
        "series": series,
        "refit": str(refit),
        "training_rows": int(eligible.sum()),
        "adjacent_pairs": int(adjacent.sum()),
        "autocorrelations": correlations,
        "undefined_features": unavailable,
    }


def _preserve_placebos(
    transform: FeatureTransform | None, columns: list[str]
) -> FeatureTransform | None:
    if transform is None:
        return None

    def wrapped(frame: pl.DataFrame, cutoff: date) -> tuple[pl.DataFrame, dict[str, Any]]:
        original = frame.select("date", *columns)
        changed, metadata = transform(frame.drop(columns), cutoff)
        if changed["date"].to_list() != original["date"].to_list():
            raise ValueError("Structural transform changed placebo origin keys")
        return changed.hstack(original.select(columns)), metadata

    return wrapped


def _save(rows: list[dict[str, Any]], output: Path, metadata: dict[str, Any]) -> pl.DataFrame:
    frame = pl.DataFrame(
        rows,
        schema={
            "series": pl.String,
            "k": pl.Int64,
            "rep": pl.Int64,
            "seed": pl.Int64,
            "persistence": pl.Float64,
            "cw_p": pl.Float64,
            "dm_p": pl.Float64,
            "r2": pl.Float64,
            "mse_difference": pl.Float64,
            "identical_forecasts": pl.Boolean,
            "refits": pl.Int64,
            "placebo_excluded_refits": pl.Int64,
            "error": pl.String,
            "elapsed_seconds": pl.Float64,
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "size_check.parquet.part"
    frame.write_parquet(temporary)
    temporary.replace(output / "size_check.parquet")
    metadata = {
        **metadata,
        "completed_rows": frame.height,
        "artifact_hash": sha256_file(output / "size_check.parquet"),
    }
    (output / "size_check.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return frame


def _excluded_placebo_refits(fits: pl.DataFrame) -> int:
    """Count refits whose selected final block is excluded, per Methodology."""
    if fits.is_empty():
        raise ValueError("Placebo exclusion audit needs fitted models")
    records = [json.loads(value) for value in fits["block_lambdas_json"].to_list()]
    if any("PLACEBO" not in record for record in records):
        raise ValueError("Missing PLACEBO penalty in refit audit")
    return sum(record["PLACEBO"] == "exclude" for record in records)


def run_size_check(
    features: pl.DataFrame,
    targets: pl.DataFrame,
    calendar: Sequence[date] | pl.DataFrame,
    settings: dict[str, Any],
    *,
    output: Path,
    transform: FeatureTransform | None = None,
) -> pl.DataFrame:
    """Run each preregistered placebo through all annual fits, logging failures (Methodology).

    Predictor seeds distinguish diagnostic specifications. They never enter the
    trading-strategy multiplicity count. Failed replications are saved, all
    remaining replications run, and an incomplete experiment then raises.
    """
    days = calendar["date"].to_list() if isinstance(calendar, pl.DataFrame) else list(calendar)
    if days != sorted(set(days)) or not days:
        raise ValueError("Size-check calendar must be sorted and unique")
    config = settings["evaluation"]
    root = Path(settings["data"]["data_root"])
    metadata = provenance(REPO, Path(settings["config_path"]), Manifest(root))
    metadata["source_hash"] = sha256_file(Path(__file__))
    last = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    requested_series = list(config["size_check_series"])
    if not requested_series or len(set(requested_series)) != len(requested_series):
        raise ValueError("Size-check series must be nonempty and unique")
    selected_all = features.filter(
        pl.col("series").is_in(requested_series),
        pl.col("date") <= last,
    )
    if set(selected_all["series"].unique()) != set(requested_series):
        raise ValueError("Missing requested size-check series")
    blocks = {
        name: list(FEATURE_BLOCKS[name]) for name in ("HAR", "CORE", "M") if FEATURE_BLOCKS[name]
    }
    columns = [column for names in blocks.values() for column in names]
    specification = {
        "target": "y1",
        "baseline": "L1",
        "blocks": blocks,
        "walkforward": settings["walkforward"],
        "features": settings.get("features", {}),
        "bdp": settings.get("bdp", {}),
        "null": "independent_stationary_gaussian_ar1",
    }
    sample = f"design:{settings['walkforward']['design_first_origin']}:{last}"
    metadata.update(
        persistence_by_series={},
        requested_series=requested_series,
        requested_replications=config["size_check_reps"],
        blocks=config["size_check_blocks"],
        master_seed=config["seed"],
        shared_innovations_across_series=True,
    )
    # The same dates and seed give shared innovations, with each series' own rho.
    simulation_days = [day for day in days if selected_all["date"].min() <= day <= last]
    rows, failures = [], 0
    for series in requested_series:
        selected = selected_all.filter(pl.col("series") == series).sort("date")
        with trial(
            settings,
            experiment="size-check-baseline",
            kind="diagnostic",
            series=series,
            sample=sample,
            specification=specification,
        ) as attempt:
            baseline_started = time.perf_counter()
            persistence, audit = training_persistence(
                selected,
                targets,
                days,
                settings,
                transform,
                series=series,
            )
            baseline = walk_forward(
                selected,
                targets,
                days,
                settings,
                series=series,
                target="y1",
                feature_set="L1",
                columns=columns,
                blocks=blocks,
                transform=transform,
            ).forecasts.select("origin", "actual", pl.col("prediction").alias("baseline"))
            attempt.metrics.update(
                persistence=persistence,
                persistence_audit=audit,
                forecast_months=baseline.height,
                elapsed_seconds=time.perf_counter() - baseline_started,
            )
        metadata["persistence_by_series"][series] = {
            "persistence": persistence,
            "audit": audit,
        }
        for k in config["size_check_blocks"]:
            for rep in range(config["size_check_reps"]):
                started = time.perf_counter()
                seed = int(np.random.SeedSequence([config["seed"], k, rep]).generate_state(1)[0])
                record = {
                    "series": series,
                    "k": k,
                    "rep": rep,
                    "seed": seed,
                    "persistence": persistence,
                    "cw_p": None,
                    "dm_p": None,
                    "r2": None,
                    "mse_difference": None,
                    "identical_forecasts": None,
                    "refits": None,
                    "placebo_excluded_refits": None,
                    "error": None,
                    "elapsed_seconds": None,
                }
                try:
                    names = [f"placebo_{index}" for index in range(k)]
                    augmented_blocks = blocks | {"PLACEBO": names}
                    with trial(
                        settings,
                        experiment="size-check-placebo",
                        kind="diagnostic",
                        series=series,
                        sample=sample,
                        specification=specification
                        | {
                            "blocks": augmented_blocks,
                            "k": k,
                            "predictor_seed": seed,
                        },
                    ) as attempt:
                        placebo = pl.DataFrame(
                            ar1_predictors(len(simulation_days), k, persistence, seed),
                            schema=names,
                        ).with_columns(pl.Series("date", simulation_days))
                        augmented = selected.join(placebo, on="date", how="left", validate="1:1")
                        run = walk_forward(
                            augmented,
                            targets,
                            days,
                            settings,
                            series=series,
                            target="y1",
                            feature_set=f"L1_placebo_{k}",
                            columns=columns + names,
                            blocks=augmented_blocks,
                            transform=_preserve_placebos(transform, names),
                        )
                        forecast = run.forecasts.select("origin", "actual", "prediction")
                        pair = baseline.join(
                            forecast,
                            on="origin",
                            suffix="_model",
                            validate="1:1",
                        )
                        if pair.height != baseline.height or pair.height != forecast.height:
                            raise ValueError("Placebo and baseline evaluation origins differ")
                        if not np.array_equal(
                            pair["actual"].to_numpy(),
                            pair["actual_model"].to_numpy(),
                        ):
                            raise ValueError("Placebo changed the real target")
                        actual, base, model = (
                            pair[name].to_numpy() for name in ("actual", "baseline", "prediction")
                        )
                        identical = bool(np.array_equal(base, model))
                        cw = clark_west(actual, base, model, config["nw_lags_monthly"])
                        dm = diebold_mariano(actual, base, model, config["nw_lags_monthly"])
                        base_loss, model_loss = (actual - base) ** 2, (actual - model) ** 2
                        if base_loss.sum() <= 0:
                            raise ValueError("Size-check baseline squared loss is zero")
                        excluded = _excluded_placebo_refits(run.fits)
                        record.update(
                            cw_p=1.0 if identical else cw["pvalue"],
                            dm_p=1.0 if identical else dm["pvalue"],
                            r2=1 - float(model_loss.sum() / base_loss.sum()),
                            mse_difference=float(np.mean(base_loss - model_loss)),
                            identical_forecasts=identical,
                            refits=run.fits.height,
                            placebo_excluded_refits=excluded,
                        )
                        attempt.metrics.update(record)
                        if record["cw_p"] is None or record["dm_p"] is None:
                            raise ValueError("Undefined size-check test statistic")
                except Exception as error:
                    record["error"] = type(error).__name__
                    failures += 1
                record["elapsed_seconds"] = time.perf_counter() - started
                rows.append(record)
                _save(rows, output, metadata)
                print(
                    f"Size check {series} k={k}: {rep + 1}/{config['size_check_reps']}; "
                    f"failed replications={failures}",
                    flush=True,
                )
    result = _save(rows, output, metadata)
    if failures:
        raise ValueError(f"Size check has {failures} failed replications; audit saved")
    return result
