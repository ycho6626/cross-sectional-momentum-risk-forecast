"""Pre-specified daily-origin robustness using saved, frozen fits (Methodology).

This module never fits a model or changes a trading decision. All learned
coefficients and structural paths come from the completed frozen holdout.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from scipy.special import expit
from threadpoolctl import threadpool_limits

from momenta.baseline_report import _h2_pair
from momenta.baseline_study import _benchmark_forecasts
from momenta.copula import CM_COLUMNS, OT_COLUMNS, Q_COLUMNS
from momenta.experiments import trial
from momenta.l2 import L2_COLUMNS
from momenta.l3_report import paired_predictions
from momenta.l3_transforms import carry
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO
from momenta.study_statistics import (
    auc,
    clark_west,
    diebold_mariano,
    holm_adjust,
    interval,
    qlike,
    stationary_indices,
)


def write_json(path: Path, value: Any) -> None:
    """Save the result and explicit provenance, independently of frozen artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")


def frozen_provenance(settings: dict) -> tuple[dict, dict]:
    """Pass the freeze check and record the frozen inputs of a post-unblinding sensitivity."""
    from momenta.frozen_holdout import FROZEN_PATH, check_freeze

    frozen = check_freeze(settings)
    root = Path(settings["data"]["data_root"])
    metadata = provenance(REPO, Path(settings["config_path"]), Manifest(root))
    metadata.update(
        frozen_config_hash=sha256_file(FROZEN_PATH),
        label="pre-specified robustness",
        executed_after_unblinding=True,
        input_hashes={str(FROZEN_PATH): sha256_file(FROZEN_PATH)},
    )
    return frozen, metadata


def saved_prediction(
    frame: pl.DataFrame, fit: dict, *, base_rate: float | None = None
) -> np.ndarray:
    """Apply intercept + clip((X−μ)/σ)β; use expit for y3.

    Excluded columns are absent from both the saved coefficients and moments.
    Intercept-only logistic fits retain the exact saved training base rate.
    """
    parameters = json.loads(fit["parameters_json"])
    if (
        frame.height
        and "refit" in fit
        and "date" in frame.columns
        and frame["date"].min() < fit["refit"]
    ):
        raise ValueError("A saved annual fit cannot serve an origin before its fit date")
    names = parameters["columns"]
    values = frame.select(names).to_numpy() if names else np.empty((frame.height, 0))
    mean, scale, coefficient = (
        np.asarray(parameters[key], dtype=float) for key in ("mean", "scale", "coefficient")
    )
    if any(value.shape != (len(names),) for value in (mean, scale, coefficient)):
        raise ValueError("Saved coefficients and standardization columns disagree")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or not np.isfinite(coefficient).all()
        or not np.isfinite(parameters["intercept"])
        or not np.isfinite(parameters["clip_sd"])
        or parameters["clip_sd"] <= 0
    ):
        raise ValueError("Saved prediction requires finite inputs and positive scales/clip")
    if fit["target"] == "y3" and not names:
        if base_rate is None or not np.isfinite(base_rate) or not 0 < base_rate < 1:
            raise ValueError("An intercept-only logistic fit needs its saved training base rate")
        return np.full(frame.height, base_rate)
    standardized = np.clip(
        (values - mean) / scale,
        -parameters["clip_sd"],
        parameters["clip_sd"],
    )
    linear = parameters["intercept"] + standardized @ coefficient
    return expit(linear) if fit["target"] == "y3" else linear


def month_resamples(
    dates: list[date],
    reps: int,
    block_months: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample consecutive calendar-month clusters, preserving every observed day.

    Return each day's month index and stationary-bootstrap month draws. Whole
    observed boundary months are retained, even when a scoring bound truncates
    their days. The number of days in a draw can vary with month length.
    """
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Daily bootstrap dates must be sorted and unique")
    months, membership = [], []
    for day in dates:
        first = day.replace(day=1)
        if not months or months[-1] != first:
            months.append(first)
        membership.append(len(months) - 1)
    return np.asarray(membership), stationary_indices(months, reps, block_months, seed)


def cluster_sums(values: np.ndarray, membership: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Sum all daily values in each drawn calendar month, including repeated months."""
    values = np.asarray(values, dtype=float)
    if values.shape != membership.shape or not np.isfinite(values).all():
        raise ValueError("Daily losses must be finite and align with their month membership")
    monthly = np.bincount(membership, weights=values, minlength=draws.shape[1])
    return monthly[draws].sum(axis=1)


def _auc_draws(y: np.ndarray, p: np.ndarray, membership: np.ndarray, draws: np.ndarray):
    """Weighted Mann–Whitney AUC after drawing whole months, with half credit for ties."""
    order = np.argsort(p, kind="stable")
    values = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    results = np.full(len(draws), np.nan)
    for i, months in enumerate(draws):
        weights = np.bincount(months, minlength=draws.shape[1])[membership][order]
        positive = np.add.reduceat(weights * y[order], starts)
        negative = np.add.reduceat(weights * (1 - y[order]), starts)
        denominator = positive.sum() * negative.sum()
        if denominator:
            results[i] = np.sum(positive * (np.cumsum(negative) - negative / 2)) / denominator
    return results


def daily_metrics(frame: pl.DataFrame, settings: dict, *, nested: bool = True) -> dict:
    """Daily losses with 21-lag HAC and month-cluster stationary intervals."""
    frame = frame.sort("origin")
    cfg = settings["evaluation"]
    y, prediction, baseline = (
        frame[name].to_numpy()
        for name in (
            "actual",
            "prediction",
            "base_prediction",
        )
    )
    if not all(np.isfinite(v).all() for v in (y, prediction, baseline)):
        raise ValueError("Daily metrics require finite paired forecasts")
    membership, draws = month_resamples(
        frame["origin"].to_list(),
        cfg["bootstrap_reps"],
        cfg["block_months"],
        cfg["seed"],
    )
    counts = cluster_sums(np.ones(len(y)), membership, draws)
    model_loss, base_loss = (y - prediction) ** 2, (y - baseline) ** 2
    model_draws = cluster_sums(model_loss, membership, draws)
    base_draws = cluster_sums(base_loss, membership, draws)
    confidence = cfg["confidence"]
    result = {
        "n": len(y),
        "calendar_months": draws.shape[1],
        "first_origin": str(frame["origin"].min()),
        "last_origin": str(frame["origin"].max()),
        "bootstrap_unit": "whole observed calendar month",
        "bootstrap_daily_count_min": int(counts.min()),
        "bootstrap_daily_count_max": int(counts.max()),
        "inference": "pre-specified robustness; daily overlap does not add independent months",
    }
    if frame["target"].unique().to_list() == ["y3"]:
        if not np.isin(y, [0, 1]).all() or np.any((prediction < 0) | (prediction > 1)):
            raise ValueError("Tail outcomes and probabilities are invalid")
        return result | {
            "brier": float(model_loss.mean()),
            "baseline_brier": float(base_loss.mean()),
            "auc": auc(y, prediction),
            "events": int(y.sum()),
            "brier_interval": interval(model_draws / counts, confidence),
            "auc_interval": interval(_auc_draws(y, prediction, membership, draws), confidence),
        }
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = 1 - model_draws / base_draws
    result.update(
        mse=float(model_loss.mean()),
        baseline_mse=float(base_loss.mean()),
        mse_improvement=float((base_loss - model_loss).mean()),
        r2_oos=float(1 - model_loss.sum() / base_loss.sum()) if base_loss.sum() > 0 else None,
        mse_interval=interval(model_draws / counts, confidence),
        mse_improvement_interval=interval((base_draws - model_draws) / counts, confidence),
        r2_oos_interval=interval(ratios, confidence),
        dm=diebold_mariano(y, baseline, prediction, cfg["nw_lags_daily"]),
    )
    if nested:
        result["cw"] = clark_west(y, baseline, prediction, cfg["nw_lags_daily"])
    if np.array_equal(prediction, baseline):
        for key in ("cw", "dm"):
            if key in result:
                result[key].update(pvalue=1.0, status="identical_nonrejection")
    if frame["target"].unique().to_list() == ["y1"]:
        result["qlike"] = float(qlike(y, prediction, frame["residual_variance"].to_numpy()).mean())
    return result


def monthly_parity(scored: pl.DataFrame, original: pl.DataFrame, tolerance: float) -> dict:
    """Require every frozen monthly prediction, benchmark, target and fit assignment."""
    keys = ["series", "target", "feature_set", "origin"]
    fields = ["refit", "target_start", "target_end", "actual", "prediction", "base_prediction"]
    matched = original.select(*keys, *fields).join(
        scored.select(*keys, *fields),
        on=keys,
        how="left",
        suffix="_daily",
        validate="1:1",
    )
    if matched["prediction_daily"].null_count():
        raise ValueError("Daily scoring omitted frozen monthly origins")
    for name in ("refit", "target_start", "target_end", "actual"):
        if not matched[name].equals(matched[f"{name}_daily"].rename(name)):
            raise ValueError(f"Daily scoring changed frozen monthly {name}")
    differences = {
        name: float((matched[name] - matched[f"{name}_daily"]).abs().max())
        for name in ("prediction", "base_prediction")
    }
    if any(value > tolerance for value in differences.values()):
        raise ValueError(f"Daily/monthly saved-fit prediction mismatch: {differences}")
    return {
        "rows": matched.height,
        "tolerance": tolerance,
        "max_absolute_error": differences,
        "passed": True,
    }


def _saved_features(
    frame: pl.DataFrame,
    fit: dict,
    stage: Path,
    copula: pl.DataFrame,
    metadata: dict,
) -> pl.DataFrame:
    """Apply the saved full-lineage paths/centroids; no estimation or smoothing."""
    structural = json.loads(fit["parameters_json"])["structural"]
    cutoff = fit["refit"]
    bdp = stage / f"derived/baseline/bdp_paths/{cutoff}.parquet"
    sidecar = bdp.with_suffix(".json")
    record = json.loads(sidecar.read_text())
    if (
        record["artifact_hash"] != sha256_file(bdp)
        or record["identity_hash"] != structural["bdp"]["identity_hash"]
        or record["artifact_hash"] != structural["bdp"]["artifact_hash"]
        or record["fit_time"] != str(cutoff)
        or date.fromisoformat(record["french_last_observation"]) >= cutoff
    ):
        raise ValueError("Saved BDP path does not match its frozen fit")
    metadata["input_hashes"].update({str(p): sha256_file(p) for p in (bdp, sidecar)})
    frame = frame.drop("bdp_ci", strict=False).join(
        pl.read_parquet(bdp).select("date", "series", "bdp_ci"),
        on=["date", "series"],
        how="left",
        maintain_order="left",
        validate="1:1",
    )
    folder = stage / "derived/final"
    if "ot" in structural:
        path = folder / f"centroids/full_{cutoff}.json"
        record = json.loads(path.read_text())
        if (
            record["fit_hash"] != structural["ot"]["fit_hash"]
            or record["lineage"] != "full"
            or record["fit_cutoff"] != str(cutoff)
        ):
            raise ValueError("Saved centroid fit differs from the frozen full lineage")
        distances = cdist(
            np.asarray(copula["embedding"].to_list()), np.asarray(record["centroids"])
        )
        frame = carry(
            frame,
            copula.select("date").with_columns(
                [pl.Series(name, distances[:, i]) for i, name in enumerate(OT_COLUMNS)],
            ),
        )
        metadata["input_hashes"][str(path)] = sha256_file(path)
    if "hmm" in structural:
        identity = structural["hmm"]["fit_hash"]
        choices = sorted((folder / "regimes").glob(f"{identity}_*.parquet"))
        if len(choices) != 1:
            raise ValueError("The frozen HMM probability path is absent or ambiguous")
        path, fit_path = choices[0], folder / "regimes" / f"{identity}.json"
        sidecar = path.with_suffix(".metadata.json")
        record, path_record = json.loads(fit_path.read_text()), json.loads(sidecar.read_text())
        if (
            record["fit_hash"] != identity
            or record["fit_time"] != str(cutoff)
            or record["series"] != fit["series"]
            or date.fromisoformat(record["training_target_end_max"]) >= cutoff
            or path_record["artifact_hash"] != sha256_file(path)
            or path_record["fit_artifact_hash"] != sha256_file(fit_path)
        ):
            raise ValueError("The frozen HMM probability path changed or violates fit timing")
        names = [f"{kind}_{i}" for kind in ("hmmf", "hmme") for i in range(record["states"] - 1)]
        frame = carry(frame, pl.read_parquet(path).select("date", *names))
        metadata["input_hashes"].update({str(p): sha256_file(p) for p in (path, sidecar, fit_path)})
    return frame


def _comparisons(forecasts: pl.DataFrame, settings: dict, best: dict) -> dict:
    """Report the same H2–H5 pairs with daily HAC/month-cluster uncertainty."""
    records = []
    for series in ("RAW", "RES"):
        pairs = [
            ("H3", None, "L1+L2", "L1"),
            ("H4", None, "L1+L2+CM+OT", "L1+L2+CM"),
            ("H4Q", None, "L1+L2+CM+Q+OT", "L1+L2+CM+Q"),
        ]
        for states in (settings["hmm"]["states"], settings["hmm"]["secondary_states"]):
            pairs += [
                ("H5a", states, f"best+HMMF{states}", best[series]),
                ("H5b", states, f"best+HMMF{states}", f"best+HMME{states}"),
            ]
        for hypothesis, states, model, benchmark in pairs:
            frame = paired_predictions(forecasts, series, model, benchmark)
            records.append(
                {
                    "hypothesis": hypothesis,
                    "states": states,
                    "series": series,
                    "model": model,
                    "benchmark": benchmark,
                    **daily_metrics(frame, settings, nested=hypothesis != "H5b"),
                }
            )
    for hypothesis, states in {(r["hypothesis"], r["states"]) for r in records}:
        members = [r for r in records if (r["hypothesis"], r["states"]) == (hypothesis, states)]
        if hypothesis != "H5b":
            for row, value in zip(
                members, holm_adjust([r["cw"]["pvalue"] for r in members]), strict=True
            ):
                row["cw_holm_pvalue"] = value
    h2 = []
    cfg = settings["evaluation"]
    for reduced in ("L1_without_M", "L1-own"):
        residual = _h2_pair(forecasts, "RES", reduced)
        for series in ("RAW_c", "RAW"):
            raw = _h2_pair(forecasts, series, reduced)
            paired = raw.join(residual, on="origin", how="inner", suffix="_res", validate="1:1")
            paired = paired.sort("origin")
            membership, draws = month_resamples(
                paired["origin"].to_list(), cfg["bootstrap_reps"], cfg["block_months"], cfg["seed"]
            )
            losses = [
                (paired[a] - paired[p]).to_numpy() ** 2
                for a, p in (
                    ("actual", "full"),
                    ("actual", "reduced"),
                    ("actual_res", "full_res"),
                    ("actual_res", "reduced_res"),
                )
            ]
            full_a, base_a, full_b, base_b = losses
            if min(base_a.sum(), base_b.sum()) <= 0:
                raise ValueError("Daily H2 requires positive reduced-model SSE")
            a, b = 1 - full_a.sum() / base_a.sum(), 1 - full_b.sum() / base_b.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                boot = -cluster_sums(full_a, membership, draws) / cluster_sums(
                    base_a, membership, draws
                ) + cluster_sums(full_b, membership, draws) / cluster_sums(
                    base_b, membership, draws
                )
            h2.append(
                {
                    "series_a": series,
                    "series_b": "RES",
                    "full_model": "L1",
                    "reduced_model": reduced,
                    "primary": series == "RAW_c" and reduced == "L1_without_M",
                    "n": paired.height,
                    "calendar_months": draws.shape[1],
                    "delta_a": float(a),
                    "delta_b": float(b),
                    "contrast": float(a - b),
                    "interval": interval(boot, cfg["confidence"]),
                    "first_origin": str(paired["origin"].min()),
                    "last_origin": str(paired["origin"].max()),
                }
            )
    return {"nested_comparisons": records, "h2": h2}


def run_daily(settings: dict, *, stage: Path, output: Path) -> dict:
    """Score every holdout trading day with saved annual fits; verify all month-ends."""
    frozen, metadata = frozen_provenance(settings)
    metadata["source_hashes"] = {str(Path(__file__)): sha256_file(Path(__file__))}
    root = Path(settings["data"]["data_root"])
    if output.resolve() == stage.resolve() or not output.resolve().is_relative_to(
        (root / "derived/robustness").resolve()
    ):
        raise ValueError("Daily robustness output must be isolated under derived/robustness")
    first, last = [
        date.fromisoformat(frozen["bounds"][key])
        for key in (
            "holdout_first_origin",
            "holdout_last_origin",
        )
    ]
    folder = stage / "derived/final"
    inputs = {
        "features": stage / "derived/baseline/features.parquet",
        "targets": stage / "derived/baseline/targets.parquet",
        "l2": stage / "derived/l2/features_monthly.parquet",
        "copula": folder / "copula_monthly.parquet",
        "fits": folder / "fits.parquet",
        "forecasts": folder / "forecasts.parquet",
        "calendar": stage / "curated/trading_calendar.parquet",
    }
    with trial(
        settings,
        experiment="robustness-daily",
        kind="diagnostic",
        series="ALL",
        sample=f"{first}:{last}",
        specification={
            "frequency": "daily",
            "fit_policy": "apply_saved_frozen_annual_fits",
            "nw_lags": settings["evaluation"]["nw_lags_daily"],
            "bootstrap": settings["evaluation"],
            "trading_frequency": "unchanged_monthly",
        },
    ) as attempt:
        metadata["input_hashes"].update({str(p): sha256_file(p) for p in inputs.values()})
        calendar = (
            pl.read_parquet(inputs["calendar"])
            .filter(pl.col("date").is_between(first, last))["date"]
            .sort()
            .to_list()
        )
        features = pl.read_parquet(inputs["features"]).filter(
            pl.col("date").is_between(first, last)
        )
        targets = pl.read_parquet(inputs["targets"]).filter(
            pl.col("origin").is_between(first, last),
            pl.col("construction") == settings["portfolios"]["primary"],
        )
        features = carry(features, pl.read_parquet(inputs["l2"]).select("date", *L2_COLUMNS))
        copula = pl.read_parquet(inputs["copula"])
        features = carry(features, copula.select("date", *CM_COLUMNS, *Q_COLUMNS))
        joined = features.join(
            targets.drop("construction"),
            left_on=["series", "date"],
            right_on=["series", "origin"],
            validate="1:1",
        ).sort("date")
        fits, original = pl.read_parquet(inputs["fits"]), pl.read_parquet(inputs["forecasts"])
        groups = ["series", "target", "feature_set"]
        frames, coverage = [], []
        with threadpool_limits(limits=settings["walkforward"]["blas_threads"]):
            for labels, selected in fits.partition_by(groups, as_dict=True).items():
                series, target, feature_set = labels
                selected = selected.sort("refit")
                parts, omitted = [], 0
                for i, fit in enumerate(selected.iter_rows(named=True)):
                    cutoff = fit["refit"]
                    if fit["training_target_end_max"] >= cutoff:
                        raise ValueError("Saved model training ends on/after its fit date")
                    end = selected["refit"][i + 1] if i + 1 < selected.height else date.max
                    frame = joined.filter(
                        pl.col("series") == series, pl.col("date") >= cutoff, pl.col("date") < end
                    )
                    frame = _saved_features(frame, fit, stage, copula, metadata)
                    params = json.loads(fit["parameters_json"])
                    eligible = np.isfinite(
                        frame.select(params["candidate_columns"]).to_numpy()
                    ).all(1)
                    eligible &= np.isfinite(frame["y2" if target == "y3" else target].to_numpy())
                    omitted += int((~eligible).sum())
                    frame = frame.filter(pl.Series(eligible))
                    old = original.filter(
                        pl.col("series") == series,
                        pl.col("target") == target,
                        pl.col("feature_set") == feature_set,
                        pl.col("refit") == cutoff,
                    )
                    base = None
                    if target != "y1":
                        values = old["base_prediction"].unique().to_list()
                        if len(values) != 1:
                            raise ValueError("Saved null forecast changes within an annual fit")
                        base = values[0]
                    predictions = saved_prediction(frame, fit, base_rate=base)
                    actual = (
                        (frame["y2"] < fit["tail_threshold"]).cast(pl.Float64)
                        if target == "y3"
                        else frame[target]
                    )
                    parts.append(
                        frame.select(
                            pl.col("date").alias("origin"), "target_start", "target_end", "y2"
                        ).with_columns(
                            pl.lit(series).alias("series"),
                            pl.lit(target).alias("target"),
                            pl.lit(feature_set).alias("feature_set"),
                            pl.lit(cutoff).alias("refit"),
                            actual.alias("actual"),
                            pl.Series("prediction", predictions),
                            pl.lit(base).cast(pl.Float64).alias("base_prediction"),
                            pl.lit(fit["residual_variance"])
                            .cast(pl.Float64)
                            .alias("residual_variance"),
                            pl.lit(fit["tail_threshold"]).cast(pl.Float64).alias("tail_threshold"),
                        )
                    )
                frame = pl.concat(parts, how="diagonal_relaxed")
                if frame["origin"].n_unique() != frame.height:
                    raise ValueError("Saved annual fits produced duplicate daily origins")
                frames.append(frame)
                coverage.append(
                    dict(zip(groups, labels, strict=True))
                    | {
                        "daily_origins": frame.height,
                        "omitted_incomplete": omitted,
                        "annual_fits": selected.height,
                        "calendar_origins": len(calendar),
                        "omitted_feature_target_days": len(calendar) - frame.height - omitted,
                    }
                )
        scored = _benchmark_forecasts(pl.concat(frames, how="diagonal_relaxed"))
        scored = scored.sort(*groups, "origin")
        parity = monthly_parity(scored, original, settings["l2"]["numerical_tolerance"])
        metrics = []
        for labels, frame in scored.partition_by(groups, as_dict=True).items():
            metrics.append(dict(zip(groups, labels, strict=True)) | daily_metrics(frame, settings))
        comparisons = _comparisons(scored, settings, frozen["best_models"])
        output.mkdir(parents=True, exist_ok=True)
        scored.write_parquet(
            output / "forecasts.parquet", compression=settings["storage"]["compression"]
        )
        write_json(output / "metrics.json", metrics)
        write_json(output / "comparisons.json", comparisons)
        summary = {
            "complete": True,
            "stage": str(stage),
            "first_origin": str(first),
            "last_origin": str(last),
            "models": len(metrics),
            "forecast_rows": scored.height,
            "monthly_parity": parity,
            "coverage": coverage,
            "best_models": frozen["best_models"],
            "daily_strategy": (
                "N/A: trading remains monthly; frozen monthly strategy results are unchanged"
            ),
            "h3": [r for r in comparisons["nested_comparisons"] if r["hypothesis"] == "H3"],
            "h2": comparisons["h2"],
            "provenance": metadata,
            "artifacts": {
                str(output / name): sha256_file(output / name)
                for name in (
                    "forecasts.parquet",
                    "metrics.json",
                    "comparisons.json",
                )
            },
        }
        write_json(output / "summary.json", summary)
        attempt.metrics.update(
            models=len(metrics),
            daily_origins=scored["origin"].n_unique(),
            forecasts=scored.height,
            monthly_parity=parity,
        )
        return summary
