"""The L2 study: cross-sectional features beyond the pinned L1 baseline (Methodology).

L2 enters as a fourth penalty block after HAR, CORE and M. The earlier blocks
keep their baseline penalties, an excluded L2 block reproduces L1 exactly, and
the redundancy audit regresses each L2 feature on L1 at the scored origins.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.experiments import read_trials, trial
from momenta.features import FEATURE_BLOCKS, FEATURE_SETS
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO
from momenta.walkforward import FeatureTransform, assert_design_origins, walk_forward


def carry_monthly_features(daily: pl.DataFrame, monthly: pl.DataFrame) -> pl.DataFrame:
    """Carry the latest monthly formation at or before each daily origin."""
    if monthly["date"].n_unique() != monthly.height:
        raise ValueError("Duplicate L2 month-end keys")
    if daily.select(pl.struct("series", "date").n_unique()).item() != daily.height:
        raise ValueError("Duplicate L1 daily feature keys")
    columns = set(daily.columns) & (set(monthly.columns) - {"date"})
    if columns:
        raise ValueError(f"L2 columns collide with existing features: {sorted(columns)}")
    return (
        daily.sort("date")
        .join_asof(
            monthly.rename({"date": "l2_formation"}).sort("l2_formation"),
            left_on="date",
            right_on="l2_formation",
            strategy="backward",
        )
        .sort("series", "date")
    )


def match_baseline(forecasts: pl.DataFrame, baseline: pl.DataFrame) -> pl.DataFrame:
    """Use the pinned L1 forecast on exactly the same series/origins."""
    keys = ["series", "origin"]
    base = baseline.filter(pl.col("target") == "y1", pl.col("feature_set") == "L1")
    if forecasts.select(pl.struct(keys).n_unique()).item() != forecasts.height:
        raise ValueError("Duplicate L2 forecast keys")
    base = base.select(
        *keys,
        pl.col("prediction").alias("base_prediction"),
        pl.col("actual").alias("baseline_actual"),
        pl.col("refit").alias("baseline_refit"),
    )
    pair = forecasts.drop("base_prediction", strict=False).join(
        base,
        on=keys,
        how="left",
        validate="1:1",
    )
    if pair["base_prediction"].is_null().any():
        raise ValueError("Missing pinned L1 prediction on a scored L2 origin")
    if not np.array_equal(pair["actual"].to_numpy(), pair["baseline_actual"].to_numpy()):
        raise ValueError("L2 and pinned L1 disagree on realized targets")
    if not pair["refit"].equals(pair["baseline_refit"]):
        raise ValueError("L2 and pinned L1 use different refit dates")
    return pair.drop("baseline_actual", "baseline_refit").sort("series", "origin")


def prefix_audit(
    fits: pl.DataFrame,
    baseline_fits: pl.DataFrame,
    forecasts: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Check unchanged earlier λs and excluded-block forecast equality."""
    base = baseline_fits.filter(pl.col("target") == "y1", pl.col("feature_set") == "L1")
    by_key = {(row["series"], row["refit"]): row for row in base.iter_rows(named=True)}
    if len(by_key) != base.height:
        raise ValueError("Duplicate pinned L1 refit keys")
    rows = []
    for fit in fits.sort("series", "refit").iter_rows(named=True):
        key = fit["series"], fit["refit"]
        if key not in by_key:
            raise ValueError(f"Missing pinned L1 refit: {key}")
        previous = by_key[key]
        selected = json.loads(fit["block_lambdas_json"])
        earlier = json.loads(previous["block_lambdas_json"])
        if any(selected[name] != earlier[name] for name in ("HAR", "CORE", "M")):
            raise ValueError(f"Earlier block penalties changed for {key}")
        for field in (
            "training_count",
            "inner_count",
            "validation_count",
            "training_target_end_max",
            "inner_target_end_max",
            "inner_fit_cutoff",
        ):
            if fit[field] != previous[field]:
                raise ValueError(f"L2 changes pinned L1 training support: {key}, {field}")
        group = forecasts.filter(pl.col("series") == key[0], pl.col("refit") == key[1])
        excluded = selected["L2"] == "exclude"
        difference = (group["prediction"] - group["base_prediction"]).abs()
        maximum = float(difference.max()) if len(difference) else 0.0
        exact = bool((difference == 0).all())
        if excluded and not exact:
            raise ValueError(f"Excluded L2 does not reproduce pinned L1: {key}")
        rows.append(
            {
                "series": key[0],
                "refit": str(key[1]),
                "origins": group.height,
                "earlier_penalties_match": True,
                "training_support_matches": True,
                "l2_penalty": selected["L2"],
                "l2_excluded": excluded,
                "excluded_prefix_exact": exact if excluded else None,
                "excluded_prefix_max_difference": maximum if excluded else None,
            }
        )
    return rows


def pinned_structural_transform(
    settings: dict[str, Any],
    enabled: bool,
    pinned: dict[str, str],
) -> FeatureTransform | None:
    """Reuse only the BDP cutoff fits the baseline recorded; never create or rewrite one."""
    if not enabled:
        return None
    from momenta.baseline_study import _structural_transform

    underlying = _structural_transform(settings, True)
    directory = Path(settings["data"]["data_root"]) / "derived/baseline/bdp_paths"
    checked: set[date] = set()

    def transform(frame: pl.DataFrame, cutoff: date):
        if cutoff not in checked:
            for path in (directory / f"{cutoff}.parquet", directory / f"{cutoff}.json"):
                if (
                    str(path) not in pinned
                    or not path.is_file()
                    or sha256_file(path) != pinned[str(path)]
                ):
                    raise ValueError(f"L2 requested an unrecorded BDP cutoff: {cutoff}")
            checked.add(cutoff)
        result, record = underlying(frame, cutoff)
        if record["fit_time"] != str(cutoff):
            raise ValueError("BDP fit time disagrees with the requested cutoff")
        return result, record

    return transform


def dated_redundancy_design(
    features: pl.DataFrame,
    forecasts: pl.DataFrame,
    transform: FeatureTransform | None,
) -> pl.DataFrame:
    """Use each scored origin's already-available January BDP fit for the audit."""
    decisions = forecasts.select("series", pl.col("origin").alias("date"), "refit").unique()
    if decisions.filter(pl.col("refit") > pl.col("date")).height:
        raise ValueError("Redundancy design contains a future fit")
    selected = features.join(decisions, on=["series", "date"], how="inner", validate="1:1")
    if selected.height != decisions.height:
        raise ValueError("Missing features for redundancy design")
    frames = []
    for (cutoff,), frame in selected.sort("refit").partition_by("refit", as_dict=True).items():
        if transform:
            changed, _ = transform(frame, cutoff)
            if not changed.select("series", "date").equals(frame.select("series", "date")):
                raise ValueError("Redundancy transform changed origin keys")
            frame = changed
        frames.append(frame)
    return pl.concat(frames).sort("series", "date")


def _feature_audit(
    monthly: pl.DataFrame,
    augmented: pl.DataFrame,
    feature_metadata: dict[str, Any],
    forecasts: pl.DataFrame,
    l2_columns: Sequence[str],
) -> dict[str, Any]:
    availability = [
        {"series": "shared", "feature": name, "first_origin": first}
        for name, first in feature_metadata["feature_availability"].items()
    ]
    availability.append(
        {
            "series": "shared",
            "feature": "L2_complete",
            "first_origin": feature_metadata["first_complete_origin"],
        }
    )
    for (series,), frame in augmented.partition_by("series", as_dict=True).items():
        complete = frame.filter(
            pl.all_horizontal(
                [pl.col(name).is_finite() for name in (*FEATURE_SETS["L1"], *l2_columns)]
            )
        )
        availability.append(
            {
                "series": series,
                "feature": "L1+L2_complete",
                "first_origin": str(complete["date"].min()) if complete.height else None,
            }
        )
    coverage = []
    for name in monthly.columns:
        if name in {"date", "breadth_m", *l2_columns}:
            continue
        values = monthly[name].drop_nulls()
        if values.dtype.is_numeric():
            coverage.append(
                {
                    "sample": "built_month_ends",
                    "field": name,
                    "origins": len(values),
                    "minimum": values.min(),
                    "mean": values.mean(),
                    "maximum": values.max(),
                }
            )
    scored_dates = forecasts["origin"].unique()
    breadth = []
    for sample, frame in (
        ("built_month_ends", monthly),
        ("scored_month_ends", monthly.filter(pl.col("date").is_in(scored_dates.implode()))),
    ):
        values = frame["breadth_m"].drop_nulls()
        breadth.append(
            {
                "sample": sample,
                "origins": len(values),
                "minimum": values.min(),
                "mean": values.mean(),
                "maximum": values.max(),
            }
        )
    return {
        "feature_availability": availability,
        "feature_coverage": coverage,
        "breadth": breadth,
        "producer_coverage": feature_metadata["coverage"],
    }


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )


def run_l2_study(settings: dict[str, Any], through: date | None = None) -> dict[str, Any]:
    """Run thin RAW or full RAW/RES H3 without changing the baseline economics."""
    from momenta.baseline_report import load_summary
    from momenta.l2 import L2_COLUMNS
    from momenta.l2_report import evaluate_l2

    last = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    bound = through or last
    full = bound == last
    series_names = ("RAW", "RES") if full else ("RAW",)
    blocks = {name: list(FEATURE_BLOCKS[name]) for name in ("HAR", "CORE", "M")}
    blocks["L2"] = list(L2_COLUMNS)
    columns = [name for group in blocks.values() for name in group]
    specification = {
        "hypothesis": "H3",
        "target": "y1",
        "feature_set": "L1+L2",
        "blocks": blocks,
        "walkforward": settings["walkforward"],
        "targets": settings["targets"],
        "features": settings["features"],
        "bdp": settings["bdp"],
        "l2": settings["l2"],
    }
    with trial(
        settings,
        experiment="h3",
        kind="diagnostic",
        series="requested",
        sample=f"design through {bound}",
        specification=specification,
    ) as outer:
        assert_design_origins([bound], settings)
        baseline_summary = load_summary(settings)
        if baseline_summary is None or not (
            baseline_summary["size_check_complete"] and baseline_summary["size_check_bias_passed"]
        ):
            raise ValueError("L2 requires a complete baseline study with a passed size check")
        root = Path(settings["data"]["data_root"])
        base, output = root / "derived/baseline", root / "derived/l2"
        monthly_path = output / "features_monthly.parquet"
        sidecar = monthly_path.with_suffix(".metadata.json")
        feature_metadata = json.loads(sidecar.read_text())
        if feature_metadata["config_hash"] != sha256_file(Path(settings["config_path"])):
            raise ValueError("L2 features use an obsolete configuration")
        if date.fromisoformat(feature_metadata["through"]) < bound:
            raise ValueError("L2 features end before the requested origin")
        feature_pins = {
            **feature_metadata["input_hashes"],
            **feature_metadata["source_hashes"],
            str(monthly_path): feature_metadata["artifact_hash"],
        }
        for source, digest in feature_pins.items():
            if not Path(source).is_file() or sha256_file(Path(source)) != digest:
                raise ValueError(f"Changed L2 feature input: {source}")
        monthly = pl.read_parquet(monthly_path).filter(pl.col("date") <= bound)
        baseline_features = pl.read_parquet(base / "features.parquet").filter(
            pl.col("date") <= bound
        )
        # Shared diagnostics may use names already present in L1; keep only new L2 fields here.
        augmented = carry_monthly_features(
            baseline_features,
            monthly.select("date", *L2_COLUMNS, "breadth_m"),
        )
        targets = pl.read_parquet(base / "targets.parquet").filter(pl.col("origin") <= bound)
        calendar_path = root / "curated/trading_calendar.parquet"
        calendar = pl.read_parquet(calendar_path)["date"].to_list()
        baseline_forecasts = pl.read_parquet(base / "forecasts.parquet").filter(
            pl.col("origin") <= bound
        )
        baseline_fits = pl.read_parquet(base / "fits.parquet")
        decision = json.loads((base / "bdp_decision.json").read_text())
        transform = pinned_structural_transform(
            settings,
            decision["full_model"],
            baseline_summary["provenance"]["input_hashes"],
        )
        inputs = [
            base / name
            for name in (
                "features.parquet",
                "features.metadata.json",
                "targets.parquet",
                "forecasts.parquet",
                "fits.parquet",
                "size_check.parquet",
                "size_check.metadata.json",
                "summary.json",
                "bdp_decision.json",
            )
        ]
        inputs.extend(
            [
                calendar_path,
                monthly_path,
                sidecar,
                Path(settings["config_path"]),
                REPO / "config/design_trial_population.json",
                root / "raw/manifest.json",
            ]
        )
        metadata = {
            **provenance(REPO, Path(settings["config_path"]), Manifest(root)),
            "mode": "full" if full else "thin",
            "through": str(bound),
            "input_hashes": {str(path): sha256_file(path) for path in inputs},
        }
        metadata["input_hashes"].update(feature_pins)
        # The baseline summary already pins these paths; preserve their identities here.
        metadata["input_hashes"].update(
            {
                name: digest
                for name, digest in baseline_summary["provenance"]["input_hashes"].items()
                if "/bdp_paths/" in name
            }
        )
        run_directory = output / "runs" / outer.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        frames, fits, run_ids = [], [], []
        for series in series_names:
            with trial(
                settings,
                experiment=f"{series}_y1_L1_L2",
                kind="forecast",
                series=series,
                sample=f"month-end origins through {bound}",
                specification=specification,
            ) as attempt:
                run_ids.append(attempt.run_id)
                print(f"H3 {series}: fit L1+L2 through {bound}", flush=True)
                result = walk_forward(
                    augmented,
                    targets,
                    calendar,
                    settings,
                    series=series,
                    target="y1",
                    feature_set="L1+L2",
                    columns=columns,
                    through=bound,
                    blocks=blocks,
                    transform=transform,
                )
                frame = match_baseline(result.forecasts, baseline_forecasts).with_columns(
                    pl.lit(attempt.run_id).alias("run_id"),
                    pl.lit(attempt.strategy_id).alias("strategy_id"),
                )
                expected = baseline_forecasts.filter(
                    pl.col("series") == series,
                    pl.col("target") == "y1",
                    pl.col("feature_set") == "L1",
                )["origin"].sort()
                if not frame["origin"].sort().equals(expected):
                    raise ValueError(f"L2 changes the pinned L1 evaluation sample for {series}")
                frame.write_parquet(run_directory / f"{series}_forecasts.parquet")
                result.fits.write_parquet(run_directory / f"{series}_fits.parquet")
                prefix_audit(result.fits, baseline_fits, frame)
                frames.append(frame)
                fits.append(result.fits)
                attempt.metrics.update(
                    origins=frame.height,
                    annual_fits=result.fits.height,
                    mse=float(((frame["actual"] - frame["prediction"]) ** 2).mean()),
                    parent_run_id=outer.run_id,
                )
        forecasts, fitted = pl.concat(frames), pl.concat(fits)
        with trial(
            settings,
            experiment="l2-redundancy",
            kind="diagnostic",
            series="RAW/RES" if full else "RAW",
            sample=f"design month-ends through {bound}",
            specification={
                "diagnostic": "L2_on_L1_OLS_intercept_unclipped",
                "l1": list(FEATURE_SETS["L1"]),
                "l2": list(L2_COLUMNS),
                "bdp": settings["bdp"],
            },
        ) as diagnostic:
            run_ids.append(diagnostic.run_id)
            design = dated_redundancy_design(augmented, forecasts, transform)
            diagnostic.metrics.update(origins=design.height, parent_run_id=outer.run_id)
        for name, frame in (
            ("forecasts", forecasts),
            ("fits", fitted),
            ("redundancy_design", design),
        ):
            frame.write_parquet(output / f"{name}.parquet")
        audit = {
            **_feature_audit(monthly, augmented, feature_metadata, forecasts, L2_COLUMNS),
            "prefix": prefix_audit(fitted, baseline_fits, forecasts),
            "trial_run_ids": run_ids,
            "parent_run_id": outer.run_id,
            "bdp_method": decision["method"],
        }
        _json(output / "_metadata.json", metadata)
        _json(output / "_audit.json", audit)
        _json(
            output / "trial_snapshot.json",
            [row for row in read_trials() if row["run_id"] in run_ids],
        )
        summary = evaluate_l2(
            settings,
            output,
            metadata,
            audit,
            baseline_summary,
            l1_columns=FEATURE_SETS["L1"],
            l2_columns=L2_COLUMNS,
        )
        outer.metrics.update(forecast_rows=forecasts.height, complete=summary["complete"])
        return summary
