"""Design-period forecast and strategy evaluation for the baseline study (Methodology).

The tables pair every candidate with its registered benchmark on identical
origins. Deflated Sharpe ratios count every design-period trading
specification in the recorded population ``config/design_trial_population.json``;
the current run's own accounts replace their recorded estimates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.manifest import sha256_file
from momenta.portfolio_report import return_summary
from momenta.settings import REPO
from momenta.study_statistics import (
    deflated_sharpe,
    forecast_metrics,
    h2_contrast,
    holm_adjust,
    interval,
    lo_sharpe,
    lo_sharpe_draws,
    stationary_indices,
)

POPULATION_PATH = REPO / "config/design_trial_population.json"
_INPUT_FILES = (
    "features.parquet",
    "targets.parquet",
    "forecasts.parquet",
    "fits.parquet",
    "training_predictions.parquet",
    "target_overlaps.parquet",
    "strategy_daily.parquet",
    "strategy_monthly.parquet",
    "strategy_rebalances.parquet",
    "overlays.parquet",
    "dm_fits.parquet",
    "bdp_strategy_moments.parquet",
    "size_check.parquet",
    "size_check.metadata.json",
    "bdp_decision.json",
    "build_metadata.json",
    "_metadata.json",
    "_audit.json",
)
_SOURCE_FILES = (
    "accounting.py",
    "signals.py",
    "portfolios.py",
    "portfolio_report.py",
    "experiments.py",
    "targets.py",
    "features.py",
    "walkforward.py",
    "strategies.py",
    "strategy_study.py",
    "bdp.py",
    "bdp_horizon.py",
    "baseline_study.py",
    "size_check.py",
    "study_statistics.py",
    "baseline_report.py",
)


def code_hash() -> str:
    """Pin the baseline computation modules and their portfolio-layer dependencies."""
    digest = hashlib.sha256()
    for name in sorted(_SOURCE_FILES):
        path = REPO / "src/momenta" / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _assert_design(frame: pl.DataFrame, settings: dict[str, Any]) -> None:
    """Refuse holdout origins, including an accidental mixed-sample input."""
    if frame.is_empty():
        raise ValueError("The design forecast table is empty")
    last = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    if frame["origin"].max() > last:
        raise ValueError("Holdout origins cannot enter a design-period report")
    keys = ["series", "feature_set", "target", "origin"]
    if frame.select(pl.struct(keys).n_unique()).item() != frame.height:
        raise ValueError("Duplicate forecast keys")
    if "refit" in frame.columns and frame.filter(pl.col("refit") > pl.col("origin")).height:
        raise ValueError("A forecast was fitted after its decision origin")


def forecast_table(
    forecasts: pl.DataFrame,
    settings: dict[str, Any],
    *,
    size_check: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Produce paired loss tables; size inflation downgrades CW evidence."""
    _assert_design(forecasts, settings)
    config = settings["evaluation"]
    records = []
    for key, frame in (
        forecasts.sort("origin")
        .partition_by(["series", "feature_set", "target"], as_dict=True)
        .items()
    ):
        series, feature_set, target = key
        own_size = [row for row in (size_check or []) if row["series"] == series]
        size_complete = {row["k"] for row in own_size} == set(config["size_check_blocks"]) and all(
            row.get("complete", False) for row in own_size
        )
        cw_inflated = any(
            row.get("cw_descriptive", False)
            or (
                row.get("cw_rejection_rate") is not None
                and row["cw_rejection_rate"] > config["size_check_max"]
            )
            for row in own_size
        )
        bias_exceeded = any(row.get("bias_exceeded", False) for row in own_size)
        metrics = forecast_metrics(
            frame,
            target=target,
            reps=config["bootstrap_reps"],
            block_months=config["block_months"],
            seed=config["seed"],
            nw_lags=config["nw_lags_monthly"],
            confidence=config["confidence"],
        )
        record = {"series": series, "feature_set": feature_set, "target": target}
        for name, value in metrics.items():
            if isinstance(value, dict):
                record.update({f"{name}_{subkey}": item for subkey, item in value.items()})
            else:
                record[name] = value
        record["cw_inference"] = (
            "size_check_unavailable"
            if series not in config["size_check_series"]
            else "invalid_placebo_bias"
            if bias_exceeded
            else "descriptive_size_failure"
            if cw_inflated
            else "size_check_pending"
            if not size_complete
            else "design_secondary"
        )
        record["dm_inference"] = "descriptive"
        record["cw_holm_pvalue"] = None
        records.append(record)
    families: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["series"] in {"RAW", "RES"} and record["target"] in {"y1", "y2"}:
            families.setdefault((record["target"], record["feature_set"]), {})[record["series"]] = (
                record
            )
    for family in families.values():
        if set(family) == {"RAW", "RES"}:
            members = [family[series] for series in ("RAW", "RES")]
            adjusted = holm_adjust([member.get("cw_pvalue") for member in members])
            for member, value in zip(members, adjusted, strict=True):
                member["cw_holm_pvalue"] = value
    return sorted(records, key=lambda row: (row["series"], row["target"], row["feature_set"]))


def _h2_pair(forecasts: pl.DataFrame, series: str, reduced: str) -> pl.DataFrame:
    selected = forecasts.filter((pl.col("target") == "y1") & (pl.col("series") == series))
    full = selected.filter(pl.col("feature_set") == "L1").select(
        "origin", "actual", pl.col("prediction").alias("full")
    )
    base = selected.filter(pl.col("feature_set") == reduced).select(
        "origin", pl.col("actual").alias("reduced_actual"), pl.col("prediction").alias("reduced")
    )
    paired = full.join(base, on="origin", how="inner", validate="1:1")
    if paired.is_empty():
        raise ValueError(f"No common H2 forecasts for {series}, {reduced}")
    if not np.array_equal(paired["actual"].to_numpy(), paired["reduced_actual"].to_numpy()):
        raise ValueError("H2 models disagree on the realized target")
    return paired.drop("reduced_actual")


def h2_table(forecasts: pl.DataFrame, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Paired calendar contrasts for market-block and whole-L1 additions."""
    _assert_design(forecasts, settings)
    config = settings["evaluation"]
    records = []
    for reduced in ("L1_without_M", "L1-own"):
        residual = _h2_pair(forecasts, "RES", reduced)
        for series in ("RAW_c", "RAW"):
            raw = _h2_pair(forecasts, series, reduced)
            pair = raw.join(residual, on="origin", how="inner", suffix="_res", validate="1:1")
            pair = pair.sort("origin")
            indices = stationary_indices(
                pair["origin"].to_list(),
                config["bootstrap_reps"],
                config["block_months"],
                config["seed"],
            )
            result = h2_contrast(
                *(
                    pair[column].to_numpy()
                    for column in (
                        "actual",
                        "full",
                        "reduced",
                        "actual_res",
                        "full_res",
                        "reduced_res",
                    )
                ),
                indices,
                confidence=config["confidence"],
            )
            confidence = result.pop("interval")
            records.append(
                {
                    "series_a": series,
                    "series_b": "RES",
                    "full_model": "L1",
                    "reduced_model": reduced,
                    "primary": series == "RAW_c" and reduced == "L1_without_M",
                    "first_origin": str(pair["origin"].min()),
                    "last_origin": str(pair["origin"].max()),
                    "excluded_a": raw.height - pair.height,
                    "excluded_b": residual.height - pair.height,
                    **result,
                    **{f"contrast_{key}": value for key, value in confidence.items()},
                }
            )
    return records


def account_sharpes(monthly: pl.DataFrame) -> dict[str, Any]:
    """Monthly SR=mean/SD(ddof=1) per account on one consecutive common month path.

    Accounts without two finite returns or with zero dispersion have no estimate.
    """
    if monthly.is_empty() or monthly["date"].is_null().any():
        raise ValueError("Strategy population requires dated monthly accounts")
    if monthly.select(pl.struct("strategy_id", "date").n_unique()).item() != monthly.height:
        raise ValueError("Duplicate strategy/month observations in the Sharpe population")
    dates = monthly["date"].unique().sort()
    periods = np.array([day.year * 12 + day.month for day in dates.to_list()])
    if len(periods) > 1 and not np.all(np.diff(periods) == 1):
        raise ValueError("Strategy monthly paths must be consecutive calendar months")
    accounts = monthly.sort("strategy_id", "date").partition_by("strategy_id")
    if any(not frame["date"].equals(dates) for frame in accounts):
        raise ValueError("Strategy accounts have misaligned monthly date paths")
    estimates = {}
    for frame in accounts:
        values = frame["monthly_return"].to_numpy()
        if len(values) < 2 or not np.isfinite(values).all():
            continue
        standard_deviation = float(values.std(ddof=1))
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            continue
        sharpe = float(values.mean() / standard_deviation)
        if np.isfinite(sharpe):
            estimates[frame["strategy_id"][0]] = sharpe
    return {
        "monthly_sharpes": estimates,
        "account_ids": sorted({frame["strategy_id"][0] for frame in accounts}),
        "first_month": str(monthly["date"].min()),
        "last_month": str(monthly["date"].max()),
    }


def recorded_population(path: Path | None = None) -> dict[str, Any]:
    """Read the recorded design-period trading population and check its structure.

    It counts every distinct design-period trading specification, including
    earlier experiments a rebuild does not rerun, with the latest monthly Sharpe
    estimate on the full design sample for each specification that has one.
    """
    population = json.loads((path or POPULATION_PATH).read_text())
    trading = population["trading_ids"]
    if (
        not trading
        or trading != sorted(set(trading))
        or population["distinct_trading_ids"] != len(trading)
        or not set(population["monthly_sharpes"]).issubset(trading)
        or not all(math.isfinite(float(value)) for value in population["monthly_sharpes"].values())
    ):
        raise ValueError("Invalid recorded design trial population")
    return population


def design_population(monthly: pl.DataFrame, path: Path | None = None) -> dict[str, Any]:
    """Keep the recorded DSR multiplicity and replace this run's account estimates.

    Every current account must be a recorded design specification and use the
    recorded monthly sample; recorded estimates of other specifications remain.
    """
    recorded = recorded_population(path)
    current = account_sharpes(monthly)
    if not set(current["account_ids"]).issubset(recorded["trading_ids"]):
        raise ValueError("A design account is absent from the recorded trading population")
    if any(current[key] != recorded[key] for key in ("first_month", "last_month")):
        raise ValueError("Design accounts use a different monthly sample from the population")
    estimates = dict(recorded["monthly_sharpes"])
    for identity in current["account_ids"]:
        estimates.pop(identity, None)
    estimates.update(current["monthly_sharpes"])
    population = {
        "distinct_trading_ids": recorded["distinct_trading_ids"],
        "trading_ids": recorded["trading_ids"],
        "monthly_sharpes": dict(sorted(estimates.items())),
        "first_month": recorded["first_month"],
        "last_month": recorded["last_month"],
        "current_accounts": len(current["account_ids"]),
        "current_estimates_match_record": all(
            recorded["monthly_sharpes"].get(identity) == value
            for identity, value in current["monthly_sharpes"].items()
        ),
    }
    population["fingerprint"] = hashlib.sha256(
        json.dumps(population, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return population


def strategy_table(
    monthly: pl.DataFrame,
    daily: pl.DataFrame,
    rebalances: pl.DataFrame,
    settings: dict[str, Any],
    population: dict[str, Any],
) -> list[dict[str, Any]]:
    """Exact ledger returns plus monthly-tail and serial-dependence metrics."""
    periods = settings["replication"]["months_per_year"]
    alpha = settings["strategy"]["cvar_alpha"]
    groups = ["series", "strategy", "strategy_id", "cost_bp"]
    results = []
    for key, frame in monthly.partition_by(groups, as_dict=True).items():
        frame = frame.sort("date")
        values = frame["monthly_return"].to_numpy()
        if np.any(values <= -1):
            raise ValueError("Strategy book is insolvent")
        selectors = dict(zip(groups, key, strict=True))
        predicate = pl.all_horizontal([pl.col(name) == value for name, value in selectors.items()])
        days = daily.filter(predicate).sort("date")
        trades = rebalances.filter(predicate)
        if days.is_empty():
            raise ValueError("Missing daily strategy NAV")
        summary = return_summary(values, periods)
        # Fractional mass at the quantile gives exactly alpha of the empirical distribution.
        mass = alpha * len(values)
        whole = int(math.floor(mass))
        ordered = np.sort(values)
        cvar = (
            ordered[:whole].sum() + (mass - whole) * ordered[min(whole, len(values) - 1)]
        ) / mass
        downside = float(np.sqrt(np.mean(np.minimum(values, 0) ** 2)))
        nav = np.r_[1.0, days["nav"].to_numpy()]
        if not np.isfinite(nav).all() or np.any(nav <= 0):
            raise ValueError("Invalid daily strategy NAV")
        summary.update(
            lo_sharpe=lo_sharpe(values, periods),
            monthly_sharpe=float(values.mean() / values.std(ddof=1))
            if len(values) > 1 and values.std(ddof=1) > 0
            else None,
            sortino=float(values.mean() / downside * np.sqrt(periods)) if downside > 0 else None,
            max_drawdown=float(np.min(nav / np.maximum.accumulate(nav) - 1)),
            monthly_cvar=cvar,
            annual_turnover=float(trades["turnover"].sum()) * periods / len(values),
            first_month=str(frame["date"].min()),
            last_month=str(frame["date"].max()),
        )
        dsr = (
            deflated_sharpe(
                values,
                list(population["monthly_sharpes"].values()),
                population["distinct_trading_ids"],
            )
            if population["distinct_trading_ids"]
            else {
                "dsr": None,
                "descriptive": True,
                "distinct_trading_strategies": 0,
                "strategies_with_sharpe": 0,
                "benchmark_monthly_sharpe": None,
            }
        )
        summary.update(
            {
                f"dsr_{name}" if name != "dsr" else name: value
                for name, value in dsr.items()
                if name != "monthly_sharpe"
            }
        )
        results.append({**selectors, **summary})
    return sorted(results, key=lambda row: (row["series"], row["strategy"], row["cost_bp"]))


def sharpe_difference_table(
    monthly: pl.DataFrame, settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Paired calendar bootstrap of Lo-adjusted annual Sharpe versus BSC."""
    config = settings["evaluation"]
    periods = settings["replication"]["months_per_year"]
    results = []
    for (series, cost), frame in monthly.partition_by(["series", "cost_bp"], as_dict=True).items():
        baseline = frame.filter(pl.col("strategy") == "BSC").select(
            "date", pl.col("monthly_return").alias("baseline")
        )
        for (strategy,), model in frame.partition_by("strategy", as_dict=True).items():
            if strategy == "BSC":
                continue
            pair = model.select("date", "monthly_return").join(baseline, on="date", validate="1:1")
            pair = pair.sort("date")
            indices = stationary_indices(
                pair["date"].to_list(),
                config["bootstrap_reps"],
                config["block_months"],
                config["seed"],
            )
            values, base = pair["monthly_return"].to_numpy(), pair["baseline"].to_numpy()
            draws = lo_sharpe_draws(values[indices], periods) - lo_sharpe_draws(
                base[indices], periods
            )
            left, right = lo_sharpe(values, periods), lo_sharpe(base, periods)
            estimate = left - right if left is not None and right is not None else None
            results.append(
                {
                    "series": series,
                    "strategy": strategy,
                    "cost_bp": cost,
                    "benchmark": "BSC",
                    "months": len(values),
                    "sharpe_difference": estimate,
                    **interval(draws, config["confidence"]),
                    "sharpe_definition": "Lo_annual",
                }
            )
    return results


def size_check_table(frame: pl.DataFrame, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Rejection, bias and block-exclusion audits separately by series/k."""
    config = settings["evaluation"]
    if frame.select(pl.struct("series", "k", "rep").n_unique()).item() != frame.height:
        raise ValueError("Duplicate size-check replications")
    rows = []
    for (series, block), group in frame.partition_by(["series", "k"], as_dict=True).items():
        identical = group["identical_forecasts"].fill_null(False).to_numpy()
        row = {
            "series": series,
            "k": block,
            "replications": group.height,
            "identical_forecasts": int(identical.sum()),
            "failed_replications": group["error"].is_not_null().sum(),
        }
        for test in ("cw", "dm"):
            values = np.where(identical, 1.0, group[f"{test}_p"].to_numpy())
            valid = np.isfinite(values) & (values >= 0) & (values <= 1)
            row[f"{test}_undefined"] = int((~valid).sum())
            row[f"{test}_rejection_rate"] = (
                float(np.mean(values[valid] < config["size_check_alpha"])) if valid.any() else None
            )
            rate = row[f"{test}_rejection_rate"]
            row[f"{test}_descriptive"] = rate is None or rate > config["size_check_max"]
        r2 = group["r2"].to_numpy()
        valid_r2 = np.isfinite(r2)
        row["r2_undefined"] = int((~valid_r2).sum())
        row["mean_r2"] = float(np.mean(r2[valid_r2])) if valid_r2.any() else None
        row["median_r2"] = float(np.median(r2[valid_r2])) if valid_r2.any() else None
        row["positive_r2_share"] = float(np.mean(r2[valid_r2] > 0)) if valid_r2.any() else None
        refits, excluded = group["refits"].to_numpy(), group["placebo_excluded_refits"].to_numpy()
        valid_refits = (
            np.isfinite(refits)
            & np.isfinite(excluded)
            & (refits > 0)
            & (excluded >= 0)
            & (excluded <= refits)
        )
        row["refits"] = int(refits[valid_refits].sum())
        row["placebo_excluded_refits"] = int(excluded[valid_refits].sum())
        row["placebo_excluded_refit_share"] = (
            row["placebo_excluded_refits"] / row["refits"] if row["refits"] else None
        )
        row["complete"] = (
            set(group["rep"].to_list()) == set(range(config["size_check_reps"]))
            and row["cw_undefined"] == row["dm_undefined"] == 0
            and row["r2_undefined"] == row["failed_replications"] == 0
            and bool(valid_refits.all())
        )
        row["bias_exceeded"] = (
            row["mean_r2"] is not None and row["mean_r2"] > config["size_check_max_mean_r2"]
        )
        row["bias_passed"] = row["complete"] and not row["bias_exceeded"]
        rows.append(row)
    return sorted(rows, key=lambda row: (row["series"], row["k"]))


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )


def _write_table(name: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[Path]:
    path = REPO / "results/tables" / f"baseline_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    sidecar = path.with_suffix(".provenance.json")
    _json(sidecar, metadata)
    return [path, sidecar]


def _overlap_table(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.group_by("series")
        .agg(
            pl.len().alias("adjacent_windows"),
            (pl.col("overlap_days") > 0).sum().alias("overlapping_pairs"),
            pl.col("overlap_days").sum().alias("total_overlap_days"),
            pl.col("overlap_days").max().alias("max_overlap_days"),
            pl.col("gap_days").sum().alias("total_gap_days"),
            pl.col("origin_distance_days").min().alias("minimum_rebalance_interval"),
            pl.col("origin_distance_days").max().alias("maximum_rebalance_interval"),
        )
        .sort("series")
        .to_dicts()
    )


def feature_coverage_table(
    features: pl.DataFrame,
    forecasts: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Summarize historical absolute-weight and balanced-panel coverage.

    Built rows weight daily origins equally; scored rows count each series/origin
    once across models and targets. Comomentum coverage is count/total at each
    origin, averaged across origins, rather than a pooled stock count ratio.
    """
    keys = ["series", "date"]
    if features.select(pl.struct(keys).n_unique()).item() != features.height:
        raise ValueError("Duplicate feature keys in coverage audit")
    scored = forecasts.select("series", pl.col("origin").alias("date")).unique()
    if scored.join(features.select(keys), on=keys, how="anti").height:
        raise ValueError("Scored origins missing from feature coverage audit")
    features = features.with_columns(
        pl.when(pl.col("book_rv_coverage").is_finite())
        .then(pl.col("book_rv_coverage"))
        .otherwise(None),
        pl.when(pl.col("comom_total") > 0)
        .then(pl.col("comom_count") / pl.col("comom_total"))
        .otherwise(None)
        .alias("comom_ratio"),
    )
    rows = []
    for sample, frame in (
        ("built_daily_origins", features),
        ("scored_month_end_origins", features.join(scored, on=keys, how="semi")),
    ):
        rows.extend(
            frame.group_by("series")
            .agg(
                pl.lit(sample).alias("sample"),
                pl.col("date").min().alias("first_origin"),
                pl.col("date").max().alias("last_origin"),
                pl.len().alias("origins"),
                pl.col("book_rv_coverage").count().alias("book_coverage_origins"),
                pl.col("book_rv_coverage").min().alias("book_coverage_min"),
                pl.col("book_rv_coverage").mean().alias("book_coverage_mean"),
                pl.col("comom_count").min().alias("comom_count_min"),
                pl.col("comom_count").mean().alias("comom_count_mean"),
                pl.col("comom_total").mean().alias("comom_total_mean"),
                pl.col("comom_ratio").count().alias("comom_ratio_origins"),
                pl.col("comom_ratio").min().alias("comom_ratio_min"),
                pl.col("comom_ratio").mean().alias("comom_ratio_mean"),
            )
            .sort("series")
            .to_dicts()
        )
    return rows


def _direct_inputs(output: Path) -> list[Path]:
    """Include the saved audits and provenance sidecars of evaluated model artifacts."""
    paths = {output / name for name in _INPUT_FILES if (output / name).is_file()}
    paths.update(
        path.with_suffix(".metadata.json")
        for path in list(paths)
        if path.suffix == ".parquet" and path.with_suffix(".metadata.json").is_file()
    )
    return sorted(paths)


def _evaluation_input_hashes(output: Path, inherited: dict[str, str]) -> dict[str, str]:
    """Pin the exact diagnostics used, checking recorded BDP artifact identities."""
    hashes = dict(inherited)
    for path in _direct_inputs(output):
        hashes[str(path)] = sha256_file(path)
        sidecar = path.with_suffix(".metadata.json")
        if path.suffix == ".parquet" and sidecar.exists():
            recorded = json.loads(sidecar.read_text()).get("artifact_hash")
            if recorded is not None and recorded != hashes[str(path)]:
                raise ValueError(f"Changed persisted model artifact: {path}")
    decision_path = output / "bdp_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        for source, digest in decision.get("artifacts", {}).items():
            path = Path(source)
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Changed BDP recovery diagnostic: {path}")
            hashes[str(path)] = digest
    fits_path = output / "fits.parquet"
    if fits_path.exists():
        fits = pl.read_parquet(fits_path)
        if "parameters_json" in fits.columns:
            checked = set()
            for serialized in fits["parameters_json"].drop_nulls().to_list():
                parameters = json.loads(serialized)
                for key in ("structural", "inner_structural"):
                    record = parameters.get(key, {})
                    if "artifact_hash" not in record or "fit_time" not in record:
                        continue
                    cutoff = record["fit_time"]
                    path = output / "bdp_paths" / f"{cutoff}.parquet"
                    sidecar = path.with_suffix(".json")
                    if (cutoff, record["artifact_hash"]) in checked:
                        continue
                    if (
                        not path.is_file()
                        or sha256_file(path) != record["artifact_hash"]
                        or not sidecar.is_file()
                        or json.loads(sidecar.read_text()) != record
                    ):
                        raise ValueError(f"Changed referenced BDP cutoff fit/path: {cutoff}")
                    hashes[str(path)] = record["artifact_hash"]
                    hashes[str(sidecar)] = sha256_file(sidecar)
                    checked.add((cutoff, record["artifact_hash"]))
    return hashes


def _raw_l1_penalties(fits: pl.DataFrame) -> list[dict[str, Any]]:
    """Export annual penalty choices and inner-validation losses, without data rows."""
    selected = fits.filter(
        pl.col("series") == "RAW",
        pl.col("target") == "y1",
        pl.col("feature_set") == "L1",
    )
    if selected["refit"].n_unique() != selected.height:
        raise ValueError("Duplicate RAW y1 L1 refit penalties")
    return (
        selected.select(
            "refit",
            "block_lambdas_json",
            "validation_losses_json",
            "training_count",
            "inner_count",
            "validation_count",
        )
        .sort("refit")
        .to_dicts()
    )


def evaluate_study(
    settings: dict[str, Any],
    output: Path,
    metadata: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate existing design outputs without refitting, and pin all results."""
    forecasts = pl.read_parquet(output / "forecasts.parquet")
    _assert_design(forecasts, settings)
    monthly = pl.read_parquet(output / "strategy_monthly.parquet")
    daily = pl.read_parquet(output / "strategy_daily.parquet")
    rebalances = pl.read_parquet(output / "strategy_rebalances.parquet")
    population = design_population(monthly)
    size_path = output / "size_check.parquet"
    size = size_check_table(pl.read_parquet(size_path), settings) if size_path.exists() else []
    availability = audit.get("feature_availability", [])
    if isinstance(availability, dict):
        availability = [
            {"feature_set": name, "series": series, "first_complete_origin": first}
            for name, series_dates in availability.items()
            for series, first in series_dates.items()
        ]
    tables = {
        "forecasts": forecast_table(forecasts, settings, size_check=size),
        "strategies": strategy_table(monthly, daily, rebalances, settings, population),
        "sharpe_differences": sharpe_difference_table(monthly, settings),
        "size_check": size,
        "feature_availability": availability,
        "feature_coverage": feature_coverage_table(
            pl.read_parquet(
                output / "features.parquet",
                columns=[
                    "series",
                    "date",
                    "book_rv_coverage",
                    "comom_count",
                    "comom_total",
                ],
            ),
            forecasts,
        ),
        "target_overlaps": _overlap_table(pl.read_parquet(output / "target_overlaps.parquet")),
        "break_even": audit.get("break_even", []),
        "raw_l1_penalties": _raw_l1_penalties(pl.read_parquet(output / "fits.parquet")),
    }
    required_h2 = {
        (series, feature_set, "y1")
        for series in ("RAW", "RAW_c", "RES")
        for feature_set in ("L1", "L1_without_M", "L1-own")
    }
    present = set(forecasts.select("series", "feature_set", "target").unique().iter_rows())
    missing_h2 = sorted(required_h2 - present)
    thin_without_h2 = metadata["mode"] == "thin" and bool(missing_h2)
    tables["h2"] = [] if thin_without_h2 else h2_table(forecasts, settings)
    inputs = _evaluation_input_hashes(output, metadata.get("input_hashes", {}))
    inputs[str(POPULATION_PATH)] = sha256_file(POPULATION_PATH)
    metadata = {**metadata, "baseline_code_hash": code_hash(), "input_hashes": inputs}
    artifacts = []
    for name, rows in tables.items():
        artifacts.extend(_write_table(name, rows, metadata))
    expected_groups = {
        (series, k)
        for series in settings["evaluation"]["size_check_series"]
        for k in settings["evaluation"]["size_check_blocks"]
    }
    size_complete = {(row["series"], row["k"]) for row in size} == expected_groups and all(
        row["complete"] for row in size
    )
    summary = {
        "provenance": metadata,
        "audit": audit,
        "tables": tables,
        "trial_population": population,
        "design_first_origin": str(forecasts["origin"].min()),
        "design_last_origin": str(forecasts["origin"].max()),
        "forecast_rows": forecasts.height,
        "size_check_complete": size_complete,
        "size_check_bias_passed": size_complete and all(row["bias_passed"] for row in size),
        "h2_status": "not_built_thin_scope" if thin_without_h2 else "built",
        "h2_missing_models": [
            dict(zip(("series", "feature_set", "target"), values, strict=True))
            for values in missing_h2
        ],
        "evaluation_settings": settings["evaluation"],
        "artifacts": {str(path): sha256_file(path) for path in artifacts},
    }
    _json(output / "summary.json", summary)
    return summary


def load_summary(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Refuse changed configuration, source, inputs, outputs or DSR population."""
    root = Path(settings["data"]["data_root"])
    path = root / "derived/baseline/summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    metadata = summary["provenance"]
    for key, source in (
        ("config_hash", Path(settings["config_path"])),
        ("data_manifest_hash", root / "raw/manifest.json"),
    ):
        if metadata.get(key) != sha256_file(source):
            raise ValueError(f"Stale baseline results: {key} changed; rerun momenta design")
    if metadata.get("baseline_code_hash") != code_hash():
        raise ValueError("Stale baseline results: computation source changed; rerun momenta design")
    for source, digest in {**metadata["input_hashes"], **summary["artifacts"]}.items():
        if not Path(source).is_file() or sha256_file(Path(source)) != digest:
            raise ValueError(f"Stale baseline input or artifact: {source}")
    for source in _direct_inputs(path.parent):
        if str(source) not in metadata["input_hashes"]:
            raise ValueError(f"Unevaluated baseline input: {source.name}")
    return summary
