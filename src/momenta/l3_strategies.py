"""Model-based L3 strategy accounts on the unchanged position ledger (Methodology).

Each account scales the momentum book by a volatility target computed from a
saved forecast, exp(prediction + training residual variance / 2), and replays
the same signed stock-level ledger as the baseline rules.
"""

from __future__ import annotations

import bisect
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.experiments import trial
from momenta.manifest import sha256_file
from momenta.strategies import _next_month, break_even_cost, simulate_strategy, volatility_multiples
from momenta.strategy_study import _monthly, _specification, _tag, _write
from momenta.walkforward import assert_design_origins


def model_specification(
    settings: dict[str, Any],
    variant: str,
    feature_set: str,
    cost: float,
    *,
    full_bdp: bool,
    best_model: str,
) -> dict[str, Any]:
    """Identify the economic model rule, including the H5 chosen prefix, not its sample."""
    result = _specification(settings, "L1", cost, full_bdp)
    result.update(
        variant=variant,
        feature_set=feature_set,
        variance_prediction="exp(prediction + training_residual_variance / 2)",
    )
    selected = best_model if "HMM" in feature_set else feature_set
    if "HMM" in feature_set:
        result["selected_prefix"] = best_model
        result["hmm"] = {
            key: value
            for key, value in settings["hmm"].items()
            if not key.startswith("synthetic_") and key not in {"secondary_states", "occupancy_min"}
        }
        result["state_probabilities"] = "filtered" if "HMMF" in feature_set else "emission_only"
    if "L2" in selected or "HMM" in feature_set:
        result["l2"] = {
            key: settings["l2"][key]
            for key in (
                "correlation_days",
                "leg_bins",
                "bowley_quantiles",
                "moors_quantiles",
                "quantile_method",
            )
        }
    if "CM" in selected:
        result["copula"] = {key: settings["ot"][key] for key in ("window_days", "quantile_method")}
    if "OT" in selected:
        result["ot"] = {
            key: settings["ot"][key]
            for key in (
                "directions",
                "grid_points",
                "k_centroids",
                "seed",
                "kmeans_n_init",
                "kmeans_max_iter",
                "kmeans_tolerance",
            )
        }
    return result


def model_overlays(
    decisions: pl.DataFrame,
    calendar: list[date],
    settings: dict[str, Any],
    *,
    bsc: bool = False,
) -> pl.DataFrame:
    """Use σ²=exp(ŷ+s²/2), or exp(own_rv_126) for BSC, and execute at t+1."""
    decisions = decisions.sort("origin")
    if calendar != sorted(set(calendar)):
        raise ValueError("Strategy execution calendar must be sorted and unique")
    if decisions.is_empty() or decisions["origin"].n_unique() != decisions.height:
        raise ValueError("Strategy origins must be present and unique")
    if decisions.filter(pl.col("refit") > pl.col("origin")).height:
        raise ValueError("A strategy forecast cannot be fitted after its origin")
    if bsc:
        log_variance = decisions["own_rv_126"].to_numpy()
    else:
        residual_variance = decisions["residual_variance"].to_numpy()
        if not np.isfinite(residual_variance).all() or np.any(residual_variance < 0):
            raise ValueError("Training residual variances must be finite and nonnegative")
        log_variance = decisions["prediction"].to_numpy() + residual_variance / 2
    with np.errstate(over="ignore", invalid="ignore"):
        variance = np.exp(log_variance)
    sizing = settings["strategy"]
    multiple = volatility_multiples(
        variance,
        sizing["vol_target_annual"],
        settings["targets"]["horizon"],
        settings["replication"]["months_per_year"] * settings["replication"]["monthly_days"],
        sizing["max_multiple"],
    )
    executions = []
    for origin in decisions["origin"].to_list():
        index = (
            bisect.bisect_right(calendar, origin) + settings["portfolios"]["rebalance_lag_days"] - 1
        )
        if index >= len(calendar):
            raise ValueError("Calendar ends before the required strategy execution")
        executions.append(calendar[index])
    return decisions.select("origin", pl.col("refit").alias("fit_date")).with_columns(
        pl.Series("execution_date", executions, dtype=pl.Date),
        pl.Series("multiple", multiple),
        *([pl.lit(None).cast(pl.Date).alias("fit_date")] if bsc else []),
    )


def run_model_strategies(
    settings: dict[str, Any],
    forecasts: pl.DataFrame,
    *,
    features: pl.DataFrame,
    variants: dict[str, dict[str, str]],
    best_models: dict[str, str],
    through: date,
    output: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Replay registered model accounts, logging each gross/net attempt and cost crossing.

    The caller supplies an isolated output directory. Holdout settings can only
    extend these bounds through the guarded runner; ordinary settings stay locked.
    Numerical Parquet contains stable strategy IDs, never attempt IDs/timestamps.
    """
    assert_design_origins([through], settings)
    if not variants or any(not names for names in variants.values()):
        raise ValueError("At least one series/model strategy is required")
    selected = forecasts.filter(
        pl.col("target") == "y1",
        pl.col("origin") <= through,
        pl.col("series").is_in(list(variants)),
    )
    first_registered = date.fromisoformat(settings["walkforward"]["design_first_origin"])
    pairs = {}
    for series, names in sorted(variants.items()):
        sample = None
        for variant, feature_set in sorted(names.items()):
            frame = selected.filter(
                pl.col("series") == series, pl.col("feature_set") == feature_set
            ).sort("origin")
            if frame.is_empty() or frame["origin"].n_unique() != frame.height:
                raise ValueError(
                    f"Missing or duplicate strategy forecasts: {series}, {feature_set}"
                )
            months = [(day.year, day.month) for day in frame["origin"].to_list()]
            if months[0] < (first_registered.year, first_registered.month):
                raise ValueError("Model-strategy forecasts precede the permitted sample")
            if sample is not None and not frame["origin"].equals(sample):
                raise ValueError(
                    "Model strategies require the same origin sample within each series"
                )
            sample = frame["origin"]
            pairs[series, variant] = frame.join(
                features.filter(pl.col("series") == series).select(
                    pl.col("date").alias("origin"),
                    "own_rv_126",
                ),
                on="origin",
                how="left",
                validate="1:1",
                maintain_order="left",
            )
    first_origin = min(frame["origin"].min() for frame in pairs.values())
    last_origin = max(frame["origin"].max() for frame in pairs.values())
    if last_origin != through:
        raise ValueError("Strategy forecasts do not reach the requested final origin")
    last_performance = _next_month(through)
    root = Path(settings["data"]["data_root"])
    portfolio_dir, curated = root / "derived/portfolios", root / "curated"
    primary = settings["portfolios"]["primary"]
    ledger_files = [
        path
        for year in range(first_origin.year, last_performance.year + 1)
        for path in sorted((portfolio_dir / "ledger" / f"year={year}").glob("*.parquet"))
    ]
    if not ledger_files:
        raise ValueError("Pinned portfolio position ledger is missing")
    decision_path = root / "derived/baseline/bdp_decision.json"
    full_bdp = json.loads(decision_path.read_text())["full_model"]
    inputs = [
        portfolio_dir / "book_daily.parquet",
        portfolio_dir / "_metadata.json",
        decision_path,
        curated / "trading_calendar.parquet",
        *ledger_files,
    ]
    metadata.setdefault("input_hashes", {}).update(
        {str(path): sha256_file(path) for path in inputs}
    )
    metadata["input_hashes"][str(Path(__file__))] = sha256_file(Path(__file__))
    calendar = pl.read_parquet(curated / "trading_calendar.parquet")["date"].sort().to_list()
    books = (
        pl.read_parquet(portfolio_dir / "book_daily.parquet")
        .filter(
            pl.col("construction") == primary,
            pl.col("series").is_in(list(variants)),
            pl.col("date").is_between(first_origin, last_performance),
        )
        .sort("series", "date")
    )
    ledger = (
        pl.scan_parquet(ledger_files)
        .filter(
            pl.col("construction") == primary,
            pl.col("series").is_in(list(variants)),
            pl.col("date") > first_origin,
            pl.col("date") <= last_performance,
        )
        .collect()
        .sort("series", "date", "permaticker")
    )
    output.mkdir(parents=True, exist_ok=True)
    ledger_output = output / "strategy_ledger"
    if ledger_output.exists():
        shutil.rmtree(ledger_output)
    costs = sorted({0.0, float(settings["strategy"]["cost_bp"])})
    frames = {
        name: []
        for name in ("strategy_daily", "strategy_monthly", "strategy_rebalances", "overlays")
    }
    initial_nav = float(settings["portfolios"]["initial_nav"])
    artifact_paths, crossings, ids = [], [], []
    for (series, variant), decisions in pairs.items():
        base = books.filter(pl.col("series") == series)
        base_ledger = ledger.filter(pl.col("series") == series)
        if base.is_empty() or (base["date"].max().year, base["date"].max().month) != (
            last_performance.year,
            last_performance.month,
        ):
            raise ValueError("Base book does not cover the final strategy holding month")
        feature_set = variants[series][variant]
        sample = f"origins:{decisions['origin'].min()}:{decisions['origin'].max()}"
        specification = model_specification(
            settings,
            variant,
            feature_set,
            0.0,
            full_bdp=full_bdp,
            best_model=best_models[series],
        )
        overlays = None
        for cost in costs:
            with trial(
                settings,
                experiment=variant,
                kind="strategy",
                series=series,
                sample=sample,
                specification={**specification, "cost_bp": cost},
            ) as attempt:
                if overlays is None:
                    overlays = model_overlays(decisions, calendar, settings)
                result = simulate_strategy(
                    base,
                    base_ledger,
                    overlays,
                    cost_bp=cost,
                    initial_nav=initial_nav,
                    last_date=last_performance,
                )
                daily = _tag(result.daily, variant, attempt.strategy_id, cost)
                monthly = _monthly(daily).sort("date")
                trades = _tag(result.rebalances, variant, attempt.strategy_id, cost)
                tagged_overlays = _tag(overlays, variant, attempt.strategy_id, cost).with_columns(
                    pl.lit(series).alias("series"),
                    pl.lit(primary).alias("construction"),
                )
                for name, frame in (
                    ("strategy_daily", daily),
                    ("strategy_monthly", monthly),
                    ("strategy_rebalances", trades),
                    ("overlays", tagged_overlays),
                ):
                    frames[name].append(frame)
                stock_ledger = _tag(result.ledger, variant, attempt.strategy_id, cost)
                for year in sorted(stock_ledger["date"].dt.year().unique().to_list()):
                    path = (
                        ledger_output
                        / f"year={year}"
                        / f"{attempt.strategy_id.split(':')[1]}.parquet"
                    )
                    _write(
                        stock_ledger.filter(pl.col("date").dt.year() == year),
                        path,
                        settings,
                        metadata,
                    )
                    artifact_paths.append(path)
                returns = monthly["monthly_return"].to_numpy()
                sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
                attempt.metrics.update(
                    first_month=str(monthly["date"].min()),
                    last_month=str(monthly["date"].max()),
                    monthly_sharpe=float(returns.mean() / sd) if sd > 0 else None,
                    months=len(returns),
                    final_nav=float(daily["nav"][-1]),
                    cost_bp=cost,
                    total_turnover=float(trades["turnover"].sum()),
                    initial_cost=float(trades["cost"][0]),
                )
                ids.append(attempt.strategy_id)
        with trial(
            settings,
            experiment="model_break_even_cost",
            kind="diagnostic",
            series=series,
            sample=sample,
            specification={
                "comparison": "annual_arithmetic_net_mean_minus_BSC",
                "strategy_specification": specification,
                "cost_grid_bp": settings["strategy"]["break_even_cost_grid_bp"],
            },
        ) as attempt:
            bsc = model_overlays(decisions, calendar, settings, bsc=True)
            compared = break_even_cost(
                base,
                base_ledger,
                overlays,
                bsc,
                settings,
                last_date=last_performance,
                initial_nav=initial_nav,
            )
            crossings.append({"series": series, "strategy": variant, **compared})
            attempt.metrics.update(compared)
    for name, parts in frames.items():
        path = output / f"{name}.parquet"
        _write(
            pl.concat(parts, how="diagonal_relaxed").sort("series", "strategy", "cost_bp", "date")
            if name != "overlays"
            else pl.concat(parts, how="diagonal_relaxed").sort(
                "series", "strategy", "cost_bp", "origin"
            ),
            path,
            settings,
            metadata,
        )
        artifact_paths.append(path)
    return {
        "series": sorted(variants),
        "variants": variants,
        "cost_bp": costs,
        "first_origin": str(first_origin),
        "last_origin": str(last_origin),
        "last_performance_date": str(last_performance),
        "initial_book": "cash",
        "initial_formation_cost_included": True,
        "strategy_ids": sorted(ids),
        "break_even": crossings,
        "artifact_hashes": {str(path): sha256_file(path) for path in artifact_paths},
    }
