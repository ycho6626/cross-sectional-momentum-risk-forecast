"""Logged strategy accounts for the registered risk-management rules (Methodology).

Each rule turns dated forecasts into monthly multiples of the momentum book,
executed at the next close through the signed stock-level ledger, with gross and
net-of-cost accounts logged as separate trading specifications.
"""

from __future__ import annotations

import bisect
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.bdp import BDPFit
from momenta.bdp_horizon import holding_moments
from momenta.experiments import trial
from momenta.manifest import sha256_file
from momenta.strategies import (
    _next_month,
    break_even_cost,
    build_dm_pairs,
    calibrate_dm,
    calibrate_training_scores,
    dm_multiples,
    fit_dm_mean,
    simulate_strategy,
    volatility_multiples,
)

BDPPaths = Callable[[date], tuple[pl.DataFrame, BDPFit, dict[str, Any]]]


def _write(
    frame: pl.DataFrame,
    path: Path,
    settings: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Persist a result and its supplied vintage/code/config provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression=settings["storage"]["compression"])
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {**metadata, "artifact_hash": sha256_file(path)},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )


def _specification(settings: dict[str, Any], variant: str, cost_bp: float, full_bdp: bool) -> dict:
    """Stable economic identity; evaluation dates and realized fit values are excluded."""
    sizing = settings["strategy"]
    result = {
        "variant": variant,
        "construction": settings["portfolios"]["primary"],
        "formation": {
            key: value for key, value in settings["portfolios"].items() if key != "initial_nav"
        },
        "universe": settings["universe"],
        "signals": settings["signals"],
        "cost_bp": cost_bp,
        "initial_book": "cash",
        "cost_sizing": "pre_cost_nav",
        "terminal_target": "cash_without_renormalization",
    }
    if variant != "Static":
        result["sizing"] = {
            "vol_target_annual": sizing["vol_target_annual"],
            "max_multiple": sizing["max_multiple"],
            "horizon": settings["targets"]["horizon"],
            "target_lag": settings["targets"]["lag"],
            "annual_trading_days": settings["replication"]["months_per_year"]
            * settings["replication"]["monthly_days"],
        }
    if variant == "BSC":
        result["variance_window"] = settings["replication"]["variance_days"]
    if variant in {"L1", "DM-transfer", "BDP-transfer"}:
        result["bdp_method"] = "full" if full_bdp else "approximation"
        result["features"] = settings["features"]
        result["walkforward"] = {
            key: settings["walkforward"][key]
            for key in (
                "refit",
                "validation_months",
                "gap_days",
                "ridge_lambdas",
            )
        }
        if full_bdp:
            result["bdp"] = settings["bdp"]
    if variant in {"L1", "DM-transfer"}:
        result["walkforward"].update({
            key: settings["walkforward"][key]
            for key in ("feature_clip_sd", "blocks", "penalty_selection")
        })
    if variant in {"DM-transfer", "BDP-transfer"}:
        result["calibration"] = {
            key: value for key, value in sizing.items() if key.startswith("calibration_")
        }
        result["calibration_sample"] = "complete_training_calendar_months_only"
    if variant == "DM-transfer":
        result["dm_first_month"] = sizing["dm_first_month"]
    return result


def _monthly(daily: pl.DataFrame) -> pl.DataFrame:
    """Compound each exact net/gross account's daily returns by calendar month."""
    return (
        daily.sort("date")
        .with_columns(pl.col("date").dt.month_end())
        .group_by(
            "series",
            "construction",
            "strategy",
            "strategy_id",
            "cost_bp",
            "date",
            maintain_order=True,
        )
        .agg(
            ((1 + pl.col("daily_return")).product() - 1).alias("monthly_return"),
            pl.col("nav").last(),
        )
    )


def _tag(frame: pl.DataFrame, strategy: str, identity: str, cost: float) -> pl.DataFrame:
    """Attach the same economic identity to each account artifact."""
    return frame.with_columns(
        pl.lit(strategy).alias("strategy"),
        pl.lit(identity).alias("strategy_id"),
        pl.lit(cost).cast(pl.Float64).alias("cost_bp"),
    )


def _moment_rows(
    selected: pl.DataFrame,
    paths: pl.DataFrame,
    fit: BDPFit,
    refit: date,
    settings: dict[str, Any],
    series: str,
) -> pl.DataFrame:
    """Evaluate lag-aligned BDP moments at supplied origins.

    The fixed seed is shared across states and refits: common random numbers
    reduce simulation noise in comparisons, without selecting on outcomes.
    """
    joined = selected.join(
        paths.filter(pl.col("series") == series).select(
            pl.col("date").alias("origin"),
            "bdp_m",
            "bdp_gamma",
            "bdp_delta",
        ),
        on="origin",
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    records = []
    for row in joined.iter_rows(named=True):
        state = np.array([row["bdp_m"], row["bdp_gamma"], row["bdp_delta"]], dtype=float)
        if not np.isfinite(state).all():
            raise ValueError("A BDP strategy origin lacks a finite causal state")
        moments = holding_moments(state, fit, settings)
        if (
            not np.isfinite(moments["mean"])
            or not np.isfinite(moments["variance"])
            or (moments["variance"] <= 0)
        ):
            raise ValueError("BDP strategy horizon moments must be finite with positive variance")
        records.append(
            {
                "series": series,
                "refit": refit,
                "origin": row["origin"],
                "target_end": row["target_end"],
                **moments,
                "score": moments["mean"] / moments["variance"],
            }
        )
    return pl.DataFrame(records, infer_schema_length=None)


def run_strategies(
    settings: dict[str, Any],
    features: pl.DataFrame,
    forecasts: pl.DataFrame,
    training_predictions: pl.DataFrame,
    *,
    last_origin: date,
    metadata: dict[str, Any],
    output: Path,
    bdp_paths: BDPPaths | None = None,
) -> dict[str, Any]:
    """Run registered origins within the scoped bounds; log each account before fitting.

    The first design execution starts from cash. Performance ends at the end
    of the calendar month following last_origin, without adding a later
    decision. Every strategy uses the same available L1 forecast origins.
    Costs 0 and the registered headline cost each have independent NAV paths.
    """
    design_end = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    if last_origin > design_end:
        raise ValueError("Strategy origins after the design boundary need the frozen holdout")
    score = forecasts.filter(
        pl.col("target") == "y1", pl.col("feature_set") == "L1", pl.col("origin") <= last_origin
    ).sort("series", "origin")
    if score.is_empty():
        raise ValueError("Strategy study requires design-period y1 L1 forecasts")
    if score.select(pl.struct("series", "origin").n_unique()).item() != score.height:
        raise ValueError("Strategy forecast origins must be unique within series")
    first_registered = date.fromisoformat(settings["walkforward"]["design_first_origin"])
    if any(
        (origin.year, origin.month) < (first_registered.year, first_registered.month)
        for origin in score["origin"].to_list()
    ):
        raise ValueError("Strategy origins precede the registered design sample")
    last_performance = _next_month(last_origin)
    root = Path(settings["data"]["data_root"])
    curated, portfolio_dir = root / "curated", root / "derived/portfolios"
    primary = settings["portfolios"]["primary"]
    books = (
        pl.read_parquet(portfolio_dir / "book_daily.parquet")
        .filter(
            pl.col("construction") == primary,
            pl.col("date") <= last_performance,
        )
        .sort("series", "date")
    )
    ledger_files = sorted((portfolio_dir / "ledger").glob("year=*/*.parquet"))
    if not ledger_files:
        raise ValueError("Checked portfolio stock-level ledger is missing")
    inputs = [
        portfolio_dir / "book_daily.parquet",
        curated / "trading_calendar.parquet",
        curated / "french/ff3_daily.parquet",
        curated / "french/momentum_deciles_monthly.parquet",
        *ledger_files,
    ]
    metadata.setdefault("input_hashes", {}).update(
        {str(path): sha256_file(path) for path in inputs}
    )
    ledger = (
        pl.scan_parquet(ledger_files)
        .filter(
            pl.col("construction") == primary,
            pl.col("date") > score["origin"].min(),
            pl.col("date") <= last_performance,
        )
        .collect()
    )
    output.mkdir(parents=True, exist_ok=True)
    artifact_names = [
        "strategy_daily",
        "strategy_monthly",
        "strategy_rebalances",
        "overlays",
        "dm_fits",
        "bdp_strategy_moments",
    ]
    for name in artifact_names:
        for suffix in (".parquet", ".metadata.json"):
            (output / f"{name}{suffix}").unlink(missing_ok=True)
    ledger_output = output / "strategy_ledger"
    if ledger_output.exists():
        shutil.rmtree(ledger_output)
    costs = sorted({0.0, float(settings["strategy"]["cost_bp"])})
    variants = ["Static", "BSC", "DM-transfer", "L1"]
    if bdp_paths is not None:
        variants.append("BDP-transfer")
    sample_name = f"design:{score['origin'].min()}:{score['origin'].max()}"
    daily_frames, monthly_frames, trade_frames, overlay_frames = [], [], [], []
    fit_records, calibration_audits, moment_frames, break_even = [], [], [], []
    dm_pairs, dm_warmup = None, None
    mean_fits, cached_bdp = {}, {}
    sizing = settings["strategy"]
    initial_nav = float(settings["portfolios"]["initial_nav"])
    for series in sorted(score["series"].unique().to_list()):
        base = books.filter(pl.col("series") == series).sort("date")
        base_ledger = ledger.filter(pl.col("series") == series)
        dates = base["date"].to_list()
        decisions = (
            score.filter(pl.col("series") == series)
            .join(
                features.filter(pl.col("series") == series).select(
                    pl.col("date").alias("origin"),
                    "own_rv_126",
                    "bear_x_var",
                ),
                on="origin",
                how="left",
                validate="1:1",
            )
            .sort("origin")
        )
        training = training_predictions.filter(pl.col("series") == series)
        states = features.filter(pl.col("series") == series).select(
            pl.col("date").alias("origin"),
            "bear_x_var",
        )
        execution_dates = []
        for origin in decisions["origin"].to_list():
            index = (
                bisect.bisect_right(dates, origin)
                + settings["portfolios"]["rebalance_lag_days"]
                - 1
            )
            if index >= len(dates):
                raise ValueError("Base history ends before a required strategy execution")
            execution_dates.append(dates[index])
        decisions = decisions.with_columns(pl.Series("execution_date", execution_dates))
        if not dates or (dates[-1].year, dates[-1].month) != (
            last_performance.year,
            last_performance.month,
        ):
            raise ValueError("Base book does not cover the final strategy holding month")
        formed: dict[str, pl.DataFrame] = {}
        for variant in variants:
            print(f"Build {series} {variant} overlays and costs", flush=True)
            for cost in costs:
                specification = _specification(settings, variant, cost, bdp_paths is not None)
                with trial(
                    settings,
                    experiment=variant,
                    kind="strategy",
                    series=series,
                    sample=sample_name,
                    specification=specification,
                ) as attempt:
                    if variant not in formed:
                        if variant == "DM-transfer" and dm_pairs is None:
                            dm_pairs, dm_warmup = build_dm_pairs(
                                pl.read_parquet(curated / "french/ff3_daily.parquet"),
                                pl.read_parquet(
                                    curated / "french/momentum_deciles_monthly.parquet"
                                ),
                                settings,
                            )
                        formed[variant] = _form_overlays(
                            variant,
                            series,
                            decisions,
                            training,
                            states,
                            base,
                            settings,
                            mean_fits,
                            cached_bdp,
                            bdp_paths,
                            fit_records,
                            calibration_audits,
                            moment_frames,
                            dm_pairs,
                        )
                    overlays = formed[variant]
                    result = simulate_strategy(
                        base,
                        base_ledger,
                        overlays,
                        cost_bp=cost,
                        initial_nav=initial_nav,
                        last_date=last_performance,
                    )
                    daily = _tag(result.daily, variant, attempt.strategy_id, cost)
                    monthly = _monthly(daily)
                    trades = _tag(result.rebalances, variant, attempt.strategy_id, cost)
                    tagged_overlays = _tag(overlays, variant, attempt.strategy_id, cost)
                    tagged_overlays = tagged_overlays.with_columns(
                        pl.lit(series).alias("series"),
                        pl.lit(primary).alias("construction"),
                    )
                    daily_frames.append(daily)
                    monthly_frames.append(monthly)
                    trade_frames.append(trades)
                    overlay_frames.append(tagged_overlays)
                    stock_ledger = _tag(result.ledger, variant, attempt.strategy_id, cost)
                    for year in sorted(stock_ledger["date"].dt.year().unique().to_list()):
                        destination = (
                            ledger_output / f"year={year}" / f"{series}_{variant}_{cost:g}.parquet"
                        )
                        _write(
                            stock_ledger.filter(pl.col("date").dt.year() == year),
                            destination,
                            settings,
                            metadata,
                        )
                    returns = monthly["monthly_return"].to_numpy()
                    attempt.metrics.update(
                        first_month=str(monthly["date"].min()),
                        last_month=str(monthly["date"].max()),
                        monthly_sharpe=float(returns.mean() / returns.std(ddof=1))
                        if len(returns) > 1 and returns.std(ddof=1) > 0
                        else None,
                        months=len(returns),
                        final_nav=float(daily["nav"][-1]),
                        cost_bp=cost,
                        total_turnover=float(trades["turnover"].sum()),
                        initial_cost=float(trades["cost"][0]),
                    )
        for variant in variants:
            if variant == "BSC":
                break_even.append(
                    {
                        "series": series,
                        "strategy": variant,
                        "break_even_bp": None,
                        "status": "benchmark",
                        "crossings_bp": [],
                    }
                )
                continue
            with trial(
                settings,
                experiment="break_even_cost",
                kind="diagnostic",
                series=series,
                sample=sample_name,
                specification={
                    "comparison": "annual_arithmetic_net_mean_minus_BSC",
                    "variant": variant,
                    "strategy_specification": _specification(
                        settings, variant, 0, bdp_paths is not None
                    ),
                    "cost_grid_bp": sizing["break_even_cost_grid_bp"],
                },
            ) as attempt:
                compared = break_even_cost(
                    base,
                    base_ledger,
                    formed[variant],
                    formed["BSC"],
                    settings,
                    last_date=last_performance,
                    initial_nav=initial_nav,
                )
                break_even.append({"series": series, "strategy": variant, **compared})
                attempt.metrics.update(compared)
    for name, frames in (
        ("strategy_daily", daily_frames),
        ("strategy_monthly", monthly_frames),
        ("strategy_rebalances", trade_frames),
        ("overlays", overlay_frames),
    ):
        _write(
            pl.concat(frames, how="diagonal_relaxed"),
            output / f"{name}.parquet",
            settings,
            metadata,
        )
    _write(
        pl.DataFrame(fit_records, infer_schema_length=None),
        output / "dm_fits.parquet",
        settings,
        metadata,
    )
    if moment_frames:
        _write(
            pl.concat(moment_frames, how="diagonal_relaxed"),
            output / "bdp_strategy_moments.parquet",
            settings,
            metadata,
        )
    return {
        "series": sorted(score["series"].unique().to_list()),
        "strategies": variants,
        "cost_bp": costs,
        "first_origin": str(score["origin"].min()),
        "last_origin": str(score["origin"].max()),
        "last_performance_date": str(last_performance),
        "initial_book": "cash",
        "initial_formation_cost_included": True,
        "bdp_included": bdp_paths is not None,
        "dm_warmup": dm_warmup,
        "calibrations": calibration_audits,
        "break_even": break_even,
    }


def _form_overlays(
    variant: str,
    series: str,
    decisions: pl.DataFrame,
    training: pl.DataFrame,
    states: pl.DataFrame,
    base: pl.DataFrame,
    settings: dict[str, Any],
    mean_fits: dict,
    cached_bdp: dict,
    bdp_paths: BDPPaths | None,
    fit_records: list,
    calibration_audits: list,
    moment_frames: list,
    dm_pairs: pl.DataFrame | None,
) -> pl.DataFrame:
    """Freeze a monthly overlay at its origin, using only its assigned prior refit."""
    sizing = settings["strategy"]
    horizon = settings["targets"]["horizon"]
    year_days = settings["replication"]["months_per_year"] * settings["replication"]["monthly_days"]
    if variant in {"Static", "BSC", "L1"}:
        if variant == "Static":
            multiples = np.ones(decisions.height)
        else:
            variance = (
                np.exp(decisions["own_rv_126"].to_numpy())
                if variant == "BSC"
                else np.exp(
                    decisions["prediction"].to_numpy()
                    + decisions["residual_variance"].to_numpy() / 2
                )
            )
            multiples = volatility_multiples(
                variance, sizing["vol_target_annual"], horizon, year_days, sizing["max_multiple"]
            )
        return decisions.select(
            "origin", "execution_date", pl.col("refit").alias("fit_date")
        ).with_columns(
            pl.Series("multiple", multiples),
            *(([pl.lit(None).cast(pl.Date).alias("fit_date")]) if variant != "L1" else []),
        )
    rows = []
    for (refit,), group in decisions.partition_by("refit", as_dict=True).items():
        selected = training.filter(pl.col("refit") == refit, pl.col("target_end") < refit)
        if variant == "DM-transfer":
            if refit not in mean_fits:
                if dm_pairs is None:
                    raise ValueError("DM-transfer requires French training pairs")
                mean_fits[refit] = fit_dm_mean(dm_pairs, refit)
            mean_fit = mean_fits[refit]
            calibration = calibrate_dm(base, selected, states, mean_fit, settings)
            variance = np.exp(
                group["prediction"].to_numpy() + group["residual_variance"].to_numpy() / 2
            )
            multiples = dm_multiples(
                group["bear_x_var"].to_numpy(),
                variance,
                mean_fit,
                calibration.scale,
                sizing["max_multiple"],
            )
            extra = {
                "intercept": mean_fit.intercept,
                "slope": mean_fit.slope,
                "mean_training_count": mean_fit.n,
                "mean_target_end": mean_fit.target_end,
            }
        else:
            if bdp_paths is None:
                raise ValueError("BDP-transfer requires the full BDP model")
            if refit not in cached_bdp:
                cached_bdp[refit] = bdp_paths(refit)
            paths, fit, fit_metadata = cached_bdp[refit]
            if (
                not fit.converged
                or date.fromisoformat(fit_metadata["fit_time"]) != refit
                or (date.fromisoformat(fit_metadata["french_last_observation"]) >= refit)
            ):
                raise ValueError(
                    "BDP strategy parameters must be fitted strictly on pre-refit returns"
                )
            train_moments = _moment_rows(selected, paths, fit, refit, settings, series)
            calibration = calibrate_training_scores(base, train_moments, refit, settings)
            forecast = _moment_rows(group, paths, fit, refit, settings, series)
            multiples = np.clip(
                calibration.scale * forecast["score"].to_numpy(),
                -sizing["max_multiple"],
                sizing["max_multiple"],
            )
            moment_frames.extend(
                [
                    train_moments.with_columns(pl.lit("training").alias("sample")),
                    forecast.with_columns(pl.lit("design").alias("sample")),
                ]
            )
            extra = {
                "nu": fit.nu,
                "mean_training_count": fit.observations,
                "mean_target_end": date.fromisoformat(fit_metadata["french_last_observation"]),
            }
        audit = {"series": series, "strategy": variant, "refit": str(refit), **asdict(calibration)}
        calibration_audits.append(audit)
        fit_records.append(
            {
                "series": series,
                "strategy": variant,
                "refit": refit,
                **extra,
                "scale": calibration.scale,
                "annual_volatility": calibration.annual_volatility,
                "attained": calibration.attained,
                "training_months": calibration.audit["training_months"],
                "training_target_end": calibration.audit["training_target_end"],
                "holding_month_last": calibration.audit["holding_month_last"],
                "audit_json": json.dumps(calibration.audit, sort_keys=True),
            }
        )
        for record, multiple in zip(group.iter_rows(named=True), multiples, strict=True):
            rows.append(
                {
                    "origin": record["origin"],
                    "execution_date": record["execution_date"],
                    "fit_date": refit,
                    "multiple": float(multiple),
                }
            )
    return pl.DataFrame(rows).sort("origin")
