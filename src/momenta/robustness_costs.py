"""Registered 5/20/40 bp sensitivity on unchanged frozen multipliers (Methodology)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from momenta.baseline_report import strategy_table
from momenta.experiments import read_trials, trial
from momenta.manifest import sha256_file
from momenta.robustness_daily import frozen_provenance, write_json
from momenta.strategies import _next_month, simulate_strategy
from momenta.strategy_study import _monthly, _tag, _write


def frozen_overlays(overlays: pl.DataFrame) -> dict[tuple[str, str], pl.DataFrame]:
    """Collapse gross/net copies only when their dated multipliers exactly agree.

    Cost sensitivities change cash deductions and future NAV-scaled positions,
    but never the previously formed decision, execution date or model fit.
    """
    result = {}
    columns = ["origin", "execution_date", "fit_date", "multiple"]
    for key, rows in overlays.partition_by(["series", "strategy"], as_dict=True).items():
        reference = None
        for _, frame in rows.partition_by("cost_bp", as_dict=True).items():
            frame = frame.select(columns).sort("origin")
            if frame["origin"].n_unique() != frame.height:
                raise ValueError("Frozen overlay origins must be unique within an account")
            if reference is not None and not reference.equals(frame):
                raise ValueError("Frozen gross/net accounts do not share identical multipliers")
            reference = frame
        if reference is None or reference.is_empty():
            raise ValueError("Missing frozen overlay decisions")
        result[key] = reference
    return result


def account_parity(actual: pl.DataFrame, expected: pl.DataFrame, tolerance: float) -> float:
    """Check exact dated account support and monthly return/NAV agreement."""
    keys = ["series", "strategy", "cost_bp", "date"]
    actual, expected = actual.sort(keys), expected.sort(keys)
    if not actual.select(keys).equals(expected.select(keys)):
        raise ValueError("Cost replay changed the frozen account sample")
    errors = [
        float((actual[name] - expected[name]).abs().max()) for name in ("monthly_return", "nav")
    ]
    if not np.isfinite(errors).all() or max(errors) > tolerance:
        raise ValueError(f"Frozen cost replay does not reproduce monthly returns/NAV: {errors}")
    return max(errors)


def _specifications(original: pl.DataFrame) -> dict[tuple[str, str, float], dict]:
    """Use the logged economic specifications themselves, retaining every frozen choice."""
    by_id = {}
    for row in read_trials():
        identity = row["strategy_id"]
        specification = json.loads(row["specification_json"])
        if identity in by_id and specification != by_id[identity]:
            raise ValueError("A logged strategy ID has inconsistent specifications")
        by_id[identity] = specification
    result = {}
    for row in (
        original.select("series", "strategy", "cost_bp", "strategy_id")
        .unique()
        .iter_rows(named=True)
    ):
        specification = by_id.get(row["strategy_id"])
        if specification is None or specification["series"] != row["series"]:
            raise ValueError(
                "Frozen strategy specification is missing from the preserved trial log"
            )
        inner = specification["specification"]
        if inner["cost_bp"] != row["cost_bp"]:
            raise ValueError("Frozen account and specification costs disagree")
        result[row["series"], row["strategy"], float(row["cost_bp"])] = inner
    return result


def run_costs(settings: dict, *, stage: Path, output: Path) -> dict:
    """Replay every frozen rule at registered costs after proving 0/10 bp parity.

    The signed base ledger nets formation and scaling trades per stock. Costs
    affect account NAV and hence future dollar positions; subtracting a fee
    from the old frozen return series would fail this accounting identity.
    """
    frozen, metadata = frozen_provenance(settings)
    metadata["source_hashes"] = {
        str(path): sha256_file(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("robustness_daily.py"),
        )
    }
    root = Path(settings["data"]["data_root"])
    if not output.resolve().is_relative_to((root / "derived/robustness").resolve()):
        raise ValueError("Cost robustness output must be isolated under derived/robustness")
    folder = stage / "derived/final"
    first, last = [
        date.fromisoformat(frozen["bounds"][name])
        for name in (
            "holdout_first_origin",
            "holdout_last_origin",
        )
    ]
    last_performance = _next_month(last)
    cost_grid = [float(value) for value in settings["strategy"]["cost_grid_bp"]]
    specification = {
        "cost_grid_bp": cost_grid,
        "decision_policy": "frozen monthly multipliers",
        "best_models": frozen["best_models"],
        "accounting": "exact signed ledger",
    }
    with trial(
        settings,
        experiment="robustness-cost-grid",
        kind="diagnostic",
        series="ALL",
        sample=f"{first}:{last}",
        specification=specification,
    ) as attempt:
        overlay_paths = [
            folder / name / "overlays.parquet" for name in ("base_strategies", "model_strategies")
        ]
        monthly_path = folder / "strategy_monthly.parquet"
        audit_path = folder / "_audit.json"
        portfolio_dir = stage / "derived/portfolios"
        book_path = portfolio_dir / "book_daily.parquet"
        ledger_paths = [
            path
            for year in range(first.year, last_performance.year + 1)
            for path in sorted((portfolio_dir / "ledger" / f"year={year}").glob("*.parquet"))
        ]
        if not ledger_paths:
            raise ValueError("Frozen cost replay requires the signed base ledger")
        metadata["input_hashes"].update(
            {
                str(path): sha256_file(path)
                for path in (
                    *overlay_paths,
                    monthly_path,
                    audit_path,
                    book_path,
                    *ledger_paths,
                )
            }
        )
        original = pl.read_parquet(monthly_path)
        raw_overlays = pl.concat(
            [pl.read_parquet(path) for path in overlay_paths], how="diagonal_relaxed"
        )
        decisions = frozen_overlays(raw_overlays)
        specifications = _specifications(original)
        books = (
            pl.scan_parquet(book_path)
            .filter(
                pl.col("construction") == settings["portfolios"]["primary"],
                pl.col("date").is_between(first, last_performance),
            )
            .collect()
        )
        ledger = (
            pl.scan_parquet(ledger_paths)
            .filter(
                pl.col("construction") == settings["portfolios"]["primary"],
                pl.col("date").is_between(first, last_performance),
            )
            .collect()
            .sort("series", "date", "permaticker")
        )
        parts = {name: [] for name in ("strategy_daily", "strategy_monthly", "strategy_rebalances")}
        validations, ids = [], []
        for (series, variant), overlays in sorted(decisions.items()):
            base = books.filter(pl.col("series") == series).sort("date")
            base_ledger = ledger.filter(pl.col("series") == series)
            for cost in sorted(
                original.filter(pl.col("series") == series, pl.col("strategy") == variant)[
                    "cost_bp"
                ].unique()
            ):
                result = simulate_strategy(
                    base,
                    base_ledger,
                    overlays,
                    cost_bp=cost,
                    initial_nav=settings["portfolios"]["initial_nav"],
                    last_date=last_performance,
                    record_ledger=False,
                )
                monthly = _monthly(_tag(result.daily, variant, "parity", cost))
                expected = original.filter(
                    pl.col("series") == series,
                    pl.col("strategy") == variant,
                    pl.col("cost_bp") == cost,
                )
                error = account_parity(monthly, expected, settings["l2"]["numerical_tolerance"])
                validations.append(
                    {
                        "series": series,
                        "strategy": variant,
                        "cost_bp": cost,
                        "months": monthly.height,
                        "max_absolute_error": error,
                    }
                )
            for cost in cost_grid:
                economic = dict(specifications[series, variant, 0.0])
                economic["cost_bp"] = cost
                with trial(
                    settings,
                    experiment=f"robustness-cost:{variant}",
                    kind="strategy",
                    series=series,
                    sample=f"origins:{first}:{last}",
                    specification=economic,
                ) as account:
                    result = simulate_strategy(
                        base,
                        base_ledger,
                        overlays,
                        cost_bp=cost,
                        initial_nav=settings["portfolios"]["initial_nav"],
                        last_date=last_performance,
                    )
                    daily = _tag(result.daily, variant, account.strategy_id, cost)
                    monthly = _monthly(daily)
                    rebalances = _tag(result.rebalances, variant, account.strategy_id, cost)
                    for name, frame in (
                        ("strategy_daily", daily),
                        ("strategy_monthly", monthly),
                        ("strategy_rebalances", rebalances),
                    ):
                        parts[name].append(frame)
                    stock_ledger = _tag(result.ledger, variant, account.strategy_id, cost)
                    for year, frame in (
                        stock_ledger.with_columns(pl.col("date").dt.year().alias("year"))
                        .partition_by("year", as_dict=True)
                        .items()
                    ):
                        path = (
                            output
                            / "strategy_ledger"
                            / f"year={year[0]}"
                            / f"{account.strategy_id.split(':')[1]}.parquet"
                        )
                        _write(frame.drop("year"), path, settings, metadata)
                    returns = monthly["monthly_return"].to_numpy()
                    account.metrics.update(
                        first_month=str(monthly["date"].min()),
                        last_month=str(monthly["date"].max()),
                        monthly_sharpe=float(returns.mean() / returns.std(ddof=1))
                        if returns.std(ddof=1) > 0
                        else None,
                        months=len(returns),
                        cost_bp=cost,
                        final_nav=float(daily["nav"][-1]),
                        total_turnover=float(rebalances["turnover"].sum()),
                    )
                    ids.append(account.strategy_id)
        frames = {
            name: pl.concat(values, how="diagonal_relaxed").sort(
                "series", "strategy", "cost_bp", "date"
            )
            for name, values in parts.items()
        }
        for name, frame in frames.items():
            _write(frame, output / f"{name}.parquet", settings, metadata)
        population = json.loads(audit_path.read_text())["strategy_population"]
        table = strategy_table(
            frames["strategy_monthly"],
            frames["strategy_daily"],
            frames["strategy_rebalances"],
            settings,
            population,
        )
        for row in table:
            row["dsr_population_scope"] = (
                "frozen pre-unblinding trading population; sensitivity descriptive"
            )
        write_json(output / "strategy_metrics.json", table)
        paths = [output / f"{name}.parquet" for name in frames] + [output / "strategy_metrics.json"]
        paths += sorted((output / "strategy_ledger").rglob("*.parquet"))
        summary = {
            "complete": True,
            "stage": str(stage),
            "first_origin": str(first),
            "last_origin": str(last),
            "first_holding_month": str(frames["strategy_monthly"]["date"].min()),
            "last_holding_month": str(frames["strategy_monthly"]["date"].max()),
            "cost_grid_bp": cost_grid,
            "frozen_rules": len(decisions),
            "accounts": len(ids),
            "best_models": frozen["best_models"],
            "monthly_parity": validations,
            "strategy_ids": ids,
            "metrics": table,
            "provenance": metadata,
            "artifacts": {str(path): sha256_file(path) for path in paths},
        }
        write_json(output / "summary.json", summary)
        attempt.metrics.update(
            frozen_rules=len(decisions),
            accounts=len(ids),
            monthly_parity_max=max(r["max_absolute_error"] for r in validations),
        )
        return summary
