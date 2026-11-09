"""Academic reconciliation and descriptive portfolio results (Methodology)."""


from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.manifest import sha256_file
from momenta.settings import REPO

_BUILD_FILES = (
    "academic_daily",
    "academic_monthly",
    "book_daily",
    "book_monthly",
    "rebalance_summary",
)


_REFERENCE_FILES = (
    "french/momentum_deciles_daily",
    "french/momentum_deciles_monthly",
    "jkp/ret_12_1_daily",
    "jkp/ret_12_1_monthly",
    "jkp/resff3_12_1_monthly",
)


_M2_SOURCE_FILES = ("accounting.py", "signals.py", "portfolios.py", "portfolio_report.py")


def code_hash() -> str:
    """Hash the portfolio computations; later modules and shared presentation may grow."""
    digest = hashlib.sha256()
    for name in sorted(_M2_SOURCE_FILES):
        path = REPO / "src" / "momenta" / name
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def return_summary(returns: np.ndarray, months_per_year: int) -> dict[str, Any]:
    """Mean×12, sample SD×√12, their ratio, adjusted skew and wealth drawdown (Methodology).

    The annualization factor is supplied by settings. Drawdown includes initial
    wealth 1. Inputs are finite monthly zero-cost returns; RF is not subtracted.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Return summaries require a finite one-dimensional series")
    n = values.size
    if not n:
        return dict(
            months=0,
            annual_mean=None,
            annual_volatility=None,
            sharpe=None,
            skew=None,
            worst_month=None,
            max_drawdown=None,
        )
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    skew = None
    if n > 2 and sd and sd > 0:
        skew = float(n / ((n - 1) * (n - 2)) * np.sum(((values - mean) / sd) ** 3))
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    return {
        "months": int(n),
        "annual_mean": mean * months_per_year,
        "annual_volatility": sd * math.sqrt(months_per_year) if sd is not None else None,
        "sharpe": mean / sd * math.sqrt(months_per_year) if sd and sd > 0 else None,
        "skew": skew,
        "worst_month": float(values.min()),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
    }


def _dated_returns(frame: pl.DataFrame, column: str, *, monthly: bool = False) -> pl.DataFrame:
    """Select decimal returns with unique observation dates; monthly keys are month ends."""
    result = frame.select(pl.col("date").cast(pl.Date), pl.col(column).cast(pl.Float64))
    if monthly:
        result = result.with_columns(pl.col("date").dt.month_end())
    if result["date"].null_count() or result["date"].n_unique() != result.height:
        raise ValueError("Missing or duplicate dates in a portfolio comparison")
    return result.sort("date")


def reconcile(
    built: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    threshold: float,
    minimum_explained: float,
    monthly: bool,
    explanation: str = "",
) -> dict[str, Any]:
    """Pearson correlation on finite, date-matched academic WML observations (Methodology)."""
    left = _dated_returns(built, "wml", monthly=monthly)
    right = _dated_returns(reference, "ret", monthly=monthly)
    overlap = left.join(right, on="date", how="inner").filter(
        pl.col("wml").is_finite() & pl.col("ret").is_finite()
    )
    correlation = None
    if overlap.height > 1:
        x, y = overlap["wml"].to_numpy(), overlap["ret"].to_numpy()
        if x.std() > 0 and y.std() > 0:
            correlation = float(np.corrcoef(x, y)[0, 1])
    strict_pass = correlation is not None and correlation >= threshold
    explained = (
        not strict_pass
        and correlation is not None
        and correlation >= minimum_explained
        and bool(explanation.strip())
    )
    return {
        "observations": overlap.height,
        "first_date": overlap["date"].min(),
        "last_date": overlap["date"].max(),
        "correlation": correlation,
        "required_correlation": threshold,
        "minimum_explained": minimum_explained,
        "strict_pass": strict_pass,
        "pass": strict_pass or explained,
        "status": "PASS" if strict_pass else "EXPLAINED" if explained else "FAIL",
        "explanation": explanation.strip(),
    }


def reconciliation_tables(
    frames: dict[str, pl.DataFrame],
    references: dict[str, pl.DataFrame],
    settings: dict[str, Any],
    build_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the four reconciliation gates; report startup-inclusive RES separately.

    The RES gate starts at the configured holding month when the panel supports
    full calendar regression windows. No holdings or per-stock eligibility change.
    """
    config = settings["reconciliation"]
    primary = settings["portfolios"]["primary"]
    comparisons = (
        ("raw_monthly", "RAW", primary, "monthly", "jkp/ret_12_1_monthly"),
        ("res_monthly", "RES", primary, "monthly", "jkp/resff3_12_1_monthly"),
        ("raw_daily", "RAW", primary, "daily", "jkp/ret_12_1_daily"),
        ("raw_decile_monthly", "RAW", "decile_vw", "monthly", "french/momentum_deciles_monthly"),
    )
    explanations = {
        **config.get("explanations", {}),
        **build_audit.get("reconciliation_explanations", {}),
    }
    gates, descriptive = [], []
    for key, series, construction, frequency, reference in comparisons:
        sample = frames[f"academic_{frequency}"].filter(
            (pl.col("series") == series) & (pl.col("construction") == construction)
        )
        benchmark = references[reference]
        if reference.startswith("french/"):
            benchmark = benchmark.with_columns((pl.col("d10") - pl.col("d1")).alias("ret"))
        definition = {
            "comparison": key,
            "series": series,
            "construction": construction,
            "frequency": frequency,
            "reference": reference,
        }
        arguments = {
            "threshold": config[key],
            "minimum_explained": config["minimum_explained"],
            "monthly": frequency == "monthly",
        }
        first_month = None
        if key == "res_monthly":
            descriptive.append(
                {
                    **definition,
                    **reconcile(sample, benchmark, **arguments),
                    "comparison": "res_monthly_all_months",
                    "pass": None,
                    "status": "DESCRIPTIVE",
                    "explanation": (
                        "Includes startup months with truncated calendar windows; not a gate."
                    ),
                }
            )
            first_month = date.fromisoformat(config["res_first_month"])
            sample = sample.filter(pl.col("date") >= first_month)
        gates.append(
            {
                **definition,
                "sample_rule": "full_calendar_windows" if first_month else "all_available",
                "holding_month_start": first_month,
                **reconcile(sample, benchmark, **arguments, explanation=explanations.get(key, "")),
            }
        )
    return gates, descriptive


def bsc_returns(
    daily: pl.DataFrame,
    monthly: pl.DataFrame,
    settings: dict[str, Any],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """w_t=min(cap,(target/√12)/√(21·Σr²/126)); use only t−1 data (Methodology).

    Require all configured daily observations, including historical Saturdays.
    Monthly WML is the published D10−D1, never compounded daily differences.
    """
    config = settings["replication"]
    window = int(config["variance_days"])
    daily = _dated_returns(daily, "wml")
    monthly = _dated_returns(monthly, "wml", monthly=True).filter(
        pl.col("date").is_between(
            date.fromisoformat(config["first_month"]), date.fromisoformat(config["last_month"])
        )
    )
    dates = daily["date"].to_numpy()
    values = daily["wml"].to_numpy()
    target = config["vol_target_annual"] / math.sqrt(config["months_per_year"])
    observations, omitted = [], []
    for row in monthly.iter_rows(named=True):
        month = row["date"]
        first = month.replace(day=1)
        stop = int(np.searchsorted(dates, np.datetime64(first)))
        sample = values[max(0, stop - window) : stop]
        reason = ""
        if len(sample) < window:
            reason = "insufficient_daily_history"
        elif not np.isfinite(sample).all():
            reason = "missing_daily_return"
        elif stop == 0 or str(dates[stop - 1])[:7] != (
            f"{first.year - (first.month == 1):04d}-{(first.month - 2) % 12 + 1:02d}"
        ):
            reason = "missing_previous_month"
        elif row["wml"] is None or not math.isfinite(row["wml"]):
            reason = "missing_monthly_return"
        if reason:
            omitted.append({"date": month, "daily_observations": len(sample), "reason": reason})
            continue
        variance = float(config["monthly_days"] / window * np.dot(sample, sample))
        multiple = (
            min(config["max_multiple"], target / math.sqrt(variance))
            if variance
            else (float(config["max_multiple"]))
        )
        observations.append(
            {
                "date": month,
                "forecast_date": date.fromisoformat(str(dates[stop - 1])),
                "daily_observations": window,
                "monthly_variance": variance,
                "multiple": multiple,
                "unscaled_return": row["wml"],
                "scaled_return": multiple * row["wml"],
            }
        )
    schema = {
        "date": pl.Date,
        "forecast_date": pl.Date,
        "daily_observations": pl.Int64,
        "monthly_variance": pl.Float64,
        "multiple": pl.Float64,
        "unscaled_return": pl.Float64,
        "scaled_return": pl.Float64,
    }
    return pl.DataFrame(observations, schema=schema), omitted


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )


def _write_table(name: str, records: list[dict[str, Any]], metadata: dict[str, Any]) -> Path:
    path = REPO / "results" / "tables" / f"m2_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    _write_json(path.with_suffix(".provenance.json"), metadata)
    return path


def _figure(
    settings: dict[str, Any],
    daily: pl.DataFrame,
    worst: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Plot unscaled primary RAW/RES book NAV and RAW's worst calendar months (Methodology)."""
    cache = Path(settings["data"]["data_root"]) / "tmp" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = settings["portfolios"]["primary"]
    frame = daily.filter(pl.col("construction") == primary).sort("date")
    fig, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for series, color in (("RAW", "#245c96"), ("RES", "#b04a39")):
        sample = frame.filter(pl.col("series") == series)
        if sample.height:
            axis.plot(
                sample["date"].to_list(),
                sample["nav"].to_numpy(),
                label=series,
                color=color,
                linewidth=1.3,
            )
    for row in worst:
        if row["series"] == "RAW" and row["construction"] == primary:
            month = row["date"]
            axis.axvspan(month.replace(day=1), month, color="#555555", alpha=0.10)
    axis.set_yscale("log")
    axis.set(title="Unscaled momentum strategy books", xlabel="Date", ylabel="Book NAV (log scale)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.supxlabel(
        f"{primary}; shaded: RAW's {settings['report']['worst_months']} worst calendar months. "
        "One trading-day execution lag; gross of costs.",
        fontsize=9,
    )
    path = REPO / "results" / "figures" / "figure_1_momentum_books.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Description": json.dumps(metadata, default=str)})
    plt.close(fig)
    _write_json(path.with_suffix(".provenance.json"), metadata)
    return path


def _series_tables(
    frames: dict[str, pl.DataFrame],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Describe each constructed object and calendar-month book losses (Methodology)."""
    coverage, summaries, worst = [], [], []
    for name in _BUILD_FILES:
        frame = frames[name]
        keys = ["date", "series", "construction"]
        if frame.select(keys).is_duplicated().any():
            raise ValueError(f"Duplicate portfolio keys: {name}")
        for (series, construction), sample in frame.partition_by(
            ["series", "construction"],
            as_dict=True,
        ).items():
            coverage.append(
                {
                    "object": name,
                    "series": series,
                    "construction": construction,
                    "observations": sample.height,
                    "first_date": sample["date"].min(),
                    "last_date": sample["date"].max(),
                }
            )
    for (series, construction), sample in (
        frames["book_monthly"]
        .partition_by(
            ["series", "construction"],
            as_dict=True,
        )
        .items()
    ):
        sample = sample.sort("date")
        rebalances = frames["rebalance_summary"].filter(
            (pl.col("series") == series) & (pl.col("construction") == construction)
        )
        summaries.append(
            {
                "series": series,
                "construction": construction,
                "first_date": sample["date"].min(),
                "last_date": sample["date"].max(),
                **return_summary(
                    sample["monthly_return"].to_numpy(), settings["replication"]["months_per_year"]
                ),
                "mean_rebalance_turnover": rebalances["turnover"].mean(),
                "total_turnover": rebalances["turnover"].sum(),
            }
        )
        selected = sample.sort(["monthly_return", "date"]).head(settings["report"]["worst_months"])
        for rank, row in enumerate(selected.iter_rows(named=True), 1):
            worst.append(
                {
                    "series": series,
                    "construction": construction,
                    "rank": rank,
                    "date": row["date"],
                    "monthly_return": row["monthly_return"],
                }
            )
    return coverage, summaries, worst


def evaluate_portfolios(
    settings: dict[str, Any],
    output: Path,
    metadata: dict[str, Any],
    build_audit: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the portfolio layer without fitting forecasts; publish only portfolio aggregates
    (Methodology)."""
    root = Path(settings["data"]["data_root"])
    input_paths = [output / f"{name}.parquet" for name in _BUILD_FILES]
    input_paths += [
        path
        for path in (
            output / "_metadata.json",
            output / "signals.parquet",
            output / "formation_weights.parquet",
        )
        if path.exists()
    ]
    input_paths += [root / "curated" / f"{name}.parquet" for name in _REFERENCE_FILES]
    metadata = {
        **metadata,
        "m2_code_hash": code_hash(),
        "input_hashes": {str(path): sha256_file(path) for path in input_paths},
    }
    frames = {name: pl.read_parquet(output / f"{name}.parquet") for name in _BUILD_FILES}
    references = {
        name: pl.read_parquet(root / "curated" / f"{name}.parquet") for name in _REFERENCE_FILES
    }
    for name in ("daily", "monthly"):
        key = f"french/momentum_deciles_{name}"
        references[key] = references[key].with_columns((pl.col("d10") - pl.col("d1")).alias("wml"))
    coverage, book_summary, worst = _series_tables(frames, settings)
    reconciliation, all_months = reconciliation_tables(frames, references, settings, build_audit)
    replication = settings["replication"]
    bsc, omitted = bsc_returns(
        references["french/momentum_deciles_daily"],
        references["french/momentum_deciles_monthly"],
        settings,
    )
    bsc_summary = []
    for strategy in ("unscaled", "scaled"):
        stats = return_summary(bsc[f"{strategy}_return"].to_numpy(), replication["months_per_year"])
        target = replication[f"{strategy}_sharpe"]
        gap = stats["sharpe"] - target if stats["sharpe"] is not None else None
        explanation = build_audit.get("bsc_explanations", {}).get(strategy, "").strip()
        strict = gap is not None and abs(gap) <= replication["sharpe_tolerance"]
        explained = gap is not None and bool(explanation)
        bsc_summary.append(
            {
                "strategy": strategy,
                "first_date": bsc["date"].min(),
                "last_date": bsc["date"].max(),
                **stats,
                "target_sharpe": target,
                "sharpe_gap": gap,
                "tolerance": replication["sharpe_tolerance"],
                "strict_pass": strict,
                "pass": strict or explained,
                "status": "PASS" if strict else "EXPLAINED" if explained else "FAIL",
                "explanation": explanation,
            }
        )
    long_start, long_end = (
        date.fromisoformat(replication[key]) for key in ("long_history_start", "long_history_end")
    )
    long = (
        references["jkp/ret_12_1_monthly"]
        .select("date", pl.col("ret").alias("RAW"))
        .join(
            references["jkp/resff3_12_1_monthly"].select("date", pl.col("ret").alias("RES")),
            on="date",
            how="inner",
        )
        .filter(pl.col("date").is_between(long_start, long_end))
        .filter(pl.col("RAW").is_finite() & pl.col("RES").is_finite())
        .sort("date")
    )
    long_summary = [
        {
            "series": series,
            "first_date": long["date"].min(),
            "last_date": long["date"].max(),
            **return_summary(long[series].to_numpy(), replication["months_per_year"]),
        }
        for series in ("RAW", "RES")
    ]
    expected = {
        (series, settings["portfolios"]["primary"]) for series in ("RAW", "RAW_c", "RES")
    } | {(series, "decile_vw") for series in ("RAW", "RES")}
    series_built = all(
        expected
        == {(row["series"], row["construction"]) for row in coverage if row["object"] == name}
        for name in _BUILD_FILES
    )
    tables = {
        "reconciliation": reconciliation,
        "reconciliation_all_months": all_months,
        "bsc_summary": bsc_summary,
        "long_history_summary": long_summary,
        "book_summary": book_summary,
        "series_coverage": coverage,
        "worst_months": worst,
    }
    artifacts = [_write_table(name, records, metadata) for name, records in tables.items()]
    artifacts.append(_figure(settings, frames["book_daily"], worst, metadata))
    bsc_path = output / "bsc_monthly.parquet"
    bsc.write_parquet(bsc_path)
    _write_json(bsc_path.with_suffix(".provenance.json"), metadata)
    summary = {
        "provenance": metadata,
        "build_audit": build_audit,
        "tables": tables,
        "series_built": series_built,
        "reconciliation_passed": all(row["pass"] for row in reconciliation),
        "bsc_passed": all(row["pass"] for row in bsc_summary),
        "bsc_requested_first_month": replication["first_month"],
        "bsc_requested_last_month": replication["last_month"],
        "bsc_daily_first_date": references["french/momentum_deciles_daily"]["date"].min(),
        "bsc_omitted_months": omitted,
        "bsc_max_multiple": bsc["multiple"].max(),
        "bsc_capped_months": bsc.filter(pl.col("multiple") == replication["max_multiple"]).height,
        "artifacts": {str(path): sha256_file(path) for path in artifacts},
        "long_history_available": long.height > 1,
    }
    summary["passed"] = (
        series_built
        and summary["reconciliation_passed"]
        and summary["bsc_passed"]
        and summary["long_history_available"]
    )
    _write_json(output / "m2_summary.json", summary)
    return summary


def load_summary(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Load the portfolio summary only if config, code, pinned data and every evaluated artifact
    still match."""
    root = Path(settings["data"]["data_root"])
    path = root / "derived" / 'portfolios' / "m2_summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    metadata = summary["provenance"]
    required = {
        "config_hash": Path(settings["config_path"]),
        "data_manifest_hash": root / "raw" / "manifest.json",
        "curated_metadata_hash": root / "curated" / "_metadata.json",
    }
    for key, source in (
        ("universe_hash", root / "curated" / "universe.parquet"),
        ("cap_repairs_hash", root / "curated" / "cap_repairs.parquet"),
        ("price_corrections_hash", REPO / "config" / "price_corrections.csv"),
    ):
        if key in metadata:
            required[key] = source
    for key, source in required.items():
        if not source.is_file() or metadata.get(key) != sha256_file(source):
            raise ValueError(f"Stale portfolio results ({key} changed); rerun momenta portfolios")
    if metadata.get("m2_code_hash") != code_hash():
        raise ValueError('Stale portfolio results (code changed); rerun momenta portfolios')
    for source, digest in {**metadata["input_hashes"], **summary["artifacts"]}.items():
        if not Path(source).is_file() or sha256_file(Path(source)) != digest:
            raise ValueError(f"Stale portfolio results ({Path(source).name} changed); rerun momenta portfolios")  # noqa: E501
    return summary
