"""Final tables and figures from saved design and holdout results (Methodology).

Headline rows copy registered comparisons and name their benchmarks; the five
figures use saved books, features and dated fit parameters. ``publish_results``
then writes the released evidence under ``results/`` and ``figures/`` with the
release export rules. No model is estimated and no result is selected here.
"""

from __future__ import annotations

import csv
import json
import os
import struct
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO

HEADLINE_MODELS = ("L1-own", "L1", "L1+L2", "L1+L2+CM", "L1+L2+CM+OT", "best+HMMF2")
HEADLINE_STRATEGIES = (
    "Static",
    "BSC",
    "DM-transfer",
    "BDP-transfer",
    "L1",
    "L1+L2",
    "L1+L2+CM+OT",
    "best+HMMF2",
)
def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _checked(path: Path, expected: str, inputs: dict[str, str]) -> Path:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Changed saved reporting input: {path.name}")
    inputs[str(path)] = actual
    return path


def _one(rows: list[dict], **keys: Any) -> dict:
    selected = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one saved result for {keys}")
    return selected[0]


def headline_forecasts(tables: dict[str, list[dict]], best: dict[str, str]) -> list[dict]:
    """Copy registered y1 comparisons and name their actual benchmarks.

    H4 uses CM and H5a the frozen best prefix; adjacent displayed rows cannot
    replace those benchmarks. No new confidence interval or hypothesis is made.
    """
    rows = []
    for series in ("RAW", "RES"):
        for model in HEADLINE_MODELS:
            if model in {"L1-own", "L1"}:
                source = _one(
                    tables["baseline_forecasts"], series=series, target="y1", feature_set=model
                )
                benchmark = "L1-own"
            elif model == "L1+L2+CM":
                source = dict(_one(tables["prefixes"], series=series, model=model))
                baseline = _one(tables["h3"], series=series, model="L1+L2")["mse"]
                benchmark = "L1+L2"
                source.update(
                    baseline_mse=baseline,
                    mse_improvement=baseline - source["mse"],
                    r2_oos=1 - source["mse"] / baseline,
                )
            else:
                name = {"L1+L2": "h3", "L1+L2+CM+OT": "h4", "best+HMMF2": "h5a"}[model]
                selectors = {"series": series, "model": model}
                if name == "h5a":
                    selectors["states"] = 2
                source = _one(tables[name], **selectors)
                benchmark = {"h3": "L1", "h4": "L1+L2+CM", "h5a": best[series]}[name]
                if source["benchmark"] != benchmark:
                    raise ValueError("Saved result changes a registered headline benchmark")
            fields = (
                "n",
                "first_origin",
                "last_origin",
                "mse",
                "baseline_mse",
                "mse_improvement",
                "mse_improvement_interval_lower",
                "mse_improvement_interval_upper",
                "r2_oos",
                "cw_pvalue",
                "cw_holm_pvalue",
                "dm_pvalue",
                "qlike",
            )
            row = {
                "series": series,
                "model": model,
                "benchmark": benchmark,
                **{key: source.get(key) for key in fields},
            }
            if not np.isclose(
                row["baseline_mse"] - row["mse"], row["mse_improvement"], rtol=1e-10, atol=1e-12
            ):
                raise ValueError("Saved MSE improvement disagrees with its named benchmark")
            if model == "L1-own":
                row.update(
                    benchmark=None,
                    mse_improvement=None,
                    mse_improvement_interval_lower=None,
                    mse_improvement_interval_upper=None,
                    r2_oos=None,
                    cw_pvalue=None,
                    cw_holm_pvalue=None,
                    dm_pvalue=None,
                )
            rows.append(row)
    return rows


def headline_strategies(tables: dict[str, list[dict]], cost_bp: float) -> list[dict]:
    """Join saved net accounts, gross Lo Sharpe and annual-mean BSC crossings."""
    rows = []
    for series in ("RAW", "RES"):
        for strategy in HEADLINE_STRATEGIES:
            net = _one(tables["strategies"], series=series, strategy=strategy, cost_bp=cost_bp)
            gross = _one(tables["strategies"], series=series, strategy=strategy, cost_bp=0)
            crossing = _one(tables["break_even"], series=series, strategy=strategy)
            rows.append(
                {
                    **net,
                    "gross_lo_sharpe": gross["lo_sharpe"],
                    "break_even_bp": crossing["break_even_bp"],
                    "break_even_status": crossing["status"],
                }
            )
    return rows


def cumulative_loss_difference(forecasts: pl.DataFrame, models: tuple[str, ...]) -> pl.DataFrame:
    """Σ through t [(actual−L1)²−(actual−model)²], exact paired origins."""
    rows = []
    for series in ("RAW", "RES"):
        sample = forecasts.filter(pl.col("series") == series, pl.col("target") == "y1")
        base = sample.filter(pl.col("feature_set") == "L1").sort("origin")
        if base.is_empty() or base["origin"].n_unique() != base.height:
            raise ValueError("L1 requires unique nonempty plotting origins")
        for model in models:
            candidate = sample.filter(pl.col("feature_set") == model).sort("origin")
            if not base.select("origin", "actual", "refit").equals(
                candidate.select("origin", "actual", "refit")
            ):
                raise ValueError("Loss plot cannot drop or misalign scored origins")
            if candidate.filter(pl.col("refit") > pl.col("origin")).height:
                raise ValueError("Loss plot uses a future fit")
            actual, left, right = (
                base["actual"].to_numpy(),
                base["prediction"].to_numpy(),
                candidate["prediction"].to_numpy(),
            )
            if not np.isfinite(np.r_[actual, left, right]).all():
                raise ValueError("Loss plot requires finite saved outcomes and forecasts")
            increments = (actual - left) ** 2 - (actual - right) ** 2
            for origin, value in zip(base["origin"], np.cumsum(increments), strict=True):
                rows.append(
                    {
                        "date": origin,
                        "series": series,
                        "model": model,
                        "cumulative_loss_improvement": float(value),
                    }
                )
    return pl.DataFrame(rows)


def serving_timeline(
    forecasts: pl.DataFrame,
    fits: pl.DataFrame,
    audit: dict,
    monthly: pl.DataFrame,
) -> pl.DataFrame:
    """Use each scored origin's recorded full fit, never a later state fit.

    Nearest centroid is argmin of Euclidean distance from the saved embedding
    to that fit's saved, aligned centroids. This applies fixed parameters only.
    """
    embeddings = {
        row["date"]: np.asarray(row["embedding"]) for row in monthly.iter_rows(named=True)
    }
    if len(embeddings) != monthly.height:
        raise ValueError("Duplicate timeline embeddings")
    regimes = {
        (r["series"], r["fit_time"]): r
        for r in audit["regime_records"]
        if r["lineage"] == "full" and r["states"] == 2
    }
    centers = {r["fit_cutoff"]: r for r in audit["centroid_records"] if r["lineage"] == "full"}
    selected = forecasts.filter(
        pl.col("target") == "y1", pl.col("feature_set") == "best+HMMF2"
    ).sort("series", "origin")
    output = []
    for row in selected.iter_rows(named=True):
        series, origin, refit = row["series"], row["origin"], row["refit"]
        if refit > origin:
            raise ValueError("A future fit cannot label the regime timeline")
        hmm = regimes[series, str(refit)]
        center = centers[str(refit)]
        for model, kind, record in (("best+HMMF2", "hmm", hmm), ("L1+L2+CM+OT", "ot", center)):
            fitted = fits.filter(
                pl.col("series") == series,
                pl.col("target") == "y1",
                pl.col("feature_set") == model,
                pl.col("refit") == refit,
            )
            if fitted.height != 1:
                raise ValueError("Timeline needs exactly one recorded serving fit")
            saved = json.loads(fitted["parameters_json"][0])["structural"][kind]
            if saved["fit_hash"] != record["fit_hash"] or saved["lineage"] != "full":
                raise ValueError("Timeline parameters differ from the scored model fit")
        probability = _one(hmm["monthly_probabilities"], date=str(origin), series=series)
        p = [probability["hmmf_0"], probability["hmmf_1"]]
        if not np.isfinite(p).all() or min(p) < 0 or max(p) > 1 or not np.isclose(sum(p), 1):
            raise ValueError("Invalid saved filtered probabilities")
        centroid_values = np.asarray(center["centroids"])
        embedding = embeddings[origin]
        if centroid_values.ndim != 2 or centroid_values.shape[1:] != embedding.shape:
            raise ValueError("Timeline centroid and embedding dimensions differ")
        if not np.isfinite(centroid_values).all() or not np.isfinite(embedding).all():
            raise ValueError("Timeline geometry must be finite")
        nearest = int(np.argmin(np.sum((centroid_values - embedding) ** 2, axis=1)))
        output.append(
            {
                "date": origin,
                "series": series,
                "refit": refit,
                "filtered_state_0": p[0],
                "nearest_centroid": nearest + 1,
            }
        )
    result = pl.DataFrame(output)
    if result.select(pl.struct("series", "date").n_unique()).item() != result.height:
        raise ValueError("Duplicate scored regime timeline origin")
    return result


def _csv(path: Path, rows: list[dict], metadata: dict) -> None:
    if not rows:
        raise ValueError("Final tables cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    _json(path.with_suffix(".provenance.json"), {**metadata, "artifact_hash": sha256_file(path)})


def _plot_figures(
    settings: dict,
    root: Path,
    stage: Path,
    forecasts: pl.DataFrame,
    fits: pl.DataFrame,
    metadata: dict,
    inputs: dict[str, str],
) -> list[Path]:
    """Five figures from saved books, features and dated fit parameters only."""
    cache = root / "tmp/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from momenta.features import FEATURE_COLUMNS
    from momenta.l2 import L2_COLUMNS

    paths = []
    destination = REPO / "results/figures"
    destination.mkdir(parents=True, exist_ok=True)

    def read_frame(path: Path) -> pl.DataFrame:
        inputs[str(path)] = sha256_file(path)
        return pl.read_parquet(path)

    def save(fig: Any, name: str) -> None:
        path = destination / name
        fig.savefig(
            path,
            dpi=settings["l2"]["heatmap_dpi"],
            metadata={"Description": json.dumps(metadata, sort_keys=True)},
        )
        plt.close(fig)
        _json(
            path.with_suffix(".provenance.json"),
            {
                **metadata,
                "input_hashes": dict(inputs),
                "artifact_hash": sha256_file(path),
            },
        )
        paths.append(path)

    colors = {"RAW": "#2364a0", "RES": "#b14a3a"}
    books = read_frame(root / "derived/portfolios/book_daily.parquet").filter(
        pl.col("construction") == settings["portfolios"]["primary"]
    )
    months = read_frame(root / "derived/portfolios/book_monthly.parquet").filter(
        pl.col("construction") == settings["portfolios"]["primary"], pl.col("series") == "RAW"
    )
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for series, color in colors.items():
        sample = books.filter(pl.col("series") == series).sort("date")
        ax.plot(sample["date"], sample["nav"], label=series, color=color, linewidth=1.3)
    for day in months.sort("monthly_return").head(settings["report"]["worst_months"])["date"]:
        ax.axvspan(day.replace(day=1), day, color="#444444", alpha=0.10)
    ax.set(yscale="log", title="Unscaled momentum books", ylabel="NAV (log scale)", xlabel="Date")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.supxlabel("Gross signed-position books; shading marks RAW's ten worst calendar months.")
    save(fig, "final_1_momentum_books.png")

    redundancy = read_frame(root / "derived/l2/redundancy_design.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained", sharey=True)
    for ax, series in zip(axes, colors, strict=True):
        frame = redundancy.filter(pl.col("series") == series)
        x, y = frame.select(FEATURE_COLUMNS).to_numpy(), frame.select(L2_COLUMNS).to_numpy()
        matrix = np.corrcoef(np.c_[x, y], rowvar=False)[
            len(FEATURE_COLUMNS) :, : len(FEATURE_COLUMNS)
        ]
        plot = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set(title=series, xticks=range(len(FEATURE_COLUMNS)), yticks=range(len(L2_COLUMNS)))
        ax.set_xticklabels(FEATURE_COLUMNS, rotation=75, ha="right", fontsize=8)
        ax.set_yticklabels(L2_COLUMNS, fontsize=9)
    fig.colorbar(plot, ax=axes, label="Pearson correlation", shrink=0.8)
    fig.suptitle("L2 against L1: accepted design-period redundancy, 120 month-ends")
    save(fig, "final_2_redundancy.png")

    audit_path = stage / "derived/final/_audit.json"
    inputs[str(audit_path)] = sha256_file(audit_path)
    monthly = read_frame(stage / "derived/final/copula_monthly.parquet")
    timeline = serving_timeline(forecasts, fits, _read(audit_path), monthly)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for series, color in colors.items():
        sample = books.filter(
            pl.col("series") == series,
            pl.col("date").is_between(timeline["date"].min(), timeline["date"].max()),
        ).sort("date")
        axes[0].plot(sample["date"], sample["nav"] / sample["nav"][0], label=series, color=color)
        sample = timeline.filter(pl.col("series") == series).sort("date")
        axes[1].step(
            sample["date"], sample["filtered_state_0"], where="post", label=series, color=color
        )
    shared = timeline.filter(pl.col("series") == "RAW").sort("date")
    axes[2].step(shared["date"], shared["nearest_centroid"], where="post", color="#65558b")
    axes[0].set(
        yscale="log",
        ylabel="Book NAV, rebased",
        title="States and geometry available at each origin",
    )
    axes[1].set(ylabel="Filtered P(state 0)", ylim=(-0.03, 1.03))
    axes[2].set(
        ylabel="Shared centroid",
        yticks=range(1, settings["ot"]["k_centroids"] + 1),
        xlabel="Holdout origin",
    )
    for ax in axes[:2]:
        ax.legend(frameon=False, ncol=2)
    fig.supxlabel(
        "Serving annual fits only; state labels are statistical, not named economic phases."
    )
    save(fig, "final_3_regime_timeline.png")

    models = ("L1+L2", "L1+L2+CM+OT", "best+HMMF2")
    losses = cumulative_loss_difference(forecasts, models)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
    for ax, series in zip(axes, colors, strict=True):
        for model in models:
            sample = losses.filter(pl.col("series") == series, pl.col("model") == model)
            ax.plot(sample["date"], sample["cumulative_loss_improvement"], label=model)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set(title=series, xlabel="Holdout origin")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Cumulative L1 squared loss − model squared loss")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Saved holdout forecast loss against L1: upward favors the added features")
    save(fig, "final_4_cumulative_loss.png")

    daily = read_frame(stage / "derived/final/strategy_daily.parquet").filter(
        pl.col("cost_bp") == settings["strategy"]["cost_bp"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    palette = plt.get_cmap("tab10")
    for ax, series in zip(axes, colors, strict=True):
        for index, strategy in enumerate(HEADLINE_STRATEGIES):
            frame = daily.filter(pl.col("series") == series, pl.col("strategy") == strategy).sort(
                "date"
            )
            ax.plot(
                [frame["date"][0] - timedelta(days=1), *frame["date"].to_list()],
                [1.0, *frame["nav"].to_list()],
                label=strategy,
                color=palette(index),
                linewidth=1.0,
            )
        ax.set(title=series, ylabel="Net strategy NAV", xlabel="Holding date", yscale="log")
        ax.grid(alpha=0.15)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Frozen risk-managed accounts: 10 bp per traded notional")
    save(fig, "final_5_net_equity.png")
    return paths


def _optional_summary(path: Path, inputs: dict[str, str]) -> dict | None:
    if not path.exists():
        return None
    inputs[str(path)] = sha256_file(path)
    return _read(path)


def _execution_summary(root: Path, name: str, inputs: dict[str, str]) -> dict | None:
    pointer = _optional_summary(root / f"derived/robustness/{name}/latest.json", inputs)
    if pointer is None:
        return None
    path = Path(pointer["output"]) / "summary.json"
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Robustness summary is outside the configured data root")
    return _read(_checked(path, pointer["summary_hash"], inputs))


def _flatten(row: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(_flatten(value, name + "_"))
        else:
            result[name] = json.dumps(value) if isinstance(value, list) else value
    return result


def robustness_rows(variants: dict | None, daily: dict | None) -> list[dict]:
    """Copy one saved row per variant, including the primary RAW_c−RES H2."""
    cases = list(variants["variants"]) if variants else []
    if daily:
        cases.append(
            {
                "variant": "daily",
                "h3": daily["h3"],
                "h2": _one(daily["h2"], primary=True),
                "strategies": [],
                "single_change": "daily forecast origins; monthly trading unchanged",
                "total_elapsed_seconds": daily.get("elapsed_seconds"),
            }
        )
    rows = []
    for case in cases:
        is_daily = case["variant"] == "daily"
        h2 = case["h2"]
        row = {
            "variant": case["variant"],
            "label": "pre-specified robustness",
            "single_change": json.dumps(case["single_change"]),
            "elapsed_seconds": case.get("total_elapsed_seconds"),
            "h2_contrast": h2["contrast"] if h2 else None,
            "h2_lower": h2["interval"]["lower"] if h2 else None,
            "h2_upper": h2["interval"]["upper"] if h2 else None,
            "trading_status": "N/A: daily forecast origins; monthly trading unchanged"
            if is_daily
            else "10 bp monthly execution",
        }
        for series in ("RAW", "RES"):
            prefix = series.lower()
            metric = _one(case["h3"], series=series)
            row.update(
                {
                    f"{prefix}_r2": metric["r2_oos"],
                    f"{prefix}_cw_pvalue": metric["cw"]["pvalue"],
                    f"{prefix}_cw_holm_pvalue": metric["cw_holm_pvalue"],
                    f"{prefix}_origins": metric["n"],
                    f"{prefix}_h3_lower": metric["mse_improvement_interval"]["lower"],
                    f"{prefix}_h3_upper": metric["mse_improvement_interval"]["upper"],
                }
            )
            for name, strategy in (("bsc", "BSC"), ("l2", "L1+L2")):
                row[f"{prefix}_{name}_net_lo_sharpe"] = (
                    None
                    if is_daily
                    else _one(case["strategies"], series=series, strategy=strategy)["lo_sharpe"]
                )
        rows.append(row)
    if len({row["variant"] for row in rows}) != len(rows):
        raise ValueError("Robustness presentation requires one row per variant")
    return rows


def cost_grid_rows(
    costs: dict, *, expected_rules: set[tuple[str, str]] | None = None
) -> list[dict]:
    """Pivot saved Lo Sharpes only after validating every rule × cost cell."""
    rates = sorted(costs["cost_grid_bp"])
    if len(rates) != 3 or len(set(rates)) != 3 or not all(np.isfinite(rates)):
        raise ValueError("Cost presentation requires three distinct finite registered rates")
    cells = {}
    for metric in costs["metrics"]:
        key = (metric["series"], metric["strategy"], metric["cost_bp"])
        if key in cells or key[2] not in rates:
            raise ValueError("Duplicate or unregistered cost-grid account")
        cells[key] = metric["lo_sharpe"]
    rules = {(series, strategy) for series, strategy, _ in cells}
    if (
        len(rules) != costs["frozen_rules"]
        or len(cells) != costs["accounts"]
        or len(cells) != len(rules) * len(rates)
        or (expected_rules is not None and rules != expected_rules)
    ):
        raise ValueError("Cost-grid coverage differs from the frozen rule × rate registry")
    rows = []
    for series, strategy in sorted(rules):
        row = {"series": series, "strategy": strategy}
        for rate in rates:
            if (series, strategy, rate) not in cells:
                raise ValueError("Missing registered cost-grid account")
            row[f"lo_sharpe_{rate:g}bp"] = cells[series, strategy, rate]
        rows.append(row)
    return rows


# Released evidence: source table under results/tables -> published file under results/.
PUBLISHED_TABLES = {
    "final_forecasts": "forecast_results",
    "final_strategies": "strategy_results",
    "holdout_h3": "primary_hypothesis",
    "holdout_h2": "market_state_contrast",
    "holdout_h4": "transport_comparison",
    "holdout_h4q": "quadratic_control",
    "holdout_h5a": "state_features",
    "holdout_h5b": "state_persistence",
    "baseline_size_check": "placebo_audit",
    "robustness_holdout": "robustness",
    "robustness_costs": "cost_sensitivity",
    "holdout_baseline_forecasts": "baseline_forecasts",
    "holdout_sharpe_differences": "sharpe_differences",
    "data_checks": "data_checks",
    "availability": "data_availability",
    "price_integrity_by_year": "price_integrity_by_year",
    "security_mapping": "security_mapping",
    "coverage_by_year": "data_coverage_by_year",
    "universe_by_month": "universe_by_month",
    "sic_by_year": "sic_by_year",
    "cap_coverage_by_year": "cap_coverage_by_year",
    "cap_repair_status": "cap_repair_status",
    "delisting_value": "delisting_value",
    "data_flags": "data_flags",
    "m2_reconciliation": "reconciliation",
    "m2_reconciliation_all_months": "reconciliation_all_months",
    "m2_bsc_summary": "bsc_replication",
    "m2_book_summary": "momentum_book_summary",
    "m2_long_history_summary": "long_history_summary",
    "bdp_recovery": "bdp_recovery",
}
# Historical records the pipeline does not rebuild: the superseded registered
# design, its placebo audit and strategies, and the summary of the private trial log.
FIXED_RECORDS = (
    "registered_forecasts_superseded",
    "registered_strategies_superseded",
    "registered_placebos_superseded",
    "research_attempts",
)
PUBLISHED_FIGURES = {
    "final_1_momentum_books": "momentum_books",
    "final_2_redundancy": "feature_redundancy",
    "final_3_regime_timeline": "state_timeline",
    "final_4_cumulative_loss": "forecast_loss",
    "final_5_net_equity": "net_equity",
}
# Free-text cells that cite review notes; the results guide explains them instead.
DROPPED_FIELDS = {"m2_reconciliation": {"explanation"}}
HOLDOUT_TABLES = ("h2", "h3", "h4", "h4q", "h5a", "h5b", "baseline_forecasts",
                  "sharpe_differences")


def strip_png_metadata(content: bytes) -> bytes:
    """Retain image chunks byte for byte, removing textual and EXIF metadata."""
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG")
    chunks = [content[:8]]
    position = 8
    while position < len(content):
        length = struct.unpack(">I", content[position:position + 4])[0]
        kind = content[position + 4:position + 8]
        end = position + length + 12
        if kind not in {b"tEXt", b"zTXt", b"iTXt", b"eXIf"}:
            chunks.append(content[position:end])
        position = end
    return b"".join(chunks)


def export_table(source: Path, target: Path, dropped: set[str] = frozenset()) -> int:
    """Copy every retained CSV cell exactly; drop strategy IDs and listed fields."""
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = [field for field in reader.fieldnames
                  if field not in {"strategy_id", *dropped}]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with target.open(newline="") as stream:
        if list(csv.DictReader(stream)) != [{key: row[key] for key in fields} for row in rows]:
            raise ValueError(f"Changed aggregate content: {target.name}")
    return len(rows)


def publish_results(destination: Path | None = None) -> dict[str, list[str]]:
    """Write released tables and figures from the generated sources that exist."""
    destination = destination or REPO
    tables = REPO / "results/tables"
    figures = REPO / "results/figures"
    written, missing = [], []
    for source, name in PUBLISHED_TABLES.items():
        path = tables / f"{source}.csv"
        if not path.is_file():
            missing.append(f"results/{name}.csv")
            continue
        export_table(path, destination / "results" / f"{name}.csv",
                     DROPPED_FIELDS.get(source, set()))
        written.append(f"results/{name}.csv")
    for source, name in PUBLISHED_FIGURES.items():
        path = figures / f"{source}.png"
        if not path.is_file():
            missing.append(f"figures/{name}.png")
            continue
        target = destination / "figures" / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(strip_png_metadata(path.read_bytes()))
        written.append(f"figures/{name}.png")
    return {"written": written, "missing": missing, "fixed_records": list(FIXED_RECORDS)}


def build_final_report(settings: dict, *, rebuild_figures: bool = True) -> dict:
    """Regenerate headline tables, figures and robustness tables, then publish them.

    Inputs are the first completed frozen holdout, the design-period summary and
    any saved robustness results; they are verified and never overwritten.
    """
    from momenta.frozen_holdout import FROZEN_PATH
    from momenta.l3_report import load_design_summary

    root = Path(settings["data"]["data_root"])
    inputs: dict[str, str] = {}
    receipt_path = root / "derived/final/holdout/completed.json"
    if not receipt_path.is_file():
        raise ValueError("The report requires a completed frozen holdout; run momenta holdout")
    receipt = _read(receipt_path)
    inputs[str(receipt_path)] = sha256_file(receipt_path)
    if receipt["identity"]["mock"]:
        raise ValueError("The report cannot present a mock holdout")
    _checked(FROZEN_PATH, receipt["identity"]["freeze_hash"], inputs)
    stage = Path(receipt["stage"])
    for relative, digest in receipt["artifact_hashes"].items():
        _checked(stage / relative, digest, inputs)
    holdout = _read(stage / "derived/final/numeric_results.json")
    design_summary = load_design_summary(settings)
    if design_summary is None:
        raise ValueError("The report requires the design-period L3 results; run momenta design")
    design_path = root / "derived/final/study/numeric_results.json"
    _checked(design_path, design_summary["artifacts"][str(design_path)], inputs)
    design = _read(design_path)
    if design["best_models"] != holdout["best_models"]:
        raise ValueError("Holdout cannot reselect the design's frozen best model")
    metadata = {
        **provenance(REPO, Path(settings["config_path"]), Manifest(root)),
        "report_source_hash": sha256_file(Path(__file__)),
        "input_hashes": inputs,
        "frozen_holdout": receipt,
        "report_only": True,
    }
    forecast_rows, strategy_rows = [], []
    for period, saved in (("design", design), ("holdout", holdout)):
        forecast_rows.extend(
            {"period": period, **row}
            for row in headline_forecasts(saved["tables"], saved["best_models"])
        )
        strategy_rows.extend(
            {"period": period, **row}
            for row in headline_strategies(saved["tables"], settings["strategy"]["cost_bp"])
        )
    tables = REPO / "results/tables"
    paths = [tables / "final_forecasts.csv", tables / "final_strategies.csv"]
    _csv(paths[0], forecast_rows, metadata)
    _csv(paths[1], strategy_rows, metadata)
    for name in HOLDOUT_TABLES:
        source = stage / "derived/final/tables" / f"{name}.csv"
        destination = tables / f"holdout_{name}.csv"
        destination.write_bytes(source.read_bytes())
        paths.append(destination)
    if rebuild_figures:
        paths.extend(
            _plot_figures(
                settings,
                root,
                stage,
                pl.read_parquet(stage / "derived/final/forecasts.parquet"),
                pl.read_parquet(stage / "derived/final/fits.parquet"),
                metadata,
                inputs,
            )
        )
    robustness = _optional_summary(root / "derived/robustness/summary.json", inputs)
    daily = _execution_summary(root, "daily", inputs)
    costs = _execution_summary(root, "costs", inputs)
    compact_rows = robustness_rows(robustness, daily)
    if compact_rows:
        destination = tables / "robustness_holdout.csv"
        _csv(destination, compact_rows, metadata)
        paths.append(destination)
    if daily:
        for name in ("metrics", "comparisons"):
            path = next(Path(p) for p in daily["artifacts"] if Path(p).name == name + ".json")
            saved = _read(_checked(path, daily["artifacts"][str(path)], inputs))
            groups = (
                {"forecasts": saved}
                if name == "metrics"
                else {"comparisons": saved["nested_comparisons"], "h2": saved["h2"]}
            )
            for kind, rows in groups.items():
                destination = tables / f"robustness_daily_{kind}.csv"
                _csv(destination, [_flatten(r) for r in rows], metadata)
                paths.append(destination)
    if costs:
        if sorted(costs["cost_grid_bp"]) != sorted(settings["strategy"]["cost_grid_bp"]):
            raise ValueError("Cost results changed the registered cost rates")
        cost_grid_rows(
            costs,
            expected_rules={(r["series"], r["strategy"]) for r in holdout["tables"]["strategies"]},
        )
        destination = tables / "robustness_costs.csv"
        _csv(destination, costs["metrics"], metadata)
        paths.append(destination)
    published = publish_results()
    summary = {
        "provenance": metadata,
        "robustness_available": robustness is not None,
        "daily_available": daily is not None,
        "costs_available": costs is not None,
        "artifacts": {str(p): sha256_file(p) for p in paths},
        "published": published,
    }
    _json(root / "derived/final/report.json", summary)
    return summary
