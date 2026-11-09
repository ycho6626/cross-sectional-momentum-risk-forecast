"""The L3 study: L2, copula, transport and regime extensions of L1 (Methodology).

``run_design`` fits every registered model on design-period origins, selects
the best feature prefix per series and runs the strategy accounts.
``execute_frozen`` runs the same production path on frozen choices inside a
guarded holdout stage. Both share one fit, account and evaluation pipeline.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.baseline_report import account_sharpes, design_population
from momenta.baseline_study import _bdp_paths, _benchmark_forecasts
from momenta.copula import CM_COLUMNS, OT_COLUMNS, Q_COLUMNS, load_copula
from momenta.experiments import trial
from momenta.features import FEATURE_BLOCKS, FEATURE_SETS
from momenta.freeze import source_config_hash
from momenta.l2 import L2_COLUMNS
from momenta.l3_inputs import prepare_inputs as prepare_inputs
from momenta.l3_transforms import LearnedTransforms, carry
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO
from momenta.walkforward import assert_design_origins, walk_forward

PREFIXES = ("L1", "L1+L2", "L1+L2+CM", "L1+L2+CM+OT")
QUADRATIC = ("L1+L2+CM+Q", "L1+L2+CM+Q+OT")
HMM_MODELS = tuple(f"best+HMM{kind}{k}" for k in (2, 3) for kind in ("F", "E"))
DIAGNOSTICS = ("hmm_recovery", "copula_synthetic", "separation")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def blocks_for(model: str, *, best: str | None = None) -> dict[str, tuple[str, ...]]:
    """Ordered registered prefix, with Q before OT and the regime block last."""
    if model in FEATURE_SETS:
        columns = FEATURE_SETS[model]
        return {
            key: tuple(c for c in names if c in columns)
            for key, names in FEATURE_BLOCKS.items()
            if any(c in columns for c in names)
        }
    prefix = best if model.startswith("best+") else model
    if prefix not in (*PREFIXES, *QUADRATIC):
        raise ValueError(f"Unknown registered L3 prefix: {prefix}")
    blocks = {key: FEATURE_BLOCKS[key] for key in ("HAR", "CORE", "M")}
    for token, names in (
        ("L2", L2_COLUMNS),
        ("CM", CM_COLUMNS),
        ("Q", Q_COLUMNS),
        ("OT", OT_COLUMNS),
    ):
        if token in prefix.split("+"):
            blocks[token] = tuple(names)
    if model.startswith("best+"):
        states = int(model[-1])
        kind = model.removeprefix("best+")[:-1]
        blocks[kind] = tuple(f"{kind.lower()}_{i}" for i in range(states - 1))
    return blocks


def select_best(forecasts: pl.DataFrame, series: str) -> str:
    """Select lowest design MSE among four prefixes, ties to shorter."""
    losses = []
    origins = None
    for name in PREFIXES:
        frame = forecasts.filter(
            pl.col("series") == series, pl.col("target") == "y1", pl.col("feature_set") == name
        ).sort("origin")
        if frame.is_empty() or (origins is not None and not frame["origin"].equals(origins)):
            raise ValueError("Best-feature selection needs matched complete prefix samples")
        origins = frame["origin"]
        losses.append(float(((frame["actual"] - frame["prediction"]) ** 2).mean()))
    return PREFIXES[int(np.argmin(losses))]


def check_prefixes(forecasts: pl.DataFrame, fits: pl.DataFrame, best: dict) -> list[dict]:
    """Require unchanged support/earlier penalties and exact exclusion nesting."""
    rows = []
    lookup = {
        (r["series"], r["feature_set"], r["refit"]): r
        for r in fits.filter(pl.col("target") == "y1").iter_rows(named=True)
    }
    for series in best:
        relationships = [(PREFIXES[i], PREFIXES[i - 1]) for i in range(1, len(PREFIXES))]
        relationships += [(QUADRATIC[0], PREFIXES[2]), (QUADRATIC[1], QUADRATIC[0])]
        relationships += [(name, best[series]) for name in HMM_MODELS]
        for larger, smaller in relationships:
            members = [
                row for (s, model, _), row in lookup.items() if s == series and model == larger
            ]
            for row in members:
                previous = lookup[series, smaller, row["refit"]]
                penalties = json.loads(row["block_lambdas_json"])
                old = json.loads(previous["block_lambdas_json"])
                if any(penalties[key] != value for key, value in old.items()):
                    raise ValueError(
                        f"Earlier penalties changed for {series}/{larger}/{row['refit']}"
                    )
                fields = (
                    "training_count",
                    "inner_count",
                    "validation_count",
                    "training_target_end_max",
                    "inner_target_end_max",
                    "inner_fit_cutoff",
                )
                if any(row[key] != previous[key] for key in fields):
                    raise ValueError("An L3 extension changed prefix training support")
                appended = [key for key in penalties if key not in old]
                excluded = all(penalties[key] == "exclude" for key in appended)
                a = forecasts.filter(
                    pl.col("series") == series,
                    pl.col("target") == "y1",
                    pl.col("feature_set") == larger,
                    pl.col("refit") == row["refit"],
                ).sort("origin")
                b = forecasts.filter(
                    pl.col("series") == series,
                    pl.col("target") == "y1",
                    pl.col("feature_set") == smaller,
                    pl.col("refit") == row["refit"],
                ).sort("origin")
                if not a["origin"].equals(b["origin"]):
                    raise ValueError("Nested forecasts differ in scored origins")
                exact = np.array_equal(a["prediction"].to_numpy(), b["prediction"].to_numpy())
                if excluded and not exact:
                    raise ValueError(f"Excluded {larger} does not reproduce {smaller} exactly")
                rows.append(
                    {
                        "series": series,
                        "model": larger,
                        "prefix": smaller,
                        "refit": str(row["refit"]),
                        "earlier_penalties_match": True,
                        "training_support_matches": True,
                        "extension_excluded": excluded,
                        "excluded_prefix_exact": exact if excluded else None,
                    }
                )
    return rows


def design_root(settings: dict) -> Path:
    """The data root holding design-period evidence; a holdout stage records its source."""
    return Path(settings.get("_design_root", settings["data"]["data_root"]))


def _baseline_summary(settings: dict) -> dict:
    """Read the baseline summary; a design run first validates it against current inputs."""
    if "_execution_provenance" in settings:
        # A holdout stage reads the design-period record as saved (sorted JSON keys).
        return json.loads((design_root(settings) / "derived/baseline/summary.json").read_text())
    from momenta.baseline_report import load_summary

    summary = load_summary(settings)
    if summary is None:
        raise ValueError("The L3 design requires the baseline study; run momenta design")
    if not (summary["size_check_complete"] and summary["size_check_bias_passed"]):
        raise ValueError("The baseline placebo size check is incomplete or failed")
    return summary


def _diagnostics(root: Path) -> dict:
    directory = root / "derived/final/diagnostics"
    result = {}
    for name in DIAGNOSTICS:
        path = directory / f"{name}.json"
        if path.exists():
            record = json.loads(path.read_text())
            result[name] = record.get("metrics", record)
    return result


def _population(monthly: pl.DataFrame, frozen: dict | None) -> dict:
    """Design: recorded design-period population; holdout: frozen trading IDs (Methodology)."""
    if frozen is None:
        return design_population(monthly)
    population = frozen["trial_population"]
    current = account_sharpes(monthly)
    if not set(current["account_ids"]).issubset(set(population["trading_ids"])):
        raise ValueError("Frozen execution introduced an unregistered trading specification")
    return {
        "distinct_trading_ids": len(population["trading_ids"]),
        "monthly_sharpes": current["monthly_sharpes"],
        "trading_ids": population["trading_ids"],
        "first_month": current["first_month"],
        "last_month": current["last_month"],
    }


def _persisted_audit(path: Path, computed: dict) -> dict:
    """Return the saved audit, keeping computed column order for the row-exported tables.

    The design tables were evaluated from this saved record after the design
    population was attached; the holdout was evaluated in the same pass. A JSON
    round trip sorts record keys, which sets the regime-restart column order.
    The prefix-check and break-even tables keep their computed column order.
    """
    saved = json.loads(path.read_text())
    for parent, original, key in (
        (saved, computed, "prefix_checks"),
        (saved["strategy_audit"], computed["strategy_audit"], "break_even"),
    ):
        fields = list(dict.fromkeys(field for row in original[key] for field in row))
        if {field for row in parent[key] for field in row} != set(fields):
            raise ValueError(f"Saved audit fields differ from the computed {key}")
        parent[key] = [{field: row[field] for field in fields if field in row}
                       for row in parent[key]]
    return saved


def _validated_inputs(settings: dict, through: date) -> dict:
    """Refuse changed build inputs before any feature is read or fitted."""
    root = Path(settings["data"]["data_root"])
    path = root / "derived/final/inputs.metadata.json"
    record = json.loads(path.read_text())
    if date.fromisoformat(record["through"]) < through:
        raise ValueError("L3 inputs end before the requested origin")
    checks = {
        str(Path(settings["config_path"])): record["config_hash"],
        str(root / "raw/manifest.json"): record["data_manifest_hash"],
        **record["input_hashes"], **record["source_hashes"],
    }
    for name, digest in checks.items():
        item = Path(name)
        if not item.is_file() or sha256_file(item) != digest:
            raise ValueError(f"L3 input changed after preparation: {item.name}")
    return record


def _execute(
    settings: dict,
    *,
    output: Path,
    through: date,
    mode: str,
    metadata: dict,
    frozen: dict | None = None,
) -> dict:
    """One shared production path for design, mock and frozen holdout."""
    from momenta.l3_report import evaluate_l3
    from momenta.l3_strategies import run_model_strategies
    from momenta.strategy_study import run_strategies

    assert_design_origins([through], settings)
    baseline_summary = _baseline_summary(settings)
    root = Path(settings["data"]["data_root"])
    input_metadata = _validated_inputs(settings, through)
    inputs = {name: Path(value) for name, value in input_metadata["paths"].items()}
    features = pl.read_parquet(inputs["features"]).filter(pl.col("date") <= through)
    targets = pl.read_parquet(inputs["targets"]).filter(pl.col("origin") <= through)
    monthly = pl.read_parquet(inputs["l2"]).filter(pl.col("date") <= through)
    copula, copula_metadata = load_copula(settings)
    copula = copula.filter(pl.col("date") <= through)
    features = carry(features, monthly.select("date", *L2_COLUMNS))
    features = carry(features, copula.select("date", *CM_COLUMNS, *Q_COLUMNS))
    features = features.with_columns(
        [pl.lit(0.0).alias(name) for name in (*OT_COLUMNS, "hmmf_0", "hmmf_1", "hmme_0", "hmme_1")]
    )
    warmup = carry(pl.read_parquet(inputs["warmup"]), monthly.select("date", *L2_COLUMNS))
    days = pl.read_parquet(root / "curated/trading_calendar.parquet")["date"].sort().to_list()
    output.mkdir(parents=True, exist_ok=True)
    metadata.update(input_hashes=dict(input_metadata["input_hashes"]))
    metadata["input_hashes"].update(copula_metadata["input_hashes"])
    diagnostic_root = design_root(settings) / "derived/final/diagnostics"
    metadata["input_hashes"].update({str(path): sha256_file(path)
                                     for path in diagnostic_root.glob("*.json")})
    learned = LearnedTransforms(settings, settings, days, copula, warmup, output, metadata)
    forecasts, fits, training = [], [], []
    series_names = ["RAW"] if mode == "thin" else ["RAW", "RAW_c", "RES"]
    l3_series = [s for s in settings["l3"]["series"] if s in series_names]

    def fit(series: str, target: str, name: str, best: str | None = None) -> None:
        blocks = blocks_for(name, best=best)
        columns = [column for names in blocks.values() for column in names]
        states = int(name[-1]) if name.startswith("best+") else None
        specification = {
            "series": series,
            "target": target,
            "feature_set": name,
            "best_model": best,
            "blocks": blocks,
            "walkforward": settings["walkforward"],
            "ot": settings["ot"],
            "hmm": settings["hmm"],
            "bdp": settings["bdp"],
        }
        print(f"L3 {mode}: {series}/{target}/{name}", flush=True)
        with trial(
            settings,
            experiment=f"l3:{name}:{target}",
            kind="forecast",
            series=series,
            sample=f"{settings['walkforward']['design_first_origin']}:{through}",
            specification=specification,
        ) as attempt:
            run = walk_forward(
                features,
                targets,
                days,
                settings,
                series=series,
                target=target,
                feature_set=name,
                columns=columns,
                blocks=blocks,
                through=through,
                transform=learned.transform(ot="OT" in blocks, states=states),
            )
            forecasts.append(run.forecasts)
            fits.append(run.fits)
            if run.training_predictions.height:
                training.append(run.training_predictions)
            attempt.metrics.update(
                origins=run.forecasts.height,
                refits=run.fits.height,
                mse=float(((run.forecasts["actual"] - run.forecasts["prediction"]) ** 2).mean()),
            )

    for series in series_names:
        for target in ("y1", "y2", "y3"):
            for name in FEATURE_SETS:
                fit(series, target, name)
    for series in l3_series:
        for name in (*PREFIXES[1:], *QUADRATIC):
            fit(series, "y1", name)
    frame = pl.concat(forecasts, how="diagonal_relaxed")
    best_models = (
        {s: frozen["best_models"][s] for s in l3_series}
        if frozen
        else {s: select_best(frame, s) for s in l3_series}
    )
    for series in l3_series:
        for name in HMM_MODELS:
            fit(series, "y1", name, best_models[series])
    frame = _benchmark_forecasts(pl.concat(forecasts, how="diagonal_relaxed"))
    fit_frame = pl.concat(fits, how="diagonal_relaxed")
    training_frame = pl.concat(training, how="diagonal_relaxed")
    prefix_checks = check_prefixes(frame, fit_frame, best_models)
    # The baseline and L2 studies' forecasts are an independent regression check.
    if mode in {"thin", "design"}:
        for folder, names in (("baseline", list(FEATURE_SETS)), ("l2", ["L1+L2"])):
            path = root / f"derived/{folder}/forecasts.parquet"
            expected = pl.read_parquet(path)
            if folder == "l2":
                expected = expected.with_columns(pl.lit("L1+L2").alias("feature_set"))
            keys = ["series", "target", "feature_set", "origin"]
            compared = frame.filter(pl.col("feature_set").is_in(names)).join(
                expected.select(*keys, pl.col("prediction").alias("study_prediction")),
                on=keys,
                how="left",
                validate="1:1",
            )
            error = (compared["prediction"] - compared["study_prediction"]).abs().max()
            if compared["study_prediction"].null_count() or error > 1e-12:
                raise ValueError(f"L3 changed the {folder} study forecasts: {error}")
            metadata[f"{folder}_max_forecast_difference"] = error
    for name, saved in (
        ("forecasts", frame.sort("series", "target", "feature_set", "origin")),
        ("fits", fit_frame),
        ("training_predictions", training_frame),
    ):
        saved.write_parquet(output / f"{name}.parquet")
        _json(output / f"{name}.metadata.json", metadata)
    baseline_audit = run_strategies(
        settings,
        features,
        frame,
        training_frame,
        last_origin=through,
        output=output / "base_strategies",
        metadata=metadata,
        bdp_paths=lambda cutoff: _bdp_paths(settings, cutoff),
    )
    variants = {
        s: {"L1+L2": "L1+L2", "L1+L2+CM+OT": "L1+L2+CM+OT", "best+HMMF2": "best+HMMF2"}
        for s in l3_series
    }
    model_audit = run_model_strategies(
        settings,
        frame,
        features=features,
        variants=variants,
        best_models=best_models,
        through=through,
        output=output / "model_strategies",
        metadata=metadata,
    )
    numeric = [
        output / "forecasts.parquet",
        output / "fits.parquet",
        output / "training_predictions.parquet",
    ]
    for name in ("strategy_daily", "strategy_monthly", "strategy_rebalances"):
        combined = pl.concat(
            [
                pl.read_parquet(output / directory / f"{name}.parquet")
                for directory in ("base_strategies", "model_strategies")
            ],
            how="diagonal_relaxed",
        )
        path = output / f"{name}.parquet"
        combined.write_parquet(path)
        numeric.append(path)
    monthly_strategies = pl.read_parquet(output / "strategy_monthly.parquet")
    audit = {
        "best_models": best_models,
        "regime_records": learned.records,
        "centroid_records": learned.centroid_records,
        "baseline_summary": baseline_summary,
        "diagnostics": _diagnostics(design_root(settings)),
        "prefix_checks": prefix_checks,
        "strategy_population": _population(monthly_strategies, frozen),
        "strategy_audit": {"break_even": baseline_audit["break_even"] + model_audit["break_even"]},
    }
    _json(output / "_audit.json", audit)
    _json(output / "_metadata.json", metadata)
    if mode in {"thin", "design"}:
        audit = _persisted_audit(output / "_audit.json", audit)
    summary = evaluate_l3(settings, output, metadata, audit)
    summary.update(
        mode=mode,
        complete=mode in {"mock_holdout", "holdout"}
        or (mode == "design" and not metadata["git_dirty"]
            and through == date.fromisoformat(settings["walkforward"]["design_last_origin"])
            and set(best_models) == set(settings["l3"]["series"])),
        fit_count=fit_frame.height,
    )
    summary["artifacts"].update({str(path): sha256_file(path) for path in numeric})
    summary["deterministic_artifacts"].extend(str(path.relative_to(root)) for path in numeric)
    _json(output / "summary.json", summary)
    return summary


def run_design(settings: dict, *, through: date | None = None, thin: bool = False) -> dict:
    """Log and execute the registered design only; preserve the pre-run tree provenance."""
    from momenta.l3_diagnostics import run_diagnostics

    through = through or date.fromisoformat(
        settings["l3"]["thin_last_origin"]
        if thin
        else settings["walkforward"]["design_last_origin"]
    )
    root = Path(settings["data"]["data_root"])
    initial = provenance(REPO, Path(settings["config_path"]), Manifest(root))
    initial.update(mode="thin" if thin else "design", source_config_hash=source_config_hash())
    with trial(
        settings,
        experiment="l3-thin" if thin else "l3",
        kind="diagnostic",
        series="RAW" if thin else "RAW/RAW_c/RES",
        sample=f"design through {through}",
        specification={"study": "L3", "settings": settings["l3"]},
    ) as attempt:
        assert_design_origins([through], settings)
        if not thin:
            run_diagnostics(settings)
        summary = _execute(
            settings,
            output=root / "derived/final" / ("thin" if thin else "study"),
            through=through,
            mode=initial["mode"],
            metadata=initial,
        )
        if not thin:
            _json(root / "derived/final/summary.json", summary)
        attempt.metrics.update(
            best_models=summary["best_models"], forecast_rows=summary["forecast_rows"]
        )
        return summary


def execute_frozen(
    settings: dict,
    *,
    frozen: dict,
    output: Path,
    first_origin: date,
    last_origin: date,
    mock: bool,
) -> dict:
    """Use the same model/account/report path with the immutable selected prefixes."""
    assert_design_origins([first_origin, last_origin], settings)
    if (
        first_origin != date.fromisoformat(settings["walkforward"]["design_first_origin"])
        or last_origin != date.fromisoformat(settings["walkforward"]["design_last_origin"])
    ):
        raise ValueError("Frozen execution arguments must match the guarded scoped bounds")
    metadata = dict(settings["_execution_provenance"])
    metadata.update(
        mode="mock_holdout" if mock else "holdout", source_config_hash=source_config_hash()
    )
    return _execute(
        settings,
        output=output,
        through=last_origin,
        mode=metadata["mode"],
        metadata=metadata,
        frozen=frozen,
    )
