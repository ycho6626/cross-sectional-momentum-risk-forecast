"""Paired H2 samples and report-level design-period guards."""

import json
from datetime import date

import polars as pl
import pytest

from momenta import baseline_report
from momenta.baseline_report import forecast_table, h2_table, size_check_table
from momenta.manifest import sha256_file


@pytest.fixture
def settings():
    return {
        "walkforward": {"design_last_origin": "2015-11-30"},
        "evaluation": {
            "bootstrap_reps": 20,
            "block_months": 12,
            "seed": 1,
            "nw_lags_monthly": 3,
            "confidence": 0.95,
            "size_check_max": 0.10,
            "size_check_blocks": [4, 10],
            "size_check_series": ["RAW", "RES"],
            "size_check_max_mean_r2": 0.0,
        },
    }


@pytest.fixture
def forecasts():
    records = []
    for series in ("RAW", "RAW_c", "RES"):
        for feature_set in ("L1", "L1_without_M", "L1-own"):
            for month in (1, 2, 3):
                # Baseline errors all2; full errors1 for RAW/RAW_c, sqrt2 for RES.
                prediction = 2.0 if feature_set != "L1" else 2**0.5 if series == "RES" else 1.0
                records.append(
                    {
                        "origin": date(2006, month, 28),
                        "refit": date(2006, 1, 2),
                        "series": series,
                        "feature_set": feature_set,
                        "target": "y1",
                        "actual": 0.0,
                        "prediction": prediction,
                        "base_prediction": 2.0,
                        "residual_variance": 0.0,
                    }
                )
    return pl.DataFrame(records)


def test_h2_common_universe_has_hand_calculated_contrast(settings, forecasts):
    rows = h2_table(forecasts, settings)
    assert len(rows) == 4
    primary = [row for row in rows if row["primary"]]
    assert len(primary) == 1
    assert primary[0]["series_a"] == "RAW_c"
    assert primary[0]["reduced_model"] == "L1_without_M"
    assert primary[0]["delta_a"] == 0.75
    assert primary[0]["delta_b"] == pytest.approx(0.5)
    assert primary[0]["contrast_lower"] == pytest.approx(0.25)
    assert primary[0]["contrast_upper"] == pytest.approx(0.25)


def test_size_failure_marks_cw_descriptive(settings, forecasts):
    size = [
        {"series": series, "k": k, "cw_rejection_rate": rate, "complete": True}
        for series, rate in (("RAW", 0.11), ("RES", 0.05))
        for k in (4, 10)
    ]
    rows = forecast_table(forecasts, settings, size_check=size)
    expected = {
        "RAW": "descriptive_size_failure",
        "RES": "design_secondary",
        "RAW_c": "size_check_unavailable",
    }
    assert all(row["cw_inference"] == expected[row["series"]] for row in rows)
    assert all(row["dm_inference"] == "descriptive" for row in rows)
    own = [row for row in rows if row["feature_set"] == "L1-own"]
    assert all(row["cw_pvalue"] is None for row in own)
    incomplete = forecast_table(
        forecasts.filter(pl.col("series") == "RAW"),
        settings,
        size_check=[{"series": "RAW", "k": 4, "cw_rejection_rate": 0.0, "complete": False}],
    )
    assert all(row["cw_inference"] == "size_check_pending" for row in incomplete)
    for row in size:
        row["bias_exceeded"] = row["series"] == "RES"
    failed_bias = forecast_table(forecasts, settings, size_check=size)
    assert all(
        row["cw_inference"] == "invalid_placebo_bias"
        for row in failed_bias
        if row["series"] == "RES"
    )


def test_holm_forecast_families_pair_raw_res_by_target_and_feature_set(
    settings,
    forecasts,
    monkeypatch,
):
    def metrics(frame, *, target, **kwargs):
        pvalues = {"RAW": 0.01, "RES": 0.04, "RAW_c": 0.001}
        if target == "y2":
            pvalues = {"RAW": 0.10, "RES": 0.02, "RAW_c": 0.001}
        return {"cw": {"pvalue": pvalues[frame["series"][0]]}}

    monkeypatch.setattr(baseline_report, "forecast_metrics", metrics)
    forecasts = pl.concat([forecasts, forecasts.with_columns(pl.lit("y2").alias("target"))])
    size = [
        {"series": series, "k": k, "cw_rejection_rate": 0.11, "complete": True}
        for series in ("RAW", "RES")
        for k in (4, 10)
    ]
    rows = forecast_table(forecasts, settings, size_check=size)
    for row in rows:
        expected = (
            None
            if row["series"] == "RAW_c"
            else {
                ("y1", "RAW"): 0.02,
                ("y1", "RES"): 0.04,
                ("y2", "RAW"): 0.10,
                ("y2", "RES"): 0.04,
            }[row["target"], row["series"]]
        )
        assert row["cw_holm_pvalue"] == expected
        assert row["cw_inference"] == (
            "size_check_unavailable" if row["series"] == "RAW_c" else "descriptive_size_failure"
        )
        if row["series"] == "RAW_c":
            assert row["cw_pvalue"] == 0.001
    incomplete = forecast_table(forecasts.filter(pl.col("series") != "RES"), settings)
    assert all(row["cw_holm_pvalue"] is None for row in incomplete)


@pytest.mark.parametrize("problem", ["holdout", "late_fit", "duplicate"])
def test_report_rejects_invalid_forecasts(settings, forecasts, problem):
    if problem == "holdout":
        forecasts = forecasts.with_columns(pl.lit(date(2016, 1, 29)).alias("origin"))
    elif problem == "late_fit":
        forecasts = forecasts.with_columns(pl.lit(date(2007, 1, 1)).alias("refit"))
    else:
        forecasts = pl.concat([forecasts, forecasts.head(1)])
    with pytest.raises(ValueError):
        forecast_table(forecasts, settings)


def test_h2_rejects_models_with_different_realized_targets(settings, forecasts):
    forecasts = forecasts.with_columns(
        pl.when(pl.col("feature_set") == "L1").then(1.0).otherwise(0.0).alias("actual")
    )
    with pytest.raises(ValueError, match="realized target"):
        h2_table(forecasts, settings)


def test_size_summary_does_not_hide_undefined_replications(settings):
    settings["evaluation"].update(size_check_reps=2, size_check_alpha=0.05)
    frame = pl.DataFrame(
        {
            "series": ["RAW"] * 2,
            "k": [4, 4],
            "rep": [0, 1],
            "cw_p": [0.01, None],
            "dm_p": [0.2, 0.3],
            "r2": [-0.1, 0.1],
            "identical_forecasts": [False] * 2,
            "error": [None] * 2,
            "refits": [2] * 2,
            "placebo_excluded_refits": [1] * 2,
        }
    )
    row = size_check_table(frame, settings)[0]
    assert row["cw_rejection_rate"] == 1
    assert row["cw_undefined"] == 1
    assert row["complete"] is False
    with pytest.raises(ValueError, match="Duplicate"):
        size_check_table(pl.concat([frame, frame.head(1)]), settings)


def test_size_summary_gates_each_series_block_and_counts_identical_nonrejections(settings):
    settings["evaluation"].update(size_check_reps=2, size_check_alpha=0.05)
    frame = pl.DataFrame(
        {
            "series": ["RAW", "RAW", "RES", "RES"],
            "k": [4] * 4,
            "rep": [0, 1, 0, 1],
            "cw_p": [0.01, 0.2, 0.01, None],
            "dm_p": [0.2, 0.3, 0.2, None],
            "r2": [0.1, 0.3, -0.2, 0.0],
            "identical_forecasts": [False, False, False, True],
            "error": [None] * 4,
            "refits": [1, 3, 1, 3],
            "placebo_excluded_refits": [0, 2, 0, 3],
        }
    )
    raw, res = size_check_table(frame, settings)
    assert raw["mean_r2"] == raw["median_r2"] == pytest.approx(0.2)
    assert raw["positive_r2_share"] == 1.0
    assert raw["placebo_excluded_refit_share"] == 0.5
    assert raw["complete"] is True and raw["bias_passed"] is False
    assert raw["bias_exceeded"] is True
    assert res["mean_r2"] == res["median_r2"] == -0.1
    assert res["positive_r2_share"] == 0.0
    assert res["placebo_excluded_refit_share"] == 0.75
    assert res["cw_rejection_rate"] == 0.5
    assert res["dm_rejection_rate"] == 0.0
    assert res["cw_undefined"] == res["dm_undefined"] == 0
    assert res["identical_forecasts"] == 1
    assert res["complete"] is True and res["bias_passed"] is True


def test_feature_coverage_counts_origins_once_and_reports_balanced_panel_shares():
    features = pl.DataFrame(
        {
            "series": ["RAW", "RAW", "RAW", "RES", "RES"],
            "date": [date(2006, month, 28) for month in (1, 2, 3, 1, 3)],
            "book_rv_coverage": [float("nan"), 0.5, 1.0, 0.8, 0.9],
            "comom_count": [1, 3, 4, 1, 4],
            "comom_total": [2, 4, 8, 2, 8],
        }
    )
    forecasts = pl.DataFrame(
        {
            "series": ["RAW", "RAW", "RAW", "RES"],
            "origin": [date(2006, month, 28) for month in (2, 2, 3, 3)],
        }
    )
    rows = baseline_report.feature_coverage_table(features, forecasts)
    keyed = {(row["series"], row["sample"]): row for row in rows}
    built = keyed["RAW", "built_daily_origins"]
    scored = keyed["RAW", "scored_month_end_origins"]
    assert built["origins"] == 3
    assert built["book_coverage_origins"] == 2
    assert built["book_coverage_min"] == 0.5
    assert built["book_coverage_mean"] == 0.75
    assert built["comom_count_min"] == 1
    assert built["comom_count_mean"] == pytest.approx(8 / 3)
    assert built["comom_ratio_mean"] == pytest.approx(7 / 12)
    assert scored["origins"] == 2
    assert scored["first_origin"] == date(2006, 2, 28)
    assert scored["last_origin"] == date(2006, 3, 28)
    assert scored["comom_count_min"] == 3
    assert scored["comom_count_mean"] == 3.5
    assert scored["comom_ratio_min"] == 0.5
    assert scored["comom_ratio_mean"] == 0.625
    assert keyed["RES", "scored_month_end_origins"]["book_coverage_mean"] == 0.9


@pytest.mark.parametrize("thin_scope", [False, True])
def test_evaluator_pins_inputs_and_refuses_changed_artifacts(
    settings,
    forecasts,
    tmp_path,
    monkeypatch,
    thin_scope,
):
    if thin_scope:
        forecasts = forecasts.filter(
            pl.col("series") == "RAW", pl.col("feature_set").is_in(["L1-own", "L1"])
        )
    root = tmp_path / "data"
    output = root / "derived/baseline"
    output.mkdir(parents=True)
    manifest = root / "raw/manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}")
    config = tmp_path / "settings.toml"
    config.write_text("fixture = true")
    population = tmp_path / "config/design_trial_population.json"
    population.parent.mkdir()
    population.write_text(json.dumps({
        "first_month": "2006-01-28", "last_month": "2006-03-28", "distinct_trading_ids": 3,
        "trading_ids": ["BSC", "Static", "earlier"], "monthly_sharpes": {"earlier": 0.25},
    }))
    settings.update(
        data={"data_root": str(root)},
        config_path=str(config),
        replication={"months_per_year": 12},
        strategy={"cvar_alpha": 0.05},
    )
    settings["evaluation"].update(
        size_check_reps=2, size_check_blocks=[4, 10], size_check_alpha=0.05
    )
    monkeypatch.setattr(baseline_report, "REPO", tmp_path)
    monkeypatch.setattr(baseline_report, "code_hash", lambda: "source")
    monkeypatch.setattr(baseline_report, "POPULATION_PATH", population)
    forecasts.write_parquet(output / "forecasts.parquet")
    pl.DataFrame(
        {
            "series": ["RAW"],
            "target": ["y1"],
            "feature_set": ["L1"],
            "refit": [date(2006, 1, 2)],
            "training_count": [30],
            "inner_count": [10],
            "validation_count": [10],
            "block_lambdas_json": ['{"M": "exclude"}'],
            "validation_losses_json": ['{"M": [{"penalty": "exclude", "loss": 1.0}]}'],
        }
    ).write_parquet(output / "fits.parquet")
    forecasts.select("series", pl.col("origin").alias("date")).unique().with_columns(
        pl.lit(0.98).alias("book_rv_coverage"),
        pl.lit(80).alias("comom_count"),
        pl.lit(100).alias("comom_total"),
    ).write_parquet(output / "features.parquet")
    rows = []
    for strategy in ("BSC", "Static"):
        for month, ret in zip((1, 2, 3), (0.1, -0.1, 0.05), strict=True):
            rows.append(
                {
                    "date": date(2006, month, 28),
                    "series": "RAW",
                    "strategy": strategy,
                    "strategy_id": strategy,
                    "cost_bp": 0,
                    "monthly_return": ret,
                    "nav": 1.0 + ret,
                    "turnover": 2.0,
                }
            )
    frame = pl.DataFrame(rows)
    for name in ("strategy_daily", "strategy_monthly", "strategy_rebalances"):
        frame.write_parquet(output / f"{name}.parquet")
    pl.DataFrame(
        {
            "series": ["RAW", "RES"],
            "overlap_days": [1, 2],
            "gap_days": [0, 0],
            "origin_distance_days": [20, 19],
        }
    ).write_parquet(output / "target_overlaps.parquet")
    metadata = {
        "mode": "thin" if thin_scope else "full",
        "config_hash": sha256_file(config),
        "git_hash": "commit",
        "git_dirty": False,
        "data_manifest_hash": sha256_file(manifest),
    }
    audit = {
        "bdp_status": "approximation",
        "calibrations": [{"attained": True}, {"attained": False}],
        "training_target_overlaps_refit": 0,
        "dm_warmup": {
            "first_supported_origin": "1928-03-31",
            "nominal_first_month": "1927-07-01",
            "warmup_omissions": 8,
            "missing_windows": 0,
        },
    }
    audit_path = output / "_audit.json"
    audit_path.write_text(json.dumps(audit))
    moments = output / "bdp_strategy_moments.parquet"
    pl.DataFrame({"expected_return": [0.01]}).write_parquet(moments)
    moments_metadata = moments.with_suffix(".metadata.json")
    moments_metadata.write_text(json.dumps({"artifact_hash": sha256_file(moments)}))
    summary = baseline_report.evaluate_study(settings, output, metadata, audit)
    # Both current accounts replace their (absent) recorded estimates; N stays three.
    assert summary["trial_population"]["distinct_trading_ids"] == 3
    assert set(summary["trial_population"]["monthly_sharpes"]) == {"BSC", "Static", "earlier"}
    counts = {row["dsr_distinct_trading_strategies"] for row in summary["tables"]["strategies"]}
    assert counts == {3}
    assert summary["tables"]["raw_l1_penalties"][0]["block_lambdas_json"] == '{"M": "exclude"}'
    assert summary["size_check_complete"] is False
    assert len(summary["tables"]["h2"]) == (0 if thin_scope else 4)
    if thin_scope:
        assert summary["h2_status"] == "not_built_thin_scope"
        assert summary["h2_missing_models"]
        with pytest.raises(ValueError, match="No common H2 forecasts"):
            baseline_report.evaluate_study(settings, output, metadata | {"mode": "full"}, {})
    assert baseline_report.load_summary(settings)["forecast_rows"] == (6 if thin_scope else 27)
    for name in ("raw_l1_penalties", "feature_coverage"):
        artifact = tmp_path / f"results/tables/baseline_{name}.csv"
        assert summary["artifacts"][str(artifact)] == sha256_file(artifact)
        assert artifact.with_suffix(".provenance.json").exists()
    for source in (audit_path, moments, moments_metadata, population):
        assert summary["provenance"]["input_hashes"][str(source)] == sha256_file(source)
        original = source.read_bytes()
        source.write_bytes(original + b"corruption")
        with pytest.raises(ValueError, match="Stale baseline input or artifact"):
            baseline_report.load_summary(settings)
        source.write_bytes(original)
    # A newly persisted diagnostic must be evaluated before it appears in the report.
    new_input = output / "build_metadata.json"
    new_input.write_text("{}")
    with pytest.raises(ValueError, match="Unevaluated baseline input: build_metadata.json"):
        baseline_report.load_summary(settings)
    new_input.unlink()
    artifact = tmp_path / "results/tables/baseline_forecasts.csv"
    artifact.write_text(artifact.read_text() + "corruption")
    with pytest.raises(ValueError, match="Stale baseline input or artifact"):
        baseline_report.load_summary(settings)


def test_evaluator_checks_recorded_recovery_and_used_cutoff_fits(tmp_path):
    recovery = tmp_path / "bdp_attempt/recovery.json"
    recovery.parent.mkdir()
    recovery.write_text('{"passed": true}')
    (tmp_path / "bdp_decision.json").write_text(
        json.dumps({"artifacts": {str(recovery): sha256_file(recovery)}})
    )
    parameters = {}
    referenced = [recovery]
    for key, cutoff in (("structural", "2005-01-03"), ("inner_structural", "2002-12-02")):
        path = tmp_path / "bdp_paths" / f"{cutoff}.parquet"
        path.parent.mkdir(exist_ok=True)
        pl.DataFrame({"bdp_ci": [0.1]}).write_parquet(path)
        record = {"fit_time": cutoff, "artifact_hash": sha256_file(path)}
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(record))
        parameters[key] = record
        referenced.extend((path, sidecar))
    pl.DataFrame({"parameters_json": [json.dumps(parameters)]}).write_parquet(
        tmp_path / "fits.parquet"
    )
    unused = tmp_path / "bdp_paths/2007-01-03.parquet"
    pl.DataFrame({"bdp_ci": [99.0]}).write_parquet(unused)
    hashes = baseline_report._evaluation_input_hashes(tmp_path, {})
    assert all(hashes[str(path)] == sha256_file(path) for path in referenced)
    assert str(unused) not in hashes
    for path in referenced:
        original = path.read_bytes()
        path.write_bytes(original + (b" " if path.suffix == ".json" else b"changed"))
        # Whitespace changes preserve JSON equality, so change the fitted parameters too.
        if path.parent.name == "bdp_paths" and path.suffix == ".json":
            record = json.loads(path.read_text())
            path.write_text(json.dumps(record | {"fit_time": "2099-01-01"}))
        with pytest.raises(ValueError, match="Changed (BDP recovery|referenced BDP)"):
            baseline_report._evaluation_input_hashes(tmp_path, {})
        path.write_bytes(original)


def test_evaluator_rejects_rewritten_artifact_with_original_sidecar(tmp_path):
    path = tmp_path / "bdp_strategy_moments.parquet"
    pl.DataFrame({"expected_return": [0.01]}).write_parquet(path)
    path.with_suffix(".metadata.json").write_text(json.dumps({"artifact_hash": sha256_file(path)}))
    pl.DataFrame({"expected_return": [0.99]}).write_parquet(path)
    with pytest.raises(ValueError, match="Changed persisted model artifact"):
        baseline_report._evaluation_input_hashes(tmp_path, {})


def test_raw_l1_penalty_audit_exports_only_selected_model_in_refit_order():
    fits = pl.DataFrame(
        {
            "series": ["RAW", "RES", "RAW", "RAW"],
            "target": ["y1"] * 4,
            "feature_set": ["L1", "L1", "L1-own", "L1"],
            "refit": [date(year, 1, 2) for year in (2007, 2007, 2007, 2006)],
            "training_count": [40, 40, 40, 30],
            "inner_count": [10] * 4,
            "validation_count": [20] * 4,
            "block_lambdas_json": ['{"M": "exclude"}'] * 4,
            "validation_losses_json": ['{"M": [{"penalty": "exclude", "loss": 1.0}]}'] * 4,
            "parameters_json": ["not exported"] * 4,
        }
    )
    rows = baseline_report._raw_l1_penalties(fits)
    assert [row["refit"] for row in rows] == [date(2006, 1, 2), date(2007, 1, 2)]
    assert [row["training_count"] for row in rows] == [30, 40]
    assert all("parameters_json" not in row for row in rows)
    with pytest.raises(ValueError, match="Duplicate RAW y1 L1 refit"):
        baseline_report._raw_l1_penalties(pl.concat([fits, fits.head(1)]))
