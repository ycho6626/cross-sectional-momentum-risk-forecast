"""Trading-population multiplicity and numerical artifacts cannot depend on later logs."""

import json
from copy import deepcopy
from datetime import date

import polars as pl
import pytest

from momenta import baseline_report, experiments, holdout, l3_study, l3_transforms


@pytest.fixture
def monthly():
    return pl.DataFrame(
        {
            "date": [date(2013, 1, 31), date(2013, 2, 28), date(2013, 3, 31)] * 2,
            "strategy_id": ["strategy:a"] * 3 + ["strategy:b"] * 3,
            "monthly_return": [0.0, 0.1, 0.2, -0.2, 0.0, 0.2],
        }
    )


@pytest.fixture
def frozen():
    return {
        "trial_population": {
            "all_ids": ["diagnostic:size", "strategy:a", "strategy:b", "strategy:failed"],
            "trading_ids": ["strategy:a", "strategy:b", "strategy:failed"],
        }
    }


def test_frozen_population_keeps_failed_specifications_and_uses_current_account_sharpes(
    monthly,
    frozen,
    monkeypatch,
):
    monkeypatch.setattr(experiments, "read_trials", lambda *a: pytest.fail("live trial log read"))
    before = deepcopy(frozen)
    population = l3_study._population(monthly, frozen)
    # A: mean=.1, sample SD=.1. B: mean=0, sample SD=.2. The failed rule stays in N.
    assert population["monthly_sharpes"] == pytest.approx({"strategy:a": 1.0, "strategy:b": 0.0})
    assert population["distinct_trading_ids"] == 3
    assert population["trading_ids"] == frozen["trial_population"]["trading_ids"]
    assert "strategy:failed" not in population["monthly_sharpes"]
    assert population["first_month"] == "2013-01-31"
    assert population["last_month"] == "2013-03-31"
    assert frozen == before


def test_later_logs_and_row_order_cannot_change_frozen_population(monthly, frozen, monkeypatch):
    monkeypatch.setattr(experiments, "read_trials", lambda *a: [{"strategy_id": "unrelated:new"}])
    first = l3_study._population(monthly, frozen)
    monkeypatch.setattr(experiments, "read_trials", lambda *a: pytest.fail("later log read"))
    second = l3_study._population(monthly.reverse(), frozen)
    assert first == second
    unregistered = monthly.filter(pl.col("strategy_id") == "strategy:a").with_columns(
        pl.lit("strategy:new").alias("strategy_id")
    )
    with pytest.raises(ValueError, match="unregistered trading"):
        l3_study._population(unregistered, frozen)


def test_undefined_sharpes_stay_missing_without_removing_their_specification(monthly, frozen):
    constant = monthly.with_columns(pl.lit(0.0).alias("monthly_return"))
    population = l3_study._population(constant, frozen)
    assert population["monthly_sharpes"] == {}
    assert population["distinct_trading_ids"] == 3
    one_month = monthly.filter(pl.col("date") == date(2013, 1, 31))
    assert l3_study._population(one_month, frozen)["monthly_sharpes"] == {}
    nonfinite = monthly.with_columns(
        pl.when(pl.col("strategy_id") == "strategy:b")
        .then(float("inf"))
        .otherwise(pl.col("monthly_return"))
        .alias("monthly_return")
    )
    assert l3_study._population(nonfinite, frozen)["monthly_sharpes"] == pytest.approx(
        {"strategy:a": 1.0}
    )
    unregistered = constant.with_columns(pl.lit("strategy:unregistered").alias("strategy_id"))
    unregistered = unregistered.unique("date")
    with pytest.raises(ValueError, match="unregistered trading"):
        l3_study._population(unregistered, frozen)


def test_population_rejects_duplicate_missing_and_nonconsecutive_account_months(monthly, frozen):
    with pytest.raises(ValueError, match="Duplicate strategy/month"):
        l3_study._population(pl.concat([monthly, monthly.head(1)]), frozen)
    with pytest.raises(ValueError, match="misaligned monthly"):
        l3_study._population(monthly.slice(1), frozen)
    with pytest.raises(ValueError, match="consecutive calendar months"):
        l3_study._population(monthly.filter(pl.col("date") != date(2013, 2, 28)), frozen)


@pytest.mark.parametrize("changed", ["first", "last"])
def test_execute_frozen_checks_arguments_before_loading_or_fitting(changed, monkeypatch):
    first, last = date(2012, 12, 31), date(2015, 11, 30)
    settings = {"walkforward": {"design_first_origin": str(first), "design_last_origin": str(last)}}
    monkeypatch.setattr(l3_study, "_execute", lambda *a, **k: pytest.fail("pipeline called"))
    with pytest.raises(ValueError, match="guarded scoped bounds"):
        l3_study.execute_frozen(
            settings,
            frozen={},
            output=None,
            first_origin=date(2013, 1, 31) if changed == "first" else first,
            last_origin=date(2015, 10, 30) if changed == "last" else last,
            mock=True,
        )


def test_compact_transformed_fit_metadata_and_numerical_paths_are_stage_independent(
    tmp_path,
    monkeypatch,
):
    cutoff = date(2012, 1, 3)
    features = pl.DataFrame({"series": ["RAW"], "date": [date(2012, 12, 31)], "bdp_ci": [0.2]})
    paths = ["derived/final/forecasts.parquet", "derived/final/fits.parquet"]
    fingerprints = []
    for label in ("first", "second"):
        stage = tmp_path / label
        output = stage / "derived/final"
        output.mkdir(parents=True)

        def factory(settings, enabled, *, artifact_output=output, run_label=label, run_stage=stage):
            def transform(frame, selected):
                return frame, {
                    "fit_time": str(selected),
                    "identity_hash": "same_inputs",
                    "artifact_hash": "same_path_bytes",
                    "fit_artifact": str(artifact_output / "bdp.json"),
                    "provenance": {
                        "run_id": run_label,
                        "config_path": str(run_stage / "settings.toml"),
                    },
                }

            return transform

        monkeypatch.setattr(l3_transforms, "_structural_transform", factory)
        transforms = l3_transforms.LearnedTransforms(
            {},
            {},
            [cutoff, date(2012, 12, 31)],
            pl.DataFrame(),
            pl.DataFrame(),
            output,
            {"run_id": label},
        )
        values, compact = transforms.transform()(features, cutoff)
        assert compact == {
            "bdp": {
                "fit_time": "2012-01-03",
                "identity_hash": "same_inputs",
                "artifact_hash": "same_path_bytes",
            }
        }
        assert str(stage) not in json.dumps(compact)
        values.write_parquet(output / "forecasts.parquet")
        pl.DataFrame(
            {
                "refit": [cutoff],
                "parameters_json": [
                    json.dumps(
                        {
                            "coefficient": [0.25],
                            "intercept": -4.0,
                            "structural": compact,
                        },
                        sort_keys=True,
                    )
                ],
            }
        ).write_parquet(output / "fits.parquet")
        (output / "provenance.json").write_text(json.dumps({"stage": str(stage), "run_id": label}))
        hashes = holdout._artifact_hashes(stage, paths)
        receipt = holdout._record_reproduction(
            tmp_path,
            stage,
            identity={"frozen": "fixed"},
            hashes=hashes,
        )
        assert receipt["reproduced"] == (label == "second")
        fingerprints.append(hashes)
    assert fingerprints[0] == fingerprints[1]
    assert set(fingerprints[0]) == set(paths)


def test_deterministic_registry_cannot_point_outside_an_attempt_or_repeat_paths(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    external = tmp_path / "external.json"
    external.write_text("{}")
    (stage / "external.json").symlink_to(external)
    with pytest.raises(ValueError, match="Invalid deterministic"):
        holdout._artifact_hashes(stage, ["external.json"])
    (stage / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="unique deterministic"):
        holdout._artifact_hashes(stage, ["result.json", "result.json"])
    with pytest.raises(ValueError, match="Invalid deterministic"):
        holdout._artifact_hashes(stage, ["../external.json"])


@pytest.fixture
def recorded(tmp_path):
    path = tmp_path / "design_trial_population.json"
    path.write_text(json.dumps({
        "first_month": "2013-01-31", "last_month": "2013-03-31",
        "distinct_trading_ids": 4,
        "trading_ids": ["strategy:a", "strategy:b", "strategy:earlier", "strategy:failed"],
        "monthly_sharpes": {"strategy:a": 0.5, "strategy:earlier": -0.25},
    }))
    return path


def test_design_population_keeps_recorded_multiplicity_and_replaces_current_accounts(
    monthly, recorded, monkeypatch,
):
    monkeypatch.setattr(experiments, "read_trials", lambda *a: pytest.fail("live trial log read"))
    population = baseline_report.design_population(monthly, recorded)
    # Current accounts: A has mean .1 and sample SD .1, B mean 0 and SD .2. The earlier
    # specification keeps its recorded estimate; the failed one stays only in N.
    assert population["monthly_sharpes"] == pytest.approx(
        {"strategy:a": 1.0, "strategy:b": 0.0, "strategy:earlier": -0.25})
    assert list(population["monthly_sharpes"]) == ["strategy:a", "strategy:b", "strategy:earlier"]
    assert population["distinct_trading_ids"] == 4
    assert population["current_accounts"] == 2
    assert not population["current_estimates_match_record"]


def test_design_population_refuses_unrecorded_accounts_or_another_sample(monthly, recorded):
    unrecorded = monthly.with_columns(
        pl.when(pl.col("strategy_id") == "strategy:b").then(pl.lit("strategy:new"))
        .otherwise(pl.col("strategy_id")).alias("strategy_id"))
    with pytest.raises(ValueError, match="absent from the recorded"):
        baseline_report.design_population(unrecorded, recorded)
    later = monthly.filter(pl.col("date") != date(2013, 1, 31))
    with pytest.raises(ValueError, match="different monthly sample"):
        baseline_report.design_population(later, recorded)
    bad = json.loads(recorded.read_text())
    bad["monthly_sharpes"]["strategy:unknown"] = 0.1
    recorded.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Invalid recorded design"):
        baseline_report.design_population(monthly, recorded)


def test_l3_design_population_is_the_recorded_population(monthly, recorded, monkeypatch):
    monkeypatch.setattr(baseline_report, "POPULATION_PATH", recorded)
    assert l3_study._population(monthly, None) == baseline_report.design_population(
        monthly, recorded)
