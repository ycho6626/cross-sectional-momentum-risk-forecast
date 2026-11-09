"""L2 availability, pinned-prefix accounting and structural-fit timing."""

import json
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

from momenta import l2_study
from momenta.manifest import sha256_file


def test_monthly_features_become_available_at_formation_and_stay_until_next():
    daily = pl.DataFrame(
        {
            "date": [date(2006, 1, 30), date(2006, 1, 31), date(2006, 2, 1), date(2006, 2, 28)],
            "series": ["RAW"] * 4,
            "own": [1.0] * 4,
        }
    )
    monthly = pl.DataFrame({"date": [date(2006, 1, 31), date(2006, 2, 28)], "shape": [2.0, 3.0]})
    result = l2_study.carry_monthly_features(daily, monthly)
    assert result["shape"].to_list() == [None, 2.0, 2.0, 3.0]
    changed = monthly.with_columns(
        pl.when(pl.col("date") > date(2006, 1, 31))
        .then(999.0)
        .otherwise(pl.col("shape"))
        .alias("shape")
    )
    assert l2_study.carry_monthly_features(daily, changed).head(3).equals(result.head(3))
    assert result.filter(pl.col("l2_formation") > pl.col("date")).is_empty()
    with pytest.raises(ValueError, match="Duplicate L2"):
        l2_study.carry_monthly_features(daily, pl.concat([monthly, monthly.head(1)]))


@pytest.fixture
def prefix():
    refit = date(2006, 1, 3)
    origins = [date(2006, 1, 31), date(2006, 2, 28)]
    fits = pl.DataFrame(
        {
            "series": ["RAW"],
            "target": ["y1"],
            "feature_set": ["L1+L2"],
            "refit": [refit],
            "training_count": [100],
            "inner_count": [50],
            "validation_count": [30],
            "training_target_end_max": [date(2005, 12, 30)],
            "inner_target_end_max": [date(2003, 11, 28)],
            "inner_fit_cutoff": [date(2003, 12, 1)],
            "block_lambdas_json": [
                json.dumps({"HAR": 0.1, "CORE": "exclude", "M": 0.3, "L2": "exclude"})
            ],
        }
    )
    baseline_fits = fits.with_columns(
        pl.lit("L1").alias("feature_set"),
        pl.lit(json.dumps({"HAR": 0.1, "CORE": "exclude", "M": 0.3})).alias("block_lambdas_json"),
    )
    forecasts = pl.DataFrame(
        {
            "series": ["RAW"] * 2,
            "target": ["y1"] * 2,
            "feature_set": ["L1+L2"] * 2,
            "origin": origins,
            "refit": [refit] * 2,
            "actual": [0.1, 0.2],
            "prediction": [0.2, 0.3],
            "base_prediction": [0.2, 0.3],
        }
    )
    baseline = forecasts.with_columns(pl.lit("L1").alias("feature_set"))
    return fits, baseline_fits, forecasts, baseline


def test_pinned_prefix_requires_exact_predictions_and_same_penalties_and_support(prefix):
    fits, baseline_fits, forecasts, baseline = prefix
    matched = l2_study.match_baseline(forecasts, baseline)
    row = l2_study.prefix_audit(fits, baseline_fits, matched)[0]
    assert row["excluded_prefix_exact"] is True
    assert row["excluded_prefix_max_difference"] == 0
    changed = matched.with_columns((pl.col("prediction") + 1e-14).alias("prediction"))
    with pytest.raises(ValueError, match="Excluded L2"):
        l2_study.prefix_audit(fits, baseline_fits, changed)
    changed_fit = fits.with_columns(
        pl.lit(json.dumps({"HAR": 0.2, "CORE": "exclude", "M": 0.3, "L2": "exclude"})).alias(
            "block_lambdas_json"
        )
    )
    with pytest.raises(ValueError, match="Earlier block penalties"):
        l2_study.prefix_audit(changed_fit, baseline_fits, matched)
    with pytest.raises(ValueError, match="training support"):
        l2_study.prefix_audit(
            fits.with_columns(pl.lit(99).alias("training_count")), baseline_fits, matched
        )
    with pytest.raises(ValueError, match="realized targets"):
        l2_study.match_baseline(forecasts, baseline.with_columns(pl.lit(999.0).alias("actual")))


def test_redundancy_bdp_uses_the_refit_serving_each_origin(prefix):
    _, _, forecasts, _ = prefix
    forecasts = forecasts.with_columns(pl.Series("refit", [date(2005, 1, 3), date(2006, 1, 3)]))
    features = forecasts.select("series", pl.col("origin").alias("date")).with_columns(
        pl.lit(0.0).alias("bdp_ci"),
        pl.lit(5.0).alias("shape"),
    )
    cutoffs = []

    def transform(frame, cutoff):
        cutoffs.append(cutoff)
        return frame.with_columns(pl.lit(float(cutoff.year)).alias("bdp_ci")), {}

    result = l2_study.dated_redundancy_design(features, forecasts, transform)
    assert result["bdp_ci"].to_list() == [2005.0, 2006.0]
    assert cutoffs == [date(2005, 1, 3), date(2006, 1, 3)]
    with pytest.raises(ValueError, match="future fit"):
        l2_study.dated_redundancy_design(
            features, forecasts.with_columns(pl.lit(date(2007, 1, 3)).alias("refit")), transform
        )


def test_only_pinned_bdp_cutoffs_can_be_reused_without_creating_fits(tmp_path, monkeypatch):
    from momenta import baseline_study

    cutoff = date(2006, 1, 3)
    directory = tmp_path / "derived/baseline/bdp_paths"
    directory.mkdir(parents=True)
    path = directory / f"{cutoff}.parquet"
    pl.DataFrame({"x": [1.0]}).write_parquet(path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text("{}")
    calls = []

    def factory(settings, enabled):
        def transform(frame, selected):
            calls.append(selected)
            return frame, {"fit_time": str(selected)}

        return transform

    monkeypatch.setattr(baseline_study, "_structural_transform", factory)
    callback = l2_study.pinned_structural_transform(
        {"data": {"data_root": str(tmp_path)}},
        True,
        {str(path): sha256_file(path), str(sidecar): sha256_file(sidecar)},
    )
    frame = pl.DataFrame({"date": [cutoff]})
    callback(frame, cutoff)
    assert calls == [cutoff]
    with pytest.raises(ValueError, match="unrecorded BDP"):
        callback(frame, date(2007, 1, 3))
    assert calls == [cutoff]


def test_holdout_attempt_is_logged_before_any_input_validation(monkeypatch):
    from momenta import baseline_report

    logs = []

    @contextmanager
    def logged(settings, **kwargs):
        logs.append(kwargs)
        yield SimpleNamespace(run_id="attempt", metrics={})

    monkeypatch.setattr(l2_study, "trial", logged)
    monkeypatch.setattr(baseline_report, "load_summary", lambda settings: pytest.fail("input read"))
    settings = {
        "walkforward": {"design_last_origin": "2015-11-30"},
        "targets": {},
        "features": {},
        "bdp": {},
        "l2": {},
    }
    with pytest.raises(ValueError, match="Holdout"):
        l2_study.run_l2_study(settings, through=date(2015, 12, 1))
    assert len(logs) == 1 and logs[0]["kind"] == "diagnostic"


def test_availability_uses_each_series_complete_augmented_features(monkeypatch):
    monkeypatch.setattr(l2_study, "FEATURE_SETS", {"L1": ("own",)})
    dates = [date(2006, 1, 31), date(2006, 2, 28)]
    monthly = pl.DataFrame(
        {"date": dates, "shape": [2.0, 3.0], "breadth_m": [0.4, 0.6], "winner_balanced": [100, 80]}
    )
    augmented = pl.DataFrame(
        {
            "series": ["RAW", "RAW", "RES", "RES"],
            "date": dates * 2,
            "own": [1.0, 2.0, None, 3.0],
            "shape": [2.0, 3.0] * 2,
        }
    )
    metadata = {
        "feature_availability": {"shape": str(dates[0])},
        "first_complete_origin": str(dates[0]),
        "coverage": {},
    }
    audit = l2_study._feature_audit(
        monthly, augmented, metadata, pl.DataFrame({"origin": [dates[1]]}), ["shape"]
    )
    complete = {
        row["series"]: row["first_origin"]
        for row in audit["feature_availability"]
        if row["feature"] == "L1+L2_complete"
    }
    assert complete == {"RAW": "2006-01-31", "RES": "2006-02-28"}
    assert audit["feature_coverage"][0]["mean"] == 90
    assert audit["breadth"][1] == {
        "sample": "scored_month_ends",
        "origins": 1,
        "minimum": 0.6,
        "mean": 0.6,
        "maximum": 0.6,
    }
