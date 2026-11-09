"""Frozen-coefficient, month-cluster and daily-HAC reference fixtures."""

import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from momenta.manifest import sha256_file
from momenta.robustness_daily import (
    _auc_draws,
    _saved_features,
    cluster_sums,
    daily_metrics,
    month_resamples,
    monthly_parity,
    saved_prediction,
)
from momenta.settings import load_settings
from momenta.walkforward import walk_forward


def _fit(target="y1", **changes):
    parameters = {
        "columns": ["a", "b"],
        "candidate_columns": ["a", "unused", "b"],
        "mean": [2.0, 5.0],
        "scale": [2.0, 1.0],
        "clip_sd": 3.0,
        "coefficient": [2.0, -0.5],
        "intercept": 1.0,
    }
    parameters.update(changes)
    return {"target": target, "parameters_json": json.dumps(parameters)}


def test_saved_coefficients_clip_in_training_units_and_drop_excluded_columns():
    frame = pl.DataFrame(
        {"a": [4.0, 100.0, -100.0], "b": [6.0, 5.0, 100.0], "unused": [float("nan")] * 3}
    )
    # z=(1,1),(3,0),(-3,3); intercept + 2*z_a - .5*z_b.
    assert_allclose(saved_prediction(frame, _fit()), [2.5, 7.0, -6.5], rtol=0, atol=0)
    assert_allclose(
        saved_prediction(frame, _fit("y3")),
        [1 / (1 + np.exp(-2.5)), 1 / (1 + np.exp(-7)), 1 / (1 + np.exp(6.5))],
    )
    damaged = frame.with_columns(pl.lit(1e100).alias("unused"))
    np.testing.assert_array_equal(
        saved_prediction(frame, _fit()), saved_prediction(damaged, _fit())
    )


def test_intercept_only_logistic_preserves_exact_frozen_base_rate():
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    fit = _fit("y3", columns=[], mean=[], scale=[], coefficient=[], intercept=-2.197224577336219)
    expected = float(13 / 127)
    np.testing.assert_array_equal(saved_prediction(frame, fit, base_rate=expected), [expected] * 3)
    with pytest.raises(ValueError, match="base rate"):
        saved_prediction(frame, fit)


def test_saved_model_rejects_future_fit_time():
    fit = _fit() | {"refit": date(2016, 1, 4)}
    frame = pl.DataFrame({"date": [date(2015, 12, 31)], "a": [1.0], "b": [2.0]})
    with pytest.raises(ValueError, match="before its fit date"):
        saved_prediction(frame, fit)


def test_saved_structural_paths_carry_only_completed_months_and_pin_their_fit(tmp_path):
    cutoff = date(2016, 1, 4)
    days = [date(2016, 1, 28), date(2016, 1, 29), date(2016, 2, 1)]
    frame = pl.DataFrame({"date": days, "series": "RAW", "bdp_ci": [0.0] * 3})
    folder = tmp_path / "derived/final"
    bdp = tmp_path / "derived/baseline/bdp_paths/2016-01-04.parquet"
    bdp.parent.mkdir(parents=True)
    frame.with_columns(pl.Series("bdp_ci", [0.1, 0.2, 0.3])).write_parquet(bdp)
    bdp.with_suffix(".json").write_text(
        json.dumps(
            {
                "fit_time": str(cutoff),
                "identity_hash": "bdp-identity",
                "french_last_observation": "2015-12-31",
                "artifact_hash": sha256_file(bdp),
            }
        )
    )
    centers = folder / "centroids/full_2016-01-04.json"
    centers.parent.mkdir(parents=True)
    centers.write_text(
        json.dumps(
            {
                "fit_hash": "ot-fit",
                "lineage": "full",
                "fit_cutoff": str(cutoff),
                "centroids": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            }
        )
    )
    month_ends = [date(2015, 12, 31), date(2016, 1, 29), date(2016, 2, 29)]
    copula = pl.DataFrame({"date": month_ends, "embedding": [[0.0, 0.0], [3.0, 4.0], [9.0, 9.0]]})
    hmm_fit = folder / "regimes/hmm-fit.json"
    hmm_fit.parent.mkdir(parents=True)
    hmm_fit.write_text(
        json.dumps(
            {
                "fit_hash": "hmm-fit",
                "fit_time": str(cutoff),
                "series": "RAW",
                "training_target_end_max": "2015-12-31",
                "states": 2,
            }
        )
    )
    hmm_path = hmm_fit.with_name("hmm-fit_path.parquet")
    pl.DataFrame(
        {"date": month_ends, "hmmf_0": [0.2, 0.8, 0.99], "hmme_0": [0.4, 0.6, 0.9]}
    ).write_parquet(hmm_path)
    hmm_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "artifact_hash": sha256_file(hmm_path),
                "fit_artifact_hash": sha256_file(hmm_fit),
            }
        )
    )
    fit = {
        "series": "RAW",
        "refit": cutoff,
        "parameters_json": json.dumps(
            {
                "structural": {
                    "bdp": {"identity_hash": "bdp-identity", "artifact_hash": sha256_file(bdp)},
                    "ot": {"fit_hash": "ot-fit"},
                    "hmm": {"fit_hash": "hmm-fit"},
                },
            }
        ),
    }
    metadata = {"input_hashes": {}}
    actual = _saved_features(frame, fit, tmp_path, copula, metadata)
    assert actual["bdp_ci"].to_list() == [0.1, 0.2, 0.3]
    assert actual["hmmf_0"].to_list() == [0.2, 0.8, 0.8]
    assert actual["hmme_0"].to_list() == [0.4, 0.6, 0.6]
    # Jan 28 still carries December, Jan 29 updates, and February 1 holds that state.
    assert actual["ot_distance_1"].to_list() == [0.0, 5.0, 5.0]
    assert len(metadata["input_hashes"]) == 6
    damaged = copula.with_columns(
        pl.when(pl.col("date") > days[-1])
        .then(pl.lit([999.0, 999.0]))
        .otherwise(pl.col("embedding"))
        .alias("embedding")
    )
    assert actual.equals(_saved_features(frame, fit, tmp_path, damaged, metadata))
    hmm_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "artifact_hash": "changed",
                "fit_artifact_hash": sha256_file(hmm_fit),
            }
        )
    )
    with pytest.raises(ValueError, match="path changed"):
        _saved_features(frame, fit, tmp_path, copula, metadata)


@pytest.mark.parametrize("target", ["y1", "y2", "y3"])
def test_daily_saved_prediction_matches_independently_executed_monthly_walker(target):
    settings = load_settings()
    settings["walkforward"].update(
        feature_start="2001-01-01",
        design_first_origin="2005-12-31",
        design_last_origin="2006-02-28",
        ridge_lambdas=[0.1],
        logit_lambdas=[0.01],
    )
    start = date(2001, 1, 1)
    days = [
        start + timedelta(days=i) for i in range(1947) if (start + timedelta(days=i)).weekday() < 5
    ]
    x = np.sin(np.arange(len(days)) * 0.17) + np.arange(len(days)) * 0.0005
    features = pl.DataFrame({"date": days, "series": "RAW", "x": x})
    targets = pl.DataFrame(
        {
            "origin": days[:-22],
            "series": "RAW",
            "target_start": days[2:-20],
            "target_end": days[22:],
            "y1": 2 + x[:-22] * 0.4,
            "y2": np.sin(np.arange(len(days) - 22) * 0.27),
        }
    )
    run = walk_forward(
        features,
        targets,
        days,
        settings,
        series="RAW",
        target=target,
        feature_set="fixture",
        columns=["x"],
        blocks={"HAR": ["x"]},
    )
    for fit in run.fits.iter_rows(named=True):
        stored = run.forecasts.filter(pl.col("refit") == fit["refit"]).sort("origin")
        scored = features.filter(pl.col("date").is_in(stored["origin"].to_list())).sort("date")
        predicted = saved_prediction(scored, fit, base_rate=stored["base_prediction"][0])
        assert_allclose(predicted, stored["prediction"], rtol=0, atol=1e-14)
    # December is explicitly served by January 2005; the next January uses the new fit.
    assert run.forecasts["refit"][0] == date(2005, 1, 3)
    assert run.forecasts["refit"][1] == date(2006, 1, 2)


def test_cluster_draws_retain_all_days_and_weight_by_number_of_days():
    dates = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 4), date(2020, 3, 2)]
    membership, draws = month_resamples(dates, 8, 2, 71)
    np.testing.assert_array_equal(membership, [0, 1, 1, 2])
    # Independent specified month draws: Jan,Feb,Feb contain five observations,
    # rather than either three days or a mean of three monthly means.
    specified = np.array([[0, 1, 1], [2, 0, 2]])
    sums = cluster_sums(np.array([1.0, 10.0, 20.0, 100.0]), membership, specified)
    counts = cluster_sums(np.ones(4), membership, specified)
    assert_allclose(sums, [61.0, 201.0])
    assert_allclose(counts, [5.0, 3.0])
    assert_allclose(sums / counts, [12.2, 67.0])
    assert draws.shape == (8, 3)
    np.testing.assert_array_equal(draws, month_resamples(dates, 8, 2, 71)[1])
    with pytest.raises(ValueError, match="consecutive"):
        month_resamples([dates[0], dates[-1]], 8, 2, 71)


def test_month_cluster_auc_matches_expanded_pairwise_tie_calculation():
    y, p, membership = (
        np.array([1, 0, 1, 0]),
        np.array([0.4, 0.4, 0.7, 0.8]),
        np.array([0, 1, 1, 2]),
    )
    draws = np.array([[0, 1, 1], [2, 0, 2], [0, 0, 0]])
    actual = _auc_draws(y, p, membership, draws)
    # First draw positives .4,.7,.7 and negatives .4,.4: (1+2+2)/6=5/6.
    # Second draw positive .4, negatives .8,.8: 0. Third has no negatives.
    assert actual[0] == pytest.approx(5 / 6)
    assert actual[1] == 0
    assert np.isnan(actual[2])


def test_daily_test_uses_21_lag_bartlett_hac_and_never_21_independent_months():
    start = date(2020, 1, 1)
    dates = [
        start + timedelta(days=i) for i in range(80) if (start + timedelta(days=i)).weekday() < 5
    ]
    n = len(dates)
    y, base = np.linspace(-1, 2, n), np.zeros(n)
    prediction = np.sin(np.arange(n) / 3) * 0.2
    frame = pl.DataFrame(
        {
            "origin": dates,
            "target": "y2",
            "actual": y,
            "prediction": prediction,
            "base_prediction": base,
        }
    )
    settings = load_settings()
    settings["evaluation"]["bootstrap_reps"] = 25
    result = daily_metrics(frame, settings)
    difference = y**2 - (y - prediction) ** 2 + prediction**2
    centered = difference - difference.mean()
    long_variance = sum(centered**2) / n
    for lag in range(1, 22):
        long_variance += 2 * (1 - lag / 22) * sum(centered[lag:] * centered[:-lag]) / n
    statistic = difference.mean() / np.sqrt(long_variance / n)
    assert result["cw"]["lags"] == 21
    assert result["cw"]["pvalue"] == pytest.approx(norm.sf(statistic))
    assert result["n"] == n and result["calendar_months"] == 3


def test_monthly_parity_rejects_missing_dates_changed_fit_and_prediction():
    original = pl.DataFrame(
        {
            "series": ["RAW"],
            "target": ["y1"],
            "feature_set": ["L1"],
            "origin": [date(2005, 12, 30)],
            "refit": [date(2005, 1, 3)],
            "target_start": [date(2006, 1, 4)],
            "target_end": [date(2006, 2, 1)],
            "actual": [2.0],
            "prediction": [1.0],
            "base_prediction": [0.0],
        }
    )
    assert monthly_parity(original, original, 1e-12)["passed"]
    with pytest.raises(ValueError, match="omitted"):
        monthly_parity(original.head(0), original, 1e-12)
    with pytest.raises(ValueError, match="refit"):
        monthly_parity(
            original.with_columns(pl.lit(date(2006, 1, 3)).alias("refit")), original, 1e-12
        )
    with pytest.raises(ValueError, match="mismatch"):
        monthly_parity(original.with_columns(pl.lit(1.01).alias("prediction")), original, 1e-12)
