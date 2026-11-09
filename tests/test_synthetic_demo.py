"""Check independent arithmetic and information timing in the runnable example."""

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from momenta.targets import forward_targets

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples/synthetic_walkforward.py"
module_spec = importlib.util.spec_from_file_location("synthetic_walkforward", EXAMPLE_PATH)
assert module_spec is not None and module_spec.loader is not None
demo = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def example():
    settings = demo.demo_settings()
    returns = demo.synthetic_returns(settings)
    features, targets, runs = demo.run_models(returns, settings)
    return settings, returns, features, targets, runs


def test_feature_and_target_use_distinct_hand_calculated_windows():
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    frame = pl.DataFrame({
        "date": [date(2001, 1, 1) + timedelta(days=i) for i in range(5)],
        "series": "SYNTHETIC", "daily_return": returns, "market_return": returns,
    })
    features = demo.feature_panel(frame, demo.demo_settings())
    # All five observed squared returns enter the trailing feature at day 4.
    assert_allclose(features["own_rv_5"][4], np.log(21 * 0.0055 / 5))
    assert np.isnan(features["own_rv_5"].to_numpy()[:4]).all()
    # A one-session lag skips days 0 and 1; a two-session target uses days 2 and 3.
    variance, log_return = forward_targets(returns, horizon=2, lag=1)
    assert_allclose(variance[0], np.log(0.0009 + 0.0016))
    assert_allclose(log_return[0], np.log(1.03 * 0.96))
    assert np.isnan(variance[2:]).all()
    assert np.isnan(log_return[2:]).all()


def test_future_returns_cannot_change_prior_features_fits_or_forecasts(example):
    settings, returns, features, _, runs = example
    cutoff = date(2007, 6, 29)
    changed = returns.with_columns(
        pl.when(pl.col("date") > cutoff).then(pl.col("daily_return") * 7)
        .otherwise(pl.col("daily_return")).alias("daily_return"),
        pl.when(pl.col("date") > cutoff).then(pl.col("market_return") * -4)
        .otherwise(pl.col("market_return")).alias("market_return"),
    )
    changed_features, _, changed_runs = demo.run_models(changed, settings)
    assert_frame_equal(features.filter(pl.col("date") <= cutoff),
                       changed_features.filter(pl.col("date") <= cutoff), check_exact=True)
    for name, original in runs.items():
        revised = changed_runs[name]
        assert_frame_equal(original.fits.filter(pl.col("refit") <= cutoff),
                           revised.fits.filter(pl.col("refit") <= cutoff), check_exact=True)
        # Realized targets may change after the decision. Predictions and every
        # fitted quantity at or before that decision must remain bit-for-bit equal.
        assert_frame_equal(
            original.forecasts.filter(pl.col("origin") <= cutoff).drop("actual", "y2"),
            revised.forecasts.filter(pl.col("origin") <= cutoff).drop("actual", "y2"),
            check_exact=True,
        )
        before = original.forecasts.filter(pl.col("origin") == cutoff)["actual"][0]
        after = revised.forecasts.filter(pl.col("origin") == cutoff)["actual"][0]
        assert before != after


def test_example_preserves_completed_windows_and_annual_training_boundaries(example):
    settings, returns, _, targets, runs = example
    days = returns["date"].to_list()
    positions = {day: i for i, day in enumerate(days)}
    assert targets["target_end"].max() <= days[-1]
    for row in targets.iter_rows(named=True):
        assert positions[row["target_start"]] - positions[row["origin"]] == 2
        assert positions[row["target_end"]] - positions[row["origin"]] == 22
    for run in runs.values():
        assert run.forecasts.height == 36
        assert run.fits["refit"].to_list() == [
            date(2005, 1, 3), date(2006, 1, 2), date(2007, 1, 1), date(2008, 1, 1),
        ]
        for row in run.fits.iter_rows(named=True):
            assert row["training_target_end_max"] < row["refit"]
            gap = positions[row["validation_origin_min"]] - positions[row["inner_target_end_max"]]
            assert gap >= settings["walkforward"]["gap_days"]
        assert run.forecasts.filter(pl.col("refit") > pl.col("origin")).is_empty()
