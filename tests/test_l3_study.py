"""Registered block identities, exact prefix support and pre-fit timing (Methodology)."""

import json
from contextlib import contextmanager
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from momenta import cli, l3_study, l3_transforms, pipeline
from momenta.regimes import REGIME_INPUTS
from momenta.settings import load_settings


def test_registered_block_order_places_quadratic_before_ot_and_hmm_last():
    assert list(l3_study.blocks_for("L1")) == ["HAR", "CORE", "M"]
    assert list(l3_study.blocks_for("L1+L2+CM+OT")) == ["HAR", "CORE", "M", "L2", "CM", "OT"]
    assert list(l3_study.blocks_for("L1+L2+CM+Q+OT")) == [
        "HAR", "CORE", "M", "L2", "CM", "Q", "OT"]
    filtered = l3_study.blocks_for("best+HMMF2", best="L1+L2+CM+OT")
    assert list(filtered) == ["HAR", "CORE", "M", "L2", "CM", "OT", "HMMF"]
    assert filtered["HMMF"] == ("hmmf_0",)
    emission = l3_study.blocks_for("best+HMME3", best="L1")
    assert list(emission) == ["HAR", "CORE", "M", "HMME"]
    assert emission["HMME"] == ("hmme_0", "hmme_1")
    assert len(l3_study.blocks_for("L1+L2+CM+Q")["Q"]) == 21


def selection_frame(predictions):
    origins = [date(2006, month, 28) for month in (1, 2)]
    return pl.DataFrame([
        {"series": "RAW", "target": "y1", "feature_set": name,
         "origin": origin, "actual": 0., "prediction": prediction}
        for name, values in predictions.items()
        for origin, prediction in zip(origins, values, strict=True)
    ])


def test_best_prefix_uses_hand_mse_and_exact_ties_go_to_shorter():
    values = {"L1": [1., -1.], "L1+L2": [0., 1.],
              "L1+L2+CM": [1., 0.], "L1+L2+CM+OT": [0., 0.]}
    assert l3_study.select_best(selection_frame(values), "RAW") == "L1+L2+CM+OT"
    values["L1+L2+CM+OT"] = [1., 0.]
    # MSEs1,.5,.5,.5; the 2nd prefix is the shortest minimizer.
    assert l3_study.select_best(selection_frame(values).reverse(), "RAW") == "L1+L2"
    values.update({"L1+L2": [1., -1.], "L1+L2+CM": [-1., 1.],
                   "L1+L2+CM+OT": [1., 1.]})
    assert l3_study.select_best(selection_frame(values), "RAW") == "L1"
    broken = selection_frame(values).slice(1)
    with pytest.raises(ValueError, match="matched complete"):
        l3_study.select_best(broken, "RAW")


@pytest.fixture
def excluded_prefixes():
    refit = date(2006, 1, 3)
    models = {
        "L1": {"HAR": .01, "CORE": .1, "M": "exclude"},
        "L1+L2": {"HAR": .01, "CORE": .1, "M": "exclude", "L2": "exclude"},
        "L1+L2+CM": {"HAR": .01, "CORE": .1, "M": "exclude",
                     "L2": "exclude", "CM": "exclude"},
    }
    fits = pl.DataFrame([
        {"series": "RAW", "target": "y1", "feature_set": name, "refit": refit,
         "block_lambdas_json": json.dumps(penalties), "training_count": 100,
         "inner_count": 40, "validation_count": 50,
         "training_target_end_max": date(2005, 12, 30),
         "inner_target_end_max": date(2003, 11, 28), "inner_fit_cutoff": date(2003, 11, 29)}
        for name, penalties in models.items()
    ])
    forecasts = pl.DataFrame([
        {"series": "RAW", "target": "y1", "feature_set": name, "refit": refit,
         "origin": date(2006, month, 28), "prediction": float(month)}
        for name in models for month in (1, 2)
    ])
    return forecasts, fits


def test_excluded_blocks_preserve_forecasts_bit_for_bit(excluded_prefixes):
    forecasts, fits = excluded_prefixes
    rows = l3_study.check_prefixes(forecasts, fits, {"RAW": "L1"})
    assert len(rows) == 2
    assert all(row["extension_excluded"] and row["excluded_prefix_exact"] for row in rows)
    changed = forecasts.with_columns(pl.when(pl.col("feature_set") == "L1+L2")
        .then(pl.col("prediction")+np.finfo(float).eps)
        .otherwise(pl.col("prediction")).alias("prediction"))
    with pytest.raises(ValueError, match="does not reproduce"):
        l3_study.check_prefixes(changed, fits, {"RAW": "L1"})


@pytest.mark.parametrize("field,value", [
    ("training_count", 99), ("inner_count", 39), ("validation_count", 49),
    ("training_target_end_max", date(2005, 12, 29)),
    ("inner_target_end_max", date(2003, 11, 27)),
    ("inner_fit_cutoff", date(2003, 11, 28)),
])
def test_changed_training_support_rejects_even_identical_predictions(
    excluded_prefixes, field, value,
):
    forecasts, fits = excluded_prefixes
    changed = fits.with_columns(pl.when(pl.col("feature_set") == "L1+L2")
        .then(pl.lit(value)).otherwise(pl.col(field)).alias(field))
    with pytest.raises(ValueError, match="training support"):
        l3_study.check_prefixes(forecasts, changed, {"RAW": "L1"})


def test_prefix_checker_rejects_changed_earlier_penalty_and_scored_dates(excluded_prefixes):
    forecasts, fits = excluded_prefixes
    penalties = json.dumps({"HAR": .1, "CORE": .1, "M": "exclude", "L2": "exclude"})
    changed = fits.with_columns(pl.when(pl.col("feature_set") == "L1+L2")
        .then(pl.lit(penalties)).otherwise(pl.col("block_lambdas_json")).alias("block_lambdas_json"))
    with pytest.raises(ValueError, match="Earlier penalties"):
        l3_study.check_prefixes(forecasts, changed, {"RAW": "L1"})
    changed_forecasts = forecasts.filter(~(
        (pl.col("feature_set") == "L1+L2") & (pl.col("origin").dt.month() == 1)))
    with pytest.raises(ValueError, match="scored origins"):
        l3_study.check_prefixes(changed_forecasts, fits, {"RAW": "L1"})


def test_locked_run_logs_before_any_input_or_fit(monkeypatch, tmp_path):
    settings = load_settings()
    settings["data"]["data_root"] = str(tmp_path)
    attempts = []

    @contextmanager
    def trial(settings, **kwargs):
        attempts.append(kwargs)
        yield SimpleNamespace(metrics={})

    def forbidden(*args, **kwargs):
        pytest.fail("The design lock must precede all input reads and fitting")

    monkeypatch.setattr(l3_study, "trial", trial)
    monkeypatch.setattr(l3_study, "provenance", lambda *args: {})
    monkeypatch.setattr(l3_study, "source_config_hash", lambda: "fixture")
    monkeypatch.setattr(l3_study, "_execute", forbidden)
    with pytest.raises(ValueError, match="Holdout origins are locked"):
        l3_study.run_design(settings, through=date(2015, 12, 1))
    assert [row["experiment"] for row in attempts] == ["l3"]
    assert not list(tmp_path.iterdir())


def test_thin_schedule_never_requests_later_bdp_fits(monkeypatch, tmp_path):
    settings = load_settings()
    start = date(2000, 11, 1)
    days = [start+timedelta(days=i) for i in range(2050)
            if (start+timedelta(days=i)).weekday() < 5]
    origins = [day for day in days if date(2001, 1, 1) <= day <= date(2006, 2, 28)]
    positions = {day: i for i, day in enumerate(days)}
    frame = pl.DataFrame({"date": origins, "series": ["RAW"]*len(origins),
        "target_end": [days[positions[day]+22] for day in origins], "y1": [1.]*len(origins),
        **{name: [0.]*len(origins) for name in REGIME_INPUTS}})
    requested = []

    def bdp(values, cutoff):
        requested.append(cutoff)
        assert cutoff <= date(2006, 2, 28)
        return values, {"fit_time": str(cutoff), "identity_hash": "x", "artifact_hash": "y"}

    monkeypatch.setattr(l3_transforms, "_structural_transform", lambda *args: bdp)
    transforms = l3_transforms.LearnedTransforms(settings, settings, days, pl.DataFrame(),
                                                pl.DataFrame(), tmp_path, {})
    schedule = transforms._schedule(frame)
    assert schedule["full"] == [date(2005, 1, 3), date(2006, 1, 2)]
    assert requested == schedule["full"]
    assert all(inner < full for inner, full in zip(
        schedule["inner"], schedule["full"], strict=True))
    with pytest.raises(ValueError, match="absent or ambiguous"):
        transforms._lineage(frame, date(2007, 1, 1))
    assert len(requested) == 2


def test_default_feature_command_uses_registered_design_end(monkeypatch):
    from momenta import baseline_study, l2, l3_inputs
    from momenta import settings as project_settings

    settings = {"walkforward": {"design_last_origin": "2015-11-30"}}
    received = []
    monkeypatch.setattr(project_settings, "load_settings", lambda: settings)
    for module, name in ((baseline_study, "build_baseline"), (l2, "build_l2"),
                         (l3_inputs, "prepare_inputs")):
        monkeypatch.setattr(module, name, lambda value, *, through, name=name:
                            received.append((name, value, through)))
    assert cli.main(["features"]) == 0
    assert received == [(name, settings, date(2015, 11, 30)) for name in (
        "build_baseline", "build_l2", "prepare_inputs")]
    assert pipeline.STAGES[:4] == ("data", "portfolios", "features", "design")


def test_centroid_months_require_finite_inputs_and_shared_cache_cannot_change_support(
    monkeypatch, tmp_path,
):
    settings = load_settings()
    dates = [date(2001, 1, 31), date(2001, 2, 28), date(2001, 3, 30),
             date(2001, 4, 30), date(2001, 5, 31)]
    frame = pl.DataFrame({"date": dates, "series": ["RAW"]*5,
        "target_end": [date(2001, month, 1) for month in range(2, 7)],
        "y1": [1., 1., 1., float("nan"), 1.],
        **{name: [0.]*5 for name in REGIME_INPUTS}}).with_columns(
            pl.Series("own_rv_5", [0., float("nan"), float("inf"), 0., 0.]))
    cutoff = date(2001, 7, 2)
    calls = []

    def fit(monthly, selected, day, settings, previous):
        calls.append(selected)
        assert day == cutoff and previous is None
        transformed = monthly.select("date").with_columns(
            [pl.lit(1.).alias(name) for name in l3_transforms.OT_COLUMNS])
        return transformed, {"training_dates": [str(value) for value in selected],
            "training_first": str(selected[0]), "training_last": str(selected[-1]),
            "training_months": len(selected), "centroids": [[1.]]}

    monkeypatch.setattr(l3_transforms, "_structural_transform", lambda *args: None)
    monkeypatch.setattr(l3_transforms, "fit_dated_centroids", fit)
    transforms = l3_transforms.LearnedTransforms(settings, settings, [], frame.select("date"),
                                                pl.DataFrame(), tmp_path, {})
    transforms.ends = dates
    monkeypatch.setattr(transforms, "_lineage", lambda *args: ("full", [cutoff]))
    monkeypatch.setattr(transforms, "_base", lambda values, day: (values, {}))
    transforms._geometry(frame, cutoff)
    residual = frame.with_columns(pl.lit("RES").alias("series"))
    transforms._geometry(residual, cutoff)
    # February/March fail finite features; April fails a finite realized training target.
    assert calls == [[date(2001, 1, 31), date(2001, 5, 31)]]
    changed = residual.with_columns(pl.col("own_rv_5").fill_nan(0.))
    with pytest.raises(ValueError, match="cannot change training-month support"):
        transforms._geometry(changed, cutoff)
    assert len(calls) == 1
