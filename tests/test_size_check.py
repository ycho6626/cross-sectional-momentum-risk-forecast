"""Placebo independence, first-cutoff persistence, and failure-preserving orchestration."""


import json
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from momenta import size_check


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setattr(size_check, "FEATURE_SETS", {"L1": ("linear", "alternating")})
    monkeypatch.setattr(
        size_check,
        "FEATURE_BLOCKS",
        {
            "HAR": ("linear",),
            "CORE": (),
            "M": ("alternating",),
        },
    )
    days = [date(2004, 12, d) for d in (27, 28, 29, 30)] + [
        date(2005, 1, 3),
        date(2005, 12, 30),
        date(2006, 1, 3),
        date(2006, 1, 31),
        date(2006, 2, 28),
        date(2006, 3, 1),
    ]
    features = pl.DataFrame(
        {
            "date": days,
            "series": ["RAW"] * len(days),
            "linear": list(range(len(days))),
            "alternating": [i % 2 for i in range(len(days))],
        }
    )
    targets = pl.DataFrame(
        {"origin": days, "series": ["RAW"] * len(days), "target_end": days, "y1": [1.0] * len(days)}
    )
    residual = features.with_columns(
        pl.lit("RES").alias("series"),
        pl.Series("linear", [0, 2, 1, 3, *range(4, len(days))]),
    )
    features = pl.concat([features, residual])
    targets = pl.concat([targets, targets.with_columns(pl.lit("RES").alias("series"))])
    settings = {
        "walkforward": {
            "design_first_origin": "2005-12-31",
            "design_last_origin": "2006-02-28",
            "feature_start": "2004-01-01",
        },
        "evaluation": {
            "size_check_blocks": [4, 10],
            "size_check_reps": 2,
            "seed": 12,
            "nw_lags_monthly": 3,
            "size_check_series": ["RAW", "RES"],
            "size_check_max_mean_r2": 0.0,
        },
    }
    return features, targets, days, settings


def test_ar1_has_registered_persistence_and_independent_columns():
    values = size_check.ar1_predictors(6000, 4, 0.6, 177)
    for column in range(4):
        assert np.corrcoef(values[:-1, column], values[1:, column])[0, 1] == pytest.approx(
            0.6, abs=0.04
        )
        assert np.var(values[:, column]) == pytest.approx(1, abs=0.09)
    correlations = np.corrcoef(values, rowvar=False)
    np.fill_diagonal(correlations, 0)
    assert np.max(np.abs(correlations)) < 0.06
    assert np.array_equal(values, size_check.ar1_predictors(6000, 4, 0.6, 177))


def test_persistence_uses_only_completed_first_refit_training(fixture):
    features, targets, days, settings = fixture
    rho, audit = size_check.training_persistence(features, targets, days, settings)
    # Linear pairs correlate+1; alternating pairs−1; their median is0.
    assert rho == pytest.approx(0)
    assert audit["refit"] == "2005-01-03"
    assert audit["training_rows"] == 4
    assert audit["adjacent_pairs"] == 3
    changed = features.with_columns(
        pl.when(pl.col("date") >= date(2005, 1, 3))
        .then(123456)
        .otherwise(pl.col("linear"))
        .alias("linear")
    )
    assert size_check.training_persistence(changed, targets, days, settings)[0] == rho
    residual_rho, residual_audit = size_check.training_persistence(
        features,
        targets,
        days,
        settings,
        series="RES",
    )
    # First four RES linear values0,2,1,3 have lag-one correlation−.5.
    assert residual_rho == pytest.approx(-0.75)
    assert residual_audit["series"] == "RES"


def test_structural_transform_preserves_placebo_values_and_order():
    frame = pl.DataFrame({"date": [date(2005, 1, 3)], "real": [1.0], "placebo_0": [2.0]})

    def transform(frame, cutoff):
        assert "placebo_0" not in frame.columns
        return frame.with_columns((pl.col("real") * 3).alias("real")), {"cutoff": str(cutoff)}

    output, metadata = size_check._preserve_placebos(transform, ["placebo_0"])(
        frame, date(2005, 1, 3)
    )
    assert output["real"].to_list() == [3.0]
    assert output["placebo_0"].to_list() == [2.0]
    assert metadata == {"cutoff": "2005-01-03"}


@pytest.mark.parametrize("case", ["ordinary", "failed", "identical"])
def test_orchestration_caches_baseline_and_saves_every_attempt(
    fixture, tmp_path, monkeypatch, case
):
    features, targets, days, settings = fixture
    settings.update(data={"data_root": str(tmp_path)}, config_path="fixture.toml")
    monkeypatch.setattr(size_check, "provenance", lambda *args: {"config_hash": "fixture"})
    calls, logs, innovations = [], [], {}

    @contextmanager
    def trial(settings, **kwargs):
        record = {**kwargs, "metrics": {}}
        logs.append(record)
        try:
            yield SimpleNamespace(metrics=record["metrics"])
        except Exception:
            record["failed"] = True
            raise

    def walk(features, supplied_targets, supplied_days, supplied_settings, **kwargs):
        calls.append(kwargs)
        assert supplied_targets.equals(targets)
        assert supplied_days == days
        assert kwargs["series"] in {"RAW", "RES"} and kwargs["target"] == "y1"
        if case == "failed" and len(calls) == 2:
            raise ValueError("one failed model")
        baseline = kwargs["feature_set"] == "L1"
        if not baseline:
            assert len([name for name in features.columns if name.startswith("placebo_")]) in (
                4,
                10,
            )
            assert list(kwargs["blocks"]) == ["HAR", "M", "PLACEBO"]
            values = features["placebo_0"].to_numpy()
            rho = 0.0 if kwargs["series"] == "RAW" else -0.75
            recovered = (values[1:] - rho * values[:-1]) / np.sqrt(1 - rho**2)
            innovations.setdefault(kwargs["series"], []).append(recovered)
        else:
            assert list(kwargs["blocks"]) == ["HAR", "M"]
        return SimpleNamespace(
            forecasts=pl.DataFrame(
                {
                    "origin": [date(2005, 12, 30), date(2006, 1, 31), date(2006, 2, 28)],
                    "actual": [1.0, 2.0, 3.0],
                    "prediction": [0.0 if baseline or case == "identical" else 0.5] * 3,
                }
            ),
            fits=pl.DataFrame(
                {
                    "block_lambdas_json": [
                        json.dumps({"PLACEBO": "exclude"}),
                        json.dumps({"PLACEBO": "exclude" if case == "identical" else 0.1}),
                    ]
                }
            ),
        )

    monkeypatch.setattr(size_check, "trial", trial)
    monkeypatch.setattr(size_check, "walk_forward", walk)
    if case == "failed":
        with pytest.raises(ValueError, match="1 failed replications"):
            size_check.run_size_check(features, targets, days, settings, output=tmp_path)
        result = pl.read_parquet(tmp_path / "size_check.parquet")
        assert result["error"].drop_nulls().to_list() == ["ValueError"]
    elif case == "identical":
        result = size_check.run_size_check(features, targets, days, settings, output=tmp_path)
        assert result["cw_p"].to_list() == result["dm_p"].to_list() == [1.0] * 8
        assert result["r2"].to_list() == [0.0] * 8
        assert result["identical_forecasts"].all()
        assert result["placebo_excluded_refits"].to_list() == [2] * 8
    else:
        result = size_check.run_size_check(features, targets, days, settings, output=tmp_path)
        # Model errors(.5,1.5,2.5) have SSE8.75 vs baseline14.
        assert result["r2"].to_list() == pytest.approx([1 - 8.75 / 14] * 8)
        assert result["placebo_excluded_refits"].to_list() == [1] * 8
    if case != "failed":
        for raw, residual in zip(innovations["RAW"], innovations["RES"], strict=True):
            np.testing.assert_allclose(raw, residual)
    assert len(calls) == 10
    assert len(logs) == 10
    assert all(row["kind"] == "diagnostic" for row in logs)
    assert result.height == 8
    assert result["seed"].n_unique() == 4
    raw_seeds = result.filter(pl.col("series") == "RAW")["seed"].to_list()
    assert raw_seeds == result.filter(pl.col("series") == "RES")["seed"].to_list()
    assert (tmp_path / "size_check.metadata.json").exists()
    assert result["elapsed_seconds"].min() >= 0
