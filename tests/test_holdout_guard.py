"""Freeze integrity and pre-build holdout protection, without real holdout data."""


import json
import tomllib
from datetime import date

import polars as pl
import pytest

from momenta import freeze, holdout


def test_toml_writer_preserves_dates_escaping_and_numeric_types(tmp_path):
    path = tmp_path / "typed.toml"
    document = {
        "settings": {
            "day": date(2015, 11, 30),
            "clip": 3.0,
            "grid": [0.0, 1.0],
            "enabled": True,
            "label": 'quote " and newline\n and λ',
        }
    }
    freeze.write_toml(path, document)
    assert tomllib.loads(path.read_text()) == document


def test_month_end_bounds_need_the_whole_lagged_target_window():
    days = [
        date(2015, 11, 27),
        date(2015, 11, 30),
        date(2015, 12, 1),
        date(2015, 12, 2),
        date(2015, 12, 3),
        date(2015, 12, 31),
        date(2016, 1, 4),
        date(2016, 1, 5),
    ]
    bounds = holdout.origin_bounds(
        days,
        first_origin=date(2015, 11, 30),
        common_end=date(2016, 1, 5),
        horizon=2,
        lag=1,
    )
    assert bounds == {
        "first_origin": "2015-11-30",
        "last_origin": "2015-11-30",
        "final_target_end": "2015-12-03",
        "common_end": "2016-01-05",
    }
    with pytest.raises(ValueError, match="No complete"):
        holdout.origin_bounds(
            days, first_origin=date(2015, 12, 31), common_end=date(2016, 1, 5), horizon=2, lag=1
        )


def test_reproduction_preserves_originals_and_fails_on_any_numeric_difference(tmp_path):
    original, repeated = tmp_path / "first", tmp_path / "second"
    for stage in (original, repeated):
        stage.mkdir()
        pl.DataFrame({"prediction": [0.1, 0.2]}).write_parquet(stage / "predictions.parquet")
    identity = {"frozen": "abc", "sample": "mock"}
    names = ["predictions.parquet"]
    first = holdout._record_reproduction(
        tmp_path,
        original,
        identity=identity,
        hashes=holdout._artifact_hashes(original, names),
    )
    second = holdout._record_reproduction(
        tmp_path,
        repeated,
        identity=identity,
        hashes=holdout._artifact_hashes(repeated, names),
    )
    assert not first["reproduced"] and second["reproduced"]
    pl.DataFrame({"prediction": [0.1, 0.20000000000001]}).write_parquet(
        repeated / "predictions.parquet"
    )
    with pytest.raises(ValueError, match="did not reproduce"):
        holdout._record_reproduction(
            tmp_path,
            repeated,
            identity=identity,
            hashes=holdout._artifact_hashes(repeated, names),
        )
    assert json.loads((tmp_path / "completed.json").read_text())["stage"] == str(original)
    (original / "predictions.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Original holdout output changed"):
        holdout._record_reproduction(
            tmp_path,
            repeated,
            identity=identity,
            hashes=holdout._artifact_hashes(repeated, names),
        )
