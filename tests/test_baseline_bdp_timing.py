"""French BDP estimation excludes the cutoff itself and every later observation."""

from datetime import date

import numpy as np
import polars as pl

from momenta import baseline_study, bdp


def test_bdp_estimator_never_receives_cutoff_or_later_french_returns(tmp_path, monkeypatch):
    french_path = tmp_path / "curated/french/momentum_deciles_daily.parquet"
    book_path = tmp_path / "derived/portfolios/book_daily.parquet"
    french_path.parent.mkdir(parents=True)
    book_path.parent.mkdir(parents=True)
    (tmp_path / "derived/baseline").mkdir()
    days = [date(2006, 1, day) for day in (1, 2, 3, 4)]
    cutoff = days[2]
    french = pl.DataFrame({"date": days, "d1": [0.0] * 4, "d10": [0.01, 0.02, 1000.0, -1000.0]})
    french.write_parquet(french_path)
    pl.DataFrame(
        {
            "date": days,
            "series": ["RAW"] * 4,
            "construction": ["primary"] * 4,
            "daily_return": [0.01] * 4,
        }
    ).write_parquet(book_path)
    settings = {
        "data": {"data_root": str(tmp_path)},
        "bdp": {"seed": 1},
        "portfolios": {"primary": "primary"},
        "walkforward": {"design_last_origin": "2006-01-04"},
    }
    calls = []

    def spy_estimator(values, current_settings):
        calls.append(values.copy())
        np.testing.assert_array_equal(values, [1.0, 2.0])  # Percent returns before Jan3.
        return bdp.BDPFit(
            loadings=(0.01, 0.01, 0.01),
            nu=8.0,
            initial=(0.0, 0.0, 0.0),
            converged=True,
            objective=0.0,
            observations=len(values),
            attempts=(),
        )

    monkeypatch.setattr(bdp, "fit_bdp", spy_estimator)
    monkeypatch.setattr(bdp, "filter_bdp", lambda values, fit: np.zeros((len(values), 3)))
    _, first_fit, first_metadata = baseline_study._bdp_paths(settings, cutoff)
    assert first_metadata["french_last_observation"] == "2006-01-02"
    assert first_metadata["fit_time"] == "2006-01-03"
    # Future edits invalidate the file cache; the second fit must get the same prefix.
    french.with_columns(
        pl.when(pl.col("date") >= cutoff).then(123456.0).otherwise(pl.col("d10")).alias("d10")
    ).write_parquet(french_path)
    _, changed_fit, changed_metadata = baseline_study._bdp_paths(settings, cutoff)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], calls[1])
    assert first_fit == changed_fit
    assert first_metadata["identity_hash"] != changed_metadata["identity_hash"]
