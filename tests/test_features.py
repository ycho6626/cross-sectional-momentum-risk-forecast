"""Hand L1 identities, factor residuals and information timing (Methodology)."""


from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from momenta.features import (
    _stock_panel,
    average_residual_correlation,
    completed_week_indices,
    cross_features,
    historical_book_variance,
    log_realized_variance,
    macro_features,
    quantile_skewness,
    rolling_beta,
    weekly_compound,
)
from momenta.settings import load_settings


def test_stock_panel_never_maps_noncalendar_prices_or_terminals_to_next_session(
    tmp_path: Path,
) -> None:
    days = [date(2001, 1, 5), date(2001, 1, 8)]
    saturday = date(2001, 1, 6)
    source = tmp_path/"prices/year=2001"
    source.mkdir(parents=True)
    # Put Saturday last: an unchecked searchsorted would overwrite Monday.
    pl.DataFrame({"permaticker": [1, 1, 1], "date": [*days, saturday],
                  "daily_return": [.1, .2, 9.], "close": [10., 11., 99.],
                  "volume": [2., 3., 99.]}).write_parquet(source/"prices.parquet")
    terminal = pl.DataFrame({"permaticker": [2], "date": [saturday],
                             "terminal_return": [-1.]})
    returns, volume, _ = _stock_panel(tmp_path, days, np.array([1, 2]), terminal)
    np.testing.assert_array_equal(returns[:, 0], [.1, .2])
    np.testing.assert_array_equal(volume[:, 0], [20., 33.])
    assert np.isnan(returns[:, 1]).all()


def test_realized_variance_uses_squared_returns_without_demeaning_and_full_windows() -> None:
    values = np.array([.1, -.2, .3, np.nan])
    actual = log_realized_variance(values, 2, 21)
    assert np.isnan(actual[0])
    assert actual[1] == pytest.approx(np.log(21*.025))
    assert actual[2] == pytest.approx(np.log(21*.065))
    assert np.isnan(actual[3])
    np.testing.assert_array_equal(actual[:3], log_realized_variance(values[:3], 2, 21))


def test_beta_includes_intercept_and_uses_current_observation() -> None:
    market = np.array([-.03, .02, .01, -.01])
    book = .02 - 2*market
    actual = rolling_beta(book, market, 3)
    assert np.isnan(actual[:2]).all()
    np.testing.assert_allclose(actual[2:], -2)


def test_book_risk_uses_fixed_signed_weights_missing_zero_and_weighted_coverage() -> None:
    panel = np.array([[.1, .2, .3], [-.1, np.nan, .1]])
    weights = np.array([.75, .25, -1.])
    # Fixed book returns -.175 and -.175; second coverage is 1.75/2.
    variance, coverage = historical_book_variance(panel, weights, 21)
    assert variance == pytest.approx(np.log(21*.175**2))
    assert coverage == pytest.approx((1+1.75/2)/2)


def test_quantile_skewness_uses_registered_quantiles_and_rejects_constant_series() -> None:
    assert quantile_skewness(np.array([0., 1., 2., 3., 8.]), [0, .5, 1]) == .5
    assert np.isnan(quantile_skewness(np.ones(5), [.05, .5, .95]))


def test_residual_correlation_removes_ff3_and_rf_and_requires_balanced_panel() -> None:
    # Orthogonal Hadamard columns give an independent exact correlation of .6.
    a = np.array([1, -1, 1, -1, 1, -1, 1, -1])*.01
    b = np.array([1, 1, -1, -1, 1, 1, -1, -1])*.01
    c = np.array([1, 1, 1, 1, -1, -1, -1, -1])*.01
    d = np.array([1, -1, -1, 1, 1, -1, -1, 1])*.01
    e = np.array([1, -1, 1, -1, -1, 1, -1, 1])*.01
    rf = np.array([1, 1, -1, -1, -1, -1, 1, 1])*.02
    ff = np.column_stack([c, d, e, rf])
    stocks = np.column_stack([.02+3*c-2*d+e+rf+a,
                              -.01-c+4*d-2*e+rf+.6*a+.8*b, a])
    stocks[3, 2] = np.nan
    result, count = average_residual_correlation(stocks, ff)
    assert count == 2
    assert result == pytest.approx(.6, abs=1e-12)


def test_weekly_returns_do_not_include_partial_future_calendar_week() -> None:
    start = date(2001, 1, 1)
    calendar = [start+timedelta(days=i) for i in range(19)
                if (start+timedelta(days=i)).weekday() < 5]
    groups = completed_week_indices(calendar, date(2001, 1, 10))
    assert len(groups) == 1
    assert calendar[groups[0][-1]] == date(2001, 1, 5)
    returns = np.full((len(calendar), 2), .1)
    returns[2, 1] = np.nan
    compounded = weekly_compound(returns, groups)
    assert compounded[0, 0] == pytest.approx(1.1**5-1)
    assert np.isnan(compounded[0, 1])


def test_month_end_book_risk_uses_new_formation_before_execution_and_comom_last_full_week() -> None:
    settings = load_settings()
    settings["features"].update(book_rv_days=3, comom_weeks=2, dispersion_days=2,
                                 illiquidity_days=2)
    start = date(2001, 1, 1)
    days = [start+timedelta(days=i) for i in range(40)
            if (start+timedelta(days=i)).weekday() < 5]
    ids = np.arange(1, 21)
    returns = np.outer(np.arange(1, len(days)+1), ids)*.0001
    formations = [date(2001, 1, 5), date(2001, 1, 31)]
    universe = pl.DataFrame([{"date": d, "permaticker": int(i)} for d in formations for i in ids])
    signals = universe.with_columns(pl.col("permaticker").cast(pl.Float64).alias("raw"))
    records = []
    for formation in formations:
        # Explicit frozen portfolios isolate formation timing from the sort.
        for name in ("RAW", "RAW_c", "RES"):
            short = 3 if formation == formations[0] else 20
            records += [{"formation": formation, "series": name,
                         "construction": "tercile_capped_vw", "permaticker": i,
                         "weight": weight} for i, weight in [(1, 1.), (short, -1.)]]
        records += [{"formation": formation, "series": "RAW", "construction": "decile_vw",
                     "permaticker": i, "weight": weight}
                    for i, weight in [(1, -.5), (2, -.5), (19, .5), (20, .5)]]
    weights = pl.DataFrame(records)
    factors = pl.DataFrame({"date": days, **{c: np.zeros(len(days))
                                           for c in ("mkt_rf", "smb", "hml", "rf")}})
    result = cross_features(days, ids, returns, np.ones_like(returns)*1e6,
                            universe, signals, weights, factors, days, settings)
    index = days.index(formations[1])
    point = result.filter((pl.col("date") == formations[1]) & (pl.col("series") == "RAW"))
    # January 31 uses its new target (1,−1 on ids1,20), not the January5 portfolio.
    expected = np.log(21*np.mean((returns[index-2:index+1, 0]
                                 - returns[index-2:index+1, 19])**2))
    assert point["book_rv_126"][0] == pytest.approx(expected)
    assert point["book_rv_coverage"][0] == 1
    assert point["comom_count"][0] == 2
    assert point["comom"][0] == pytest.approx(1)
    assert point["mom_gap"][0] == pytest.approx(9.5)


def test_macro_features_forward_fill_observed_values_without_future_or_recession_input() -> None:
    dates = pl.DataFrame({"date": [date(2001, 1, 2), date(2001, 1, 3), date(2001, 1, 4)]})
    data = pl.DataFrame({"date": [date(2001, 1, 1), date(2001, 1, 3), date(2001, 1, 5)],
                         "value": [10., None, 1000.]})
    fred = {key: data for key in ("VIXCLS", "BAA10Y", "T10Y3M", "USREC")}
    actual = macro_features(dates, fred)
    np.testing.assert_allclose(actual["log_vix"].to_numpy(), np.log(10))
    assert actual["baa10y"].to_list() == [10, 10, 10]
    assert "USREC" not in actual.columns
