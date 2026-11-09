"""Hand-derived momentum scores, OLS recovery, and formation timing."""


import calendar
from datetime import date
from math import sqrt

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from momenta.signals import build_signals, raw_momentum, residual_momentum

SETTINGS = {
    "signals": {"formation_months": 11, "res_window_months": 36, "res_min_months": 24},
}


def _dates(count: int, start_year: int = 1998) -> list[date]:
    return [
        date(
            start_year + i // 12, i % 12 + 1,
            calendar.monthrange(start_year + i // 12, i % 12 + 1)[1],
        )
        for i in range(count)
    ]


def _panel(count: int = 36) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dates = _dates(count)
    monthly = pl.DataFrame({
        "permaticker": [7] * count,
        "date": dates,
        "monthly_return": [0.02 + (0.01 if i % 2 == 0 else -0.01) for i in range(count)],
    })
    factors = pl.DataFrame({
        "date": dates, "mkt_rf": [0.0] * count, "smb": [0.0] * count,
        "hml": [0.0] * count, "rf": [0.0] * count,
    })
    universe = pl.DataFrame({
        "permaticker": [7], "date": [date(2000, 12, 29)], "res_eligible": [True],
        "marketcap": [500.0], "rank": [1], "cap_source": ["observed"],
        "dollar_volume_rank": [2],
    })
    return monthly, factors, universe


def test_raw_compounds_total_returns_and_accepts_total_loss():
    assert raw_momentum(np.array([0.10, -0.10, 0.0])) == pytest.approx(-0.01)
    assert raw_momentum(np.array([0.10, -1.0, 0.30])) == -1.0


def test_residual_ols_recovers_known_noise_with_intercept_and_risk_free():
    # Each four-month block has zero factor sums. Noise is constant per block,
    # and its nine block levels sum to zero, making it orthogonal to all X.
    ff3 = np.tile(np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]), (9, 1))
    ff3 = ff3 * np.array([0.01, 0.02, 0.03])
    noise = np.repeat(np.arange(-4, 5), 4) * 0.01
    rf = np.arange(36) * 0.0001 + 0.001
    returns = 0.03 + ff3 @ np.array([2.0, -1.0, 0.5]) + noise + rf
    recent = (np.arange(36) >= 24) & (np.arange(36) < 35)
    # Recent noise / .01 is [2]*4 + [3]*4 + [4]*3: sum32, squared sum100.
    expected = (32 / 11) / sqrt((100 - 32**2 / 11) / 10)
    result = residual_momentum(returns, np.column_stack((ff3, rf)), recent, 24)
    assert result == pytest.approx(expected, abs=1e-12)


def test_raw_uses_exact_calendar_window_skips_formation_and_preserves_universe():
    monthly, factors, universe = _panel(38)
    monthly = monthly.with_columns(
        pl.when(pl.col("date").is_between(date(2000, 1, 1), date(2000, 11, 30)))
        .then(pl.when(pl.col("date") == date(2000, 1, 31)).then(0.10)
              .when(pl.col("date") == date(2000, 2, 29)).then(-0.10).otherwise(0.0))
        .otherwise(9.0).alias("monthly_return"),
    )
    universe = universe.with_columns(pl.lit(False).alias("res_eligible"))
    result = build_signals(monthly.reverse(), factors.reverse(), universe, SETTINGS)
    assert result["raw"].item() == pytest.approx(-0.01)
    assert result["res"].item() is None
    assert_frame_equal(result.select(universe.columns), universe)


def test_res_sample_sd_and_current_month_included_in_fit_but_not_raw():
    monthly, factors, universe = _panel()
    result = build_signals(monthly, factors, universe, SETTINGS)
    # Eleven recent residuals: six +.01 and five -.01. Mean=.01/11,
    # sample variance=.01**2 * 12/11, so signal=1/sqrt(132).
    assert result["res"].item() == pytest.approx(1 / sqrt(132), abs=1e-12)
    changed = monthly.with_columns(
        pl.when(pl.col("date") == date(2000, 12, 31))
        .then(pl.col("monthly_return") + 0.36).otherwise(pl.col("monthly_return"))
        .alias("monthly_return"),
    )
    changed_result = build_signals(changed, factors, universe, SETTINGS)
    # Adding .36 in the fit's final month moves its intercept by .01.
    assert changed_result["res"].item() == pytest.approx(-10 / sqrt(132), abs=1e-12)
    assert changed_result["raw"].item() == result["raw"].item()


def test_fit_uses_36_calendar_months_and_ignores_future_data():
    monthly, factors, universe = _panel(49)
    universe = universe.with_columns(pl.lit(date(2001, 12, 31)).alias("date"))
    baseline = build_signals(monthly, factors, universe, SETTINGS)
    # The fit is Jan1999..Dec2001. Dec1998 and Jan2002 must not enter it.
    outside = (pl.col("date") < date(1999, 1, 1)) | (pl.col("date") > date(2001, 12, 31))
    changed_returns = monthly.with_columns(
        pl.when(outside).then(99.0).otherwise(pl.col("monthly_return")).alias("monthly_return"),
    )
    changed_factors = factors.with_columns([
        pl.when(outside).then(999.0).otherwise(pl.col(c)).alias(c)
        for c in ["mkt_rf", "smb", "hml", "rf"]
    ])
    assert_frame_equal(
        build_signals(changed_returns, changed_factors, universe, SETTINGS), baseline,
    )
    first_in_window = monthly.with_columns(
        pl.when(pl.col("date") == date(1999, 1, 31)).then(pl.col("monthly_return") + 0.36)
        .otherwise(pl.col("monthly_return")).alias("monthly_return"),
    )
    result = build_signals(first_in_window, factors, universe, SETTINGS)
    assert result["res"].item() == pytest.approx(-10 / sqrt(132), abs=1e-12)


def test_first_res_formation_has_24_months_in_december_1999():
    monthly, factors, universe = _panel(24)
    november = universe.with_columns(
        pl.lit(date(1999, 11, 30)).alias("date"), pl.lit(False).alias("res_eligible"),
    )
    december = universe.with_columns(pl.lit(date(1999, 12, 31)).alias("date"))
    result = build_signals(monthly, factors, pl.concat([november, december]), SETTINGS)
    assert result["res"].to_list()[0] is None
    assert result["res"].to_list()[1] == pytest.approx(1 / sqrt(132), abs=1e-12)
    with pytest.raises(ValueError, match="23 complete observations; requires 24"):
        build_signals(
            monthly, factors, november.with_columns(pl.lit(True).alias("res_eligible")), SETTINGS,
        )


def test_zero_returns_are_retained_in_residual_regression():
    monthly, factors, universe = _panel(24)
    monthly = monthly.with_columns((pl.col("monthly_return") - 0.01).alias("monthly_return"))
    universe = universe.with_columns(pl.lit(date(1999, 12, 31)).alias("date"))
    # Twelve genuine zero monthly returns remain valid observations.
    result = build_signals(monthly, factors, universe, SETTINGS)
    assert result["res"].item() == pytest.approx(1 / sqrt(132), abs=1e-12)


def test_missing_formation_month_is_not_replaced_by_older_return():
    monthly, factors, universe = _panel()
    monthly = monthly.filter(pl.col("date") != date(2000, 6, 30))
    with pytest.raises(ValueError, match="Incomplete RAW formation months"):
        build_signals(monthly, factors, universe, SETTINGS)


def test_missing_factor_cannot_silently_shrink_recent_residual_window():
    monthly, factors, universe = _panel()
    factors = factors.filter(pl.col("date") != date(2000, 6, 30))
    with pytest.raises(ValueError, match="complete recent residual months"):
        build_signals(monthly, factors, universe, SETTINGS)


def test_duplicate_month_cannot_inflate_history_count():
    monthly, factors, universe = _panel()
    duplicate = monthly.tail(1).with_columns(pl.lit(date(2000, 12, 29)).alias("date"))
    with pytest.raises(ValueError, match="duplicate calendar-month keys"):
        build_signals(pl.concat([monthly, duplicate]), factors, universe, SETTINGS)


def test_perfect_fit_fails_instead_of_ranking_floating_point_noise():
    monthly, factors, universe = _panel()
    monthly = monthly.with_columns(pl.lit(0.03).alias("monthly_return"))
    with pytest.raises(ValueError, match="standard deviation is zero"):
        build_signals(monthly, factors, universe, SETTINGS)
