"""Hand-computed forward-window and tail-timing fixtures (Methodology)."""


from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from momenta.targets import build_targets, forward_targets, overlap_counts, tail_labels


def test_execution_day_is_excluded_and_window_is_not_demeaned():
    values = np.array([.1, .2, .3, .4, .5])
    y1, y2 = forward_targets(values, horizon=3, lag=1)
    assert y1[0] == pytest.approx(np.log(.09 + .16 + .25))
    assert y2[0] == pytest.approx(np.log(1.3 * 1.4 * 1.5))
    assert np.isnan(y1[1:]).all() and np.isnan(y2[1:]).all()
    values[1] = 20
    changed = forward_targets(values, horizon=3, lag=1)
    assert changed[0][0] == y1[0] and changed[1][0] == y2[0]


def test_zero_variance_has_no_arbitrary_log_floor():
    y1, y2 = forward_targets(np.zeros(5), horizon=3, lag=1)
    assert np.isnan(y1[0]) and y2[0] == 0


def test_tail_cutoff_is_supplied_and_equality_is_not_a_tail_event():
    assert tail_labels(np.array([-.2, -.1, 0]), -.1).tolist() == [1, 0, 0]
    assert tail_labels(np.array([-.2, -.1, 0]), 0).tolist() == [1, 1, 0]


def test_dated_targets_preserve_calendar_holidays_and_fail_on_missing_book_days():
    dates = [date(2005, 12, 30), date(2006, 1, 3), date(2006, 1, 4),
             date(2006, 1, 5), date(2006, 1, 6)]
    book = pl.DataFrame({"date": dates, "series": ["RAW"] * 5,
                         "construction": ["primary"] * 5,
                         "daily_return": [.1, .2, .3, .4, .5]})
    result = build_targets(book, dates, {"targets": {"horizon": 3, "lag": 1}})
    assert result.height == 1
    assert result["origin"].item() == dates[0]
    assert result["target_start"].item() == dates[2]
    assert result["target_end"].item() == dates[4]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        build_targets(book.filter(pl.col("date") != dates[2]), dates,
                      {"targets": {"horizon": 3, "lag": 1}})


def test_overlap_counts_use_trading_indices_and_actual_month_ends():
    # Deliberately small month calendars make 19/20/22-session spacing explicit.
    dates = [date(2000, 1, 31)]
    for month, count in [(2, 19), (3, 20), (4, 22), (5, 23)]:
        dates += [date(2000, month, 1) + timedelta(days=i) for i in range(count)]
    origins = [0, 19, 39, 61]
    targets = pl.DataFrame({
        "series": ["RAW"] * 4, "construction": ["primary"] * 4,
        "origin": [dates[i] for i in origins],
        "target_start": [dates[i + 2] for i in origins],
        "target_end": [dates[i + 22] for i in origins],
    })
    result = overlap_counts(targets, dates)
    assert result["origin_distance_days"].to_list() == [19, 20, 22]
    assert result["overlap_days"].to_list() == [2, 1, 0]
    assert result["gap_days"].to_list() == [0, 0, 1]


def test_origin_bound_does_not_cut_off_its_future_target_window():
    dates = [date(2015, 11, 30) + timedelta(days=i) for i in range(8)]
    book = pl.DataFrame({"date": dates, "series": ["RAW"] * 8,
                         "construction": ["primary"] * 8, "daily_return": [.01] * 8})
    result = build_targets(book, dates, {"targets": {"horizon": 3, "lag": 1}},
                           last_origin=dates[0])
    assert result.height == 1 and result["target_end"].item() == dates[4]
    assert result["y1"].item() == pytest.approx(np.log(.0003))


def test_design_bound_precedes_target_computation_and_rejects_holdout_origins():
    dates = [date(2015, 11, 30) + timedelta(days=i) for i in range(8)]
    book = pl.DataFrame({"date": dates, "series": ["RAW"] * 8,
                         "construction": ["primary"] * 8,
                         "daily_return": [.01] * 5 + [np.nan] * 3})
    settings = {"targets": {"horizon": 3, "lag": 1},
                "walkforward": {"design_last_origin": "2015-11-30"}}
    result = build_targets(book, dates, settings)
    assert result.height == 1 and result["y2"].item() == pytest.approx(3 * np.log(1.01))
    with pytest.raises(ValueError, match="Holdout-origin"):
        build_targets(book, dates, settings, last_origin=dates[1])
