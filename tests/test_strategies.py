"""Hand accounting, cost and training-cutoff fixtures for Methodology."""


from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest

from momenta.settings import load_settings
from momenta.strategies import (
    DMMeanFit,
    break_even_cost,
    build_dm_pairs,
    calibrate_dm,
    calibrate_scale,
    dm_multiples,
    fit_dm_mean,
    gross_scaled_monthly,
    simulate_strategy,
    volatility_multiples,
)


def _base():
    days = [date(2006, 1, 3), date(2006, 1, 4), date(2006, 1, 31),
            date(2006, 2, 1), date(2006, 2, 28)]
    nav = [1., 1.1, 1.21, 1.331, 1.4641]
    book = pl.DataFrame({
        "date": days, "nav": nav, "daily_return": [0., .1, .1, .1, .1],
        "cash": [1., 1., 1., 1.331, 1.331],
        "gross_exposure": [2., 2.1 / 1.1, 2.21 / 1.21, 2., 2.7951 / 1.4641],
        "terminal_cash": [0.] * 5,
    })
    ledger = pl.DataFrame({
        "date": [days[0]] * 2 + [days[3]] * 2,
        "formation": [date(2005, 12, 30)] * 2 + [days[2]] * 2,
        "permaticker": [1, 2, 1, 2], "pretrade": [0., 0., 1.331, -1.],
        "posttrade": [1., -1., 1.331, -1.331],
    })
    overlays = pl.DataFrame({
        "origin": [date(2005, 12, 30), days[2]], "execution_date": [days[0], days[3]],
        "multiple": [2., 1.], "fit_date": [date(2005, 1, 3), date(2006, 1, 3)],
    })
    return book, ledger, overlays


def test_exact_scaled_book_drifts_and_old_book_earns_execution_day_return():
    base, ledger, overlays = _base()
    result = simulate_strategy(base, ledger, overlays, cost_bp=0, initial_nav=1,
                               last_date=date(2006, 2, 28))
    # Jan: x_long=2 -> 2.2 -> 2.42 and cash/short stay fixed. On Feb 1,
    # old x_long earns .242 before the new one-times target executes.
    assert result.daily["nav"].to_list() == pytest.approx([1, 1.2, 1.42, 1.662, 1.8282])
    assert result.daily["daily_return"][2] == pytest.approx(.22 / 1.2)
    assert result.daily["daily_return"][2] != pytest.approx(2 * base["daily_return"][2])
    assert result.rebalances["turnover"].to_list() == pytest.approx([4, 1.338 / 1.662])


def test_fees_are_cash_deductions_on_the_actual_path_not_subtracted_return_fractions():
    base, ledger, overlays = _base()
    result = simulate_strategy(base, ledger, overlays, cost_bp=10, initial_nav=1,
                               last_date=date(2006, 2, 28))
    # Entry fee .004; positions still +/-2. January profits .2 and .22.
    # Feb 1 NAV before cost=1.658, target +/-1.658, pretrade=2.662/-2.
    # Net trades -1.004/+0.342 cost .001346; future profit=.1658.
    assert result.daily["nav"].to_list() == pytest.approx(
        [.996, 1.196, 1.416, 1.656654, 1.822454]
    )
    assert result.rebalances["cost"].to_list() == pytest.approx([.004, .001346])
    assert result.daily["daily_return"][3] == pytest.approx(1.656654 / 1.416 - 1)
    assert result.daily["gross_exposure"][0] == pytest.approx(4 / .996)
    assert result.ledger["cost"].sum() == pytest.approx(.005346)


def test_offsetting_formation_and_scaling_trades_are_netted_per_name():
    days = [date(2006, 1, 3), date(2006, 2, 1)]
    base = pl.DataFrame({"date": days, "nav": [100., 100.], "daily_return": [0., 0.],
                         "cash": [100., 100.], "gross_exposure": [2., 2.],
                         "terminal_cash": [0., 0.]})
    ledger = pl.DataFrame({"date": [days[0]] * 4 + [days[1]] * 4,
                           "permaticker": [1, 2, 3, 4] * 2,
                           "pretrade": [0.] * 4 + [60., 40., -50., -50.],
                           "posttrade": [60., 40., -50., -50., 30., 70., -25., -75.]})
    overlays = pl.DataFrame({"origin": [date(2005, 12, 30), date(2006, 1, 31)],
                             "execution_date": days, "multiple": [1., 2.]})
    result = simulate_strategy(base, ledger, overlays, cost_bp=0, initial_nav=100,
                               last_date=days[-1])
    trades = result.ledger.filter(pl.col("date") == days[1]).sort("permaticker")
    assert trades["trade"].to_list() == pytest.approx([0., 100., 0., -100.])
    assert result.rebalances["turnover"][-1] == pytest.approx(2.)


@pytest.mark.parametrize("multiple,nav,cash,terminal_cash", [
    (2., [1., .8, .92], [1., 1.6, 1.6], .6),
    (-1., [1., 1.1, 1.04], [1., .7, .7], -.3),
])
def test_terminal_cash_and_negative_overlay_settlement(multiple, nav, cash, terminal_cash):
    days = [date(2006, 1, i) for i in [3, 4, 5]]
    base = pl.DataFrame({"date": days, "nav": [1., .9, .96],
                         "daily_return": [0., -.1, .06 / .9], "cash": [1., 1.3, 1.3],
                         "gross_exposure": [2., 1.6 / .9, 1.66 / .96],
                         "terminal_cash": [0., .3, 0.]})
    ledger = pl.DataFrame({"date": [days[0]] * 3, "permaticker": [1, 2, 3],
                           "pretrade": [0.] * 3, "posttrade": [.5, .5, -1.]})
    overlays = pl.DataFrame({"origin": [date(2005, 12, 30)], "execution_date": [days[0]],
                             "multiple": [multiple]})
    result = simulate_strategy(base, ledger, overlays, cost_bp=0, initial_nav=1,
                               last_date=days[-1])
    assert result.daily["nav"].to_list() == pytest.approx(nav)
    assert result.daily["cash"].to_list() == pytest.approx(cash)
    assert result.daily["terminal_cash"][1] == pytest.approx(terminal_cash)
    assert result.daily["gross_exposure"][0] == 2 * abs(multiple)


def test_model_fit_after_decision_and_missing_monthly_overlay_are_errors():
    base, ledger, overlays = _base()
    future_fit = overlays.with_columns(pl.lit(date(2006, 1, 3)).alias("fit_date"))
    with pytest.raises(ValueError, match="fitted after"):
        simulate_strategy(base, ledger, future_fit, cost_bp=0, initial_nav=1,
                           last_date=date(2006, 2, 28))
    with pytest.raises(ValueError, match="Every base rebalance"):
        simulate_strategy(base, ledger, overlays.head(1), cost_bp=0, initial_nav=1,
                           last_date=date(2006, 2, 28))


def test_future_book_and_ledger_values_do_not_change_earlier_holdings():
    base, ledger, overlays = _base()
    cutoff = date(2006, 1, 31)
    before = simulate_strategy(base, ledger, overlays, cost_bp=10, initial_nav=1, last_date=cutoff)
    damaged_base = base.with_columns(
        pl.when(pl.col("date") > cutoff).then(float("nan")).otherwise(pl.col(column)).alias(column)
        for column in ("nav", "daily_return", "cash", "gross_exposure", "terminal_cash")
    )
    damaged_ledger = ledger.with_columns(
        pl.when(pl.col("date") > cutoff).then(1e9).otherwise(pl.col(column)).alias(column)
        for column in ("pretrade", "posttrade")
    )
    after = simulate_strategy(damaged_base, damaged_ledger, overlays, cost_bp=10,
                              initial_nav=1, last_date=cutoff)
    assert before.daily.equals(after.daily) and before.ledger.equals(after.ledger)


def test_monthly_calibration_return_formula_includes_previous_book_first_day():
    result = gross_scaled_monthly(np.array([1., 1.331]), np.array([1.21, 1.4641]),
                                   np.array([2., 1.]))
    assert result == pytest.approx([.42, 1.8282 / 1.42 - 1])


def _calibration_arrays(returns):
    ends = np.cumprod(1 + np.asarray(returns))
    return np.r_[1., ends[:-1]], ends


def test_scale_calibration_hits_hand_calculated_target_and_reports_unattainable_cap():
    settings = load_settings()
    executions, ends = _calibration_arrays([.1, -.1, .1, -.1])
    result = calibrate_scale(np.ones(4), executions, ends, np.ones(4, dtype=bool), settings)
    # Sample sd of [+.1,-.1,+.1,-.1] times sqrt(12) is .4; c=.12/.4.
    assert result.attained and result.scale == pytest.approx(.3, abs=1e-7)
    assert result.annual_volatility == pytest.approx(.12, abs=1e-8)
    settings["strategy"]["vol_target_annual"] = 2.
    failed = calibrate_scale(np.ones(4), executions, ends, np.ones(4, dtype=bool), settings)
    assert not failed.attained and failed.annual_volatility == pytest.approx(.8)
    assert failed.scale == pytest.approx(2) and failed.audit["fully_saturated"]


def test_first_bracketed_crossing_survives_nonmonotone_clipped_volatility():
    settings = load_settings()
    executions, ends = _calibration_arrays([.1] * 4)
    result = calibrate_scale(np.array([4., 1., 1., 1.]), executions, ends,
                             np.ones(4, dtype=bool), settings)
    # At c=2 all returns equal .2, so capped-endpoint volatility is zero.
    # Before clipping sd([.4c,.1c,.1c,.1c])=.15c; the first root is below c=.5.
    assert result.attained
    assert result.scale == pytest.approx(.12 / (.15 * np.sqrt(12)), abs=1e-7)


def test_french_state_retains_saturdays_and_uses_next_month_published_deciles():
    settings = load_settings()
    settings["features"].update(bear_days=4, market_rv_days=2)
    settings["targets"]["horizon"] = 1
    settings["strategy"]["dm_first_month"] = "1999-12-01"
    daily = pl.DataFrame({"date": [date(1999, 12, 31)] +
                          [date(2000, 1, i) for i in [26, 27, 28, 29, 31]],
                          "mkt_rf": [0., -.5, 0., 0., -.1, 0.], "rf": [0.] * 6})
    monthly = pl.DataFrame({"date": [date(2000, 2, 29)], "d10": [.2], "d1": [.1]})
    pairs, audit = build_dm_pairs(daily, monthly, settings)
    assert audit["warmup_omissions"] == 1
    assert pairs["origin"].item() == date(2000, 1, 31)
    assert pairs["target_end"].item() == date(2000, 2, 29)
    assert pairs["bear_x_var"].item() == pytest.approx(.005)
    assert pairs["next_month_return"].item() == pytest.approx(.1)
    # Positive total-market return despite negative excess return turns bear off.
    total_positive = daily.with_columns(
        pl.when(pl.col("date") == date(2000, 1, 29)).then(.2).otherwise(0.).alias("rf")
    )
    assert build_dm_pairs(total_positive, monthly, settings)[0]["bear_x_var"].item() == 0


def test_french_mean_fit_uses_strictly_completed_outcomes_only():
    refit = date(2001, 1, 2)
    pairs = pl.DataFrame({"origin": [date(2000, i, 30) for i in [8, 9, 10, 11, 12]],
                          "target_end": [date(2000, i, 30) for i in [9, 10, 11]] +
                          [refit, date(2001, 1, 31)],
                          "bear_x_var": [0., 1., 2., np.nan, np.nan],
                          "next_month_return": [.1, .3, .5, np.nan, np.nan]})
    fitted = fit_dm_mean(pairs, refit)
    assert (fitted.intercept, fitted.slope) == pytest.approx((.1, .2))
    assert fitted.n == 3 and fitted.target_end < refit
    weights = dm_multiples(np.array([0., 1., 2.]), np.ones(3), fitted, 10, 2)
    assert weights == pytest.approx([1, 2, 2])
    negative = replace(fitted, intercept=-.1, slope=0)
    assert dm_multiples(np.array([0.]), np.array([.01]), negative, 1, 2)[0] == -2


def test_french_pairs_stop_at_origin_bound_and_preserve_completed_pairs():
    settings = load_settings()
    settings["features"].update(bear_days=2, market_rv_days=2)
    settings["targets"]["horizon"] = 1
    settings["strategy"]["dm_first_month"] = "2000-12-01"
    settings["walkforward"]["design_last_origin"] = "2001-01-31"
    cutoff = date(2001, 1, 31)
    daily = pl.DataFrame({
        "date": [date(2000, 12, 28), date(2000, 12, 29), date(2001, 1, 2),
                 date(2001, 1, 15), cutoff, date(2001, 2, 1), date(2001, 2, 28),
                 date(2001, 3, 1), date(2001, 3, 30)],
        "mkt_rf": [-.2, 0., 0., -.1, 0., -.3, 0., -.4, 0.], "rf": [0.] * 9,
    })
    monthly = pl.DataFrame({
        "date": [date(2001, 1, 31), date(2001, 2, 28), date(2001, 3, 31),
                 date(2001, 4, 30)],
        "d10": [.3, .4, .5, .6], "d1": [.1] * 4,
    })
    pairs, audit = build_dm_pairs(daily, monthly, settings)
    assert pairs["origin"].to_list() == [date(2000, 12, 29), cutoff]
    assert pairs["bear_x_var"].to_list() == pytest.approx([.02, .005])
    assert pairs["next_month_return"].to_list() == pytest.approx([.2, .3])
    assert pairs["origin"].max() <= cutoff
    assert audit["source_sessions"] == 9

    # Future factor damage cannot enter even an unused post-cutoff pair.
    damaged_daily = daily.with_columns([
        pl.when(pl.col("date") > cutoff).then(float("nan")).otherwise(pl.col(name)).alias(name)
        for name in ("mkt_rf", "rf")
    ])
    damaged_pairs, damaged_audit = build_dm_pairs(damaged_daily, monthly, settings)
    assert damaged_pairs.equals(pairs) and damaged_audit == audit

    # A final allowed origin has a future target by definition. Only completed
    # targets are invariant to damaging future published monthly outcomes.
    damaged_monthly = monthly.with_columns([
        pl.when(pl.col("date") > cutoff).then(100.).otherwise(pl.col(name)).alias(name)
        for name in ("d10", "d1")
    ])
    completed = pairs.filter(pl.col("target_end") <= cutoff)
    damaged_pairs, _ = build_dm_pairs(damaged_daily, damaged_monthly, settings)
    assert damaged_pairs.filter(pl.col("target_end") <= cutoff).equals(completed)

    # Keeping the untruncated source calendar prevents January 15 becoming a
    # spurious month-end when a caller narrows the permitted origin boundary.
    settings["walkforward"]["design_last_origin"] = "2001-01-15"
    partial, _ = build_dm_pairs(daily, monthly, settings)
    assert partial["origin"].to_list() == [date(2000, 12, 29)]
    assert partial.equals(completed)


def test_dm_scale_cutoff_excludes_equal_and_future_target_ends_and_future_nav():
    settings = load_settings()
    refit = date(2001, 4, 2)
    origins = [date(2000, 12, 29), date(2001, 1, 31), date(2001, 2, 28), date(2001, 3, 30)]
    training = pl.DataFrame({"origin": origins, "refit": [refit] * 4,
                             "target_end": origins[1:] + [refit],
                             "variance_prediction": [.01, .01, .01, np.nan]})
    states = pl.DataFrame({"origin": origins, "bear_x_var": [0., 0., 0., np.nan]})
    base = pl.DataFrame({"date": [origins[0], date(2001, 1, 2), origins[1], date(2001, 2, 1),
                                   origins[2], date(2001, 3, 1), origins[3], refit],
                         "nav": [1., 1., 1.1, 1.1, .99, .99, 1.089, np.nan]})
    fit = DMMeanFit(.01, 0., refit, date(1928, 7, 31), date(2000, 11, 30),
                    date(2000, 12, 31), 800)
    calibration = calibrate_dm(base, training, states, fit, settings)
    assert calibration.attained and calibration.scale == pytest.approx(.3, abs=1e-7)
    assert calibration.audit["training_months"] == 3
    assert calibration.audit["holding_month_last"] == "2001-03-31"
    assert calibration.audit["training_target_end"] == "2001-03-30"


def test_volatility_multiple_uses_21_day_risk_and_caps_zero_risk():
    result = volatility_multiples(np.array([.0012, .0048, 0]), .12, 21, 252, 2)
    assert result == pytest.approx([1., .5, 2.])


@pytest.mark.parametrize("market_return,expected", [(.01, 50.), (.1, None)])
def test_break_even_replays_net_ledger_costs_within_fixed_search_only(market_return, expected):
    settings = load_settings()
    days = [date(2006, 1, 3), date(2006, 1, 31)]
    base = pl.DataFrame({"date": days, "nav": [1., 1. + market_return],
                         "daily_return": [0., market_return], "cash": [1., 1.],
                         "gross_exposure": [2., (2. + market_return) / (1. + market_return)],
                         "terminal_cash": [0., 0.]})
    ledger = pl.DataFrame({"date": [days[0]] * 2, "permaticker": [1, 2],
                           "pretrade": [0., 0.], "posttrade": [1., -1.]})
    bsc = pl.DataFrame({"origin": [date(2005, 12, 30)], "execution_date": [days[0]],
                        "multiple": [1.]})
    variant = bsc.with_columns(pl.lit(2.).alias("multiple"))
    result = break_even_cost(base, ledger, variant, bsc, settings,
                             last_date=days[-1], initial_nav=1)
    # Monthly net difference is market_return - 2*c: initial trades are 4 vs 2.
    if expected is None:
        assert result["break_even_bp"] is None and result["status"] == "no_crossing_in_cost_grid"
    else:
        assert result["break_even_bp"] == pytest.approx(expected)
        assert result["status"] == "crossing" and result["crossings_bp"] == [expected]
    unchanged = break_even_cost(base, ledger, bsc, bsc, settings,
                                last_date=days[-1], initial_nav=1)
    assert unchanged["break_even_bp"] is None and unchanged["status"] == "equal_on_cost_grid"
