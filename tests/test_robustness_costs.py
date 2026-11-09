"""Frozen decision identity and exact net-cost accounting."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from momenta.robustness_costs import account_parity, frozen_overlays
from momenta.strategies import simulate_strategy


def test_frozen_gross_net_overlay_copies_must_match_including_fit_time():
    frame = pl.DataFrame(
        {
            "series": ["RAW"] * 4,
            "strategy": ["L1"] * 4,
            "cost_bp": [0.0, 0.0, 10.0, 10.0],
            "origin": [date(2015, 12, 31), date(2016, 1, 29)] * 2,
            "execution_date": [date(2016, 1, 4), date(2016, 2, 1)] * 2,
            "fit_date": [date(2015, 1, 2), date(2016, 1, 4)] * 2,
            "multiple": [0.7, 1.2, 0.7, 1.2],
        }
    )
    selected = frozen_overlays(frame)
    assert selected["RAW", "L1"]["multiple"].to_list() == [0.7, 1.2]
    bad = frame.with_columns(
        pl.when(pl.col("cost_bp") == 10).then(2.0).otherwise(pl.col("multiple")).alias("multiple")
    )
    with pytest.raises(ValueError, match="identical"):
        frozen_overlays(bad)


@pytest.mark.parametrize("bp", [5.0, 20.0, 40.0])
def test_registered_cost_grid_uses_changed_nav_and_nets_offsetting_trades(bp):
    # Gross book starts +/-1 and gains .1 on its long before a new half-sized
    # target. At cost fraction c, entry NAV=1-2c; next pre-cost NAV=1.1-2c.
    # Target long/short = ±(.55-c), pretrade=1.1/-1, total trade=1+2c.
    # The next day's 10% long gain is .055-.1c, on the actual after-cost NAV.
    dates = [date(2016, 1, 4), date(2016, 2, 1), date(2016, 2, 2)]
    base = pl.DataFrame(
        {
            "date": dates,
            "nav": [1.0, 1.1, 1.21],
            "daily_return": [0.0, 0.1, 0.1],
            "cash": [1.0, 1.1, 1.1],
            "gross_exposure": [2.0, 2.0, 2.31 / 1.21],
            "terminal_cash": [0.0] * 3,
        }
    )
    ledger = pl.DataFrame(
        {
            "date": [dates[0]] * 2 + [dates[1]] * 2,
            "permaticker": [1, 2, 1, 2],
            "pretrade": [0.0, 0.0, 1.1, -1.0],
            "posttrade": [1.0, -1.0, 1.1, -1.1],
        }
    )
    overlays = pl.DataFrame(
        {
            "origin": [date(2015, 12, 31), date(2016, 1, 29)],
            "execution_date": dates[:2],
            "multiple": [1.0, 0.5],
        }
    )
    actual = simulate_strategy(
        base, ledger, overlays, cost_bp=bp, initial_nav=1, last_date=dates[-1]
    )
    c = bp / 10000
    expected = [1 - 2 * c, 1.1 - 2 * c - c * (1 + 2 * c), 1.155 - 3.1 * c - 2 * c * c]
    assert actual.daily["nav"].to_list() == pytest.approx(expected)
    assert actual.rebalances["cost"].to_list() == pytest.approx([2 * c, c * (1 + 2 * c)])
    assert actual.ledger.filter(pl.col("date") == dates[1])["trade"].to_list() == pytest.approx(
        [-0.55 - c, 0.45 + c]
    )
    assert np.prod(1 + actual.daily["daily_return"].to_numpy()) == pytest.approx(expected[-1])


def test_cost_parity_detects_missing_month_and_changed_net_path():
    expected = pl.DataFrame(
        {
            "series": ["RAW"],
            "strategy": ["Static"],
            "cost_bp": [10.0],
            "date": [date(2016, 1, 31)],
            "monthly_return": [0.02],
            "nav": [1.02],
        }
    )
    assert account_parity(expected, expected, 1e-12) == 0
    with pytest.raises(ValueError, match="sample"):
        account_parity(expected.head(0), expected, 1e-12)
    with pytest.raises(ValueError, match="reproduce"):
        account_parity(expected.with_columns(pl.lit(1.021).alias("nav")), expected, 1e-12)
