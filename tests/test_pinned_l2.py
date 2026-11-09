"""Bounded pinned-vintage leakage for shared monthly L2 snapshots (Methodology)."""


import bisect
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from momenta.features import _stock_panel
from momenta.l2 import L2_COLUMNS, monthly_features
from momenta.settings import load_settings

pytestmark = pytest.mark.slow


def test_pinned_l2_ignores_later_stock_signal_universe_weight_and_economic_month_data():
    """Friday 2005-12-30 may use December's 2005-12-31 return label, but no January data.

    Two snapshots precede or equal the cutoff; two follow it. Read only their
    universe union and a 126-session warmup, never the full price table. Damage
    each later input in memory, retaining the current economic month's return
    even when its calendar label is later than its final trading session.
    """
    settings = load_settings()
    root = Path(settings['data']['data_root'])
    curated, portfolio_dir = root/'curated', root/'derived/portfolios'
    required = [
        curated/'universe.parquet', curated/'monthly_returns.parquet',
        curated/'delistings.parquet', curated/'trading_calendar.parquet',
        portfolio_dir/'signals.parquet', portfolio_dir/'formation_weights.parquet',
        curated/'prices/year=2005/part.parquet', curated/'prices/year=2006/part.parquet',
    ]
    if any(not path.is_file() for path in required):
        pytest.skip('Pinned curation and portfolio inputs are absent')
    cutoff, current_month_label = date(2005, 12, 30), date(2005, 12, 31)
    first, last, prior = date(2005, 11, 30), date(2006, 2, 28), date(2005, 10, 31)
    formations = [first, cutoff, date(2006, 1, 31), last]
    assert cutoff.weekday() == 4 and current_month_label > cutoff
    calendar = pl.read_parquet(curated/'trading_calendar.parquet')['date'].sort().to_list()
    start = bisect.bisect_left(calendar, first) - settings['l2']['correlation_days'] + 1
    assert start >= 0 and all(day in calendar for day in formations)
    days = calendar[start:bisect.bisect_right(calendar, last)]
    universe = pl.scan_parquet(curated/'universe.parquet').filter(
        pl.col('date').is_between(prior, last),
    ).select('date', 'permaticker').collect()
    signals = pl.scan_parquet(portfolio_dir/'signals.parquet').filter(
        pl.col('date').is_between(prior, last),
    ).select('date', 'permaticker', 'raw').collect()
    weights = pl.scan_parquet(portfolio_dir/'formation_weights.parquet').filter(
        pl.col('formation').is_between(prior, last), pl.col('series') == 'RAW',
        pl.col('construction') == settings['portfolios']['primary'],
    ).select('formation', 'series', 'construction', 'permaticker', 'weight').collect()
    ids = np.sort(universe['permaticker'].unique().to_numpy())
    monthly = pl.scan_parquet(curated/'monthly_returns.parquet').filter(
        pl.col('date').is_between(date(2005, 11, 1), last), pl.col('permaticker').is_in(ids),
    ).select('date', 'permaticker', 'monthly_return').collect()
    terminal = pl.scan_parquet(curated/'delistings.parquet').filter(
        pl.col('date').is_between(days[0], last), pl.col('permaticker').is_in(ids),
    ).select('date', 'permaticker', 'terminal_return').collect()
    returns, dollar_volume, _ = _stock_panel(curated, days, ids, terminal)
    del dollar_volume
    future_days = np.array([day > cutoff for day in days])
    assert np.isfinite(returns[future_days]).sum() > len(ids)
    changed_returns = returns.copy()
    changed_returns[future_days] = 1 + np.abs(changed_returns[future_days])
    assert not np.array_equal(changed_returns, returns, equal_nan=True)
    np.testing.assert_array_equal(changed_returns[~future_days], returns[~future_days])

    changed_signals = signals.with_columns(
        pl.when(pl.col('date') > cutoff).then(-100-pl.col('raw'))
        .otherwise(pl.col('raw')).alias('raw'),
    )
    changed_monthly = monthly.with_columns(
        pl.when(pl.col('date') > current_month_label)
        .then(10+pl.col('monthly_return').abs().fill_null(0))
        .otherwise(pl.col('monthly_return')).alias('monthly_return'),
    )
    assert monthly.filter(pl.col('date') == current_month_label).height > 0
    assert_frame_equal(monthly.filter(pl.col('date') == current_month_label),
                       changed_monthly.filter(pl.col('date') == current_month_label),
                       check_exact=True)
    changed_weights = weights.with_columns(
        pl.when(pl.col('formation') > cutoff).then(-pl.col('weight'))
        .otherwise(pl.col('weight')).alias('weight'),
    )
    # Remove one middle-tercile name per later formation. Its pinned extreme
    # legs remain valid subsets, while universe membership itself is damaged.
    middle = universe.filter(pl.col('date') > cutoff).join(
        weights.select(pl.col('formation').alias('date'), 'permaticker'),
        on=['date', 'permaticker'], how='anti',
    )
    removed = middle.sort('date', 'permaticker').group_by('date').agg(pl.col('permaticker').first())
    assert removed.height == 2
    changed_universe = universe.join(removed, on=['date', 'permaticker'], how='anti')
    for original, changed, time_column, preserved_through in (
        (signals, changed_signals, 'date', cutoff),
        (monthly, changed_monthly, 'date', current_month_label),
        (weights, changed_weights, 'formation', cutoff),
        (universe, changed_universe, 'date', cutoff),
    ):
        assert not original.sort(time_column, 'permaticker').equals(
            changed.sort(time_column, 'permaticker'),
        ), 'Every supplied future input must actually change'
        assert_frame_equal(
            original.filter(pl.col(time_column) <= preserved_through)
            .sort(time_column, 'permaticker'),
            changed.filter(pl.col(time_column) <= preserved_through)
            .sort(time_column, 'permaticker'),
            check_exact=True,
        )

    original = monthly_features(days, ids, returns, universe, signals, monthly, weights,
                                settings, formations=formations)
    changed = monthly_features(days, ids, changed_returns, changed_universe, changed_signals,
                               changed_monthly, changed_weights, settings, formations=formations)
    earlier = original.filter(pl.col('date') <= cutoff)
    assert earlier['date'].to_list() == [first, cutoff]
    assert earlier.drop_nulls(L2_COLUMNS).height == 2, 'Require complete earlier real features'
    assert_frame_equal(earlier, changed.filter(pl.col('date') <= cutoff), check_exact=True)
    assert not original.filter(pl.col('date') > cutoff).select('date', *L2_COLUMNS).equals(
        changed.filter(pl.col('date') > cutoff).select('date', *L2_COLUMNS),
    ), 'Future damage must change a later predictor snapshot, not only an audit count'
