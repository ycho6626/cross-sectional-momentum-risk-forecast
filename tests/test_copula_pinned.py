"""Bounded pinned-vintage leakage for monthly copula geometry (Methodology)."""


import bisect
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from momenta.copula import CM_COLUMNS, Q_COLUMNS, monthly_clouds, unit_directions
from momenta.features import _stock_panel
from momenta.settings import load_settings

pytestmark = pytest.mark.slow


def test_pinned_copula_ignores_later_stock_market_signal_universe_and_economic_month_data():
    """December's Saturday label is current information at Friday's formation.

    Read four real month-end universes and their 126-session warmup. Damage
    every consumed future input in memory; require exact earlier point clouds,
    embeddings, CM and Q, and nonvacuous changes to later geometry.
    """
    settings = load_settings()
    root = Path(settings['data']['data_root'])
    curated, portfolio_dir = root/'curated', root/'derived/portfolios'
    required = [
        curated/'universe.parquet', curated/'monthly_returns.parquet',
        curated/'delistings.parquet', curated/'trading_calendar.parquet',
        curated/'french/ff3_daily.parquet', portfolio_dir/'signals.parquet',
        curated/'prices/year=2005/part.parquet', curated/'prices/year=2006/part.parquet',
    ]
    if any(not path.is_file() for path in required):
        pytest.skip('Pinned curation and portfolio inputs are absent')
    first, cutoff, last = date(2005, 11, 30), date(2005, 12, 30), date(2006, 2, 28)
    current_month_label = date(2005, 12, 31)
    formations = [first, cutoff, date(2006, 1, 31), last]
    assert cutoff.weekday() == 4 and current_month_label > cutoff
    calendar = pl.read_parquet(curated/'trading_calendar.parquet')['date'].sort().to_list()
    start = bisect.bisect_left(calendar, first)-settings['ot']['window_days']+1
    assert start >= 0 and all(day in calendar for day in formations)
    days = calendar[start:bisect.bisect_right(calendar, last)]
    universe = pl.scan_parquet(curated/'universe.parquet').filter(
        pl.col('date').is_in(formations),
    ).select('date', 'permaticker').collect()
    signals = pl.scan_parquet(portfolio_dir/'signals.parquet').filter(
        pl.col('date').is_in(formations),
    ).select('date', 'permaticker', 'raw').collect()
    ids = np.sort(universe['permaticker'].unique().to_numpy())
    monthly = pl.scan_parquet(curated/'monthly_returns.parquet').filter(
        pl.col('date').is_between(date(2005, 11, 1), last), pl.col('permaticker').is_in(ids),
    ).select('date', 'permaticker', 'monthly_return').collect()
    terminal = pl.scan_parquet(curated/'delistings.parquet').filter(
        pl.col('date').is_between(days[0], last), pl.col('permaticker').is_in(ids),
    ).select('date', 'permaticker', 'terminal_return').collect()
    returns, dollar_volume, _ = _stock_panel(curated, days, ids, terminal)
    del dollar_volume
    factors = pl.scan_parquet(curated/'french/ff3_daily.parquet').filter(
        pl.col('date').is_between(days[0], last),
    ).select('date', 'mkt_rf').collect()
    market = pl.DataFrame({'date': days}).join(
        factors, on='date', how='left', validate='1:1',
    )['mkt_rf'].to_numpy()
    assert np.isfinite(market).all()
    future_days = np.array([day > cutoff for day in days])
    assert np.isfinite(returns[future_days]).sum() > len(ids)
    changed_returns, changed_market = returns.copy(), market.copy()
    changed_returns[future_days] = 1+np.abs(changed_returns[future_days])
    changed_market[future_days] = -1+7*changed_market[future_days]
    for original, changed in ((returns, changed_returns), (market, changed_market)):
        assert not np.array_equal(original, changed, equal_nan=True)
        np.testing.assert_array_equal(original[~future_days], changed[~future_days])
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
    removed = universe.filter(pl.col('date') > cutoff).sort('date', 'permaticker').group_by(
        'date',
    ).agg(pl.col('permaticker').first())
    assert removed.height == 2
    changed_universe = universe.join(removed, on=['date', 'permaticker'], how='anti')
    for original, changed, preserved_through in (
        (signals, changed_signals, cutoff),
        (monthly, changed_monthly, current_month_label),
        (universe, changed_universe, cutoff),
    ):
        assert not original.sort('date', 'permaticker').equals(changed.sort('date', 'permaticker'))
        assert_frame_equal(
            original.filter(pl.col('date') <= preserved_through).sort('date', 'permaticker'),
            changed.filter(pl.col('date') <= preserved_through).sort('date', 'permaticker'),
            check_exact=True,
        )
    directions = unit_directions(settings['ot']['directions'], 4, settings['ot']['seed'])
    points, original = monthly_clouds(
        days, ids, returns, market, universe, signals, monthly, directions,
        settings, formations=formations,
    )
    changed_points, changed = monthly_clouds(
        days, ids, changed_returns, changed_market, changed_universe, changed_signals,
        changed_monthly, directions, settings, formations=formations,
    )
    earlier = original.filter(pl.col('date') <= cutoff)
    assert earlier['date'].to_list() == [first, cutoff]
    assert earlier.drop_nulls(CM_COLUMNS+Q_COLUMNS).height == 2
    assert earlier['cloud_size'].min() == settings['universe']['n_universe']
    assert_frame_equal(earlier, changed.filter(pl.col('date') <= cutoff), check_exact=True)
    assert_frame_equal(points.filter(pl.col('date') <= cutoff),
                       changed_points.filter(pl.col('date') <= cutoff), check_exact=True)
    for columns in (['embedding'], list(CM_COLUMNS)):
        assert not original.filter(pl.col('date') > cutoff).select(columns).equals(
            changed.filter(pl.col('date') > cutoff).select(columns),
        ), 'Future damage must change later geometry, not only omission counts'
