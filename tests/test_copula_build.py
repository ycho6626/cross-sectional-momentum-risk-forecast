"""Copula month-end timing, missing coordinates and bounded input provenance (Methodology)."""


import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from momenta import copula
from momenta.manifest import sha256_file
from momenta.settings import load_settings


@pytest.fixture
def panel():
    settings = load_settings()
    first, end = date(1998, 6, 1), date(1999, 3, 5)
    days = [first+timedelta(days=i) for i in range((end-first).days+1)
            if (first+timedelta(days=i)).weekday() < 5]
    ids = np.arange(1, 7)
    market = .005*np.sin(np.arange(len(days))/3)+.003*np.cos(np.arange(len(days))/4)
    returns = ids[None, :]*market[:, None] + ids[None, :]*.0001
    # ID6 is missing inside January's complete-window requirement.
    returns[days.index(date(1999, 1, 15)), 5] = np.nan
    formations = [date(1998, 12, 31), date(1999, 1, 29), date(1999, 2, 26)]
    universe = pl.DataFrame([{'date': day, 'permaticker': int(p)}
                             for day in formations for p in ids])
    signals = universe.with_columns(pl.Series('raw', [*ids, *ids[::-1], *ids]/np.float64(10)))
    monthly = pl.DataFrame([{'date': day, 'permaticker': int(p),
                            'monthly_return': None if (day.month == 1 and p == 5) else float(p)/100}
        for day in (date(1998, 12, 31), date(1999, 1, 31), date(1999, 2, 28)) for p in ids])
    directions = copula.unit_directions(settings['ot']['directions'], 4, settings['ot']['seed'])
    return settings, days, ids, returns, market, universe, signals, monthly, directions


def snapshots(panel):
    settings, days, ids, returns, market, universe, signals, monthly, directions = panel
    return copula.monthly_clouds(days, ids, returns, market, universe, signals, monthly,
                                 directions, settings)


def test_clouds_use_current_month_and_one_complete_four_coordinate_subset(panel):
    points, monthly = snapshots(panel)
    assert monthly['date'].to_list() == [date(1998, 12, 31), date(1999, 1, 29), date(1999, 2, 26)]
    assert monthly['cloud_size'].to_list() == [6, 4, 5]
    january = points.filter(pl.col('date') == date(1999, 1, 29))
    assert january['permaticker'].to_list() == [1, 2, 3, 4]
    # January excludes ID5's missing R and ID6's incomplete daily history;
    # every marginal is reranked on the same four stocks, denominator5.
    assert_allclose(january['mom'], [.8, .6, .4, .2])
    assert_allclose(january['r1m'], [.2, .4, .6, .8])
    assert_allclose(january['beta'], [.2, .4, .6, .8])
    assert_allclose(january['raw_beta'], [1., 2., 3., 4.], atol=1e-13)
    assert monthly['cm_mom_beta'].to_list() == pytest.approx([1., -1., 1.])
    assert monthly['missing_current_return'].to_list() == [0, 1, 0]
    assert monthly['incomplete_daily_window'].to_list() == [0, 1, 1]


def test_later_rows_cannot_change_an_earlier_cloud_embedding_or_cm(panel):
    points, original = snapshots(panel)
    settings, days, ids, returns, market, universe, signals, monthly, directions = panel
    cutoff = date(1999, 1, 29)
    later = np.array([day > cutoff for day in days])
    returns[later] = 1+np.abs(returns[later])
    market[later] = -1+market[later]*7
    signals = signals.with_columns(pl.when(pl.col('date') > cutoff).then(-100-pl.col('raw'))
                                   .otherwise(pl.col('raw')).alias('raw'))
    monthly = monthly.with_columns(pl.when(pl.col('date') > date(1999, 1, 31))
                                   .then(10+pl.col('monthly_return'))
                                   .otherwise(pl.col('monthly_return')).alias('monthly_return'))
    changed_points, changed = snapshots(
        (settings, days, ids, returns, market, universe, signals, monthly, directions),
    )
    assert_frame_equal(points.filter(pl.col('date') <= cutoff),
                       changed_points.filter(pl.col('date') <= cutoff), check_exact=True)
    assert_frame_equal(original.filter(pl.col('date') <= cutoff),
                       changed.filter(pl.col('date') <= cutoff), check_exact=True)
    assert not original.equals(changed)


def test_builder_preserves_cross_year_inputs_and_validates_every_pinned_artifact(
    panel, tmp_path, monkeypatch,
):
    settings, days, ids, returns, market, universe, signals, monthly, directions = panel
    root = tmp_path/'data'
    settings['data']['data_root'] = str(root)
    curated, portfolio_dir = root/'curated', root/'derived/portfolios'
    (curated/'french').mkdir(parents=True)
    portfolio_dir.mkdir(parents=True)
    for directory in (curated, portfolio_dir):
        (directory/'_metadata.json').write_text('{}\n')
    universe.write_parquet(curated/'universe.parquet')
    signals.write_parquet(portfolio_dir/'signals.parquet')
    monthly.write_parquet(curated/'monthly_returns.parquet')
    pl.DataFrame({'date': days}).write_parquet(curated/'trading_calendar.parquet')
    pl.DataFrame({'date': days, 'mkt_rf': market}).write_parquet(curated/'french/ff3_daily.parquet')
    pl.DataFrame(schema={'date': pl.Date, 'permaticker': pl.Int64,
                          'terminal_return': pl.Float64}).write_parquet(
        curated/'delistings.parquet',
    )
    (root/'raw').mkdir()
    (root/'raw/manifest.json').write_text('{"schema_version":1,"entries":{}}\n')
    for year in (1998, 1999):
        folder = curated/f'prices/year={year}'
        folder.mkdir(parents=True)
        pl.DataFrame([{'date': day, 'permaticker': int(p), 'daily_return': returns[t, c],
                       'close': 10., 'volume': 100.}
            for t, day in enumerate(days) if day.year == year for c, p in enumerate(ids)]
        ).write_parquet(folder/'part.parquet')
    read_windows = []
    original_reader = copula._stock_panel

    def read(curated, panel_days, panel_ids, delistings):
        read_windows.append(panel_days)
        return original_reader(curated, panel_days, panel_ids, delistings)

    monkeypatch.setattr(copula, '_stock_panel', read)
    path = copula.build_copula(settings, date(1999, 1, 29))
    actual, metadata = copula.load_copula(settings)
    expected = snapshots(panel)[1].filter(pl.col('date') <= date(1999, 1, 29))
    assert_frame_equal(actual, expected, check_exact=True)
    assert len(read_windows) == 2 and all(len(window) == 126 for window in read_windows)
    assert metadata['embedding_shape'] == [128, 101]
    assert_allclose(metadata['probability_grid'], np.linspace(0., 1., 101))
    assert metadata['embedding_scale'] == pytest.approx(1/np.sqrt(128*101))
    assert metadata['artifact_hash'] == sha256_file(path)
    assert all(sha256_file(Path(name)) == digest
               for name, digest in metadata['cloud_hashes'].items())
    assert json.loads(path.with_suffix('.metadata.json').read_text()) == metadata
    # Corrupt a consumed input without touching the monthly artifact: the
    # validator must reject the build before a learned transform can use it.
    (curated/'french/ff3_daily.parquet').write_bytes(b'changed source bytes')
    with pytest.raises(ValueError, match='Stale copula input'):
        copula.load_copula(settings)


def test_builder_rejects_actual_holdout_before_any_input_read():
    settings = load_settings()
    settings['data']['data_root'] = '/does/not/exist'
    with pytest.raises(ValueError, match='Holdout copula'):
        copula.build_copula(settings, date(2015, 12, 31))
