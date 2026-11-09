"""Hand-calculated monthly observables and formation timing (Methodology)."""


import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from momenta import l2
from momenta.manifest import sha256_file
from momenta.settings import load_settings


def test_shape_and_rank_ties_have_hand_calculated_values():
    config = load_settings()['l2']
    bowley, moors = l2.shape(
        np.array([0., 1., 2., 5., 10., np.nan]), config['bowley_quantiles'],
        config['moors_quantiles'], method=config['quantile_method'],
    )
    assert bowley == pytest.approx(.5)
    assert moors == pytest.approx(1.25)
    assert np.isnan(l2.shape(np.ones(6), config['bowley_quantiles'],
                            config['moors_quantiles'], method=config['quantile_method'])).all()
    assert l2.rank_correlation(np.array([1., 1., 2.]), np.array([1., 2., 3.])) == pytest.approx(
        np.sqrt(3)/2,
    )
    assert np.isnan(l2.rank_correlation(np.ones(3), np.arange(3)))
    assert l2.breadth(np.array([-2., 0., 0., 1., 1., np.nan])) == pytest.approx(1/3)
    assert 'breadth_m' not in l2.L2_COLUMNS and len(l2.L2_COLUMNS) == 9


def test_balancing_separates_missing_and_constant_stocks():
    a = np.array([-1., -1., 1., 1.])
    b = np.array([-1., 1., -1., 1.])
    panel = np.column_stack([a, .6*a+.8*b, np.full(4, .1), [0., np.nan, 1., 2.]])
    balanced = l2.balanced_standardize(panel)
    assert balanced.columns.tolist() == [0, 1]
    counts = balanced.complete_count, balanced.missing_count, balanced.zero_variance_count
    assert counts == (3, 1, 1)
    assert_allclose(balanced.z.mean(axis=0), [0., 0.], atol=1e-15)
    assert_allclose(np.mean(balanced.z**2, axis=0), [1., 1.])
    assert l2.within_correlation(balanced.z) == pytest.approx(.6)
    loser = l2.balanced_standardize(np.column_stack([-a, b]))
    assert l2.within_correlation(loser.z) == pytest.approx(0.)
    # The four cross correlations are -1, 0, -.6, +.8: their mean is -.2.
    assert l2.cross_correlation(balanced.z, loser.z) == pytest.approx(-.2)


@pytest.fixture
def monthly_panel():
    settings = load_settings()
    settings['l2']['correlation_days'] = 4
    start = date(1998, 11, 1)
    days = [start+timedelta(days=i) for i in range(125)
            if (start+timedelta(days=i)).weekday() < 5]
    dates = [date(1998, 12, 31), date(1999, 1, 29), date(1999, 2, 26)]
    ids = np.arange(1, 8)
    members = [np.arange(1, 7), np.arange(2, 8), np.arange(2, 8)]
    raws = [[-3., -2., -1., 1., 2., 3.], [-3., 3., -1., 1., -2., 2.],
            [3., 2., 1., -1., -2., -3.]]
    universe = pl.DataFrame([{'date': d, 'permaticker': int(p)}
                             for d, group in zip(dates, members, strict=True) for p in group])
    signals = universe.with_columns(pl.Series('raw', [r for group in raws for r in group]))
    returns = np.outer(np.arange(1, len(days)+1), ids)*.00001
    a, b = np.array([-1., -1., 1., 1.])*.01, np.array([-1., 1., -1., 1.])*.01
    december = slice(days.index(dates[0])-3, days.index(dates[0])+1)
    returns[december, 0], returns[december, 1] = a, a
    returns[december, 4], returns[december, 5] = a, -a
    january = slice(days.index(dates[1])-3, days.index(dates[1])+1)
    returns[january, 2], returns[january, 6] = a, .6*a+.8*b
    returns[january, 1], returns[january, 5] = -a, b
    legs = [[(1, -.5), (2, -.5), (5, .5), (6, .5)],
            [(2, -.5), (6, -.5), (3, .5), (7, .5)],
            [(6, -.5), (7, -.5), (2, .5), (3, .5)]]
    weights = pl.DataFrame([{
        'formation': d, 'permaticker': p, 'weight': weight,
        'series': 'RAW', 'construction': settings['portfolios']['primary'],
    } for d, leg in zip(dates, legs, strict=True) for p, weight in leg])
    monthly = pl.DataFrame([{'date': d, 'permaticker': int(p), 'monthly_return': value}
        for d, group, values in zip(
            [dates[0], date(1999, 1, 31), date(1999, 2, 28)], members,
            [[-.03, -.02, -.01, .01, .02, .03], [.03, .02, None, -.01, .0, .01],
             [-.03, -.02, -.01, .01, .02, .03]], strict=True,
        ) for p, value in zip(group, values, strict=True)])
    return settings, days, ids, returns, universe, signals, monthly, weights


def _snapshots(inputs):
    settings, days, ids, returns, universe, signals, monthly, weights = inputs
    return l2.monthly_features(days, ids, returns, universe, signals, monthly, weights, settings)


def test_snapshots_use_current_raw_legs_and_previous_universe_intersection(monthly_panel):
    result = _snapshots(monthly_panel)
    december, january = result.row(0, named=True), result.row(1, named=True)
    assert december['date'] == date(1998, 12, 31)
    assert december['rank_persist'] is None and december['rank_ic'] is None
    assert december['rho_ww'] == pytest.approx(-1.)
    assert december['rho_ll'] == pytest.approx(1.)
    assert january['date'] == date(1999, 1, 29)
    assert january['rho_ww'] == pytest.approx(.6)
    assert january['rho_ll'] == pytest.approx(0.)
    assert january['rho_wl'] == pytest.approx(-.2)
    assert january['shape_mom_bowley'] == pytest.approx(0.)
    assert january['shape_mom_moors'] == pytest.approx(5/7)
    assert january['shape_r1m_moors'] == pytest.approx(1.)
    # Intersection IDs2..6: old ranks1..5, current ranks[1,5,3,4,2].
    # Rank-IC omits missing ID4: old ranks1..4, R ranks[4,3,1,2].
    assert january['rank_persist'] == pytest.approx(.1)
    assert january['rank_ic'] == pytest.approx(-.8)
    assert january['rank_intersection_count'] == 5 and january['rank_ic_count'] == 4
    assert january['shape_r1m_count'] == 5 and january['universe_size'] == 6
    assert january['breadth_m'] == 0.


def test_snapshot_audits_omissions_and_never_fills_missing_returns(monthly_panel):
    settings, days, ids, returns, universe, signals, monthly, weights = monthly_panel
    end = days.index(date(1999, 1, 29))+1
    returns[end-2, 6] = np.nan
    returns[end-4:end, 5] = .1
    result = _snapshots((settings, days, ids, returns, universe, signals, monthly, weights))
    point = result.row(1, named=True)
    assert point['winner_total'] == 2 and point['winner_balanced'] == 1
    assert point['winner_complete'] == 1 and point['winner_missing'] == 1
    assert point['loser_complete'] == 2 and point['loser_zero_variance'] == 1
    assert point['loser_balanced'] == 1
    assert point['rho_ww'] is None and point['rho_ll'] is None
    assert point['rho_wl'] == pytest.approx(-1.)


def test_future_inputs_cannot_change_an_earlier_snapshot(monthly_panel):
    original = _snapshots(monthly_panel)
    settings, days, ids, returns, universe, signals, monthly, weights = monthly_panel
    cutoff = date(1999, 1, 29)
    returns[np.array([day > cutoff for day in days])] = 1e9
    signals = signals.with_columns(pl.when(pl.col('date') > cutoff).then(-99999.)
                                   .otherwise(pl.col('raw')).alias('raw'))
    monthly = monthly.with_columns(
        pl.when(pl.col('date') >= date(1999, 2, 1)).then(99.)
        .otherwise(pl.col('monthly_return')).alias('monthly_return'),
    )
    weights = weights.with_columns(pl.when(pl.col('formation') > cutoff).then(-pl.col('weight'))
                                   .otherwise(pl.col('weight')).alias('weight'))
    damaged = _snapshots((settings, days, ids, returns, universe, signals, monthly, weights))
    assert not original.equals(damaged)
    assert_frame_equal(original.filter(pl.col('date') <= cutoff),
                       damaged.filter(pl.col('date') <= cutoff), check_exact=True)


def test_bounded_build_keeps_year_boundary_rank_history_and_terminal_credit(
    monthly_panel, tmp_path, monkeypatch,
):
    settings, days, ids, returns, universe, signals, monthly, weights = monthly_panel
    root = tmp_path/'data'
    settings['data']['data_root'] = str(root)
    curated, portfolio_dir = root/'curated', root/'derived/portfolios'
    curated.mkdir(parents=True)
    portfolio_dir.mkdir(parents=True)
    for folder in (curated, portfolio_dir):
        (folder/'_metadata.json').write_text('{}\n')
    universe.write_parquet(curated/'universe.parquet')
    signals.write_parquet(portfolio_dir/'signals.parquet')
    monthly.write_parquet(curated/'monthly_returns.parquet')
    weights.write_parquet(portfolio_dir/'formation_weights.parquet')
    pl.DataFrame({'date': days}).write_parquet(curated/'trading_calendar.parquet')
    terminal_date, terminal_id = date(1999, 1, 29), 7
    terminal_return = returns[days.index(terminal_date), terminal_id-1]
    pl.DataFrame({'date': [terminal_date], 'permaticker': [terminal_id],
                  'terminal_return': [terminal_return]}).write_parquet(curated/'delistings.parquet')
    for year in (1998, 1999):
        folder = curated/f'prices/year={year}'
        folder.mkdir(parents=True)
        pl.DataFrame([{'date': day, 'permaticker': int(p), 'daily_return': returns[t, c],
                       'close': 10., 'volume': 100.}
            for t, day in enumerate(days) if day.year == year for c, p in enumerate(ids)
            if (day, p) != (terminal_date, terminal_id)]).write_parquet(folder/'part.parquet')
    monkeypatch.setattr(l2, 'provenance', lambda *args: {
        'config_hash': 'fixture', 'git_hash': 'fixture', 'git_dirty': False,
        'data_manifest_hash': 'fixture', 'data_vintage': {'fixture': 'synthetic'},
    })
    windows = []
    source_reader = l2._stock_panel

    def tracked_reader(curated, panel_days, panel_ids, delistings):
        windows.append((panel_days, panel_ids))
        return source_reader(curated, panel_days, panel_ids, delistings)

    monkeypatch.setattr(l2, '_stock_panel', tracked_reader)
    path = l2.build_l2(settings, date(1999, 1, 29))
    result = pl.read_parquet(path)
    expected = _snapshots(monthly_panel).filter(pl.col('date') <= date(1999, 1, 29))
    assert_frame_equal(result, expected, check_exact=True)
    assert result['rank_ic'][1] == pytest.approx(-.8)
    assert result['rho_ww'][1] == pytest.approx(.6)
    assert len(windows) == 2
    assert all(len(window) == 4 for window, _ in windows)
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    assert metadata['first_origin'] == '1998-12-31'
    assert metadata['first_complete_origin'] == '1999-01-29'
    assert metadata['feature_availability']['rank_ic'] == '1999-01-29'
    assert metadata['artifact_hash'] == sha256_file(path)
    assert all(sha256_file(Path(name)) == digest
               for name, digest in metadata['input_hashes'].items())
    assert {'l2.py', 'features.py', 'signals.py', 'portfolios.py'}.issubset(
        {Path(name).name for name in metadata['source_hashes']},
    )


def test_builder_refuses_holdout_before_reading_inputs():
    settings = load_settings()
    settings['data']['data_root'] = '/does/not/exist'
    with pytest.raises(ValueError, match='Holdout L2'):
        l2.build_l2(settings, date(2015, 12, 31))
