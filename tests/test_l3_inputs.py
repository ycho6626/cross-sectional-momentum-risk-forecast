"""L3 pre-read origin guards and isolated real-month HMM initialization (Methodology)."""

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from momenta import l3_inputs
from momenta.freeze import write_toml
from momenta.manifest import sha256_file
from momenta.settings import load_settings


@pytest.mark.parametrize('rebuild', [False, True])
def test_post_design_request_fails_before_any_input_read_or_builder(rebuild, monkeypatch):
    settings = load_settings()
    settings['data']['data_root'] = '/does/not/exist'

    def forbidden(*args, **kwargs):
        pytest.fail('The origin guard must precede reads and builds')

    for name in ('_design_inputs', '_copy_bdp', 'build_features', 'build_targets', 'build_l2',
                 'load_copula', 'build_copula'):
        monkeypatch.setattr(l3_inputs, name, forbidden)
    monkeypatch.setattr(l3_inputs.pl, 'read_parquet', forbidden)
    with pytest.raises(ValueError, match='scoped design bounds'):
        l3_inputs.prepare_inputs(settings, date(2015, 12, 31), rebuild)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    settings = load_settings()
    root = tmp_path/'original'
    settings['data']['data_root'] = str(root)
    settings['walkforward']['design_last_origin'] = '2001-01-03'
    settings['targets'].update(horizon=2, lag=1)
    for relative in ('raw', 'curated', 'derived/portfolios', 'derived/baseline', 'derived/l2',
                     'derived/final'):
        (root/relative).mkdir(parents=True)
    (root/'raw/manifest.json').write_text('{"schema_version":1,"entries":{}}\n')
    config = root/'settings.toml'
    document = {key: value for key, value in settings.items() if key != 'config_path'}
    write_toml(config, document)
    settings['config_path'] = str(config)
    days = [date(2000, 12, 28), date(2000, 12, 29), date(2001, 1, 2), date(2001, 1, 3),
            date(2001, 1, 4), date(2001, 1, 5), date(2001, 1, 8), date(2001, 1, 9)]
    pl.DataFrame({'date': days}).write_parquet(root/'curated/trading_calendar.parquet')
    books = pl.DataFrame([{'date': day, 'series': series,
                           'construction': settings['portfolios']['primary'],
                           'daily_return': (i+1)/100}
                          for i, day in enumerate(days) for series in ('RAW', 'RAW_c', 'RES')])
    books.write_parquet(root/'derived/portfolios/book_daily.parquet')
    (root/'derived/baseline/bdp_decision.json').write_text('{"full_model":true}\n')
    cache = root/'derived/baseline/bdp_paths'
    cache.mkdir()
    (cache/'2001-01-02.parquet').write_bytes(b'design-period cached path bytes')
    (cache/'2001-01-02.json').write_text('{"fit_time":"2001-01-02"}\n')
    calls = []

    def write_frame(path, frame, bound):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        path.with_suffix('.metadata.json').write_text(json.dumps({
            'through': str(bound), 'config_hash': sha256_file(config),
            'data_manifest_hash': sha256_file(root/'raw/manifest.json'), 'input_hashes': {
                str(root/'curated/trading_calendar.parquet'):
                sha256_file(root/'curated/trading_calendar.parquet'),
            }, 'artifact_hash': sha256_file(path),
        }))
        return path

    def features(scoped, through=None):
        destination = Path(scoped['data']['data_root'])/'derived/baseline/features.parquet'
        calls.append((destination, through, scoped['walkforward']['feature_start']))
        frame = pl.DataFrame([{'date': day, 'series': series, 'own_rv_5': float(i),
                               'bdp_ci': float(i)/10}
                              for i, day in enumerate(days) for series in ('RAW', 'RAW_c', 'RES')
                              if day <= through])
        return write_frame(destination, frame, through)

    def l2(scoped, through=None):
        destination = Path(scoped['data']['data_root'])/'derived/l2/features_monthly.parquet'
        return write_frame(destination, pl.DataFrame({'date': [days[1]], 'rank_ic': [.25]}),
                           through)

    def copula(scoped, through=None):
        destination = Path(scoped['data']['data_root'])/'derived/final/copula_monthly.parquet'
        return write_frame(destination, pl.DataFrame({'date': [days[1]], 'embedding': [[1.]]}),
                           through)

    def load_copula(scoped):
        path = Path(scoped['data']['data_root'])/'derived/final/copula_monthly.parquet'
        return pl.read_parquet(path), json.loads(path.with_suffix('.metadata.json').read_text())

    bound = date(2001, 1, 3)
    features(settings, bound)
    l2(settings, bound)
    targets = write_frame(root/'derived/baseline/targets.parquet',
                          pl.DataFrame({'origin': days[2:4], 'y1': [-5., -4.]}), bound)
    (root/'derived/baseline/build_metadata.json').write_bytes(
        targets.with_suffix('.metadata.json').read_bytes())
    calls.clear()
    monkeypatch.setattr(l3_inputs, 'build_features', features)
    monkeypatch.setattr(l3_inputs, 'build_l2', l2)
    monkeypatch.setattr(l3_inputs, 'build_copula', copula)
    monkeypatch.setattr(l3_inputs, 'load_copula', load_copula)
    return settings, root, days, calls


def test_warmup_uses_actual_pre2001_snapshot_without_modifying_baseline_inputs(
    inputs, monkeypatch,
):
    settings, root, days, calls = inputs
    baseline = [root/name for name in ('derived/baseline/features.parquet',
                                       'derived/baseline/targets.parquet',
                                       'derived/l2/features_monthly.parquet')]
    before = {path: sha256_file(path) for path in baseline}
    validations = []
    checked = l3_inputs._design_inputs

    def validate(root_path, scoped, through):
        validations.append(through)
        return checked(root_path, scoped, through)

    monkeypatch.setattr(l3_inputs, '_design_inputs', validate)
    original_settings = deepcopy(settings)
    paths = l3_inputs.prepare_inputs(settings, date(2001, 1, 3))
    assert settings == original_settings and validations == [date(2001, 1, 3)]
    assert set(paths) == {'features', 'targets', 'l2', 'copula', 'warmup'}
    assert before == {path: sha256_file(path) for path in baseline}
    rows = pl.read_parquet(paths['warmup'])
    assert rows['date'].to_list() == [date(2000, 12, 29)]*3
    assert rows['series'].to_list() == ['RAW', 'RAW_c', 'RES']
    assert rows['own_rv_5'].to_list() == [1., 1., 1.]
    assert len(calls) == 1 and calls[0][1:] == (days[1], '2000-12-01')
    assert calls[0][0].is_relative_to(root/'derived/final/warmup_inputs')
    metadata = json.loads(paths['warmup'].with_suffix('.metadata.json').read_text())
    assert metadata['initialization_only'] and metadata['rows'] == 3
    assert all(sha256_file(Path(name)) == digest
               for name, digest in metadata['input_hashes'].items())
    # Identical inputs reuse the snapshot; changed pinned input bytes invalidate it.
    l3_inputs.prepare_inputs(settings, date(2001, 1, 3))
    assert len(calls) == 1
    calendar = root/'curated/trading_calendar.parquet'
    pl.read_parquet(calendar).with_columns(pl.lit('new vintage').alias('audit')).write_parquet(
        calendar,
    )
    l3_inputs.prepare_inputs(settings, date(2001, 1, 3))
    assert len(calls) == 2


def test_frozen_rebuild_stays_in_stage_keeps_target_lag_and_refits_every_bdp_path(
    inputs, monkeypatch,
):
    settings, original, days, calls = inputs
    stage = original.parent/'stage'
    for relative in ('raw', 'curated', 'derived/portfolios'):
        link = stage/relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(original/relative, target_is_directory=True)
    frozen = {key: value for key, value in settings.items() if key != 'config_path'}
    write_toml(stage/'frozen.toml', {'settings': frozen})
    scoped = deepcopy(settings)
    scoped['data']['data_root'] = str(stage)
    runtime = stage/'settings.toml'
    write_toml(runtime, {key: value for key, value in scoped.items() if key != 'config_path'})
    scoped['config_path'] = str(runtime)
    scoped['_design_root'] = str(original)
    before = {path: sha256_file(path) for path in (original/'derived/baseline').rglob('*')
              if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail('A staged rebuild must not validate the design root as its own input')

    monkeypatch.setattr(l3_inputs, '_design_inputs', forbidden)
    paths = l3_inputs.prepare_inputs(scoped, days[3], rebuild=True)
    assert all(path.is_relative_to(stage) for path in paths.values())
    assert before == {path: sha256_file(path) for path in before}
    # The design-period decision is copied; every cutoff-specific BDP path is refitted.
    assert (stage/'derived/baseline/bdp_decision.json').read_bytes() == (
        original/'derived/baseline/bdp_decision.json').read_bytes()
    assert not (stage/'derived/baseline/bdp_paths').exists()
    target = pl.read_parquet(paths['targets']).filter(
        pl.col('series') == 'RAW', pl.col('origin') == days[3],
    ).row(0, named=True)
    assert target['target_start'] == days[5] and target['target_end'] == days[6]
    assert target['y1'] == pytest.approx(np.log(.06**2+.07**2))
    assert target['y2'] == pytest.approx(np.log(1.06*1.07))
    assert all(destination.is_relative_to(stage) for destination, _, _ in calls)


def test_rebuild_refuses_the_design_root_or_a_missing_one(inputs):
    settings, root, _, _ = inputs
    with pytest.raises(ValueError, match='isolated root and its design root'):
        l3_inputs.prepare_inputs(settings, date(2001, 1, 3), rebuild=True)
    settings['_design_root'] = str(root)
    with pytest.raises(ValueError, match='isolated root and its design root'):
        l3_inputs.prepare_inputs(settings, date(2001, 1, 3), rebuild=True)


def test_design_inputs_must_come_from_this_config_and_reach_the_origin(inputs):
    settings, root, _, _ = inputs
    later = deepcopy(settings)
    later['walkforward']['design_last_origin'] = '2001-01-05'
    with pytest.raises(ValueError, match='ends before the requested origin'):
        l3_inputs.prepare_inputs(later, date(2001, 1, 4))
    path = root/'derived/l2/features_monthly.metadata.json'
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, 'config_hash': 'another configuration'}))
    with pytest.raises(ValueError, match='Stale design input'):
        l3_inputs.prepare_inputs(settings, date(2001, 1, 3))
