"""Separation evidence never survives changed numerical inputs or lost provenance."""

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from momenta import l3_diagnostics
from momenta.copula import CM_COLUMNS, COORDINATES
from momenta.manifest import sha256_file
from momenta.settings import REPO, load_settings


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    settings = load_settings()
    settings['data']['data_root'] = str(tmp_path)
    settings['ot'].update(directions=2, grid_points=5, separation_splits=2)
    config = tmp_path/'settings.toml'
    config.write_text('source = "small diagnostic fixture"\n')
    settings['config_path'] = str(config)
    (tmp_path/'raw').mkdir()
    manifest = tmp_path/'raw/manifest.json'
    manifest.write_text('{"schema_version":1,"entries":{}}\n')
    output = tmp_path/'derived/final'
    cloud_path = output/'copula_clouds/year=2001/part.parquet'
    cloud_path.parent.mkdir(parents=True)
    dates = [date(2001, 1, 31), date(2001, 2, 28), date(2001, 3, 30)]
    # Every month's eight points have balanced, finite marginals; the middle
    # month's reverse coordinate changes dependence without changing marginals.
    rows = []
    for month, day in enumerate(dates):
        for i in range(8):
            values = [(i+1)/9]*4
            if month == 1:
                values[1] = 1-values[1]
            rows.append({'date': day, 'permaticker': i,
                         **dict(zip(COORDINATES, values, strict=True))})
    pl.DataFrame(rows).write_parquet(cloud_path)
    directions = output/'copula_directions.parquet'
    pl.DataFrame({'direction': [0, 1], 'mom': [1., 0.], 'r1m': [0., 1.],
                  'beta': [0., 0.], 'sigma': [0., 0.]}).write_parquet(directions)
    monthly = output/'copula_monthly.parquet'
    pl.DataFrame({'date': dates, 'embedding': [list(np.linspace(0., 1., 10))]*3,
                  **{name: [0., 0., 0.] for name in CM_COLUMNS}}).write_parquet(monthly)
    metadata = {
        'artifact_hash': sha256_file(monthly), 'config_hash': sha256_file(config),
        'data_manifest_hash': sha256_file(manifest), 'dependency_hash': sha256_file(REPO/'uv.lock'),
        'directions_hash': sha256_file(directions),
        'cloud_hashes': {str(cloud_path): sha256_file(cloud_path)},
        'input_hashes': {}, 'source_hashes': {}, 'rows': 3,
    }
    monthly.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    attempts = []

    @contextmanager
    def log(*args, **kwargs):
        attempt = SimpleNamespace(run_id=f'fixture-{len(attempts)}', metrics={}, status='running')
        attempts.append(attempt)
        try:
            yield attempt
        except Exception:
            attempt.status = 'failed'
            raise
        else:
            attempt.status = 'completed'

    monkeypatch.setattr(l3_diagnostics, 'trial', log)
    return settings, output, cloud_path, directions, monthly, attempts


def test_cache_keeps_original_evidence_but_invalidates_settings_source_and_result_changes(
    pinned, monkeypatch,
):
    settings, output, _, _, _, attempts = pinned
    first = l3_diagnostics.run_separation(settings)
    destination = output/'diagnostics/separation.json'
    preserved = destination.read_bytes()
    second = l3_diagnostics.run_separation(settings)
    assert second == first and destination.read_bytes() == preserved
    assert [attempt.metrics['reused'] for attempt in attempts] == [False, True]
    assert first['metrics']['months'] == 3 and first['metrics']['half_size'] == 4
    assert first['metrics']['counts'] == {'within': 6, 'adjacent': 4, 'all_pairs': 6}
    assert first['metrics']['first_complete_cm'] == '2001-01-31'
    assert first['metrics']['first_complete_embedding'] == '2001-01-31'
    assert not first['metrics']['is_gate']
    settings['ot']['separation_ratio'] += 1
    l3_diagnostics.run_separation(settings)
    assert not attempts[-1].metrics['reused']
    original_hash = l3_diagnostics.sha256_file

    def changed_source(path):
        digest = original_hash(path)
        return 'changed implementation' if Path(path).name == 'l3_diagnostics.py' else digest

    monkeypatch.setattr(l3_diagnostics, 'sha256_file', changed_source)
    l3_diagnostics.run_separation(settings)
    assert not attempts[-1].metrics['reused']
    record = json.loads(destination.read_text())
    record['metrics']['months'] = 999
    destination.write_text(json.dumps(record))
    repaired = l3_diagnostics.run_separation(settings)
    assert repaired['metrics']['months'] == 3 and not attempts[-1].metrics['reused']


@pytest.mark.parametrize('kind', ['cloud', 'direction', 'config'])
def test_changed_pinned_input_fails_loudly_before_cached_result_is_used(pinned, kind):
    settings, output, cloud, directions, _, attempts = pinned
    l3_diagnostics.run_separation(settings)
    destination = output/'diagnostics/separation.json'
    previous = destination.read_bytes()
    changed = {'cloud': cloud, 'direction': directions,
               'config': Path(settings['config_path'])}[kind]
    changed.write_bytes(changed.read_bytes()+b'changed pinned input')
    with pytest.raises(ValueError, match='Stale copula input'):
        l3_diagnostics.run_separation(settings)
    assert attempts[-1].status == 'failed'
    assert destination.read_bytes() == previous


def test_synthetic_checks_are_saved_and_reused_only_for_an_identical_identity(
    tmp_path, monkeypatch,
):
    from momenta import copula, regimes

    settings = load_settings()
    settings['data']['data_root'] = str(tmp_path)
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/manifest.json').write_text('{"schema_version":1,"entries":{}}\n')
    calls, logged = [], []

    @contextmanager
    def trial(settings, **kwargs):
        logged.append(kwargs['experiment'])
        yield SimpleNamespace(run_id=f'run{len(logged)}', metrics={})

    def recovery(current):
        calls.append('hmm')
        return {'passed': True, 'filtered_accuracy': 0.9}, pl.DataFrame({'state': [0, 1]})

    def distances(current):
        calls.append('copula')
        return {'noise_threshold': 0.5, 'shift': {'detection_rate': 1.0}}

    monkeypatch.setattr(l3_diagnostics, 'trial', trial)
    monkeypatch.setattr(regimes, 'synthetic_recovery', recovery)
    monkeypatch.setattr(copula, 'synthetic_diagnostics', distances)
    first = {name: l3_diagnostics.run_synthetic(settings, name)
             for name in l3_diagnostics.SYNTHETIC}
    assert first['hmm_recovery']['metrics'] == {'passed': True, 'filtered_accuracy': 0.9}
    assert first['copula_synthetic']['metrics']['shift'] == {'detection_rate': 1.0}
    folder = tmp_path/'derived/final/diagnostics'
    assert pl.read_parquet(folder/'hmm_recovery.parquet')['state'].to_list() == [0, 1]
    again = l3_diagnostics.run_synthetic(settings, 'hmm_recovery')
    assert again == first['hmm_recovery'] and calls == ['hmm', 'copula']
    settings['hmm']['synthetic_n'] += 1
    l3_diagnostics.run_synthetic(settings, 'hmm_recovery')
    assert calls == ['hmm', 'copula', 'hmm']
    assert logged == ['l3-hmm-recovery', 'l3-copula-synthetic', 'l3-hmm-recovery',
                      'l3-hmm-recovery']
