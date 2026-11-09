"""Sensitivity isolation must not silently change the registered design."""

from copy import deepcopy
from pathlib import Path

import pytest

from momenta.freeze import write_toml
from momenta.manifest import sha256_file
from momenta.robustness import variant_settings
from momenta.robustness_worker import verify_request


def test_every_registered_sensitivity_changes_one_economic_setting() -> None:
    original = {'portfolios': {'primary': 'tercile_capped_vw', 'cap_percentile': 0.8},
                'universe': {'n_universe': 1000, 'cap_repair': True},
                'delisting': {'performance_return': -0.3, 'unexplained_return': 0.0}}
    cases = [('decile', 'portfolios', 'primary', 'decile_vw'),
             ('universe_2000', 'universe', 'n_universe', 2000),
             ('performance_zero', 'delisting', 'performance_return', 0.0),
             ('performance_minus_one', 'delisting', 'performance_return', -1.0),
             ('unexplained_minus_30', 'delisting', 'unexplained_return', -0.3),
             ('no_cap_repairs', 'universe', 'cap_repair', False)]
    before = deepcopy(original)
    for label, section, name, expected in cases:
        result = variant_settings(original, label)
        differences = [(s,k,v) for s,table in result.items() for k,v in table.items()
                       if v != original[s][k]]
        assert differences == [(section,name,expected)]
    assert original == before
    with pytest.raises(ValueError, match='Unknown'):
        variant_settings(original, 'retuned')


def test_worker_rejects_an_extra_change_despite_allowed_root_and_bound_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import momenta.robustness_worker as worker

    repo, stage = tmp_path/'freeze', tmp_path/'stage'
    (repo/'config').mkdir(parents=True)
    stage.mkdir()
    original = {'data': {'data_root': '/frozen-data'},
                'walkforward': {'design_first_origin': '2005-12-31',
                                'design_last_origin': '2015-11-30', 'feature_clip_sd': 3.0},
                'universe': {'n_universe': 1000, 'cap_repair': True}}
    write_toml(repo/'config/frozen.toml', {'settings':original})
    request = {'single_change':['universe','n_universe',2000],
               'bounds': {'first_origin':'2015-12-31', 'last_origin':'2026-06-30'},
               'frozen_config_sha256':sha256_file(repo/'config/frozen.toml')}
    (stage/'request.json').write_text(json.dumps(request))
    routed = deepcopy(original)
    routed['data']['data_root'] = str(stage)
    routed['walkforward'].update(design_first_origin='2015-12-31',
                                 design_last_origin='2026-06-30')
    routed['universe']['n_universe'] = 2000
    write_toml(stage/'settings.toml', routed)
    monkeypatch.setattr(worker, 'REPO', repo)
    actual, _, frozen = verify_request(stage)
    assert actual['universe']['n_universe'] == 2000
    assert frozen['settings'] == original
    routed['walkforward']['feature_clip_sd'] = 4.0
    write_toml(stage/'settings.toml', routed)
    with pytest.raises(ValueError, match='more than one'):
        verify_request(stage)
