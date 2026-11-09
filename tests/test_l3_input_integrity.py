"""Refuse changed prepared inputs before the L3 study learns from them."""

import json
from datetime import date

import pytest

from momenta.l3_study import _validated_inputs
from momenta.manifest import sha256_file


def test_input_integrity_and_bound_precede_consumption(tmp_path):
    """A one-byte change invalidates the original pin rather than replacing it."""
    root = tmp_path/'data'
    (root/'derived/final').mkdir(parents=True)
    (root/'raw').mkdir()
    config = tmp_path/'settings.toml'
    config.write_text('registered = true\n')
    manifest = root/'raw/manifest.json'
    manifest.write_text('{}\n')
    feature = root/'features.parquet'
    feature.write_bytes(b'original input bytes')
    record = {'through': '2015-11-30', 'config_hash': sha256_file(config),
              'data_manifest_hash': sha256_file(manifest),
              'input_hashes': {str(feature): sha256_file(feature)},
              'source_hashes': {}, 'paths': {'features': str(feature)}}
    (root/'derived/final/inputs.metadata.json').write_text(json.dumps(record))
    settings = {'data': {'data_root': str(root)}, 'config_path': str(config)}
    assert _validated_inputs(settings, date(2015, 11, 30)) == record
    with pytest.raises(ValueError, match='end before'):
        _validated_inputs(settings, date(2015, 12, 31))
    feature.write_bytes(b'changed input bytes')
    with pytest.raises(ValueError, match='changed after preparation'):
        _validated_inputs(settings, date(2015, 11, 30))
