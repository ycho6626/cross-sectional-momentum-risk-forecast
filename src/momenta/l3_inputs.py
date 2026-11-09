"""Feature inputs for the L3 study: L1, targets, L2, copula clouds and HMM warmup (Methodology).

A design run reuses the baseline and L2 inputs built through the design
boundary. A holdout stage rebuilds every input inside its isolated root, with
bounds widened only by the guarded holdout runner, and copies the design-period
BDP recovery decision; every BDP path is then fitted afresh.
"""

from __future__ import annotations

import bisect
import json
import shutil
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from momenta.copula import build_copula, load_copula
from momenta.features import build_features
from momenta.freeze import write_toml
from momenta.l2 import build_l2
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO
from momenta.targets import build_targets


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False)+'\n')


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in sorted(set(paths))}


def _copy_bdp(source: Path, destination: Path) -> list[Path]:
    """Copy the design-period BDP recovery decision, retaining its recorded bytes."""
    decision = source/'bdp_decision.json'
    if not decision.is_file():
        raise ValueError('The design-period BDP recovery decision is required')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(decision, destination/decision.name)
    return [decision]


def _valid_artifact(path: Path, identity: dict[str, Any]) -> bool:
    """Reuse only an identical artifact and all of its recorded input bytes."""
    try:
        metadata = json.loads(path.with_suffix('.metadata.json').read_text())
        return (metadata['identity'] == identity
                and metadata['artifact_hash'] == sha256_file(path)
                and all(Path(name).is_file() and sha256_file(Path(name)) == digest
                        for name, digest in metadata['input_hashes'].items()))
    except (OSError, ValueError, KeyError):
        return False


def _warmup(settings: dict[str, Any], calendar: list[date], output: Path) -> Path:
    """Build the preceding month-end L1 snapshot solely to initialize HMM state.

    The feature builder runs in its own staged root. Its BDP column is the
    static placeholder; each learned fit replaces it causally later. No target,
    regression observation or baseline artifact is added or changed.
    """
    root = Path(settings['data']['data_root'])
    feature_start = date.fromisoformat(settings['walkforward']['feature_start'])
    previous = feature_start.replace(day=1)-timedelta(days=1)
    first = previous.replace(day=1)
    candidates = [day for day in calendar if first <= day <= previous]
    if not candidates:
        raise ValueError('The HMM warmup month has no pinned trading sessions')
    last = candidates[-1]
    destination = output/'warmup.parquet'
    identity = {
        'config_hash': sha256_file(Path(settings['config_path'])),
        'data_manifest_hash': sha256_file(root/'raw/manifest.json'),
        'first': str(first), 'last': str(last),
        'source_hashes': _hashes([REPO/f'src/momenta/{name}.py' for name in (
            'l3_inputs', 'features', 'portfolios', 'signals', 'manifest', 'settings', 'freeze',
        )]),
    }
    if _valid_artifact(destination, identity):
        return destination
    stage = output/'warmup_inputs'
    stage.mkdir(parents=True, exist_ok=True)
    for relative in ('raw', 'curated', 'derived/portfolios'):
        source, link = root/relative, stage/relative
        if not source.is_dir():
            raise ValueError(f'Missing warmup input directory: {relative}')
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError(f'Warmup input link changed: {relative}')
        elif link.exists():
            raise ValueError(f'Warmup input must be an isolated symbolic link: {relative}')
        else:
            link.symlink_to(source.resolve(), target_is_directory=True)
    scoped = deepcopy(settings)
    scoped = {key: value for key, value in scoped.items()
              if key != 'config_path' and not key.startswith('_')}
    scoped['data']['data_root'] = str(stage)
    scoped['walkforward']['feature_start'] = str(first)
    scoped['walkforward']['design_last_origin'] = str(last)
    config_path = stage/'settings.toml'
    write_toml(config_path, scoped)
    scoped['config_path'] = str(config_path)
    copied = _copy_bdp(root/'derived/baseline', stage/'derived/baseline')
    feature_path = build_features(scoped, through=last)
    rows = pl.read_parquet(feature_path).filter(
        pl.col('date') == last, pl.col('series').is_in(['RAW', 'RAW_c', 'RES']),
    ).sort('series')
    if rows.height != 3 or rows['series'].to_list() != ['RAW', 'RAW_c', 'RES']:
        raise ValueError('HMM initialization requires one actual warmup row for every series')
    rows.write_parquet(destination, compression=settings['storage']['compression'])
    source_metadata = json.loads(feature_path.with_suffix('.metadata.json').read_text())
    _write_json(destination.with_suffix('.metadata.json'), {
        **provenance(REPO, Path(settings['config_path']), Manifest(root)),
        'identity': identity, 'initialization_only': True, 'date': str(last), 'rows': rows.height,
        'bdp_method': 'static placeholder replaced by each cutoff-specific fit',
        'input_hashes': {
            **source_metadata['input_hashes'],
            **_hashes([*copied, config_path, feature_path,
                       feature_path.with_suffix('.metadata.json')]),
        },
        'artifact_hash': sha256_file(destination),
    })
    return destination


def _design_inputs(root: Path, settings: dict[str, Any], through: date) -> list[Path]:
    """Require baseline and L2 inputs built from this config and data vintage."""
    checked = []
    for path in (root/'derived/baseline/build_metadata.json',
                 root/'derived/baseline/features.metadata.json',
                 root/'derived/l2/features_monthly.metadata.json'):
        if not path.is_file():
            raise ValueError(f'Missing design input {path.name}; run momenta features')
        metadata = json.loads(path.read_text())
        if (metadata['config_hash'] != sha256_file(Path(settings['config_path']))
                or metadata['data_manifest_hash'] != sha256_file(root/'raw/manifest.json')):
            raise ValueError(f'Stale design input {path.name}; run momenta features')
        if date.fromisoformat(metadata['through']) < through:
            raise ValueError(f'Design input {path.name} ends before the requested origin')
        checked.append(path)
    return checked


def prepare_inputs(
    settings: dict[str, Any], through: date, rebuild: bool = False,
) -> dict[str, Path]:
    """Prepare guarded L3 paths without overwriting the baseline design outputs.

    A design run reuses the checked baseline and L2 builds. A holdout stage
    passes rebuild=True: its settings name the design root from which the BDP
    recovery decision is copied, and every feature is rebuilt inside the stage.
    The requested bound is checked before any input is read or built.
    """
    design_end = date.fromisoformat(settings['walkforward']['design_last_origin'])
    feature_start = date.fromisoformat(settings['walkforward']['feature_start'])
    if not feature_start <= through <= design_end:
        raise ValueError('L3 feature and target origins must stay within the scoped design bounds')
    root = Path(settings['data']['data_root'])
    output = root/'derived/final'
    paths = {'features': root/'derived/baseline/features.parquet',
             'targets': root/'derived/baseline/targets.parquet',
             'l2': root/'derived/l2/features_monthly.parquet',
             'copula': output/'copula_monthly.parquet'}
    copied = []
    if rebuild:
        source = Path(settings['_design_root']) if '_design_root' in settings else None
        if source is None or not source.is_dir() or source.resolve() == root.resolve():
            raise ValueError('A holdout rebuild requires an isolated root and its design root')
        copied = _copy_bdp(source/'derived/baseline', root/'derived/baseline')
    else:
        copied = _design_inputs(root, settings, through)
    calendar_path = root/'curated/trading_calendar.parquet'
    calendar = pl.read_parquet(calendar_path)['date'].sort().to_list()
    last_index = bisect.bisect_right(calendar, through)-1
    if last_index < 0 or calendar != sorted(set(calendar)):
        raise ValueError('L3 inputs require a unique calendar covering the requested dates')
    if rebuild:
        paths['features'] = build_features(settings, through=through)
        book_path = root/'derived/portfolios/book_daily.parquet'
        end_index = min(len(calendar)-1, last_index+settings['targets']['lag']
                        + settings['targets']['horizon'])
        books = pl.scan_parquet(book_path).filter(
            pl.col('construction') == settings['portfolios']['primary'],
            pl.col('date') <= calendar[end_index],
        ).collect()
        targets = build_targets(books, calendar, settings, last_origin=through)
        targets.write_parquet(paths['targets'], compression=settings['storage']['compression'])
        _write_json(paths['targets'].with_suffix('.metadata.json'), {
            **provenance(REPO, Path(settings['config_path']), Manifest(root)),
            'through': str(through), 'rows': targets.height,
            'input_hashes': _hashes([book_path, calendar_path]),
            'source_hash': sha256_file(REPO/'src/momenta/targets.py'),
            'artifact_hash': sha256_file(paths['targets']),
        })
        paths['l2'] = build_l2(settings, through=through)
    for name, column in (('features', 'date'), ('targets', 'origin'), ('l2', 'date')):
        maximum = pl.scan_parquet(paths[name]).select(pl.col(column).max()).collect().item()
        required = calendar[last_index]
        if name == 'l2':
            months = [day for i, day in enumerate(calendar[:-1]) if day <= through
                      and (day.year, day.month) != (calendar[i+1].year, calendar[i+1].month)]
            if not months:
                raise ValueError('L3 has no completed formation month')
            required = months[-1]
        if maximum is None or maximum < required:
            raise ValueError(f'L3 {name} inputs end before the requested cutoff')
    try:
        _, metadata = load_copula(settings)
        stale = date.fromisoformat(metadata['through']) < through
    except (OSError, ValueError, KeyError):
        stale = True
    if stale:
        paths['copula'] = build_copula(settings, through=through)
    output.mkdir(parents=True, exist_ok=True)
    paths['warmup'] = _warmup(settings, calendar, output)
    inputs = [calendar_path, *copied]
    for path in paths.values():
        inputs.extend([path, path.with_suffix('.metadata.json')])
    if rebuild:
        inputs.append(root/'frozen.toml')
    _write_json(output/'inputs.metadata.json', {
        **provenance(REPO, Path(settings['config_path']), Manifest(root)),
        'through': str(through), 'rebuild': rebuild,
        'paths': {key: str(path) for key, path in paths.items()},
        'input_hashes': _hashes(inputs),
        'source_hashes': _hashes([REPO/f'src/momenta/{name}.py' for name in (
            'l3_inputs', 'features', 'targets', 'l2', 'copula',
        )]),
    })
    return paths
