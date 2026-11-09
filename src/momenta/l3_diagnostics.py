"""Logged L3 diagnostics: copula separation and the synthetic recovery checks (Methodology).

The separation ratio compares matched half-cloud sliced Wasserstein distances
and never gates H4. The synthetic HMM and copula checks run on generated data
with the registered seeds. Each result is reused only on an identical identity.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.copula import CM_COLUMNS, COORDINATES, load_copula, separation_diagnostic
from momenta.experiments import trial
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO


def run_separation(settings: dict[str, Any]) -> dict[str, Any]:
    """Record fixed half-split SW2 distances; a ratio below two never blocks H4.

    Reuse only byte-identical clouds, directions, config and implementation.
    Each invocation, including a reused or failed attempt, enters TRIALS. The
    cached artifact retains the original producing attempt's provenance.
    """
    root = Path(settings['data']['data_root'])
    directory = root/'derived/final'
    destination = directory/'diagnostics/separation.json'
    sidecar = destination.with_suffix('.metadata.json')
    specification = {
        'operation': 'matched half-cloud sliced Wasserstein separation',
        'settings': settings['ot'], 'is_gate': False,
        'matching': 'global minimum cloud size, disjoint halves, original full-cloud ranks',
    }
    with trial(settings, experiment='l3-copula-separation', kind='diagnostic', series='RAW',
               sample=f"formation clouds through {settings['walkforward']['design_last_origin']}",
               specification=specification) as attempt:
        monthly, copula = load_copula(settings)
        dates = monthly['date'].to_list()
        if (dates != sorted(set(dates)) or len(dates) < 2 or dates[-1]
                > date.fromisoformat(settings['walkforward']['design_last_origin'])):
            raise ValueError('Separation requires chronological design-period copula months')
        inputs = {
            **copula['cloud_hashes'],
            str(directory/'copula_directions.parquet'): copula['directions_hash'],
            str(directory/'copula_monthly.parquet'): copula['artifact_hash'],
            str(directory/'copula_monthly.metadata.json'):
                sha256_file(directory/'copula_monthly.metadata.json'),
            str(Path(settings['config_path'])): sha256_file(Path(settings['config_path'])),
            str(Manifest(root).path): sha256_file(Manifest(root).path),
            str(REPO/'uv.lock'): sha256_file(REPO/'uv.lock'),
        }
        source_hashes = {str(REPO/f'src/momenta/{name}.py'):
                         sha256_file(REPO/f'src/momenta/{name}.py')
                         for name in ('l3_diagnostics', 'copula', 'experiments', 'manifest')}
        identity = {'settings': settings['ot'], 'input_hashes': inputs,
                    'source_hashes': source_hashes,
                    'blas_threads': settings['walkforward']['blas_threads']}
        try:
            previous = json.loads(sidecar.read_text())
            cached = (previous['identity'] == identity
                      and previous['artifact_hash'] == sha256_file(destination))
        except (OSError, ValueError, KeyError):
            cached = False
        if cached:
            record = json.loads(destination.read_text())
            attempt.metrics.update(record['metrics'], reused=True)
            return record
        points = pl.concat([pl.read_parquet(name).select('date', 'permaticker', *COORDINATES)
                            for name in sorted(copula['cloud_hashes'])]).sort('date', 'permaticker')
        if (points['date'].unique().sort().to_list() != dates
                or points.select(pl.struct('date', 'permaticker').n_unique()).item()
                != points.height):
            raise ValueError('Cloud observations must match unique monthly copula keys')
        clouds = [frame.select(COORDINATES).to_numpy()
                  for frame in points.partition_by('date', maintain_order=True)]
        directions = pl.read_parquet(directory/'copula_directions.parquet').sort('direction')
        if (directions['direction'].to_list() != list(range(settings['ot']['directions']))
                or not all(np.isfinite(cloud).all() for cloud in clouds)):
            raise ValueError('Invalid fixed direction indices or nonfinite cloud coordinates')
        values = np.asarray(monthly['embedding'].to_list(), dtype=float)
        complete = np.isfinite(values).all(axis=1)
        if values.shape[1:] != (settings['ot']['directions']*settings['ot']['grid_points'],):
            raise ValueError('Stored embedding dimension differs from registered geometry')
        first_cm = monthly.drop_nulls(CM_COLUMNS)['date'].min()
        metrics = {
            **separation_diagnostic(clouds, directions.select(COORDINATES).to_numpy(), settings),
            'first_month': str(dates[0]), 'last_month': str(dates[-1]),
            'first_complete_cm': str(first_cm) if first_cm is not None else None,
            'first_complete_embedding': (
                str(np.asarray(dates)[complete][0]) if complete.any() else None
            ),
        }
        record = {'metrics': metrics, 'identity': identity, 'run_id': attempt.run_id,
                  'provenance': provenance(REPO, Path(settings['config_path']), Manifest(root))}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
        sidecar.write_text(json.dumps(
            {'identity': identity, 'artifact_hash': sha256_file(destination)},
            indent=2, sort_keys=True, allow_nan=False,
        )+'\n')
        attempt.metrics.update(metrics, reused=False)
        return record


SYNTHETIC = {
    "hmm_recovery": ("hmm", "regimes", "T=300,d=3,persistence=.95,separation=1.5"),
    "copula_synthetic": ("ot", "copula",
                         "registered n=1000, 100 matched null and 100 t3 simulations"),
}


def run_synthetic(settings: dict[str, Any], name: str) -> dict[str, Any]:
    """Run one seeded synthetic recovery check and save its metrics.

    hmm_recovery fits the registered HMM to a simulated persistent Markov chain;
    copula_synthetic measures sliced Wasserstein distances under matched nulls,
    a location shift and t3 tails. Neither reads market data.
    """
    from momenta.copula import synthetic_diagnostics
    from momenta.regimes import synthetic_recovery

    section, module, sample = SYNTHETIC[name]
    root = Path(settings['data']['data_root'])
    destination = root/'derived/final/diagnostics'/f'{name}.json'
    identity = {
        'settings': settings[section],
        'source_hashes': {str(REPO/f'src/momenta/{source}.py'):
                          sha256_file(REPO/f'src/momenta/{source}.py')
                          for source in (module, 'l3_diagnostics')},
        'dependency_hash': sha256_file(REPO/'uv.lock'),
    }
    with trial(settings, experiment=f'l3-{name.replace("_", "-")}', kind='diagnostic',
               series='synthetic', sample=sample,
               specification={'diagnostic': name, 'settings': settings[section]}) as attempt:
        try:
            record = json.loads(destination.read_text())
            if record['identity'] == identity:
                attempt.metrics.update(record['metrics'], reused=True)
                return record
        except (OSError, ValueError, KeyError):
            pass
        if name == 'hmm_recovery':
            metrics, frame = synthetic_recovery(settings)
        else:
            metrics, frame = synthetic_diagnostics(settings), None
        metrics = json.loads(json.dumps(metrics, default=str))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if frame is not None:
            frame.write_parquet(destination.with_suffix('.parquet'))
        record = {'metrics': metrics, 'identity': identity, 'run_id': attempt.run_id,
                  'provenance': provenance(REPO, Path(settings['config_path']), Manifest(root))}
        destination.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
        attempt.metrics.update(reused=False)
        return record


def run_diagnostics(settings: dict[str, Any]) -> dict[str, Any]:
    """Run the separation and both synthetic diagnostics the L3 evaluation reports."""
    return {'separation': run_separation(settings),
            **{name: run_synthetic(settings, name) for name in SYNTHETIC}}
