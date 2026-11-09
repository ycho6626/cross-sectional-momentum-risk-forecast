"""Pre-specified sensitivities, isolated from the frozen headline (Methodology).

Six variants each change exactly one registered setting and rebuild the affected
layers in a fresh stage. The daily-origin and cost-grid sensitivities reuse the
completed frozen holdout without refitting. All run after unblinding and never
alter the frozen headline, chosen prefixes or inference settings.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from momenta.experiments import trial
from momenta.freeze import write_toml
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO

VARIANTS = {
    "decile": ("portfolios", "primary", "decile_vw"),
    "universe_2000": ("universe", "n_universe", 2000),
    "performance_zero": ("delisting", "performance_return", 0.0),
    "performance_minus_one": ("delisting", "performance_return", -1.0),
    "unexplained_minus_30": ("delisting", "unexplained_return", -0.30),
    "no_cap_repairs": ("universe", "cap_repair", False),
}


def variant_settings(settings: dict[str, Any], variant: str) -> dict[str, Any]:
    """Change exactly one registered setting; routing/bounds are applied separately."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown pre-specified robustness variant: {variant}")
    result = deepcopy(settings)
    section, name, value = VARIANTS[variant]
    result[section][name] = value
    return result


def _completed_holdout(root: Path) -> dict[str, Any]:
    path = root / "derived/final/holdout/completed.json"
    if not path.is_file():
        raise ValueError("Robustness requires the completed frozen holdout; run momenta holdout")
    return json.loads(path.read_text())


def run_variant(
    settings: dict[str, Any], variant: str, *, prepare_only: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Run one sensitivity in a fresh stage through a separate worker process."""
    from momenta.frozen_holdout import FROZEN_PATH, check_freeze

    check_freeze(settings)
    variant_settings(settings, variant)
    original = Path(settings["data"]["data_root"]).resolve()
    metadata = provenance(REPO, Path(settings["config_path"]), Manifest(original))
    specification = {"label": "pre-specified robustness", "variant": variant,
                     "single_change": VARIANTS[variant], "freeze": "frozen design",
                     "phase": "prepare" if prepare_only else "complete"}
    with trial(settings, experiment=f"robustness:{variant}", kind="diagnostic",
               series="RAW/RES", sample="holdout", specification=specification) as attempt:
        directory = original / "derived/robustness/variants"
        stage = resume or directory / variant / attempt.run_id
        if resume is None:
            stage.mkdir(parents=True, exist_ok=False)
            document = _completed_holdout(original)
            scoped = variant_settings(settings, variant)
            scoped.pop("config_path")
            scoped["data"]["data_root"] = str(stage)
            bounds = document["identity"]["bounds"]
            scoped["walkforward"]["design_first_origin"] = bounds["first_origin"]
            scoped["walkforward"]["design_last_origin"] = bounds["last_origin"]
            write_toml(stage / "settings.toml", scoped)
            (stage / "request.json").write_text(json.dumps({
                "variant": variant, "original": str(original),
                "single_change": list(VARIANTS[variant]), "provenance": metadata,
                "frozen_config_sha256": sha256_file(FROZEN_PATH),
                "original_settings_hash": sha256_file(Path(settings["config_path"])),
                "bounds": document["identity"]["bounds"],
            }, indent=2, sort_keys=True) + "\n")
        else:
            if not stage.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Resume must name an isolated robustness stage")
            request = json.loads((stage / "request.json").read_text())
            if request["variant"] != variant or (stage / "result.json").exists():
                raise ValueError("Resume variant differs or completed result is immutable")
        started = time.monotonic()
        log = stage / ("prepare.log" if prepare_only else "run.log")
        command = [sys.executable, "-m", "momenta.robustness_worker", str(stage),
                   "prepare" if prepare_only else "complete"]
        worker_hash = sha256_file(REPO / "src/momenta/robustness_worker.py")
        try:
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=REPO, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"Robustness {variant} failed; see {log}")
        finally:
            attempt.metrics.update(stage=str(stage), elapsed_seconds=time.monotonic()-started,
                                   phase=command[-1], log=str(log))
        result_path = stage / ("prepared.json" if prepare_only else "result.json")
        record = json.loads(result_path.read_text())
        elapsed = time.monotonic()-started
        preparation_elapsed = (
            json.loads((stage / "prepared.json").read_text()).get("elapsed_seconds", 0.0)
            if resume else 0.0
        )
        record.update(run_id=attempt.run_id, elapsed_seconds=elapsed,
                      preparation_elapsed_seconds=preparation_elapsed,
                      total_elapsed_seconds=preparation_elapsed+elapsed,
                      runtime_config_hash=sha256_file(stage / "settings.toml"),
                      worker_hash=worker_hash)
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"Robustness {variant}: {result_path}", flush=True)
        return record


def write_summary(settings: dict[str, Any]) -> dict[str, Any]:
    """Collect saved registered variant statistics without reselecting a result."""
    import polars as pl

    root = Path(settings['data']['data_root'])
    directory = root/'derived/robustness'
    rows, sources, records = [], {}, []
    for name in VARIANTS:
        paths = sorted((directory/'variants'/name).glob('*/result.json'),
                       key=lambda p: p.stat().st_mtime)
        if not paths:
            continue
        path = paths[-1]
        result = json.loads(path.read_text())
        sources[str(path)] = sha256_file(path)
        records.append(result)
        row = {'variant': name, 'label': 'pre-specified robustness',
               'single_change': json.dumps(result['single_change']),
               'elapsed_seconds': result['total_elapsed_seconds'],
               'h2_contrast': None, 'h2_lower': None, 'h2_upper': None}
        for metric in result['h3']:
            prefix = metric['series'].lower()
            row.update({f'{prefix}_r2': metric['r2_oos'],
                        f'{prefix}_cw_pvalue': metric['cw']['pvalue'],
                        f'{prefix}_cw_holm_pvalue': metric['cw_holm_pvalue'],
                        f'{prefix}_origins': metric['n']})
        if result['h2']:
            row.update(h2_contrast=result['h2']['contrast'],
                       h2_lower=result['h2']['interval']['lower'],
                       h2_upper=result['h2']['interval']['upper'])
        for account in result['strategies']:
            rule = 'bsc' if account['strategy'] == 'BSC' else 'l2'
            row[f"{account['series'].lower()}_{rule}_net_lo_sharpe"] = account['lo_sharpe']
        rows.append(row)
    provenance_record = provenance(REPO, Path(settings['config_path']), Manifest(root))
    summary = {'label': 'pre-specified robustness', 'completed_variants': len(records),
               'required_variants': list(VARIANTS), 'variants': records,
               'input_hashes': sources, 'provenance': provenance_record}
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    if rows:
        target = REPO/'results/tables/robustness_variants.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(target)
        target.with_suffix('.provenance.json').write_text(json.dumps({
            **provenance_record, 'label': 'pre-specified robustness', 'input_hashes': sources,
            'artifact_hash': sha256_file(target),
        }, indent=2, sort_keys=True)+'\n')
    return summary


def run_robustness(settings: dict[str, Any], variant: str = 'all') -> dict[str, Any]:
    """Serialize the pre-specified rebuilds, fixed-fit daily scores and cost replays."""
    from uuid import uuid4

    if variant not in {*VARIANTS, 'all', 'daily', 'costs'}:
        raise ValueError(f'Unknown robustness variant: {variant}')
    names = list(VARIANTS) if variant == 'all' else [variant] if variant in VARIANTS else []
    for name in names:
        run_variant(settings, name)
        write_summary(settings)
    root = Path(settings['data']['data_root']).resolve()
    stage = Path(_completed_holdout(root)['stage'])
    for name in ('daily', 'costs'):
        if variant not in ('all', name):
            continue
        output = root/'derived/robustness'/name/uuid4().hex
        started = time.monotonic()
        if name == 'daily':
            from momenta.robustness_daily import run_daily

            result = run_daily(settings, stage=stage, output=output)
        else:
            from momenta.robustness_costs import run_costs

            result = run_costs(settings, stage=stage, output=output)
        result['elapsed_seconds'] = time.monotonic()-started
        (output/'summary.json').write_text(
            json.dumps(result, indent=2, sort_keys=True)+'\n'
        )
        (output.parent/'latest.json').write_text(json.dumps({
            'output': str(output), 'summary_hash': sha256_file(output/'summary.json'),
            'result': result,
        }, indent=2, sort_keys=True)+'\n')
    return write_summary(settings)
