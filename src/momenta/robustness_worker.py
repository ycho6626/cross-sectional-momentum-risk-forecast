"""Worker for one pre-specified robustness variant, run in its own process (Methodology).

``python -m momenta.robustness_worker STAGE PHASE`` prepares the variant's
data, books and features in an isolated stage, then fits the H3/H2 models and
accounts with the frozen rules. It never writes through a linked original input.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import tomllib
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO, connect, load_settings, sql_path


def save_json(path: Path, value: Any) -> None:
    """Write deterministic audit values with source provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False)+'\n')


def link(source: Path, destination: Path) -> None:
    """Link only frozen immutable inputs into a fresh generated root."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def verify_request(stage: Path) -> tuple[dict, dict, dict]:
    """Allow one economic change and explicit root/origin routing; keep all else frozen."""
    request = json.loads((stage/'request.json').read_text())
    frozen = tomllib.loads((REPO/'config/frozen.toml').read_text())
    if sha256_file(REPO/'config/frozen.toml') != request['frozen_config_sha256']:
        raise ValueError('Frozen configuration changed')
    expected = frozen['settings']
    expected['data']['data_root'] = str(stage)
    expected['walkforward']['design_first_origin'] = request['bounds']['first_origin']
    expected['walkforward']['design_last_origin'] = request['bounds']['last_origin']
    section, name, value = request['single_change']
    expected[section][name] = value
    settings = load_settings(stage/'settings.toml')
    if {k:v for k,v in settings.items() if k != 'config_path'} != expected:
        raise ValueError('A robustness variant changed more than one frozen setting')
    # Read a fresh frozen document: do not let runtime routing relabel frozen choices.
    frozen = tomllib.loads((REPO/'config/frozen.toml').read_text())
    return settings, request, frozen


def prepare(stage: Path, settings: dict, request: dict) -> dict:
    """Rebuild the affected data and portfolio layer and every required feature."""
    from momenta.copula import build_copula
    from momenta.curate import _corrections, _delistings, _monthly, _view
    from momenta.features import build_features
    from momenta.l2 import build_l2
    from momenta.portfolios import form_portfolios, run_portfolios
    from momenta.signals import build_signals
    from momenta.targets import build_targets
    from momenta.universe import build_universe

    original = Path(request['original'])
    variant = request['variant']
    timings = {}
    start = time.monotonic()
    link(original/'raw', stage/'raw')
    output = stage/'curated'
    if variant == 'decile':
        # The portfolio layer already stores the true plain-VW decile construction.
        # Changing primary in form_portfolios would relabel its bin dispatch.
        link(original/'curated', output)
        link(original/'derived/portfolios', stage/'derived/portfolios')
        timings['reuse_registered_decile_book_seconds'] = time.monotonic()-start
    else:
        output.mkdir()
        changed = {'universe', 'universe_audit', 'observed_cap_exceptions',
                   'pit_sic', 'sic_audit', 'sic_issues', '_metadata'}
        events = request['single_change'][0] == 'delisting'
        if events:
            changed |= {'delistings', 'exceptions', 'monthly_returns'}
        for source in (original/'curated').iterdir():
            stem = source.name.split('.')[0]
            if stem not in changed:
                link(source, output/source.name)
        with connect(settings) as con:
            for name in ('master', 'cap_repairs'):
                _view(con, name, output/f'{name}.parquet')
            for name in ('prices', 'caps'):
                _view(con, name, output/name/'year=*/*.parquet')
            con.execute(f'''CREATE VIEW calendar AS SELECT date,
                row_number() OVER(ORDER BY date)-1 AS day_index
                FROM read_parquet({sql_path(output/'trading_calendar.parquet')})''')
            con.execute(f'''CREATE VIEW actions_flat AS
                SELECT a.permaticker,a.date,e.action,e.value,e.contraticker
                FROM read_parquet({sql_path(output/'actions.parquet')}) a,
                UNNEST(a.events) AS event(e)''')
            common_end = date.fromisoformat(request['bounds']['common_end'])
            if events:
                _corrections(con)
                _delistings(con, settings, output, None)
                years = sorted(int(path.name.split('=')[1])
                               for path in (output/'prices').glob('year=*'))
                _monthly(con, settings, output, years, common_end)
            else:
                _view(con, 'monthly_returns', output/'monthly_returns.parquet')
            universe_audit = build_universe(con, settings, output, through=common_end)
        metadata = {**json.loads((original/'curated/_metadata.json').read_text()),
                    **provenance(REPO, Path(settings['config_path']), Manifest(stage)),
                    'robustness': request['single_change'], 'label': 'pre-specified robustness'}
        save_json(output/'_metadata.json', metadata)
        timings['curation_seconds'] = time.monotonic()-start
        start = time.monotonic()
        portfolio_dir = stage/'derived/portfolios'
        portfolio_dir.mkdir(parents=True)
        universe = pl.read_parquet(output/'universe.parquet')
        monthly = pl.scan_parquet(output/'monthly_returns.parquet').filter(
            pl.col('permaticker').is_in(universe['permaticker'].unique().implode())
        ).select('permaticker','date','monthly_return').collect()
        factors = pl.read_parquet(output/'french/ff3_monthly.parquet')
        signals = build_signals(monthly, factors, universe, settings)
        signals.write_parquet(portfolio_dir/'signals.parquet')
        weights = form_portfolios(signals, settings)
        weights.write_parquet(portfolio_dir/'formation_weights.parquet')
        calendar = pl.read_parquet(output/'trading_calendar.parquet').filter(
            pl.col('date') > universe['date'].min(), pl.col('date') <= common_end
        )['date'].to_list()
        audit = run_portfolios(weights, calendar, output, portfolio_dir, settings)
        save_json(portfolio_dir/'_metadata.json', metadata)
        save_json(portfolio_dir/'robustness_audit.json',
                  {'accounting': audit, 'universe': universe_audit})
        timings['portfolios_seconds'] = time.monotonic()-start
    start = time.monotonic()
    baseline = stage/'derived/baseline'
    baseline.mkdir(parents=True)
    shutil.copy2(original/'derived/baseline/bdp_decision.json', baseline/'bdp_decision.json')
    last = date.fromisoformat(request['bounds']['last_origin'])
    build_features(settings, through=last)
    calendar = pl.read_parquet(output/'trading_calendar.parquet')['date'].sort().to_list()
    books = pl.read_parquet(stage/'derived/portfolios/book_daily.parquet').filter(
        pl.col('construction') == settings['portfolios']['primary'],
        pl.col('date') <= date.fromisoformat(request['bounds']['final_target_end']),
    )
    targets = build_targets(books, calendar, settings, last_origin=last)
    targets.write_parquet(baseline/'targets.parquet',
                          compression=settings['storage']['compression'])
    # L2 is always the shared RAW-tercile dictionary, even when the predicted
    # book is a decile strategy. The frozen builder uses this selector to find
    # its legs; changing it would confound the one-setting sensitivity.
    l2_settings = deepcopy(settings)
    l2_settings['portfolios']['primary'] = tomllib.loads(
        (REPO/'config/frozen.toml').read_text()
    )['settings']['portfolios']['primary']
    l2_path = build_l2(l2_settings, through=last)
    if variant == 'decile':
        receipt = json.loads((original/'derived/final/holdout/completed.json').read_text())
        expected_l2 = Path(receipt['stage'])/'derived/l2/features_monthly.parquet'
        if not pl.read_parquet(l2_path).equals(pl.read_parquet(expected_l2)):
            raise ValueError('Decile book sensitivity changed the shared L2 dictionary')
    build_copula(settings, through=last)
    timings['features_targets_seconds'] = time.monotonic()-start
    artifact_names = ['derived/portfolios/book_daily.parquet',
                      'derived/portfolios/formation_weights.parquet',
                      'derived/portfolios/signals.parquet', 'curated/universe.parquet',
                      'curated/delistings.parquet', 'derived/baseline/features.parquet',
                      'derived/baseline/targets.parquet', 'derived/l2/features_monthly.parquet',
                      'derived/final/copula_monthly.parquet']
    record = {'label': 'pre-specified robustness', 'variant': variant,
              'stage': str(stage), 'timings': timings,
              'artifact_hashes': {name:sha256_file(stage/name) for name in artifact_names},
              'l2_reference_construction': l2_settings['portfolios']['primary'],
              'decile_l2_matches_frozen': True if variant == 'decile' else None,
              'provenance': request['provenance'], 'single_change': request['single_change']}
    save_json(stage/'prepared.json', record)
    return record


def complete(stage: Path, settings: dict, request: dict, frozen: dict) -> dict:
    """Run the requested H3/H2 models and accounts with the frozen fitted-step rules."""
    import numpy as np

    from momenta.baseline_study import _benchmark_forecasts, _structural_transform
    from momenta.copula import CM_COLUMNS
    from momenta.experiments import trial
    from momenta.features import FEATURE_SETS
    from momenta.l2 import L2_COLUMNS
    from momenta.l3_strategies import model_overlays, model_specification
    from momenta.l3_study import blocks_for
    from momenta.l3_transforms import carry
    from momenta.strategies import _next_month, simulate_strategy
    from momenta.strategy_study import _monthly, _specification, _tag
    from momenta.study_statistics import (
        forecast_metrics,
        h2_contrast,
        holm_adjust,
        lo_sharpe,
        stationary_indices,
    )
    from momenta.walkforward import walk_forward

    prepared = json.loads((stage/'prepared.json').read_text())
    for name, digest in prepared['artifact_hashes'].items():
        if sha256_file(stage/name) != digest:
            raise ValueError(f'Prepared robustness input changed: {name}')
    output = stage/'results'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    features = pl.read_parquet(stage/'derived/baseline/features.parquet')
    features = carry(features, pl.read_parquet(stage/'derived/l2/features_monthly.parquet')
                     .select('date', *L2_COLUMNS))
    features = carry(features, pl.read_parquet(stage/'derived/final/copula_monthly.parquet')
                     .select('date', *CM_COLUMNS))
    targets = pl.read_parquet(stage/'derived/baseline/targets.parquet')
    calendar = pl.read_parquet(stage/'curated/trading_calendar.parquet')['date'].sort().to_list()
    last = date.fromisoformat(request['bounds']['last_origin'])
    first = date.fromisoformat(request['bounds']['first_origin'])
    names = ['RAW', 'RES'] if request['variant'] == 'decile' else ['RAW', 'RAW_c', 'RES']
    transform = _structural_transform(settings, True)
    forecasts, fits = [], []
    for series in names:
        models = [name for name in FEATURE_SETS if name != 'L1m']
        if series in ('RAW', 'RES'):
            models.append('L1+L2')
            if frozen['best_models'][series] not in models:
                models.append(frozen['best_models'][series])
        for model in models:
            print(f"Robustness {request['variant']}: {series}/y1/{model}", flush=True)
            blocks = blocks_for(model)
            columns = [c for block in blocks.values() for c in block]
            with trial(settings, experiment=f'robustness:{model}:y1', kind='forecast',
                       series=series, sample=f'{first}:{last}', specification={
                           'label': 'pre-specified robustness', 'variant': request['variant'],
                           'single_change': request['single_change'], 'model': model,
                           'target': 'y1', 'blocks': blocks, 'walkforward': settings['walkforward'],
                       }) as attempt:
                run = walk_forward(features, targets, calendar, settings, series=series,
                                   target='y1', feature_set=model, columns=columns, blocks=blocks,
                                   through=last, transform=transform)
                forecasts.append(run.forecasts)
                fits.append(run.fits)
                attempt.metrics.update(origins=run.forecasts.height, refits=run.fits.height,
                                       mse=float(((run.forecasts['actual']
                                                   -run.forecasts['prediction'])**2).mean()))
    forecasts = _benchmark_forecasts(pl.concat(forecasts, how='diagonal_relaxed'))
    forecasts.sort('series','target','feature_set','origin').write_parquet(output/'forecasts.parquet')
    pl.concat(fits, how='diagonal_relaxed').write_parquet(output/'fits.parquet')
    forecast_seconds = time.monotonic()-started
    cfg = settings['evaluation']
    pairs, h3 = {}, []
    for series in names:
        for model in forecasts.filter(pl.col('series') == series)['feature_set'].unique():
            pairs[series,model] = forecasts.filter(
                pl.col('series') == series, pl.col('feature_set') == model,
            ).sort('origin')
        if series == 'RAW_c':
            continue
        f, base = pairs[series,'L1+L2'], pairs[series,'L1']
        if not f['origin'].equals(base['origin']):
            raise ValueError('H3 origins differ')
        metrics = forecast_metrics(f.with_columns(base['prediction'].alias('base_prediction')),
                                   target='y1', reps=cfg['bootstrap_reps'],
                                   block_months=cfg['block_months'], seed=cfg['seed'],
                                   nw_lags=cfg['nw_lags_monthly'], confidence=cfg['confidence'])
        if f['prediction'].equals(base['prediction']):
            for statistic in ('cw', 'dm'):
                metrics[statistic].update(pvalue=1.0, status='identical_nonrejection')
        h3.append({'series':series, **metrics})
    for row, pvalue in zip(h3, holm_adjust([r['cw']['pvalue'] for r in h3]), strict=True):
        row['cw_holm_pvalue'] = pvalue
    h2 = None
    if 'RAW_c' in names:
        a, ar = pairs['RAW_c','L1'], pairs['RAW_c','L1_without_M']
        b, br = pairs['RES','L1'], pairs['RES','L1_without_M']
        if any(not f['origin'].equals(a['origin']) for f in (ar,b,br)):
            raise ValueError('H2 origins differ')
        idx = stationary_indices(a['origin'].to_list(), cfg['bootstrap_reps'],
                                 cfg['block_months'], cfg['seed'])
        h2 = h2_contrast(a['actual'], a['prediction'], ar['prediction'],
                         b['actual'], b['prediction'], br['prediction'], idx, cfg['confidence'])
    started = time.monotonic()
    primary = settings['portfolios']['primary']
    end = _next_month(last)
    books = pl.read_parquet(stage/'derived/portfolios/book_daily.parquet').filter(
        pl.col('construction') == primary, pl.col('date').is_between(first,end),
        pl.col('series').is_in(['RAW','RES']),
    )
    ledger_files = [path for year in range(first.year, end.year+1)
                    for path in (stage/'derived/portfolios/ledger'/f'year={year}')
                    .glob('*.parquet')]
    ledger = pl.scan_parquet(ledger_files).filter(
        pl.col('construction') == primary, pl.col('date') > first, pl.col('date') <= end,
        pl.col('series').is_in(['RAW','RES']),
    ).collect().sort('series','date','permaticker')
    strategies, monthly_rows = [], []
    for series in ('RAW','RES'):
        decisions = pairs[series,'L1+L2'].join(
            features.filter(pl.col('series') == series).select(
                pl.col('date').alias('origin'), 'own_rv_126'), on='origin', how='left',
            validate='1:1', maintain_order='left')
        for name in ('BSC', 'L1+L2'):
            overlays = model_overlays(decisions, calendar, settings, bsc=name == 'BSC')
            cost = settings['strategy']['cost_bp']
            spec = (_specification(settings, name, cost, True) if name == 'BSC' else
                    model_specification(settings, name, name, cost, full_bdp=True,
                                        best_model=frozen['best_models'][series]))
            spec.update(label='pre-specified robustness', single_change=request['single_change'],
                        delisting=settings['delisting'])
            with trial(settings, experiment=f'robustness:{name}', kind='strategy', series=series,
                       sample=f'{first}:{last}', specification=spec) as attempt:
                result = simulate_strategy(books.filter(pl.col('series') == series),
                                           ledger.filter(pl.col('series') == series), overlays,
                                           cost_bp=cost,
                                           initial_nav=settings['portfolios']['initial_nav'],
                                           last_date=end)
                daily = _tag(result.daily, name, attempt.strategy_id, cost)
                monthly = _monthly(daily).sort('date')
                monthly_rows.append(monthly)
                returns = monthly['monthly_return'].to_numpy()
                row = {'series':series, 'strategy':name, 'cost_bp':cost,
                       'lo_sharpe':lo_sharpe(returns), 'months':len(returns),
                       'strategy_id':attempt.strategy_id,
                       'annual_mean':float(np.mean(returns)*12),
                       'annual_turnover':float(result.rebalances['turnover'].sum()*12/len(returns))}
                strategies.append(row)
                attempt.metrics.update(row)
                daily.write_parquet(output/f'{series}_{name}_daily.parquet')
    pl.concat(monthly_rows).write_parquet(output/'strategy_monthly.parquet')
    strategy_seconds = time.monotonic()-started
    result = {**prepared, 'h3':h3, 'h2':h2, 'strategies':strategies,
              'best_models':frozen['best_models'],
              'model_scope':'y1 H3 and H2 prefixes plus each frozen best; BSC and L1+L2 accounts',
              'timings':{**prepared['timings'], 'forecasts_seconds':forecast_seconds,
                         'strategies_seconds':strategy_seconds},
              'result_hashes':{p.name:sha256_file(p) for p in sorted(output.glob('*.parquet'))}}
    save_json(stage/'result.json', result)
    return result


def main() -> None:
    """Dispatch isolated preparation or requested frozen forecast comparisons."""
    stage = Path(sys.argv[1]).resolve()
    settings, request, frozen = verify_request(stage)
    if not (stage/'prepared.json').exists():
        prepare(stage, settings, request)
    if sys.argv[2] == 'complete':
        complete(stage, settings, request, frozen)


if __name__ == '__main__':
    main()
