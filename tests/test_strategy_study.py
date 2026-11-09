"""Synthetic strategy-baseline_study timing, artifact and failed-attempt integration."""

import calendar
import json
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from momenta import experiments, strategy_study
from momenta.bdp import BDPFit
from momenta.manifest import sha256_file
from momenta.settings import load_settings


@pytest.mark.parametrize('variant', ['Static', 'BSC', 'BDP-transfer', 'L1', 'DM-transfer'])
@pytest.mark.parametrize('cost_bp', [0.0, 10.0])
def test_amended_forecast_rules_change_only_dependent_trading_identities(variant, cost_bp):
    """DSR counts amended L1/DM rules separately, retaining unchanged strategy IDs."""
    amended_settings = load_settings()
    registered_settings = deepcopy(amended_settings)
    new_fields = ('feature_clip_sd', 'blocks', 'penalty_selection')
    for field in new_fields:
        registered_settings['walkforward'].pop(field)
    amended = strategy_study._specification(amended_settings, variant, cost_bp, full_bdp=True)

    def identity(specification):
        return experiments.strategy_id(
            'strategy', {'series': 'RAW', 'specification': specification},
        )

    if variant in {'L1', 'DM-transfer'}:
        # The registered economic rule used this four-field shared-ridge model
        # specification. No archived data or current-config hash enters its ID.
        registered = deepcopy(amended)
        registered['walkforward'] = {
            key: registered_settings['walkforward'][key]
            for key in ('refit', 'validation_months', 'gap_days', 'ridge_lambdas')
        }
        assert set(amended['walkforward']) - set(registered['walkforward']) == set(new_fields)
        for field in new_fields:
            assert amended['walkforward'][field] == amended_settings['walkforward'][field]
            omitted = deepcopy(amended)
            omitted['walkforward'].pop(field)
            assert identity(omitted) != identity(amended)
        assert identity(registered) != identity(amended)
    else:
        # These rules do not consume L1 regression forecasts, so deleting every
        # amendment field must leave their economic specification and ID intact.
        registered = strategy_study._specification(
            registered_settings, variant, cost_bp, full_bdp=True,
        )
        assert registered == amended
        assert identity(registered) == identity(amended)


def _month_dates(year, month):
    first, last = date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    while first.weekday() > 4:
        first += timedelta(days=1)
    while last.weekday() > 4:
        last -= timedelta(days=1)
    if month == 1:
        first = date(year, 1, 3)
    return first, last


@pytest.fixture
def study_inputs(tmp_path, monkeypatch):
    settings = load_settings()
    settings["data"]["data_root"] = str(tmp_path / "data")
    settings["features"].update(bear_days=2, market_rv_days=2)
    settings["strategy"]["dm_first_month"] = "2003-11-01"
    monkeypatch.setattr(experiments, "REPO", tmp_path)
    metadata = {
        "git_hash": "fixture",
        "git_dirty": False,
        "config_hash": "config",
        "data_manifest_hash": "manifest",
        "data_vintage": {"fixture": "pinned"},
    }
    pinned_provenance = dict(metadata)
    monkeypatch.setattr(experiments, "provenance", lambda *args: pinned_provenance)
    root = tmp_path / "data"
    portfolio_dir, curated = root / "derived/portfolios", root / "curated/french"
    portfolio_dir.mkdir(parents=True)
    curated.mkdir(parents=True)
    months = [
        (year, month)
        for year in range(2003, 2007)
        for month in range(1, 13)
        if (2003, 11) <= (year, month) <= (2006, 2)
    ]
    primary = settings["portfolios"]["primary"]
    books, ledger, features, factors, french = [], [], [], [], []
    nav, previous_execution, previous_origin = 1.0, 1.0, date(2003, 10, 31)
    for index, (year, month) in enumerate(months):
        execution, origin = _month_dates(year, month)
        monthly_return = 0.1 if month % 2 else -0.1
        ledger.extend(
            [
                {
                    "date": execution,
                    "formation": previous_origin,
                    "series": "RAW",
                    "construction": primary,
                    "permaticker": p,
                    "nav": nav,
                    "pretrade": pre if index else 0.0,
                    "posttrade": post,
                }
                for p, pre, post in [(1, nav, nav), (2, -previous_execution, -nav)]
            ]
        )
        books.append(
            {
                "date": execution,
                "series": "RAW",
                "construction": primary,
                "daily_return": 0.0,
                "nav": nav,
                "cash": nav,
                "gross_exposure": 2.0,
                "terminal_cash": 0.0,
            }
        )
        previous_execution = nav
        nav *= 1 + monthly_return
        books.append(
            {
                "date": origin,
                "series": "RAW",
                "construction": primary,
                "daily_return": monthly_return,
                "nav": nav,
                "cash": previous_execution,
                "gross_exposure": (nav + previous_execution) / nav,
                "terminal_cash": 0.0,
            }
        )
        previous_origin = origin
        features.append(
            {
                "date": origin,
                "series": "RAW",
                "own_rv_126": np.log(0.0012),
                "bear_x_var": 0.0021 if month % 2 else 0.0,
            }
        )
        for day in (execution, origin):
            factors.append({"date": day, "mkt_rf": -0.01 if month % 2 else 0.01, "rf": 0.0})
        # This is the next-month response to the previous month's bear state.
        french.append(
            {
                "date": date(year, month, calendar.monthrange(year, month)[1]),
                "d10": 0.01 + 0.1 * (0.0021 if (month - 1) % 2 else 0.0),
                "d1": 0.0,
            }
        )
    pl.DataFrame(books).write_parquet(portfolio_dir / "book_daily.parquet")
    pl.DataFrame(books).select("date").unique().sort("date").write_parquet(
        root / "curated/trading_calendar.parquet"
    )
    ledger_frame = pl.DataFrame(ledger)
    for year in range(2003, 2007):
        selected = ledger_frame.filter(pl.col("date").dt.year() == year)
        if selected.height:
            destination = portfolio_dir / f"ledger/year={year}"
            destination.mkdir(parents=True)
            selected.write_parquet(destination / "part.parquet")
    pl.DataFrame(factors).write_parquet(curated / "ff3_daily.parquet")
    pl.DataFrame(french).write_parquet(curated / "momentum_deciles_monthly.parquet")
    origins = [date(2005, 12, 30), date(2006, 1, 31)]
    refits = [date(2005, 1, 3), date(2006, 1, 3)]
    forecasts = pl.DataFrame(
        {
            "series": ["RAW"] * 2,
            "target": ["y1"] * 2,
            "feature_set": ["L1"] * 2,
            "origin": origins,
            "refit": refits,
            "target_end": [date(2006, 1, 31), date(2006, 2, 28)],
            "prediction": [np.log(0.0012)] * 2,
            "residual_variance": [2 * np.log(4)] * 2,
        }
    )
    training = []
    for refit in refits:
        for row in features:
            origin = row["date"]
            year, month = (
                (origin.year + 1, 1) if origin.month == 12 else (origin.year, origin.month + 1)
            )
            end = date(year, month, calendar.monthrange(year, month)[1])
            if end < refit:
                training.append(
                    {
                        "series": "RAW",
                        "refit": refit,
                        "origin": origin,
                        "target_end": end,
                        "variance_prediction": 0.0012,
                        "bear_x_var": row["bear_x_var"],
                    }
                )
    return settings, pl.DataFrame(features), forecasts, pl.DataFrame(training), metadata


def test_registered_strategies_write_exact_paths_and_reuse_ids_across_thin_and_full(study_inputs):
    settings, features, forecasts, training, metadata = study_inputs
    output = Path(settings["data"]["data_root"]) / "derived/baseline"
    thin = strategy_study.run_strategies(
        settings,
        features,
        forecasts,
        training,
        last_origin=date(2005, 12, 31),
        metadata=metadata,
        output=output,
    )
    before = pl.read_parquet(output / "strategy_daily.parquet")
    ids = set(before["strategy_id"].to_list())
    assert len(ids) == 8 and not thin["bdp_included"]
    pinned = metadata["input_hashes"]
    data_root = Path(settings["data"]["data_root"])
    consumed = [
        data_root / "derived/portfolios/book_daily.parquet",
        data_root / "curated/trading_calendar.parquet",
        data_root / "curated/french/ff3_daily.parquet",
        data_root / "curated/french/momentum_deciles_monthly.parquet",
        *sorted((data_root / "derived/portfolios/ledger").glob("year=*/*.parquet")),
    ]
    assert set(pinned) == {str(path) for path in consumed}
    assert all(pinned[str(path)] == sha256_file(path) for path in consumed)
    assert thin["last_performance_date"] == "2006-01-31"
    overlays = pl.read_parquet(output / "overlays.parquet")
    l1 = overlays.filter(pl.col("strategy") == "L1")
    assert l1["multiple"].to_list() == pytest.approx([0.5, 0.5])
    assert l1["fit_date"].to_list() == [date(2005, 1, 3)] * 2
    full = strategy_study.run_strategies(
        settings,
        features,
        forecasts,
        training,
        last_origin=date(2006, 1, 31),
        metadata=metadata,
        output=output,
    )
    after = pl.read_parquet(output / "strategy_daily.parquet")
    assert set(after["strategy_id"].to_list()) == ids
    assert after.filter(pl.col("date") <= date(2006, 1, 31)).equals(before)
    monthly = pl.read_parquet(output / "strategy_monthly.parquet")
    static = monthly.filter(pl.col("strategy") == "Static", pl.col("cost_bp") == 0).sort("date")
    assert static["monthly_return"].to_list() == pytest.approx([0.1, -0.1])
    assert full["last_performance_date"] == "2006-02-28"
    assert all(
        date.fromisoformat(row["audit"]["training_target_end"]) < date.fromisoformat(row["refit"])
        for row in full["calibrations"]
    )
    attempts = [row for row in experiments.read_trials() if row["kind"] == "strategy"]
    assert len(attempts) == 16 and len({row["strategy_id"] for row in attempts}) == 8
    assert all(row["status"] == "completed" for row in attempts)
    assert all("monthly_sharpe" in json.loads(row["metrics_json"]) for row in attempts)
    for path in output.rglob("*.parquet"):
        assert path.with_suffix(".metadata.json").exists()


def test_failed_overlay_is_logged_before_it_can_produce_returns(study_inputs):
    settings, features, forecasts, training, metadata = study_inputs
    features = features.with_columns(pl.lit(float("nan")).alias("own_rv_126"))
    output = Path(settings["data"]["data_root"]) / "derived/baseline"
    with pytest.raises(ValueError, match="Variance forecasts"):
        strategy_study.run_strategies(
            settings,
            features,
            forecasts,
            training,
            last_origin=date(2005, 12, 31),
            metadata=metadata,
            output=output,
        )
    attempts = experiments.read_trials()
    assert [row["status"] for row in attempts] == ["completed", "completed", "failed"]
    assert attempts[-1]["experiment"] == "BSC" and attempts[-1]["kind"] == "strategy"


def test_full_bdp_branch_uses_cutoff_paths_and_persists_training_moment_audit(
    study_inputs, monkeypatch
):
    settings, features, forecasts, training, metadata = study_inputs
    output = Path(settings["data"]["data_root"]) / "derived/baseline"
    calls = []

    def paths(cutoff):
        calls.append(cutoff)
        states = features.select("date", "series").with_columns(
            pl.lit(0.1).alias("bdp_m"),
            pl.lit(0.0).alias("bdp_gamma"),
            pl.lit(0.0).alias("bdp_delta"),
            pl.lit(0.0).alias("bdp_ci"),
        )
        fit = BDPFit((0.01, 0.01, 0.01), 8.0, (0.0, 0.0, 0.0), True, 0.0, 500, ())
        return (
            states,
            fit,
            {"fit_time": str(cutoff), "french_last_observation": str(cutoff - timedelta(days=1))},
        )

    monkeypatch.setattr(
        strategy_study,
        "holding_moments",
        lambda state, fit, config: {
            "mean": 0.01,
            "variance": 0.0012,
            "nonpositive_wealth_paths": 1,
            "simulations": 4096,
            "seed": 17,
            "horizon": 21,
            "lag": 1,
        },
    )
    result = strategy_study.run_strategies(
        settings,
        features,
        forecasts,
        training,
        last_origin=date(2005, 12, 31),
        metadata=metadata,
        output=output,
        bdp_paths=paths,
    )
    assert calls == [date(2005, 1, 3)]
    assert result["bdp_included"] and "BDP-transfer" in result["strategies"]
    moments = pl.read_parquet(output / "bdp_strategy_moments.parquet")
    assert moments.filter(pl.col("sample") == "training")["target_end"].max() < calls[0]
    assert moments.filter(pl.col("sample") == "design")["origin"].to_list() == [date(2005, 12, 30)]
    assert moments["nonpositive_wealth_paths"].to_list() == [1] * moments.height
    fits = pl.read_parquet(output / "dm_fits.parquet")
    assert set(fits["strategy"].to_list()) == {"DM-transfer", "BDP-transfer"}
