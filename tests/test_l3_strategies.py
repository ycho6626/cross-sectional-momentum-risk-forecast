"""L3 variance sizing, exact ledger costs, identities and failed-run logging (Methodology)."""

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from momenta import experiments, l3_strategies
from momenta.settings import load_settings


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    settings = load_settings()
    root = tmp_path / "data"
    settings["data"]["data_root"] = str(root)
    primary = settings["portfolios"]["primary"]
    portfolio_dir = root / "derived/portfolios"
    baseline_dir, curated = root / "derived/baseline", root / "curated"
    for path in (portfolio_dir / "ledger/year=2006", baseline_dir, curated):
        path.mkdir(parents=True)
    (portfolio_dir / "_metadata.json").write_text("{}")
    (baseline_dir / "bdp_decision.json").write_text('{"full_model":false}')
    dates = [date(2006, 1, 3), date(2006, 1, 31), date(2006, 2, 1), date(2006, 2, 28)]
    books = pl.DataFrame(
        {
            "date": dates,
            "series": ["RAW"] * 4,
            "construction": [primary] * 4,
            "daily_return": [0.0, 0.1, 0.0, -0.1],
            "nav": [1.0, 1.1, 1.1, 0.99],
            "cash": [1.0, 1.0, 1.1, 1.1],
            "gross_exposure": [2.0, 2.1 / 1.1, 2.0, 2.09 / 0.99],
            "terminal_cash": [0.0] * 4,
        }
    )
    books.write_parquet(portfolio_dir / "book_daily.parquet")
    calendar = [date(2005, 12, 30), *dates]
    pl.DataFrame({"date": calendar}).write_parquet(curated / "trading_calendar.parquet")
    pl.DataFrame(
        {
            "date": [dates[0]] * 2 + [dates[2]] * 2,
            "formation": [date(2005, 12, 30)] * 2 + [date(2006, 1, 31)] * 2,
            "series": ["RAW"] * 4,
            "construction": [primary] * 4,
            "permaticker": [1, 2, 1, 2],
            "pretrade": [0.0, 0.0, 1.1, -1.0],
            "posttrade": [1.0, -1.0, 1.1, -1.1],
        }
    ).write_parquet(portfolio_dir / "ledger/year=2006/part.parquet")
    forecasts = pl.DataFrame(
        {
            "series": ["RAW"] * 2,
            "target": ["y1"] * 2,
            "feature_set": ["L1+L2"] * 2,
            "origin": [date(2005, 12, 30), date(2006, 1, 31)],
            "refit": [date(2005, 1, 3), date(2006, 1, 3)],
            "prediction": [np.log(0.0012)] * 2,
            "residual_variance": [2 * np.log(4)] * 2,
        }
    )
    features = forecasts.select("series", pl.col("origin").alias("date")).with_columns(
        pl.lit(np.log(0.0012)).alias("own_rv_126"),
    )
    metadata = {
        "git_hash": "synthetic",
        "git_dirty": False,
        "config_hash": "config",
        "data_manifest_hash": "manifest",
        "data_vintage": {"fixture": "pinned"},
    }
    monkeypatch.setattr(experiments, "REPO", tmp_path)
    initial_provenance = deepcopy(metadata)
    monkeypatch.setattr(experiments, "provenance", lambda *args: deepcopy(initial_provenance))
    return settings, forecasts, features, metadata, calendar


def _run(accounts, output, forecasts=None):
    settings, original, features, metadata, _ = accounts
    return l3_strategies.run_model_strategies(
        settings,
        original if forecasts is None else forecasts,
        features=features,
        variants={"RAW": {"L1+L2": "L1+L2"}},
        best_models={"RAW": "L1+L2"},
        through=date(2006, 1, 31),
        output=output,
        metadata=metadata,
    )


def test_model_variance_sizing_and_net_ledger_paths_are_hand_calculated(accounts, tmp_path):
    output = tmp_path / "model_strategies"
    audit = _run(accounts, output)
    # exp(log(.0012) + log(4))=.0048; .12*sqrt(21/252/.0048)=0.5.
    overlays = pl.read_parquet(output / "overlays.parquet")
    assert overlays["multiple"].to_list() == pytest.approx([0.5] * 4)
    assert overlays["execution_date"].to_list() == [date(2006, 1, 3), date(2006, 2, 1)] * 2
    daily = pl.read_parquet(output / "strategy_daily.parquet")
    gross = daily.filter(pl.col("cost_bp") == 0)
    net = daily.filter(pl.col("cost_bp") == 10)
    assert gross["nav"].to_list() == pytest.approx([1.0, 1.05, 1.05, 0.9975])
    # Initial |+.5|+|-.5| costs .001. In February the offsetting trades total .05,
    # so the additional fee is .00005 and final long-side loss is .05245.
    assert net["nav"].to_list() == pytest.approx([0.999, 1.049, 1.04895, 0.9965])
    assert net["cost"].to_list() == pytest.approx([0.001, 0, 0.00005, 0])
    assert audit["last_performance_date"] == "2006-02-28"
    assert len(audit["break_even"]) == 1
    assert audit["break_even"][0]["grid_bp"] == [0, 5, 10, 20, 40, 80, 160, 200]
    assert abs(audit["break_even"][0]["annual_mean_differences"][0]) < 1e-12
    rows = experiments.read_trials()
    assert [row["kind"] for row in rows] == ["strategy", "strategy", "diagnostic"]
    assert all(row["status"] == "completed" for row in rows)
    for row in rows[:2]:
        metrics = json.loads(row["metrics_json"])
        assert metrics["first_month"] == "2006-01-31" and metrics["last_month"] == "2006-02-28"
        assert "monthly_sharpe" in metrics
    for path in output.rglob("*.parquet"):
        assert path.with_suffix(".metadata.json").exists()
        assert "run_id" not in pl.read_parquet_schema(path)
    pinned = accounts[3]["input_hashes"]
    assert any("ledger/year=2006" in path for path in pinned)
    assert any(path.endswith("bdp_decision.json") for path in pinned)


def test_repeated_model_accounts_are_identical_and_future_decisions_cannot_change_positions(
    accounts,
    tmp_path,
):
    first, second = tmp_path / "first", tmp_path / "second"
    left, right = _run(accounts, first), _run(accounts, second)
    left_hashes = {
        str(Path(path).relative_to(first)): value for path, value in left["artifact_hashes"].items()
    }
    right_hashes = {
        str(Path(path).relative_to(second)): value
        for path, value in right["artifact_hashes"].items()
    }
    assert left_hashes == right_hashes
    forecasts = accounts[1].with_columns(
        pl.when(pl.col("origin") > date(2005, 12, 30))
        .then(np.log(0.0192))
        .otherwise(pl.col("prediction"))
        .alias("prediction"),
    )
    changed = tmp_path / "changed"
    _run(accounts, changed, forecasts)
    before = pl.read_parquet(first / "strategy_daily.parquet").filter(
        pl.col("date") <= date(2006, 1, 31),
    )
    after = pl.read_parquet(changed / "strategy_daily.parquet").filter(
        pl.col("date") <= date(2006, 1, 31),
    )
    assert before.equals(after)
    assert (
        len({row["strategy_id"] for row in experiments.read_trials() if row["kind"] == "strategy"})
        == 2
    )


def test_only_hmm_strategy_identity_depends_on_the_selected_prefix(accounts):
    settings = accounts[0]
    for model in ("L1+L2", "L1+L2+CM+OT", "best+HMMF2"):
        short = l3_strategies.model_specification(
            settings,
            model,
            model,
            10,
            full_bdp=False,
            best_model="L1",
        )
        long = l3_strategies.model_specification(
            settings,
            model,
            model,
            10,
            full_bdp=False,
            best_model="L1+L2+CM+OT",
        )
        identity = lambda value: experiments.strategy_id("strategy", value)  # noqa: E731
        assert (identity(short) != identity(long)) == (model == "best+HMMF2")
        resampled = deepcopy(settings)
        resampled["walkforward"]["design_first_origin"] = "2012-12-31"
        resampled["l2"]["heatmap_dpi"] = 200
        same = l3_strategies.model_specification(
            resampled,
            model,
            model,
            10,
            full_bdp=False,
            best_model="L1",
        )
        assert identity(short) == identity(same)


def test_bad_model_variance_is_logged_and_future_refits_are_rejected(accounts, tmp_path):
    damaged = accounts[1].with_columns(pl.lit(-1.0).alias("residual_variance"))
    with pytest.raises(ValueError, match="residual variances"):
        _run(accounts, tmp_path / "failure", damaged)
    rows = experiments.read_trials()
    assert len(rows) == 1 and rows[0]["status"] == "failed" and rows[0]["kind"] == "strategy"
    decisions = accounts[1].with_columns(pl.lit(date(2007, 1, 3)).alias("refit"))
    with pytest.raises(ValueError, match="after its origin"):
        l3_strategies.model_overlays(decisions, accounts[4], accounts[0])


def test_ordinary_strategy_entry_keeps_the_holdout_locked(accounts, tmp_path):
    with pytest.raises(ValueError, match="Holdout origins are locked"):
        l3_strategies.run_model_strategies(
            accounts[0],
            accounts[1],
            features=accounts[2],
            variants={"RAW": {"L1+L2": "L1+L2"}},
            best_models={"RAW": "L1+L2"},
            through=date(2015, 12, 31),
            output=tmp_path / "blocked",
            metadata=accounts[3],
        )
    assert not (tmp_path / "blocked").exists()
