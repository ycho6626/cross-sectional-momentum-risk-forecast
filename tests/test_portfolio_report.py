"""Hand calculations for portfolio matching, risk scaling, summaries and stale results."""


from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from momenta import portfolio_report
from momenta.manifest import sha256_file
from momenta.portfolio_report import (
    bsc_returns,
    evaluate_portfolios,
    load_summary,
    reconcile,
    reconciliation_tables,
    return_summary,
)


@pytest.fixture
def replication():
    return {
        "variance_days": 126,
        "monthly_days": 21,
        "months_per_year": 12,
        "vol_target_annual": 0.12,
        "max_multiple": 2.0,
        "first_month": "2000-01-01",
        "last_month": "2000-03-31",
        "long_history_start": "2000-01-01",
        "long_history_end": "2000-03-31",
        "unscaled_sharpe": 0.53,
        "scaled_sharpe": 0.97,
        "sharpe_tolerance": 0.08,
    }


def test_monthly_summary_annualization_and_zero_cost_sharpe():
    result = return_summary(np.array([0.01, 0.02, 0.03]), 12)
    assert result["annual_mean"] == pytest.approx(0.24)
    assert result["annual_volatility"] == pytest.approx(np.sqrt(12) * 0.01)
    assert result["sharpe"] == pytest.approx(np.sqrt(48))
    assert result["skew"] == pytest.approx(0)
    assert result["max_drawdown"] == 0


def test_drawdown_includes_initial_wealth_and_recovery():
    # Wealth is 1 -> .8 -> 1 -> .9: the initial loss is the largest drawdown.
    assert return_summary(np.array([-0.2, 0.25, -0.1]), 12)["max_drawdown"] == pytest.approx(-0.2)


def test_constant_or_empty_returns_have_no_defined_sharpe():
    assert return_summary(np.array([]), 12)["sharpe"] is None
    assert return_summary(np.zeros(3), 12)["sharpe"] is None
    with pytest.raises(ValueError, match="finite"):
        return_summary(np.array([0.01, np.nan]), 12)


def test_reconciliation_matches_calendar_months_and_only_common_finite_dates():
    built = pl.DataFrame(
        {
            "date": [date(2000, 1, 28), date(2000, 2, 28), date(2000, 3, 30), date(2000, 4, 30)],
            "wml": [0.01, 0.02, 0.03, 99.0],
        }
    )
    reference = pl.DataFrame(
        {
            "date": [date(2000, 1, 31), date(2000, 2, 29), date(2000, 3, 31), date(2000, 5, 31)],
            "ret": [0.02, 0.04, 0.06, -99.0],
        }
    )
    result = reconcile(built, reference, threshold=0.90, minimum_explained=0.80, monthly=True)
    assert result["observations"] == 3
    assert result["first_date"] == date(2000, 1, 31)
    assert result["last_date"] == date(2000, 3, 31)
    assert result["correlation"] == pytest.approx(1.0)
    assert result["pass"]
    daily = reconcile(built, reference, threshold=0.90, minimum_explained=0.80, monthly=False)
    assert daily["observations"] == 0
    assert not daily["pass"]


@pytest.mark.parametrize(
    "correlation,explanation,passed",
    [
        (0.85, "", False),
        (0.85, "Specific reviewed universe difference", True),
        (0.79, "Specific reviewed universe difference", False),
        (0.95, "", True),
    ],
)
def test_reconciliation_never_excuses_below_floor(correlation, explanation, passed):
    # Equal-length orthogonal centered vectors give the requested correlation exactly.
    x = np.array([-1.0, 0.0, 1.0])
    perpendicular = np.array([1.0, -2.0, 1.0]) / np.sqrt(3)
    y = correlation * x + np.sqrt(1 - correlation**2) * perpendicular
    dates = [date(2000, month, 1) for month in (1, 2, 3)]
    result = reconcile(
        pl.DataFrame({"date": dates, "wml": x}),
        pl.DataFrame({"date": dates, "ret": y}),
        threshold=0.90,
        minimum_explained=0.80,
        monthly=True,
        explanation=explanation,
    )
    assert result["correlation"] == pytest.approx(correlation)
    assert result["pass"] is passed


def test_monthly_matching_refuses_duplicate_months():
    frame = pl.DataFrame({"date": [date(2000, 1, 1), date(2000, 1, 2)], "wml": [0.1, 0.2]})
    with pytest.raises(ValueError, match="duplicate dates"):
        reconcile(
            frame, frame.rename({"wml": "ret"}), threshold=0.9, minimum_explained=0.8, monthly=True
        )


@pytest.mark.parametrize(
    "correlation,explanation,passed",
    [
        (0.82, "Approved construction explanation; docs/DECISIONS.md", True),
        (0.82, "", False),
        (0.79, "Approved construction explanation; docs/DECISIONS.md", False),
    ],
)
def test_res_gate_uses_holding_cutoff_and_preserves_all_months(
    correlation,
    explanation,
    passed,
):
    dates = [date(2000, 12, 31), date(2001, 1, 31), date(2001, 2, 28), date(2001, 3, 31)]
    # The three full-window observations have known correlation; startup is adverse.
    x = np.array([-1.0, 0.0, 1.0])
    y = correlation * x + np.sqrt(1 - correlation**2) * np.array([1.0, -2.0, 1.0]) / np.sqrt(3)
    built = pl.DataFrame({"date": dates, "wml": [10.0, *x]})
    benchmark = pl.DataFrame({"date": dates, "ret": [-10.0, *y]})
    monthly = pl.concat(
        [
            built.with_columns(
                pl.lit(series).alias("series"), pl.lit(construction).alias("construction")
            )
            for series, construction in (
                ("RAW", "tercile_capped_vw"),
                ("RES", "tercile_capped_vw"),
                ("RAW", "decile_vw"),
            )
        ]
    )
    references = {
        name: benchmark
        for name in (
            "jkp/ret_12_1_monthly",
            "jkp/resff3_12_1_monthly",
            "jkp/ret_12_1_daily",
        )
    }
    references["french/momentum_deciles_monthly"] = benchmark.rename({"ret": "d10"}).with_columns(
        pl.lit(0.0).alias("d1")
    )
    settings = {
        "portfolios": {"primary": "tercile_capped_vw"},
        "reconciliation": {
            "res_first_month": "2001-01-01",
            "raw_monthly": 0.90,
            "res_monthly": 0.85,
            "raw_daily": 0.85,
            "raw_decile_monthly": 0.90,
            "minimum_explained": 0.80,
            "explanations": {"res_monthly": explanation},
        },
    }
    original = monthly.clone()
    gates, descriptive = reconciliation_tables(
        {"academic_monthly": monthly, "academic_daily": monthly},
        references,
        settings,
        {},
    )
    assert len(gates) == 4
    res = next(row for row in gates if row["comparison"] == "res_monthly")
    assert res["first_date"] == date(2001, 1, 31)
    assert res["observations"] == 3
    assert res["correlation"] == pytest.approx(correlation)
    assert res["required_correlation"] == 0.85
    assert res["minimum_explained"] == 0.80
    assert not res["strict_pass"]
    assert res["pass"] is passed
    assert res["status"] == ("EXPLAINED" if passed else "FAIL")
    assert len(descriptive) == 1
    assert descriptive[0]["first_date"] == date(2000, 12, 31)
    assert descriptive[0]["observations"] == 4
    assert descriptive[0]["correlation"] < 0.80
    assert descriptive[0]["pass"] is None
    assert descriptive[0]["status"] == "DESCRIPTIVE"
    assert monthly.equals(original)
    assert next(row for row in gates if row["comparison"] == "raw_monthly")["observations"] == 4


def test_m2_hash_ignores_unrelated_modules_but_detects_economic_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio_report, "REPO", tmp_path)
    package = tmp_path / "src" / "momenta"
    package.mkdir(parents=True)
    for name in portfolio_report._M2_SOURCE_FILES:
        (package / name).write_text("original\n")
    original = portfolio_report.code_hash()
    for name in ("models.py", "features.py", "report.py", "cli.py"):
        (package / name).write_text("new milestone\n")
    assert portfolio_report.code_hash() == original
    (package / "accounting.py").write_text("changed economics\n")
    assert portfolio_report.code_hash() != original


def _bsc_daily():
    # 126 observations end January 31; 29 February observations must not affect February.
    first = date(2000, 1, 31) - timedelta(days=125)
    return pl.DataFrame(
        {
            "date": [first + timedelta(days=day) for day in range(155)],
            "wml": [0.01] * 126 + [0.1] * 29,
        }
    )


def test_bsc_prior_month_timing_full_window_and_monthly_returns(replication):
    monthly = pl.DataFrame(
        {
            "date": [date(2000, 1, 31), date(2000, 2, 29), date(2000, 3, 31)],
            "wml": [0.4, 0.1, -0.1],
        }
    )
    result, omitted = bsc_returns(_bsc_daily(), monthly, {"replication": replication})
    assert omitted == [
        {
            "date": date(2000, 1, 31),
            "daily_observations": 95,
            "reason": "insufficient_daily_history",
        }
    ]
    february, march = result.to_dicts()
    assert february["forecast_date"] == date(2000, 1, 31)
    assert february["monthly_variance"] == pytest.approx(0.0021)
    assert february["multiple"] == pytest.approx(np.sqrt(4 / 7))
    assert february["scaled_return"] == pytest.approx(0.1 * np.sqrt(4 / 7))
    assert february["unscaled_return"] == 0.1
    assert march["monthly_variance"] == pytest.approx(21 / 126 * (97 * 0.01**2 + 29 * 0.1**2))
    changed_future = _bsc_daily().with_columns(
        pl.when(pl.col("date") > date(2000, 1, 31)).then(0.9).otherwise(pl.col("wml")).alias("wml")
    )
    perturbed, _ = bsc_returns(changed_future, monthly, {"replication": replication})
    assert perturbed.row(0, named=True) == february


def test_bsc_zero_variance_caps_weight_and_never_subtracts_rf(replication):
    daily = _bsc_daily().with_columns(pl.lit(0.0).alias("wml"))
    monthly = pl.DataFrame({"date": [date(2000, 2, 29)], "wml": [0.1], "rf": [0.03]})
    result, omitted = bsc_returns(daily, monthly, {"replication": replication})
    assert not omitted
    assert result["multiple"][0] == 2.0
    assert result["scaled_return"][0] == 0.2


def test_bsc_missing_return_does_not_reach_further_back(replication):
    daily = _bsc_daily().with_columns(
        pl.when(pl.col("date") == date(2000, 1, 10))
        .then(None)
        .otherwise(pl.col("wml"))
        .alias("wml")
    )
    monthly = pl.DataFrame({"date": [date(2000, 2, 29)], "wml": [0.1]})
    result, omitted = bsc_returns(daily, monthly, {"replication": replication})
    assert result.is_empty()
    assert omitted[0]["reason"] == "missing_daily_return"


@pytest.fixture
def evaluated(tmp_path, monkeypatch, replication):
    monkeypatch.setattr(portfolio_report, "REPO", tmp_path / "repo")
    package = portfolio_report.REPO / "src" / "momenta"
    package.mkdir(parents=True)
    for name in portfolio_report._M2_SOURCE_FILES:
        (package / name).write_text("fixture\n")
    root = tmp_path / "data"
    output = root / "derived" / 'portfolios'
    output.mkdir(parents=True)
    config = tmp_path / "settings.toml"
    config.write_text("fixture = true\n")
    for name in ("raw/manifest.json", "curated/_metadata.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    settings = {
        "data": {"data_root": str(root)},
        "config_path": str(config),
        "replication": replication,
        "portfolios": {"primary": "tercile_capped_vw"},
        "report": {"worst_months": 2},
        "reconciliation": {
            "raw_monthly": 0.9,
            "res_monthly": 0.85,
            "raw_daily": 0.85,
            "raw_decile_monthly": 0.9,
            "minimum_explained": 0.8,
            "res_first_month": "1999-01-01",
        },
    }
    dates = [date(2000, 1, 31), date(2000, 2, 29), date(2000, 3, 31)]
    pairs = [(series, "tercile_capped_vw") for series in ("RAW", "RAW_c", "RES")]
    pairs += [(series, "decile_vw") for series in ("RAW", "RES")]
    academic, book, rebalances = [], [], []
    for series, construction in pairs:
        nav = 1.0
        for day, ret in zip(dates, (0.01, -0.02, 0.03), strict=True):
            base = {"date": day, "series": series, "construction": construction}
            academic.append({**base, "winner_return": ret, "loser_return": 0.0, "wml": ret})
            nav *= 1 + ret
            book.append(
                {**base, "daily_return": ret, "nav": nav, "cash": 1.0, "gross_exposure": 2.0}
            )
            rebalances.append(
                {**base, "formation": day, "turnover": 0.5, "nav": nav, "unavailable_targets": 0}
            )
    for frequency in ("daily", "monthly"):
        pl.DataFrame(academic).write_parquet(output / f"academic_{frequency}.parquet")
        frame = pl.DataFrame(book)
        if frequency == "monthly":
            frame = frame.rename({"daily_return": "monthly_return"})
        frame.write_parquet(output / f"book_{frequency}.parquet")
    pl.DataFrame(rebalances).write_parquet(output / "rebalance_summary.parquet")
    for name in portfolio_report._REFERENCE_FILES:
        path = root / "curated" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "french/momentum_deciles_daily":
            frame = _bsc_daily().rename({"wml": "d10"}).with_columns(pl.lit(0.0).alias("d1"))
        else:
            frame = pl.DataFrame({"date": dates, "ret": [0.01, -0.02, 0.03]})
            if name.startswith("french/"):
                frame = frame.rename({"ret": "d10"}).with_columns(pl.lit(0.0).alias("d1"))
            elif name == "jkp/resff3_12_1_monthly":
                # The long-history table must use a common sample, omitting RAW's January.
                frame = frame.tail(2)
        frame.write_parquet(path)
    metadata = {
        "config_hash": sha256_file(config),
        "git_hash": "fixture",
        "git_dirty": False,
        "data_manifest_hash": sha256_file(root / "raw" / "manifest.json"),
        "curated_metadata_hash": sha256_file(root / "curated" / "_metadata.json"),
        "data_vintage": {"french": "fixture"},
        "mode": "thin",
        "common_end": "2000-03-31",
    }
    summary = evaluate_portfolios(settings, output, metadata, {"missing_held_returns": 0})
    return settings, output, summary


@pytest.mark.parametrize("changed", ["config", "inputs", "artifacts", "code"])
def test_m2_report_refuses_stale_inputs_or_code(evaluated, changed):
    settings, output, summary = evaluated
    if changed == "config":
        target = Path(settings["config_path"])
    elif changed == "inputs":
        target = output / "book_daily.parquet"
    elif changed == "artifacts":
        target = Path(next(iter(summary["artifacts"])))
    else:
        target = portfolio_report.REPO / "src" / "momenta" / "signals.py"
    target.write_text("changed\n")
    with pytest.raises(ValueError, match='Stale portfolio results'):
        load_summary(settings)
