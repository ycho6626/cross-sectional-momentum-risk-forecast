"""Saved-comparison semantics and causal plotting, with independent hand expectations."""

import json
from copy import deepcopy
from datetime import date

import polars as pl
import pytest

from momenta.final_report import (
    HEADLINE_STRATEGIES,
    cost_grid_rows,
    cumulative_loss_difference,
    headline_forecasts,
    headline_strategies,
    robustness_rows,
    serving_timeline,
)


def _metric(model, benchmark, mse, baseline):
    return {
        "model": model,
        "benchmark": benchmark,
        "mse": mse,
        "baseline_mse": baseline,
        "mse_improvement": baseline - mse,
        "mse_improvement_interval_lower": -7.0,
        "mse_improvement_interval_upper": 11.0,
        "r2_oos": 1 - mse / baseline,
        "cw_pvalue": 0.04,
        "cw_holm_pvalue": 0.08,
        "dm_pvalue": 0.7,
        "n": 127,
        "first_origin": "2015-12-31",
        "last_origin": "2026-06-30",
    }


def test_headlines_retain_registered_benchmarks_and_saved_uncertainty():
    tables = {name: [] for name in ("baseline_forecasts", "prefixes", "h3", "h4", "h5a")}
    for series in ("RAW", "RES"):
        for model, mse in (("L1-own", 10.0), ("L1", 8.0)):
            tables["baseline_forecasts"].append(
                {
                    "series": series,
                    "target": "y1",
                    "feature_set": model,
                    **_metric(model, "L1-own", mse, 10.0),
                }
            )
        tables["h3"].append({"series": series, **_metric("L1+L2", "L1", 7.0, 8.0)})
        tables["prefixes"].append({"series": series, "model": "L1+L2+CM", "mse": 6.0})
        tables["h4"].append({"series": series, **_metric("L1+L2+CM+OT", "L1+L2+CM", 5.0, 6.0)})
        best = "L1+L2+CM" if series == "RAW" else "L1"
        tables["h5a"].append(
            {
                "series": series,
                "states": 2,
                **_metric("best+HMMF2", best, 4.0, 6.0 if series == "RAW" else 8.0),
            }
        )
    chosen = {"RAW": "L1+L2+CM", "RES": "L1"}
    rows = headline_forecasts(tables, chosen)
    raw = {r["model"]: r for r in rows if r["series"] == "RAW"}
    # OT contributes 6−5=1; filtered HMM contributes 6−4=2, never 5−4=1.
    assert raw["L1+L2+CM+OT"]["mse_improvement"] == 1
    assert raw["best+HMMF2"]["mse_improvement"] == 2
    assert raw["best+HMMF2"]["benchmark"] == "L1+L2+CM"
    assert raw["L1+L2"]["r2_oos"] == 0.125
    assert raw["L1+L2"]["mse_improvement_interval_lower"] == -7
    assert raw["L1+L2"]["cw_holm_pvalue"] == 0.08
    assert raw["L1-own"]["cw_holm_pvalue"] is None
    assert raw["L1-own"]["mse_improvement"] is None
    assert raw["L1+L2+CM"]["mse_improvement"] == 1
    assert raw["L1+L2+CM"]["r2_oos"] == pytest.approx(1 / 7)
    assert raw["L1+L2+CM"]["mse_improvement_interval_lower"] is None
    assert raw["L1+L2+CM"]["cw_holm_pvalue"] is None
    assert len(rows) == 12
    altered = deepcopy(tables)
    altered["h5a"][0]["benchmark"] = "L1+L2+CM+OT"
    with pytest.raises(ValueError, match="registered headline benchmark"):
        headline_forecasts(altered, chosen)


def test_robustness_keeps_raw_adjusted_p_and_primary_h2_without_daily_trading():
    h3 = [
        {
            "series": series,
            "n": 127,
            "r2_oos": gain,
            "cw": {"pvalue": unadjusted},
            "cw_holm_pvalue": 0.050728,
            "mse_improvement_interval": {"lower": -0.1, "upper": 0.2},
        }
        for series, gain, unadjusted in (("RAW", 0.04, 0.025364), ("RES", 0.03, 0.036754))
    ]
    case = {
        "variant": "universe_2000",
        "single_change": ["universe", "n_universe", 2000],
        "total_elapsed_seconds": 500,
        "h3": h3,
        "h2": {"contrast": -0.05, "interval": {"lower": -0.15, "upper": 0.01}},
        "strategies": [
            {"series": series, "strategy": strategy, "lo_sharpe": value}
            for series, strategy, value in (
                ("RAW", "BSC", 0.3),
                ("RAW", "L1+L2", 0.1),
                ("RES", "BSC", -0.03),
                ("RES", "L1+L2", -0.15),
            )
        ],
    }
    variants = {"variants": [case], "completed_variants": 1, "required_variants": ["a", "b"]}
    daily = {
        "h3": h3,
        "h2": [
            {"primary": False, "contrast": 99, "interval": {"lower": 98, "upper": 100}},
            {"primary": True, "contrast": -0.02, "interval": {"lower": -0.1, "upper": 0.1}},
        ],
        "first_origin": "2015-12-31",
        "last_origin": "2026-06-30",
        "forecast_rows": 5000,
        "models": 54,
        "daily_strategy": "N/A: monthly trading unchanged",
    }
    rows = robustness_rows(variants, daily)
    assert len(rows) == 2
    assert rows[0]["raw_cw_pvalue"] == 0.025364
    assert rows[0]["raw_cw_holm_pvalue"] == 0.050728
    assert rows[0]["res_cw_pvalue"] == 0.036754
    assert rows[0]["res_l2_net_lo_sharpe"] == -0.15
    assert rows[1]["h2_contrast"] == -0.02
    assert rows[1]["raw_bsc_net_lo_sharpe"] is None
    assert rows[1]["res_l2_net_lo_sharpe"] is None
    assert "monthly trading unchanged" in rows[1]["trading_status"]
    assert rows[0]["elapsed_seconds"] == 500
    assert json.loads(rows[0]["single_change"]) == ["universe", "n_universe", 2000]


def _cost_fixture():
    # Unordered input and negative/undefined values must retain their saved meanings.
    metrics = [
        {"series": s, "strategy": rule, "cost_bp": cost, "lo_sharpe": value}
        for s, rule, values in (
            ("RAW", "BSC", (0.3, 0.1, -0.2)),
            ("RES", "L1+L2", (0.05, -0.1, None)),
        )
        for cost, value in zip((5, 20, 40), values, strict=True)
    ]
    return {
        "cost_grid_bp": [40, 5, 20],
        "frozen_rules": 2,
        "accounts": 6,
        "metrics": list(reversed(metrics)),
    }


def test_cost_table_preserves_complete_saved_grid_and_undefined_sharpe():
    costs = _cost_fixture()
    original = deepcopy(costs)
    rows = cost_grid_rows(costs, expected_rules={("RAW", "BSC"), ("RES", "L1+L2")})
    assert rows == [
        {
            "series": "RAW",
            "strategy": "BSC",
            "lo_sharpe_5bp": 0.3,
            "lo_sharpe_20bp": 0.1,
            "lo_sharpe_40bp": -0.2,
        },
        {
            "series": "RES",
            "strategy": "L1+L2",
            "lo_sharpe_5bp": 0.05,
            "lo_sharpe_20bp": -0.1,
            "lo_sharpe_40bp": None,
        },
    ]
    assert costs == original


@pytest.mark.parametrize("damage", ["missing_cell", "duplicate", "extra_rate", "wrong_rule"])
def test_cost_table_rejects_incomplete_or_changed_frozen_grid(damage):
    costs = _cost_fixture()
    if damage == "missing_cell":
        costs["metrics"].pop()
    elif damage == "duplicate":
        costs["metrics"].append(costs["metrics"][0])
    elif damage == "extra_rate":
        costs["metrics"][0]["cost_bp"] = 80
    else:
        for row in costs["metrics"]:
            if row["series"] == "RES":
                row["strategy"] = "L1"
    with pytest.raises(ValueError, match="cost-grid|Cost-grid"):
        cost_grid_rows(costs, expected_rules={("RAW", "BSC"), ("RES", "L1+L2")})


def _loss_frame():
    rows = []
    for series, baseline, candidate in (
        ("RAW", [1.0, 2.0], [0.0, 3.0]),
        ("RES", [2.0, 0.0], [1.0, 1.0]),
    ):
        for model, values in (("L1", baseline), ("L1+L2", candidate)):
            for day, prediction in zip((date(2020, 1, 31), date(2020, 2, 28)), values, strict=True):
                rows.append(
                    {
                        "series": series,
                        "target": "y1",
                        "feature_set": model,
                        "origin": day,
                        "refit": date(2020, 1, 2),
                        "actual": 0.0,
                        "prediction": prediction,
                    }
                )
    return pl.DataFrame(rows)


def test_cumulative_loss_uses_paired_squared_losses_with_positive_gain():
    result = cumulative_loss_difference(_loss_frame(), ("L1+L2",))
    assert result.filter(pl.col("series") == "RAW")["cumulative_loss_improvement"].to_list() == [
        1.0,
        -4.0,
    ]
    assert result.filter(pl.col("series") == "RES")["cumulative_loss_improvement"].to_list() == [
        3.0,
        2.0,
    ]


@pytest.mark.parametrize("damage", ["missing", "actual", "future"])
def test_loss_plot_refuses_silent_sample_or_timing_changes(damage):
    frame = _loss_frame()
    select = (pl.col("series") == "RAW") & (pl.col("feature_set") == "L1+L2")
    if damage == "missing":
        frame = frame.filter(~(select & (pl.col("origin") == date(2020, 1, 31))))
    elif damage == "actual":
        frame = frame.with_columns(
            pl.when(select).then(99.0).otherwise(pl.col("actual")).alias("actual")
        )
    else:
        frame = frame.with_columns(pl.lit(date(2021, 1, 4)).alias("refit"))
    with pytest.raises(ValueError, match="misalign|future"):
        cumulative_loss_difference(frame, ("L1+L2",))


def test_timeline_uses_serving_fit_and_aligned_centroids_not_future_records():
    origins = (date(2006, 6, 30), date(2007, 6, 29))
    cutoffs = (date(2006, 1, 3), date(2007, 1, 3))
    forecast = pl.DataFrame(
        [
            {
                "series": "RAW",
                "target": "y1",
                "feature_set": "best+HMMF2",
                "origin": origin,
                "refit": cutoff,
            }
            for origin, cutoff in zip(origins, cutoffs, strict=True)
        ]
    )
    fits, regimes, centers = [], [], []
    for index, (origin, cutoff) in enumerate(zip(origins, cutoffs, strict=True)):
        for model, kind in (("best+HMMF2", "hmm"), ("L1+L2+CM+OT", "ot")):
            fits.append(
                {
                    "series": "RAW",
                    "target": "y1",
                    "feature_set": model,
                    "refit": cutoff,
                    "parameters_json": json.dumps(
                        {"structural": {kind: {"fit_hash": f"{kind}{index}", "lineage": "full"}}}
                    ),
                }
            )
        regimes.append(
            {
                "series": "RAW",
                "lineage": "full",
                "states": 2,
                "fit_time": str(cutoff),
                "fit_hash": f"hmm{index}",
                "monthly_probabilities": [
                    {
                        "series": "RAW",
                        "date": str(origin),
                        "hmmf_0": 0.25 + index * 0.5,
                        "hmmf_1": 0.75 - index * 0.5,
                    }
                ],
            }
        )
        centers.append(
            {
                "lineage": "full",
                "fit_cutoff": str(cutoff),
                "fit_hash": f"ot{index}",
                "centroids": [[0.0, 0.0], [10.0, 0.0]]
                if index == 0
                else [[20.0, 0.0], [10.0, 0.0]],
            }
        )
    monthly = pl.DataFrame({"date": origins, "embedding": [[0.0, 0.0], [10.0, 0.0]]})
    audit = {"regime_records": regimes, "centroid_records": centers}
    result = serving_timeline(forecast, pl.DataFrame(fits), audit, monthly)
    assert result["filtered_state_0"].to_list() == [0.25, 0.75]
    assert result["nearest_centroid"].to_list() == [1, 2]
    future = deepcopy(regimes[0])
    future.update(fit_time="2008-01-02", fit_hash="future")
    future["monthly_probabilities"][0].update(hmmf_0=0.99, hmmf_1=0.01)
    audit["regime_records"].append(future)
    assert serving_timeline(forecast, pl.DataFrame(fits), audit, monthly).equals(result)
    audit["regime_records"][0]["fit_hash"] = "unserved"
    with pytest.raises(ValueError, match="scored model fit"):
        serving_timeline(forecast, pl.DataFrame(fits), audit, monthly)


def test_break_even_absence_retains_status_and_gross_net_identity():
    accounts, crossings = [], []
    for series in ("RAW", "RES"):
        for strategy in HEADLINE_STRATEGIES:
            accounts.extend(
                {"series": series, "strategy": strategy, "cost_bp": cost, "lo_sharpe": value}
                for cost, value in ((0, 0.8), (10, 0.3))
            )
            crossings.append(
                {
                    "series": series,
                    "strategy": strategy,
                    "break_even_bp": None,
                    "status": "no_crossing_in_cost_grid",
                }
            )
    rows = headline_strategies({"strategies": accounts, "break_even": crossings}, 10)
    assert len(rows) == 16
    assert all(row["gross_lo_sharpe"] == 0.8 and row["lo_sharpe"] == 0.3 for row in rows)
    assert all(
        row["break_even_bp"] is None and row["break_even_status"] == "no_crossing_in_cost_grid"
        for row in rows
    )


def test_publication_drops_ids_and_review_notes_and_strips_image_metadata(tmp_path, monkeypatch):
    import struct
    import zlib

    from momenta import final_report

    monkeypatch.setattr(final_report, "REPO", tmp_path)
    tables, figures = tmp_path / "results/tables", tmp_path / "results/figures"
    tables.mkdir(parents=True)
    figures.mkdir()
    (tables / "m2_reconciliation.csv").write_bytes(
        b"comparison,strategy_id,correlation,explanation\r\nraw,s:1,0.975,see the notes\r\n")
    (tables / "holdout_h3.csv").write_bytes(b"series,mse,strategy_id\r\nRAW,0.41,s:2\r\n")

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    image = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", b"\x00" * 13)
             + chunk(b"tEXt", b"Description\x00{\"run\": 1}") + chunk(b"IDAT", b"pixels")
             + chunk(b"IEND", b""))
    (figures / "final_5_net_equity.png").write_bytes(image)
    result = final_report.publish_results()
    assert (tmp_path / "results/reconciliation.csv").read_bytes() == (
        b"comparison,correlation\nraw,0.975\n")
    assert (tmp_path / "results/primary_hypothesis.csv").read_bytes() == b"series,mse\nRAW,0.41\n"
    published = (tmp_path / "figures/net_equity.png").read_bytes()
    assert b"tEXt" not in published and b"pixels" in published
    assert published == image.replace(chunk(b"tEXt", b"Description\x00{\"run\": 1}"), b"")
    assert "results/forecast_results.csv" in result["missing"]
    assert "research_attempts" in result["fixed_records"]
