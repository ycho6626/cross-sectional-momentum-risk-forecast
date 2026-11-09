"""Hand-calculated H3 evidence, descriptive OLS, and L2 input integrity (Methodology)."""

import json
from datetime import date
from math import erfc, sqrt

import numpy as np
import polars as pl
import pytest

from momenta import l2_report
from momenta.manifest import sha256_file


@pytest.fixture
def evidence():
    origins = [date(2006, month, 28) for month in (1, 2, 3)]
    forecasts = pl.DataFrame(
        {
            "series": ["RAW"] * 3 + ["RES"] * 3,
            "origin": origins * 2,
            "refit": [date(2006, 1, 2)] * 6,
            "actual": [0.0] * 6,
            "prediction": [0.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "base_prediction": [2.0] * 6,
            "residual_variance": [0.0] * 6,
        }
    )
    placebo = pl.DataFrame(
        {
            "series": ["RAW"] * 4 + ["RES"] * 4,
            "k": [10] * 8,
            "rep": [0, 1, 2, 3] * 2,
            "r2": [-0.2, 0.4, 0.7, 0.8, -0.2, 0.0, 0.0, 0.1],
            "error": [None] * 8,
        }
    )
    settings = {
        "walkforward": {"design_first_origin": "2006-01-31", "design_last_origin": "2006-03-31"},
        "evaluation": {
            "bootstrap_reps": 20,
            "block_months": 1e100,
            "seed": 5,
            "nw_lags_monthly": 3,
            "confidence": 0.95,
            "size_check_blocks": [4, 10],
            "size_check_max": 0.10,
            "size_check_reps": 4,
        },
        "l2": {
            "placebo_k": 10,
            "numerical_tolerance": 1e-10,
            "heatmap_dpi": 40,
            "heatmap_size_inches": [4, 4],
        },
    }
    baseline = {
        "design_first_origin": str(origins[0]),
        "design_last_origin": str(origins[-1]),
        "tables": {
            "size_check": [
                {
                    "series": series,
                    "k": k,
                    "complete": True,
                    "bias_passed": True,
                    "cw_rejection_rate": 0.085 if series == "RAW" else 0.025,
                }
                for series in ("RAW", "RES")
                for k in (4, 10)
            ]
        },
    }
    return forecasts, placebo, baseline, settings


def test_h3_hand_losses_cw_holm_and_same_series_placebo_ties(evidence):
    forecasts, placebo, baseline, settings = evidence
    raw, res = l2_report.h3_table(forecasts, placebo, baseline, settings, full=True)
    # RAW squared errors0,1,4 vs4,4,4: improvement7/3, incremental R2=7/12.
    assert raw["mse"] == pytest.approx(5 / 3)
    assert raw["baseline_mse"] == 4
    assert raw["mse_improvement"] == pytest.approx(7 / 3)
    assert raw["r2_oos"] == pytest.approx(7 / 12)
    assert raw["mse_improvement_interval_lower"] == pytest.approx(7 / 3)
    assert raw["mse_improvement_interval_upper"] == pytest.approx(7 / 3)
    # CW differences8,4,0: Bartlett HAC L2 gives t=3sqrt(3)/2.
    pvalue = erfc((3 * sqrt(3) / 2) / sqrt(2)) / 2
    assert raw["cw_pvalue"] == pytest.approx(pvalue)
    assert raw["cw_holm_pvalue"] == pytest.approx(2 * pvalue)
    assert raw["placebo_position"] == 0.5
    assert res["r2_oos"] == 0
    assert res["cw_pvalue"] == res["cw_holm_pvalue"] == 1.0
    assert res["placebo_position"] == 0.75
    assert res["identical_forecasts"] is True
    assert all(row["cw_inference"] == "confirmatory_size_supported" for row in (raw, res))


def test_h3_thin_sample_is_descriptive_and_full_requires_pinned_months(evidence):
    forecasts, placebo, baseline, settings = evidence
    thin = forecasts.filter(pl.col("series") == "RAW").head(2)
    row = l2_report.h3_table(thin, placebo, baseline, settings, full=False)[0]
    assert row["cw_inference"] == "descriptive_thin_scope"
    assert row["cw_holm_pvalue"] is None
    assert row["placebo_position"] is None
    with pytest.raises(ValueError, match="complete pinned month-end sample"):
        l2_report.h3_table(thin, placebo, baseline, settings, full=True)
    with pytest.raises(ValueError, match="Incomplete k=10"):
        l2_report.h3_table(forecasts, placebo.head(7), baseline, settings, full=True)
    with pytest.raises(ValueError, match="Duplicate"):
        l2_report.h3_table(
            pl.concat([forecasts, forecasts.head(1)]), placebo, baseline, settings, full=True
        )


def test_redundancy_r2_vif_and_thin_degrees_of_freedom_are_hand_calculated():
    x = np.array([-1, -1, 0, 0, 1, 1], dtype=float)
    orthogonal = np.array([-1, 1, -1, 1, -1, 1], dtype=float)
    design = pl.DataFrame(
        {
            "date": [date(2006, month, 28) for month in range(1, 7)],
            "series": ["RAW"] * 6,
            "x": x,
            "mixed": x + sqrt(2 / 3) * orthogonal,
            "perfect": 2 * x,
            "constant": [1.0] * 6,
        }
    )
    rows, matrices = l2_report.redundancy_audit(
        design,
        ["x"],
        ["mixed", "perfect", "constant"],
        1e-10,
    )
    assert rows[0]["r2"] == pytest.approx(0.5)
    assert rows[0]["vif"] == pytest.approx(2.0)
    assert rows[1]["r2"] == 1.0 and rows[1]["status"] == "infinite_vif"
    assert rows[1]["vif"] is None
    assert rows[2]["status"] == "constant_feature"
    assert matrices["RAW"][1][0, 1] == pytest.approx(sqrt(0.5))
    assert np.isnan(matrices["RAW"][1][-1]).all()
    thin = design.gather([0, 4])
    thin_rows, _ = l2_report.redundancy_audit(thin, ["x"], ["perfect"], 1e-10)
    assert thin_rows[0]["status"] == "insufficient_residual_degrees_of_freedom"


def test_l2_report_pins_forecasts_config_and_numeric_outputs(evidence, tmp_path, monkeypatch):
    forecasts, placebo, baseline, settings = evidence
    output = tmp_path / "data/derived/l2"
    output.mkdir(parents=True)
    baseline_dir = output.parent / "baseline"
    baseline_dir.mkdir()
    placebo.write_parquet(baseline_dir / "size_check.parquet")
    config = tmp_path / "settings.toml"
    config.write_text("test = true")
    settings.update(config_path=str(config), data={"data_root": str(tmp_path / "data")})
    monkeypatch.setattr(l2_report, "REPO", tmp_path)
    monkeypatch.setattr(l2_report, "code_hash", lambda: "source")
    forecasts.write_parquet(output / "forecasts.parquet")
    pl.DataFrame({"refit": [date(2006, 1, 2)]}).write_parquet(output / "fits.parquet")
    design = forecasts.select("series", pl.col("origin").alias("date")).with_columns(
        pl.Series("x", [1.0, 2.0, 3.0] * 2),
        pl.Series("z", [2.0, 1.0, 2.0] * 2),
    )
    design.write_parquet(output / "redundancy_design.parquet")
    metadata = {
        "mode": "full",
        "config_hash": sha256_file(config),
        "git_hash": "commit",
        "git_dirty": False,
        "data_manifest_hash": "manifest",
        "input_hashes": {
            str(baseline_dir / "size_check.parquet"):
            sha256_file(baseline_dir / "size_check.parquet"),
        },
    }
    audit = {"prefix": [], "feature_availability": [], "feature_coverage": [], "breadth": []}
    for name in ("_metadata.json", "_audit.json", "trial_snapshot.json"):
        (output / name).write_text("{}")
    summary = l2_report.evaluate_l2(
        settings, output, metadata, audit, baseline, l1_columns=["x"], l2_columns=["z"]
    )
    assert summary["complete"] is True
    assert l2_report.load_summary(settings)["forecast_rows"] == 6
    assert (tmp_path / "results/figures/l2_redundancy_RAW.png").exists()
    saved = (output / "forecasts.parquet").read_bytes()
    (output / "forecasts.parquet").write_bytes(saved + b"damage")
    with pytest.raises(ValueError, match="Stale L2 input"):
        l2_report.load_summary(settings)
    (output / "forecasts.parquet").write_bytes(saved)
    config.write_text("test = false")
    with pytest.raises(ValueError, match="configuration"):
        l2_report.load_summary(settings)
    # The saved summary is strict JSON, including unavailable/infinite-VIF audit values.
    assert json.loads((output / "summary.json").read_text())["complete"] is True
