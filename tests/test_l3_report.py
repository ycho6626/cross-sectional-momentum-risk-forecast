"""Independent loss calculations, frozen selection and causal regime reporting."""

import json
from copy import deepcopy
from datetime import date
from math import erfc, sqrt

import numpy as np
import polars as pl
import pytest

from momenta import l3_report
from momenta.manifest import sha256_file


@pytest.fixture
def evidence():
    origins = [date(2006, month, 28) for month in (1, 2, 3)]
    predictions = {
        "L1-own": [3., 3., 3.], "L1_without_M": [2.5, 2.5, 2.5],
        "L1": [2., 2., 2.], "L1+L2": [0., 1., 2.],
        "L1+L2+CM": [2., 2., 2.], "L1+L2+CM+OT": [0., 1., 2.],
        "L1+L2+CM+Q": [3., 3., 3.], "L1+L2+CM+Q+OT": [1., 1., 2.],
        "best+HMMF2": [0., 1., 2.], "best+HMME2": [1., 1., 1.],
        "best+HMMF3": [2., 1., 0.], "best+HMME3": [1., 1., 1.],
    }
    frame = pl.DataFrame([
        {"series": series, "target": "y1", "feature_set": model,
         "origin": origin, "refit": date(2006, 1, 2), "actual": 0.,
         "prediction": value if series == "RAW" else 2.,
         "base_prediction": 3., "residual_variance": 0.}
        for series in ("RAW", "RES") for model, values in predictions.items()
        for origin, value in zip(origins, values, strict=True)
    ])
    settings = {
        "walkforward": {"design_first_origin": "2006-01-31",
                        "design_last_origin": "2015-11-30"},
        "hmm": {"states": 2, "secondary_states": 3},
        "evaluation": {"bootstrap_reps": 20, "block_months": 1e100, "seed": 5,
                       "nw_lags_monthly": 3, "confidence": .95,
                       "size_check_blocks": [4, 10], "size_check_max": .10,
                       "size_check_series": ["RAW", "RES"]},
    }
    size = [{"series": series, "k": k, "complete": True,
             "bias_passed": True, "cw_rejection_rate": .05}
            for series in ("RAW", "RES") for k in (4, 10)]
    best = {"RAW": "L1+L2+CM", "RES": "L1"}
    return frame, settings, size, best


def test_nested_losses_pair_correct_prefixes_and_holm_only_within_hypothesis(evidence):
    frame, settings, size, best = evidence
    rows = l3_report.comparison_table(frame, settings, best, size, mode="design")
    chosen = {(row["hypothesis"], row["states"], row["series"]): row for row in rows}
    p = erfc((3*sqrt(3)/2)/sqrt(2))/2  # CW d=[8,4,0], Bartlett HAC with 2 lags.
    for hypothesis, states in [("H3", None), ("H4", None), ("H5a", 2), ("H5a", 3)]:
        raw = chosen[hypothesis, states, "RAW"]
        res = chosen[hypothesis, states, "RES"]
        assert raw["mse"] == pytest.approx(5/3)
        assert raw["baseline_mse"] == 4
        assert raw["mse_improvement"] == pytest.approx(7/3)
        assert raw["r2_oos"] == pytest.approx(7/12)
        assert raw["mse_improvement_interval_lower"] == pytest.approx(7/3)
        assert raw["mse_improvement_interval_upper"] == pytest.approx(7/3)
        assert raw["cw_pvalue"] == pytest.approx(p)
        assert raw["cw_holm_pvalue"] == pytest.approx(2*p)
        assert res["cw_pvalue"] == res["cw_holm_pvalue"] == 1
    assert chosen["H4Q", None, "RAW"]["benchmark"] == "L1+L2+CM+Q"
    assert chosen["H4Q", None, "RAW"]["mse_improvement"] == 7
    assert chosen["H5a", 2, "RAW"]["benchmark"] == best["RAW"]
    assert chosen["H5a", 2, "RES"]["benchmark"] == best["RES"]


def test_h5b_is_unadjusted_two_sided_dm_without_any_cw_calculation(evidence, monkeypatch):
    frame, settings, _, _ = evidence
    pair = l3_report.paired_predictions(frame, "RAW", "best+HMMF2", "best+HMME2")

    def reject(*args, **kwargs):
        raise AssertionError("H5b must not calculate nested CW statistics")

    monkeypatch.setattr(l3_report, "forecast_metrics", reject)
    result = l3_report._non_nested_metrics(pair, settings)
    # Unadjusted d=[1,0,-3], HAC variance148/81, t=-sqrt(27/37).
    assert result["mse_improvement"] == pytest.approx(-2/3)
    assert result["r2_oos"] == pytest.approx(-2/3)
    assert result["dm"]["pvalue"] == pytest.approx(erfc(sqrt(27/37)/sqrt(2)))
    assert result["mse_improvement_interval"]["lower"] == pytest.approx(-2/3)
    assert not any(key.startswith("cw") for key in result)


def test_identical_h5b_nonrejection_and_incomplete_size_is_not_confirmatory(evidence):
    frame, settings, size, best = evidence
    incomplete = deepcopy(size)
    incomplete[0].update(complete=False, cw_rejection_rate=None)
    rows = l3_report.comparison_table(frame, settings, best, incomplete, mode="design")
    assert all(not any(key.startswith("cw") for key in row) for row in rows
               if row["hypothesis"] == "H5b")
    assert all(row["dm_pvalue"] == 1 for row in rows
               if row["hypothesis"] == "H5b" and row["series"] == "RES")
    assert all(row["cw_inference"] == "size_check_unavailable" for row in rows
               if row["hypothesis"] != "H5b" and row["series"] == "RAW")
    thin = l3_report.comparison_table(frame, settings, best, size, mode="thin")
    assert all(row["cw_holm_pvalue"] is None for row in thin if row["hypothesis"] != "H5b")


@pytest.mark.parametrize("damage", ["drop", "target", "refit"])
def test_paired_comparisons_refuse_different_support_or_realized_targets(evidence, damage):
    frame, _, _, _ = evidence
    predicate = (pl.col("feature_set") == "L1+L2") & (pl.col("origin").dt.month() == 1)
    if damage == "drop":
        frame = frame.filter(~predicate)
    else:
        column, value = ("actual", 1.) if damage == "target" else ("refit", date(2005, 1, 3))
        frame = frame.with_columns(pl.when(predicate).then(pl.lit(value))
                                   .otherwise(pl.col(column)).alias(column))
    with pytest.raises(ValueError, match="support or targets differ"):
        l3_report.paired_predictions(frame, "RAW", "L1+L2", "L1")


def test_prefix_table_preserves_frozen_best_even_if_another_model_now_wins(evidence):
    frame, _, _, best = evidence
    table = l3_report.prefix_table(frame, best)
    selected = [row for row in table if row["selected_best"]]
    assert {(row["series"], row["model"]) for row in selected} == set(best.items())
    raw = {row["model"]: row for row in table if row["series"] == "RAW"}
    assert raw["L1+L2+CM"]["mse"] == 4
    assert raw["L1+L2+CM+OT"]["mse"] == pytest.approx(5/3)
    assert not raw["L1+L2+CM+OT"]["selected_best"]
    assert not raw["L1+L2+CM+Q"]["selection_candidate"]
    assert not raw["best+HMMF2"]["selection_candidate"]


def test_diagnostic_summary_retains_detection_rates_and_failed_restart_counts():
    result = l3_report.diagnostic_summary({
        "hmm_recovery": {"passed": True, "fit": {"log_likelihood": -2., "restarts": [
            {"status": "converged"}, {"status": "nonconverged"}]}},
        "copula_synthetic": {"noise_distances": [1.]*100, "noise_threshold": 1.,
                             "tail": {"reps": 100, "detection_rate": .95,
                                      "replications": [{"distance": 2.}]*100}},
    })
    assert result["hmm_recovery"]["fit"]["restart_statuses"] == {
        "converged": 1, "nonconverged": 1}
    assert result["copula_synthetic"] == {
        "noise_threshold": 1., "tail": {"reps": 100, "detection_rate": .95}}


def regime_fixture():
    first, second = "2005-01-03", "2006-01-02"
    base = {"lineage": "full", "series": "RAW", "states": 2, "fit_hash": "a",
            "fit_time": first, "training_months": 47,
            "hmm": {"transition": [[.9, .1], [.2, .8]],
                    "restarts": [{"restart": 0, "converged": True}]},
            "occupancy": [.96, .04], "posterior_occupancy": [.9, .1],
            "low_occupancy": [False, True], "mean_duration_months": [10., 5.],
            "infinite_duration": [False, False],
            "monthly_probabilities": [
                {"date": "2005-12-30", "filtered_state": 1, "y1": -2.},
                {"date": "2006-01-31", "filtered_state": 1, "y1": -1.}]}
    later = deepcopy(base)
    later.update(fit_time=second, fit_hash="b", training_months=59)
    later["hmm"]["transition"] = [[.8, .2], [.3, .7]]
    later["monthly_probabilities"] = [
        {"date": "2005-12-30", "filtered_state": 0, "y1": -2.},
        {"date": "2006-01-31", "filtered_state": 0, "y1": -1.}]
    inner = deepcopy(base)
    inner.update(lineage="inner", fit_time="2002-12-03", training_months=22)
    frame = pl.DataFrame({"origin": [date(2005, 12, 30), date(2006, 1, 31)],
        "refit": [date.fromisoformat(first), date.fromisoformat(second)],
        "series": ["RAW"]*2, "target": ["y1"]*2, "feature_set": ["best+HMMF2"]*2,
        "actual": [-2., -1.]})
    return [base, later, inner], frame


def test_regime_risk_uses_only_the_fit_serving_the_origin_and_reports_inner_flags():
    records, frame = regime_fixture()
    tables = l3_report.regime_tables(records[::-1], frame)
    assert len(tables["regime_occupancy"]) == 6
    assert len(tables["regime_restarts"]) == 3
    assert tables["regime_occupancy"][-1]["below_five_percent"] is True
    stability = tables["regime_stability"]
    assert len(stability) == 2
    assert stability[0]["frobenius_change"] is None
    assert stability[1]["frobenius_change"] == pytest.approx(.2)
    assert stability[1]["max_change"] == pytest.approx(.1)
    risk = tables["regime_subsequent_risk"]
    assert [row["origins"] for row in risk] == [1, 1]
    assert [row["mean_log_realized_variance"] for row in risk] == [-1., -2.]
    assert [row["mean_realized_variance"] for row in risk] == pytest.approx(np.exp([-1., -2.]))
    with pytest.raises(ValueError, match="cover every"):
        l3_report.regime_tables(records[1:], frame)
    duplicate = deepcopy(records[0])
    duplicate["fit_hash"] = "conflicting"
    with pytest.raises(ValueError, match="Conflicting"):
        l3_report.regime_tables([*records, duplicate], frame)


def test_mock_report_keeps_numeric_outputs_identical_and_never_writes_repo(evidence, tmp_path,
                                                                         monkeypatch):
    frame, settings, size, best = evidence
    frame = frame.filter(~pl.col("feature_set").str.starts_with("best+HMM"))
    settings["data"] = {"data_root": str(tmp_path)}
    output = tmp_path/"derived/mock"
    output.mkdir(parents=True)
    frame.write_parquet(output/"forecasts.parquet")
    repo = tmp_path/"repository"
    monkeypatch.setattr(l3_report, "REPO", repo)
    audit = {"best_models": best, "baseline_summary": {"tables": {"size_check": size}},
             "diagnostics": {"hmm_recovery": {"passed": True, "provenance": {"run_id": "a"},
                                               "elapsed_seconds": 10.}}}
    metadata = {"mode": "mock_holdout", "run_id": "first", "config_hash": "pinned"}
    first = l3_report.evaluate_l3(settings, output, metadata, audit)
    hashes = {path: sha256_file(tmp_path/path) for path in first["deterministic_artifacts"]}
    metadata["run_id"] = "second"
    audit["diagnostics"]["hmm_recovery"].update(elapsed_seconds=99., provenance={"run_id": "b"})
    second = l3_report.evaluate_l3(settings, output, metadata, audit)
    assert hashes == {path: sha256_file(tmp_path/path)
                      for path in second["deterministic_artifacts"]}
    assert not repo.exists()
    assert second["provenance"]["run_id"] == "second"
    assert json.loads((output/"numeric_results.json").read_text())["diagnostics"] == {
        "hmm_recovery": {"passed": True}}
    assert all("provenance" not in path for path in second["deterministic_artifacts"])
    assert json.loads((output/"numeric_results.provenance.json").read_text())["run_id"] == "second"
    assert len(second["tables"]["baseline_forecasts"]) == 6
    settings["walkforward"]["design_last_origin"] = "2006-02-28"
    metadata["mode"] = "design"
    with pytest.raises(ValueError, match="Holdout origins"):
        l3_report.evaluate_l3(settings, output, metadata, audit)
