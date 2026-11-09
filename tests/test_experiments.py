"""Trials must survive failed work and repeated specifications (Methodology)."""


import json

import pytest

from momenta import experiments


@pytest.fixture
def setup_log(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "REPO", tmp_path)
    monkeypatch.setattr(experiments, "provenance", lambda *args: {
        "git_hash": "commit", "git_dirty": True, "config_hash": "config",
        "data_manifest_hash": "manifest", "data_vintage": {"input": "vintage"},
    })
    return {"data": {"data_root": str(tmp_path / "data")}, "config_path": "settings.toml"}


def test_identity_ignores_execution_sample_and_key_order():
    first = {"model": "ridge", "lambda": [0.1, 1], "series": "RAW"}
    second = {"series": "RAW", "lambda": [0.1, 1], "model": "ridge",
              "sample_window": "thin", "timestamp": "later", "results": {"mse": 1}}
    assert experiments.strategy_id("forecast", first) == experiments.strategy_id("forecast", second)
    assert experiments.strategy_id("strategy", first) != experiments.strategy_id("forecast", first)
    assert experiments.strategy_id("forecast", first) != experiments.strategy_id(
        "forecast", first | {"lambda": [0.1, 2]}
    )


def test_failure_preserves_running_row_partial_metrics_and_does_not_leak(setup_log):
    with pytest.raises(RuntimeError, match="sensitive"):
        with experiments.trial(
            setup_log, experiment="recovery", kind="diagnostic", series="French",
            sample="synthetic", specification={"model": "skew-t", "seed": 17},
        ) as attempt:
            before = experiments.read_trials()
            assert before[0]["status"] == "running"
            assert before[0]["strategy_id"] == attempt.strategy_id
            attempt.metrics["converged"] = False
            raise RuntimeError("sensitive credential")
    after = experiments.read_trials()
    assert len(after) == 1
    assert after[0]["status"] == "failed"
    assert after[0]["error_type"] == "RuntimeError"
    assert json.loads(after[0]["metrics_json"]) == {"converged": False}
    assert "sensitive" not in str(after)
    assert after[0]["finished_at"]


def test_nested_attempts_and_repeated_specifications_preserve_history(setup_log):
    arguments = dict(experiment="static", kind="strategy", series="RAW",
                     sample="design", specification={"rule": "static", "cost_bp": 10})
    with experiments.trial(setup_log, **arguments) as first:
        with experiments.trial(setup_log, **(arguments | {"sample": "thin"})) as second:
            assert len(experiments.read_trials()) == 2
            second.metrics["monthly_sharpe"] = float("nan")
        first.metrics["monthly_sharpe"] = 0.1
    rows = experiments.read_trials()
    assert len({row["strategy_id"] for row in rows}) == 1
    assert len({row["run_id"] for row in rows}) == 2
    assert all(row["status"] == "completed" for row in rows)
    assert json.loads(rows[1]["metrics_json"])["monthly_sharpe"] is None


def test_invalid_or_credential_specifications_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        experiments.strategy_id("diagnostic", {"timestamp": "today"})
    with pytest.raises(ValueError, match="Credentials"):
        experiments.strategy_id("strategy", {"api_key": "never persisted"})
