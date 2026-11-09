"""Pipeline commands keep their order and refuse holdout requests before reading inputs."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from momenta import baseline_study, cli, frozen_holdout, l2_study, l3_study
from momenta import settings as project_settings


@pytest.fixture
def fixture_settings(tmp_path, monkeypatch):
    settings = project_settings.load_settings()
    settings["data"]["data_root"] = str(tmp_path / "data")
    monkeypatch.setattr(project_settings, "load_settings", lambda: settings)
    return settings


def test_feature_build_refuses_a_holdout_origin_before_targets_or_features(
    fixture_settings, monkeypatch, tmp_path, capsys,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("The design lock must precede every target and feature build")

    monkeypatch.setattr(baseline_study, "checked_portfolio_summary", lambda s: {"passed": True})
    for name in ("build_targets", "build_features"):
        monkeypatch.setattr(baseline_study, name, forbidden)
    monkeypatch.setattr("momenta.bdp.ensure_bdp_decision", forbidden)
    assert cli.main(["features", "--through", "2015-12-01"]) == 1
    assert "Holdout origins are locked" in capsys.readouterr().err
    assert not list((tmp_path / "data").rglob("*.parquet"))


def test_design_runs_the_baseline_l2_and_l3_studies_in_order(fixture_settings, monkeypatch):
    order = []
    monkeypatch.setattr(baseline_study, "run_study", lambda s, name: order.append(name))
    monkeypatch.setattr(l2_study, "run_l2_study", lambda s: order.append("h3"))
    monkeypatch.setattr(l3_study, "run_design", lambda s: order.append("l3"))
    assert cli.main(["design"]) == 0
    assert order == ["baseline", "h3", "l3"]
    order.clear()
    assert cli.main(["design", "--study", "l2"]) == 0
    assert order == ["h3"]


@pytest.mark.parametrize("flags,experiment", [([], "holdout"), (["--mock"], "mock-holdout")])
def test_holdout_command_logs_and_stops_at_the_freeze_check(
    flags, experiment, fixture_settings, monkeypatch, capsys,
):
    logged = []

    @contextmanager
    def trial(settings, **kwargs):
        logged.append(kwargs)
        yield SimpleNamespace(metrics={}, run_id="fixture")

    def check(settings, *, frozen_path):
        raise ValueError("Frozen configuration differs from its recorded hash")

    monkeypatch.setattr(frozen_holdout, "provenance", lambda *args: {})
    monkeypatch.setattr(frozen_holdout, "trial", trial)
    monkeypatch.setattr(frozen_holdout, "check_freeze", check)
    monkeypatch.setattr(frozen_holdout.pl, "read_parquet",
                        lambda *a, **k: pytest.fail("calendar read before the freeze check"))
    assert cli.main(["holdout", *flags]) == 1
    assert "recorded hash" in capsys.readouterr().err
    assert [row["experiment"] for row in logged] == [experiment]


def test_freeze_check_command_reports_the_frozen_prefixes(fixture_settings, monkeypatch, capsys):
    monkeypatch.setattr(frozen_holdout, "check_freeze",
                        lambda settings: {"best_models": {"RAW": "L1+L2+CM", "RES": "L1"}})
    assert cli.main(["freeze-check"]) == 0
    assert "L1+L2+CM" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [["robustness", "retuned"], ["reproduce"],
                                       ["design", "--study", "h3"]])
def test_unknown_choices_and_missing_paths_are_usage_errors(arguments):
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
