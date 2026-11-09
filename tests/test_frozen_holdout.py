"""The public freeze check needs no git tag and runs before any holdout-period read."""

import json
import subprocess
from copy import deepcopy
from datetime import date, timedelta

import polars as pl
import pytest

from momenta import experiments, freeze, frozen_holdout, holdout
from momenta.manifest import sha256_file
from momenta.settings import load_settings

BOUNDS = {
    "design_first_origin": "2005-12-30",
    "design_last_origin": "2015-11-30",
    "holdout_first_origin": "2015-12-31",
    "holdout_last_origin": "2016-12-30",
    "final_target_end": "2017-01-31",
    "common_end": "2017-01-31",
}
REGISTRY = {"forecasts": ["L1", "L1+L2"], "strategies": ["Static", "BSC"],
            "tables": ["forecasts", "strategies"]}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    repo, root = tmp_path / "repo", tmp_path / "data"
    (repo / "src/momenta").mkdir(parents=True)
    (repo / "config").mkdir()
    (repo / "docs").mkdir()
    (repo / "src/momenta/example.py").write_text("value = 1\n")
    for relative in ("raw", "curated", "derived/portfolios"):
        (root / relative).mkdir(parents=True)
    (root / "raw/manifest.json").write_text('{"schema_version":1,"entries":{}}\n')
    document = {
        "data": {"data_root": str(root)},
        "walkforward": {"design_first_origin": "2005-12-31", "design_last_origin": "2015-11-30"},
        "holdout": {"mock_first_origin": "2012-12-31", "mock_last_origin": "2015-11-30"},
        "targets": {"horizon": 21, "lag": 1},
        "numerical": {"clip": 3.0, "grid": [0.0, 0.1, 1.0], "enabled": True},
    }
    config = repo / "config/settings.toml"
    freeze.write_toml(config, document)
    days, day = [], date(2012, 12, 1)
    while day <= date(2017, 2, 28):
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    pl.DataFrame({"date": days}).write_parquet(root / "curated/trading_calendar.parquet")
    published = deepcopy(document)
    published["data"]["data_root"] = "data"
    frozen = {
        "freeze": {"schema_version": 1, "approval_tag": "model-freeze",
                   "data_manifest_hash": sha256_file(root / "raw/manifest.json")},
        "bounds": BOUNDS, "best_models": {"RAW": "L1+L2", "RES": "L1"},
        "registry": REGISTRY, "settings": published,
        "trial_population": {"trading_ids": ["strategy:a"]},
    }
    path = repo / "config/frozen.toml"
    freeze.write_toml(path, frozen)
    (repo / "docs/code_provenance.json").write_text(json.dumps({"configuration": [
        {"path": "config/frozen.toml", "exported_source_sha256": sha256_file(path)}]}))
    _git(repo, "init")
    _git(repo, "config", "user.name", "Freeze tests")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Create synthetic frozen inputs")
    for module in (freeze, experiments, frozen_holdout):
        monkeypatch.setattr(module, "REPO", repo)
    monkeypatch.setattr(frozen_holdout, "FROZEN_PATH", path)
    monkeypatch.setattr(frozen_holdout, "PROVENANCE_PATH", repo / "docs/code_provenance.json")
    return repo, root, load_settings(config), path


def test_freeze_check_passes_when_only_the_data_location_differs(frozen_inputs):
    _, _, settings, path = frozen_inputs
    frozen = frozen_holdout.check_freeze(settings)
    assert frozen["best_models"] == {"RAW": "L1+L2", "RES": "L1"}
    assert frozen["settings"]["data"]["data_root"] == "data"
    assert settings["data"]["data_root"] != "data"
    assert frozen_holdout.recorded_frozen_hash() == sha256_file(path)


@pytest.mark.parametrize("changed", ["frozen", "manifest", "setting", "file", "record"])
def test_freeze_check_rejects_each_changed_input(frozen_inputs, changed):
    repo, root, settings, path = frozen_inputs
    if changed == "frozen":
        path.write_text(path.read_text().replace('"L1+L2"', '"L1+L2+CM"', 1))
    elif changed == "manifest":
        (root / "raw/manifest.json").write_text('{"schema_version":1,"entries":{"new":{}}}\n')
    elif changed == "setting":
        document = {key: value for key, value in settings.items() if key != "config_path"}
        document["numerical"]["clip"] = 3.5
        freeze.write_toml(repo / "config/settings.toml", document)
        settings = load_settings(repo / "config/settings.toml")
    elif changed == "file":
        settings["numerical"]["clip"] = 3.5
    else:
        (repo / "docs/code_provenance.json").write_text(json.dumps({"configuration": []}))
    message = {"frozen": "recorded hash", "manifest": "Data manifest", "setting": "Study settings",
               "file": "In-memory settings", "record": "not recorded"}[changed]
    with pytest.raises(ValueError, match=message):
        frozen_holdout.check_freeze(settings)


def test_exempt_settings_are_the_data_location_only():
    assert frozen_holdout.EXEMPT_SETTINGS == (("data", "data_root"),)
    left = {"data": {"data_root": "a", "sharadar_tables": ["stocks"]}, "config_path": "x"}
    right = {"data": {"data_root": "b", "sharadar_tables": ["stocks"]}}
    assert frozen_holdout.comparable_settings(left) == frozen_holdout.comparable_settings(right)
    right["data"]["sharadar_tables"] = ["daily"]
    assert frozen_holdout.comparable_settings(left) != frozen_holdout.comparable_settings(right)


def test_failed_check_is_logged_before_the_calendar_or_pipeline_is_read(
    frozen_inputs, monkeypatch,
):
    repo, root, settings, path = frozen_inputs
    path.write_text(path.read_text() + "\n")
    monkeypatch.setattr(frozen_holdout.pl, "read_parquet", lambda *a, **k: pytest.fail("read"))
    with pytest.raises(ValueError, match="recorded hash"):
        frozen_holdout.run_holdout(settings)
    rows = experiments.read_trials(repo / "docs/TRIALS.csv")
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert rows[0]["strategy_id"].startswith("diagnostic:")
    assert not (root / "derived/final").exists()


def test_failed_check_imports_no_study_module(frozen_inputs, monkeypatch):
    """A failed freeze check stops the run before a builder or design reader is imported."""
    import sys

    _, _, settings, path = frozen_inputs
    path.write_text(path.read_text() + "\n")
    study_modules = {"momenta.l3_study", "momenta.l3_report", "momenta.l3_inputs"}
    for name in study_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(ValueError, match="recorded hash"):
        frozen_holdout.run_holdout(settings)
    assert not study_modules & set(sys.modules)


def test_holdout_requires_a_current_design_that_selected_the_frozen_prefixes(
    frozen_inputs, monkeypatch,
):
    from momenta import l3_report

    _, root, settings, _ = frozen_inputs
    monkeypatch.setattr(frozen_holdout.pl, "read_parquet", lambda *a, **k: pytest.fail("read"))
    monkeypatch.setattr(l3_report, "load_design_summary", lambda current: None)
    with pytest.raises(ValueError, match="design run must precede"):
        frozen_holdout.run_holdout(settings)
    monkeypatch.setattr(l3_report, "load_design_summary",
                        lambda current: {"best_models": {"RAW": "L1", "RES": "L1"}})
    with pytest.raises(ValueError, match="select the frozen prefixes"):
        frozen_holdout.run_holdout(settings)
    assert not (root / "derived/final").exists()


def test_source_directory_symlinks_cannot_escape_the_execution_hash(frozen_inputs, tmp_path):
    repo, _, _, _ = frozen_inputs
    external = tmp_path / "external_source"
    external.mkdir()
    (repo / "src/extra").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="regular files"):
        freeze.source_config_hash(repo)


def test_staging_never_changes_normal_command_bounds_or_design_inputs(frozen_inputs, tmp_path):
    repo, root, settings, path = frozen_inputs
    before = deepcopy(settings)
    stage = tmp_path / "stage"
    scoped = holdout._stage_settings(
        settings, stage, date(2012, 12, 31), date(2015, 11, 30), {"git_dirty": False}, path,
    )
    assert settings == before
    assert scoped["walkforward"]["design_first_origin"] == "2012-12-31"
    assert scoped["_registered_design_first_origin"] == "2005-12-31"
    assert scoped["data"]["data_root"] == str(stage)
    assert (stage / "derived/portfolios").resolve() == (root / "derived/portfolios").resolve()
    assert not (stage / "derived/baseline").is_symlink()
    (stage / "derived/baseline/model.json").write_text("new fitted model")
    assert not (root / "derived/baseline/model.json").exists()
    assert (stage / "frozen.toml").read_bytes() == path.read_bytes()
