"""The design-period L3 summary must be current before a holdout or report uses it."""

import json

import pytest

from momenta import l3_report
from momenta.manifest import sha256_file


@pytest.fixture
def pinned_report(tmp_path, monkeypatch):
    (tmp_path/"raw").mkdir()
    (tmp_path/"derived/final").mkdir(parents=True)
    config = tmp_path/"settings.toml"
    config.write_text("[l3]\n")
    manifest = tmp_path/"raw/manifest.json"
    manifest.write_text("{}")
    source = tmp_path/"input.parquet"
    source.write_bytes(b"pinned input")
    result = tmp_path/"derived/final/table.csv"
    result.write_text("value\n1\n")
    summary = {
        "complete": True, "mode": "design",
        "provenance": {"mode": "design", "config_hash": sha256_file(config),
            "data_manifest_hash": sha256_file(manifest), "source_config_hash": "source-fixture",
            "git_dirty": False, "git_hash": "original-producer",
            "input_hashes": {str(source): sha256_file(source)}},
        "artifacts": {str(result): sha256_file(result)},
    }
    path = tmp_path/"derived/final/summary.json"
    path.write_text(json.dumps(summary))
    settings = {"config_path": str(config), "data": {"data_root": str(tmp_path)}, "l3": {}}
    monkeypatch.setattr(l3_report, "source_config_hash", lambda: "source-fixture")
    return settings, summary, path, source, result


@pytest.mark.parametrize("changed", ["config", "manifest", "input", "result", "source"])
def test_report_rejects_changed_producer_inputs_and_artifacts(pinned_report, changed, monkeypatch):
    settings, summary, _, source, result = pinned_report
    assert l3_report.load_design_summary(settings) == summary
    if changed == "source":
        monkeypatch.setattr(l3_report, "source_config_hash", lambda: "different-tree")
    else:
        path = {"config": source.parent/"settings.toml",
                "manifest": source.parent/"raw/manifest.json",
                "input": source, "result": result}[changed]
        path.write_text("changed")
    with pytest.raises(ValueError, match="changed|Changed"):
        l3_report.load_design_summary(settings)


def test_result_registry_cannot_replace_a_conflicting_original_input_hash(pinned_report):
    settings, summary, path, source, _ = pinned_report
    source.write_text("input overwritten by an output")
    summary["artifacts"][str(source)] = sha256_file(source)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Changed L3 input"):
        l3_report.load_design_summary(settings)


def test_dirty_initial_tree_is_provisional_and_holdout_cannot_replace_design(pinned_report):
    settings, summary, path, _, _ = pinned_report
    summary["provenance"]["git_dirty"] = True
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="clean initial"):
        l3_report.load_design_summary(settings)
    summary["complete"] = False
    path.write_text(json.dumps(summary))
    assert l3_report.load_design_summary(settings)["complete"] is False
    summary["provenance"]["mode"] = "holdout"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="cannot load a holdout"):
        l3_report.load_design_summary(settings)
