"""Raw-only isolation, exact artifact comparison and the saved-audit evaluation order."""

import hashlib
import json

import pytest

from momenta import l3_study, reproduce
from momenta.l3_report import _csv
from momenta.manifest import Manifest, sha256_file


def _raw(tmp_path):
    source = tmp_path / "source"
    path = source / "raw/sharadar/stocks.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"pinned compressed source bytes")
    Manifest(source).record("sharadar/stocks", path, "https://example.invalid/stocks", "v1")
    generated = source / "raw/staged/stocks/source_year=2000/part.parquet"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"must never be reused as an input")
    return source, path


def test_raw_reproduction_links_only_verified_downloads_and_preserves_original(tmp_path):
    source, path = _raw(tmp_path)
    original = {item: item.read_bytes() for item in source.rglob("*") if item.is_file()}
    destination = tmp_path / "rebuild"
    reproduce.stage_raw_downloads(source, destination)
    assert (destination / "raw/sharadar/stocks.zip").resolve() == path
    assert (destination / "raw/manifest.json").read_bytes() == original[
        source / "raw/manifest.json"
    ]
    assert not (destination / "raw/staged").exists()
    assert not (destination / "curated").exists()
    assert all(item.read_bytes() == value for item, value in original.items())
    with pytest.raises(ValueError, match="fresh isolated"):
        reproduce.stage_raw_downloads(source, destination)
    with pytest.raises(ValueError, match="fresh isolated"):
        reproduce.stage_raw_downloads(source, source)


def test_raw_reproduction_refuses_changed_or_escaping_pinned_inputs(tmp_path):
    source, path = _raw(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256"):
        reproduce.stage_raw_downloads(source, tmp_path / "changed-rebuild")
    path.unlink()
    elsewhere = tmp_path / "elsewhere.zip"
    elsewhere.write_bytes(b"pinned compressed source bytes")
    path.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="escapes"):
        reproduce.stage_raw_downloads(source, tmp_path / "escaping-rebuild")


def test_saved_audit_round_trip_keeps_computed_csv_bytes_and_values(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    expected = {
        "prefix_checks": (
            b"series,model,prefix,extension_excluded,excluded_prefix_exact\r\n"
            b"RAW,L1+L2,L1,True,True\r\n"
        ),
        "break_even": (
            b"series,strategy,break_even_bp,status,crossings_bp,evaluations\r\n"
            b"RAW,Static,12.5,crossing,[12.5],9\r\n"
            b"RAW,BSC,,benchmark,[],\r\n"
        ),
    }
    audit = {
        "prefix_checks": [{"series": "RAW", "model": "L1+L2", "prefix": "L1",
                           "extension_excluded": True, "excluded_prefix_exact": True}],
        "strategy_audit": {"break_even": [
            {"series": "RAW", "strategy": "Static", "break_even_bp": 12.5,
             "status": "crossing", "crossings_bp": [12.5], "evaluations": 9},
            {"series": "RAW", "strategy": "BSC", "break_even_bp": None,
             "status": "benchmark", "crossings_bp": []},
        ]},
    }
    saved = tmp_path / "_audit.json"
    saved.write_text(json.dumps(audit, sort_keys=True))
    reloaded = json.loads(saved.read_text())
    restored = l3_study._persisted_audit(saved, audit)
    assert restored == reloaded == audit
    assert list(reloaded["prefix_checks"][0])[0] == "excluded_prefix_exact"
    assert list(restored["prefix_checks"][0])[0] == "series"
    for name, rows in (
        ("prefix_checks", restored["prefix_checks"]),
        ("break_even", restored["strategy_audit"]["break_even"]),
    ):
        _csv(tables / f"{name}.csv", rows)
        assert (tables / f"{name}.csv").read_bytes() == expected[name]


def test_saved_audit_refuses_changed_row_schemas(tmp_path):
    saved = tmp_path / "_audit.json"
    saved.write_text(json.dumps({"prefix_checks": [{"series": "RAW"}],
                                 "strategy_audit": {"break_even": []}}))
    computed = {"prefix_checks": [{"series": "RAW", "model": "L1+L2"}],
                "strategy_audit": {"break_even": []}}
    with pytest.raises(ValueError, match="Saved audit fields differ"):
        l3_study._persisted_audit(saved, computed)


def test_registry_reports_changed_and_missing_outputs_and_refuses_outside_paths(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    path = stage / "table.csv"
    path.write_text("mse\n0.25\n")
    expected = {"table.csv": sha256_file(path)}
    assert reproduce.verify_artifacts(stage, expected) == {
        "table.csv": {"sha256": expected["table.csv"], "match": True}}
    path.write_text("mse\n0.2500000001\n")
    assert reproduce.verify_artifacts(stage, expected)["table.csv"]["match"] is False
    missing = reproduce.verify_artifacts(stage, {"missing.csv": expected["table.csv"]})
    assert missing == {"missing.csv": {"sha256": None, "match": False}}
    with pytest.raises(ValueError, match="empty"):
        reproduce.verify_artifacts(stage, {})
    outside = tmp_path / "outside.csv"
    outside.write_text("mse\n0.25\n")
    (stage / "linked.csv").symlink_to(outside)
    for name in ("linked.csv", "../outside.csv", str(outside)):
        with pytest.raises(ValueError, match="Invalid"):
            reproduce.verify_artifacts(stage, {name: sha256_file(outside)})


def test_artifact_map_digest_hashes_the_sorted_compact_name_map():
    hashes = {"b.csv": "2", "a.csv": "1"}
    canonical = b'{"a.csv":"1","b.csv":"2"}'
    assert reproduce.artifact_map_digest(hashes) == hashlib.sha256(canonical).hexdigest()
    assert reproduce.artifact_map_digest(dict(reversed(list(hashes.items())))) == (
        reproduce.artifact_map_digest(hashes))


def test_published_files_match_their_released_digests_or_differ_only_in_timing(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(reproduce, "REPO", tmp_path)
    table = tmp_path / "results/robustness.csv"
    table.parent.mkdir(parents=True)
    table.write_text("variant,elapsed_seconds,raw_r2\ndecile,352.9,-0.02\n")
    figure = tmp_path / "figures/net_equity.png"
    figure.parent.mkdir()
    figure.write_bytes(b"pixels")
    evidence = tmp_path / "results/manifest.json"
    evidence.write_text(json.dumps({"artifacts": [
        {"path": "results/robustness.csv", "sha256": sha256_file(table)},
        {"path": "figures/net_equity.png", "sha256": sha256_file(figure)},
    ]}))
    published = tmp_path / "published.csv"
    published.write_bytes(table.read_bytes())
    monkeypatch.setattr(reproduce, "_published_bytes", lambda name: published)
    table.write_text("variant,elapsed_seconds,raw_r2\ndecile,301.0,-0.02\n")
    result = reproduce.verify_published(["results/robustness.csv", "figures/net_equity.png"],
                                        evidence)
    assert result["figures/net_equity.png"]["match"] is True
    assert result["results/robustness.csv"] == {
        "sha256": sha256_file(table), "match": False, "match_except_timing": True}
    table.write_text("variant,elapsed_seconds,raw_r2\ndecile,301.0,-0.03\n")
    assert reproduce.verify_published(["results/robustness.csv"], evidence)[
        "results/robustness.csv"]["match_except_timing"] is False


def test_reproduction_fails_on_any_mismatch_or_unexpected_missing_release():
    robustness_tables = ["results/robustness.csv", "results/cost_sensitivity.csv"]
    receipt = {
        "matches": {"portfolio_artifacts": 37}, "expected": {"portfolio_artifacts": 37},
        "holdout_map_matches": True, "published": {"results/forecast_results.csv": {}},
        "published_matches": 1, "missing_published": robustness_tables,
    }
    assert reproduce.reproduction_passed(receipt, robustness=False)
    assert not reproduce.reproduction_passed(receipt, robustness=True)
    assert reproduce.reproduction_passed({**receipt, "missing_published": []}, robustness=True)
    for change in ({"missing_published": [*robustness_tables, "results/forecast_results.csv"]},
                   {"matches": {"portfolio_artifacts": 36}},
                   {"holdout_map_matches": False},
                   {"published_matches": 0}):
        assert not reproduce.reproduction_passed({**receipt, **change}, robustness=False)


def test_reproduction_refuses_a_used_output_or_absolute_root_before_any_stage(
    tmp_path, monkeypatch,
):
    from momenta import settings as project_settings

    output = tmp_path / "output"
    output.mkdir()
    (output / "previous.txt").write_text("earlier rebuild")
    monkeypatch.setattr(reproduce, "stage_raw_downloads",
                        lambda *args: pytest.fail("staged before the output was checked"))
    with pytest.raises(ValueError, match="new or empty output"):
        reproduce.run_reproduction(tmp_path / "source", output)
    loaded = project_settings.load_settings()
    loaded["data"]["data_root"] = str(tmp_path / "absolute")
    monkeypatch.setattr(reproduce, "load_settings", lambda: loaded)
    with pytest.raises(ValueError, match="default relative data_root"):
        reproduce.run_reproduction(tmp_path / "source", tmp_path / "fresh")
