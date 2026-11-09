"""Independent aggregate arithmetic and mutation checks for the released evidence.

The fixture uses hand-chosen losses and probabilities. It contains no market
data, forecasts or statistical estimates copied from the study's output.
"""

import csv
import hashlib
import json

import pytest

from momenta.evidence import EvidenceError, main, verify_release


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(root):
    # Rehashing a corrupted fixture isolates the arithmetic checks from SHA checks.
    entries = [{"path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": "aggregate_table" if path.suffix == ".csv" else "receipt"}
               for path in sorted((root / "results").iterdir()) if path.name != "manifest.json"]
    (root / "results/manifest.json").write_text(json.dumps({
        "schema_version": 1, "artifacts": entries,
    }))


@pytest.fixture
def release(tmp_path):
    forecasts = []
    # The losses, p values and Holm results below are independent hand values.
    losses = {"L1-own": 1., "L1": .8, "L1+L2": .7, "L1+L2+CM": .68,
              "L1+L2+CM+OT": .67, "best+HMMF2": .72}
    for period, n, first, last in [
        ("design", 120, "2005-12-30", "2015-11-30"),
        ("holdout", 127, "2015-12-31", "2026-06-30"),
    ]:
        for series in ("RAW", "RES"):
            for model, loss in losses.items():
                base = {"L1-own": "", "L1": "L1-own", "L1+L2": "L1",
                        "L1+L2+CM": "L1+L2", "L1+L2+CM+OT": "L1+L2+CM",
                        "best+HMMF2": "L1+L2" if series == "RAW" else "L1"}[model]
                baseline = losses[base] if base else loss
                tested = model not in {"L1-own", "L1+L2+CM"}
                forecasts.append({
                    "period": period, "series": series, "model": model, "benchmark": base,
                    "n": n, "first_origin": first, "last_origin": last, "mse": loss,
                    "baseline_mse": baseline,
                    "mse_improvement": baseline - loss if base else "",
                    "r2_oos": (baseline - loss) / baseline if base else "",
                    "cw_pvalue": (.03 if series == "RAW" else .04) if tested else "",
                    "cw_holm_pvalue": .06 if tested else "",
                    "mse_improvement_interval_lower": -.1 if base else "",
                    "mse_improvement_interval_upper": .2 if base else "",
                })
    primary = []
    for row in forecasts:
        if row["period"] == "holdout" and row["model"] == "L1+L2":
            primary.append({"hypothesis": "H3", "cw_n": 127, "dm_n": 127,
                            **{key: value for key, value in row.items() if key != "period"}})
    strategies = []
    for period, first, last, months in [
        ("design", "2006-01-31", "2015-12-31", 120),
        ("holdout", "2016-01-31", "2026-07-31", 127),
    ]:
        for series in ("RAW", "RES"):
            for rule in ("Static", "BSC", "DM-transfer", "BDP-transfer", "L1",
                         "L1+L2", "L1+L2+CM+OT", "best+HMMF2"):
                strategies.append({
                    "period": period, "series": series, "strategy": rule,
                    "first_month": first, "last_month": last, "months": months,
                    "annual_mean": .06, "annual_volatility": .12, "sharpe": .5,
                    "lo_sharpe": .4, "cost_bp": 10, "annual_turnover": 12,
                    "max_drawdown": -.2, "dsr": .3,
                })
    _write_csv(tmp_path / "results/forecast_results.csv", forecasts)
    _write_csv(tmp_path / "results/primary_hypothesis.csv", primary)
    _write_csv(tmp_path / "results/strategy_results.csv", strategies)
    (tmp_path / "results/reproduction_receipt.json").write_text('{"status":"recorded"}\n')
    _manifest(tmp_path)
    return tmp_path


def _mutate(root, name, change):
    path = root / "results" / name
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    change(rows)
    _write_csv(path, rows)
    _manifest(root)


def test_independent_losses_holm_and_ordinary_sharpe_pass(release):
    # Loss1 - loss2 = .8 - .7 = .1; relative gain .1/.8 = .125.
    # Ordered p=.03,.04 gives Holm max(.06,.04)=.06 for the second test.
    # Ordinary Sharpe .06/.12=.5; the distinct Lo value .4 is not re-estimated.
    summary = verify_release(release)
    assert summary == {"artifacts_verified": 4, "tables_verified": 3,
                       "forecast_rows": 24, "strategy_rows": 32,
                       "primary_rows": 2, "statistical_refits": False}


@pytest.mark.parametrize("operation", ["change", "delete"])
def test_changed_or_missing_artifact_is_rejected(release, operation):
    receipt = release / "results/reproduction_receipt.json"
    if operation == "delete":
        receipt.unlink()
    else:
        receipt.write_text('{"status":"altered"}')
    with pytest.raises(EvidenceError, match="artifact|Artifact"):
        verify_release(release)


def test_mse_sign_error_is_detected_even_after_rehashing(release):
    _mutate(release, "forecast_results.csv",
            lambda rows: rows[1].update(mse_improvement=-.2))
    with pytest.raises(EvidenceError, match="MSE improvement"):
        verify_release(release)


def test_wrong_holm_family_is_detected_even_after_rehashing(release):
    _mutate(release, "primary_hypothesis.csv",
            lambda rows: rows[1].update(cw_holm_pvalue=.08))
    with pytest.raises(EvidenceError, match="Holm"):
        verify_release(release)


def test_duplicate_natural_key_with_different_values_is_rejected(release):
    _mutate(release, "forecast_results.csv",
            lambda rows: rows.append({**rows[1], "mse": .81}))
    with pytest.raises(EvidenceError, match="Duplicate result keys"):
        verify_release(release)


def test_unmatched_series_or_changed_sample_is_rejected(release):
    _mutate(release, "primary_hypothesis.csv", lambda rows: rows[1].update(n=128))
    with pytest.raises(EvidenceError, match="sample"):
        verify_release(release)


def test_primary_table_cannot_disagree_with_headline_interval(release):
    _mutate(release, "primary_hypothesis.csv",
            lambda rows: rows[0].update(mse_improvement_interval_lower=-.2))
    with pytest.raises(EvidenceError, match="primary/headline"):
        verify_release(release)


def test_manifest_cannot_escape_release_root(release):
    path = release / "results/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][0]["path"] = "../outside.csv"
    path.write_text(json.dumps(manifest))
    with pytest.raises(EvidenceError, match="artifact path"):
        verify_release(release)


def test_command_fails_closed_and_explains_verification_scope(release, capsys):
    assert main(["verify", "--root", str(release)]) == 0
    output = capsys.readouterr().out
    assert "were not recomputed" in output
    (release / "results/primary_hypothesis.csv").unlink()
    assert main(["verify", "--root", str(release)]) == 1
    assert "Verification failed" in capsys.readouterr().err
