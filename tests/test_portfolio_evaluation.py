"""Portfolio evaluation keeps a measured target miss visible and pins its provenance."""

from datetime import date
from pathlib import Path

from test_portfolio_report import evaluated, replication  # noqa: F401

from momenta.portfolio_report import load_summary


def test_evaluator_common_samples_worst_calendar_months_and_provenance(evaluated):  # noqa: F811
    settings, _, summary = evaluated
    assert summary["series_built"]
    assert summary["reconciliation_passed"]
    assert not summary["bsc_passed"]  # A measured target miss must remain visible.
    assert not summary["passed"]
    assert all(row["months"] == 2 for row in summary["tables"]["long_history_summary"])
    assert all(
        row["first_date"] == date(2000, 2, 29) for row in summary["tables"]["long_history_summary"]
    )
    worst = summary["tables"]["worst_months"]
    assert len(worst) == 10  # Two months for each of five constructed books.
    assert all(row["date"] == date(2000, 2, 29) for row in worst if row["rank"] == 1)
    assert all(Path(path).with_suffix(".provenance.json").exists() for path in summary["artifacts"])
    assert load_summary(settings)["reconciliation_passed"]
    tables = summary["tables"]
    reconciliation = {row["comparison"]: row for row in tables["reconciliation_all_months"]}
    assert reconciliation["res_monthly_all_months"]["status"] == "DESCRIPTIVE"
