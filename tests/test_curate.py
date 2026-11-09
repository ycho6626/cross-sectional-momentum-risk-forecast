"""Hand-calculated fixtures for price timing, monthly returns, and diagnostics."""


import csv
import io
import zipfile
from datetime import date
from pathlib import Path

import duckdb
import pytest

import momenta.curate as curate_module
from momenta.curate import _corrections, _flags, _master_actions, _monthly, _yearly_prices
from momenta.settings import load_settings, sql_path


def _correction_file(repo: Path, rows: list[tuple]) -> None:
    path = repo / "config/price_corrections.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["permaticker", "date", "field", "old_value", "new_value", "evidence"])
        writer.writerows(rows)


@pytest.fixture
def price_panel(tmp_path: Path, monkeypatch):
    con = duckdb.connect()
    settings = load_settings()
    settings["data"]["data_root"] = str(tmp_path)
    output = tmp_path / "curated"
    con.execute("""CREATE TABLE calendar AS
        SELECT date::DATE AS date, day_index::BIGINT AS day_index FROM (VALUES
        ('2020-12-30', 0), ('2020-12-31', 1), ('2021-01-04', 2),
        ('2021-01-05', 3), ('2021-01-06', 4)) AS c(date, day_index)""")
    con.execute("""CREATE TABLE master AS SELECT * FROM (VALUES
        (101::BIGINT, 'CURRENT'), (202::BIGINT, 'OLD1')) AS m(permaticker, ticker)""")
    con.execute("""CREATE TABLE raw_prices AS
        SELECT ticker, date::DATE AS date, price AS open, price + 1 AS high,
            price - 1 AS low, price AS close, 1000.0 AS volume,
            price AS closeadj, price AS closeunadj
        FROM (VALUES
            ('CURRENT', '2020-12-30', 100.0), ('CURRENT', '2020-12-31', 110.0),
            ('CURRENT', '2021-01-04', 132.0), ('CURRENT', '2021-01-05', 118.8),
            ('CURRENT', '2021-01-06', 130.68),
            ('OLD1', '2020-12-30', 100.0), ('OLD1', '2021-01-04', 121.0),
            ('OLD1', '2021-01-05', 133.1)) AS p(ticker,date,price)""")
    for year in (2020, 2021):
        path = tmp_path / f"raw/staged/stocks/source_year={year}/part.parquet"
        path.parent.mkdir(parents=True)
        con.execute(
            f"COPY (SELECT * FROM raw_prices WHERE year(date) = {year}) TO {sql_path(path)}"
        )
    monkeypatch.setattr(curate_module, "REPO", tmp_path / "repo")
    yield con, settings, output, tmp_path / "repo"
    con.close()


def test_corrected_cross_year_total_return_and_missing_calendar_day(price_panel):
    con, settings, output, repo = price_panel
    _correction_file(repo, [(101, "2020-12-31", "closeadj", 110, 120, "Independent fixture")])
    assert _corrections(con) == 1
    _yearly_prices(con, settings, output, through=None)
    rows = con.execute("""SELECT date, close, closeadj, daily_return FROM prices
        WHERE permaticker = 101 ORDER BY date""").fetchall()
    assert rows[0][-1] is None
    assert rows[1][1:3] == (110, 120)
    assert [r[-1] for r in rows[1:]] == pytest.approx([0.20, 0.10, -0.10, 0.10])
    historical = con.execute("""SELECT date,daily_return FROM prices
        WHERE permaticker = 202 ORDER BY date""").fetchall()
    # 121/100 − 1 spans the absent December 31 row, so it cannot count as a daily return.
    assert historical[1] == (date(2021, 1, 4), None)
    assert historical[2][1] == pytest.approx(0.10)


def test_correction_cannot_apply_to_a_different_pinned_old_value(price_panel):
    con, settings, output, repo = price_panel
    _correction_file(repo, [(101, "2020-12-31", "closeadj", 109, 120, "Independent fixture")])
    _corrections(con)
    with pytest.raises(ValueError, match="old values do not match"):
        _yearly_prices(con, settings, output, through=None)


def test_correction_without_evidence_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(curate_module, "REPO", tmp_path)
    _correction_file(tmp_path, [(101, "2020-12-31", "closeadj", 110, 120, " ")])
    with duckdb.connect() as con, pytest.raises(ValueError, match="evidence"):
        _corrections(con)


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [("closeadj", 110, "nan"), ("closeadj", 110, "inf"), ("volume", 100, -1)],
)
def test_invalid_correction_cannot_create_nonfinite_price_or_negative_volume(
    tmp_path: Path, monkeypatch, field, old, new
):
    monkeypatch.setattr(curate_module, "REPO", tmp_path)
    _correction_file(tmp_path, [(101, "2020-12-31", field, old, new, "Independent fixture")])
    with duckdb.connect() as con, pytest.raises(ValueError):
        _corrections(con)


@pytest.fixture
def monthly_panel():
    con = duckdb.connect()
    # Nineteen actual US trading days in January 2021; New Year and MLK Day are absent.
    con.execute("""CREATE TABLE calendar AS
        SELECT date::DATE AS date FROM
            generate_series(DATE '2021-01-04', DATE '2021-01-29', INTERVAL 1 DAY) AS c(date)
        WHERE isodow(date) <= 5 AND date <> DATE '2021-01-18'""")
    con.execute("""CREATE TABLE prices AS
        SELECT permaticker::BIGINT AS permaticker,date,
            CASE WHEN date = DATE '2021-01-04' THEN 0.10
                WHEN date = DATE '2021-01-05' THEN -0.10 ELSE 0 END::DOUBLE AS daily_return
        FROM calendar CROSS JOIN range(1, 6) AS p(permaticker)
        WHERE permaticker IN (1, 5) OR date <= DATE '2021-01-05'""")
    con.execute("DELETE FROM prices WHERE permaticker = 3 AND date = DATE '2021-01-04'")
    con.execute("UPDATE prices SET daily_return=NULL WHERE permaticker=5 AND date='2021-01-04'")
    con.execute("""CREATE TABLE delistings AS
        SELECT permaticker::BIGINT AS permaticker, DATE '2021-01-06' AS date,
            terminal_return::DOUBLE AS terminal_return,false AS censored
        FROM (VALUES (2, -0.30), (3, -0.30), (4, -1.00)) AS d(permaticker,terminal_return)""")
    yield con, load_settings()
    con.close()


def test_monthly_compounds_returns_and_credits_terminal_once(monthly_panel, tmp_path: Path):
    con, settings = monthly_panel
    _monthly(con, settings, tmp_path, [2021], date(2021, 1, 29))
    returns = dict(con.execute("SELECT permaticker,monthly_return FROM monthly_returns").fetchall())
    # (1.10 × 0.90) − 1 = −1%; after a 30% terminal loss, 0.99 × 0.70 − 1 = −30.7%.
    assert returns[1] == pytest.approx(-0.01)
    assert returns[2] == pytest.approx(-0.307)
    assert returns[4] == -1
    assert returns[5] is None
    observed = con.execute("""SELECT observed_days,expected_days FROM monthly_returns
        WHERE permaticker=2""").fetchone()
    assert observed == (3, 19)


def test_terminal_month_with_a_missing_precredit_return_stays_missing(
    monthly_panel, tmp_path: Path
):
    con, settings = monthly_panel
    _monthly(con, settings, tmp_path, [2021], date(2021, 1, 29))
    assert con.execute("""SELECT monthly_return FROM monthly_returns
        WHERE permaticker=3""").fetchone()[0] is None


def test_partial_calendar_month_is_not_a_valid_formation_month(monthly_panel, tmp_path: Path):
    con, settings = monthly_panel
    _monthly(con, settings, tmp_path, [2021], date(2021, 1, 5))
    available = con.execute("""SELECT count(*) FROM monthly_returns
        WHERE monthly_return IS NOT NULL""").fetchone()[0]
    assert available == 0


def test_flags_cross_year_reversal_action_suppression_and_prices_are_unchanged(tmp_path: Path):
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE prices AS
            SELECT permaticker::BIGINT AS permaticker,date::DATE AS date,
                close::DOUBLE AS close,low::DOUBLE AS low,high::DOUBLE AS high,
                volume::DOUBLE AS volume,daily_return::DOUBLE AS daily_return
            FROM (VALUES
                (1,'2020-12-30',100,99,101,10,0.0),
                (1,'2020-12-31',160,159,161,10,0.6),
                (1,'2021-01-04',100,99,101,10,-0.375),
                (2,'2020-12-30',100,99,101,10,0.0),
                (2,'2020-12-31',160,159,161,10,0.6),
                (2,'2021-01-04',100,99,101,10,-0.375),
                (3,'2020-12-30',100,99,101,10,0.0),
                (3,'2020-12-31',112,100,110,10,0.12),
                (4,'2020-12-30',100,99,101,10,0.0),
                (4,'2020-12-31',110,109,111,0,0.10)
            ) AS p(permaticker,date,close,low,high,volume,daily_return)""")
        con.execute("""CREATE TABLE actions_flat AS SELECT 2::BIGINT AS permaticker,
            DATE '2021-01-04' AS date,'dividend' AS action""")
        con.execute("""CREATE TABLE universe AS SELECT permaticker::BIGINT AS permaticker,
            DATE '2020-11-30' AS date FROM range(1, 5) AS p(permaticker)""")
        con.execute("""CREATE TABLE calendar AS SELECT DISTINCT date FROM prices
            UNION SELECT DATE '2020-11-30' UNION SELECT DATE '2021-01-29'""")
        before = con.execute("SELECT * FROM prices ORDER BY permaticker,date").fetchall()
        _flags(con, load_settings(), tmp_path, [2020, 2021])
        flags = con.execute("""SELECT permaticker,date,flag,in_universe
            FROM read_parquet(?) ORDER BY permaticker,date,flag""", [
            str(tmp_path / "flags.parquet")
        ]).fetchall()
        assert flags == [
            (1, date(2020, 12, 31), "spike_reversal", True),
            (3, date(2020, 12, 31), "close_outside_range", True),
            (4, date(2020, 12, 31), "price_change_zero_volume", True),
        ]
        assert con.execute("SELECT * FROM prices ORDER BY permaticker,date").fetchall() == before


def test_empty_formation_ends_prior_flag_membership_and_cutoff_limits_report(tmp_path: Path):
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE prices AS SELECT 1::BIGINT AS permaticker,
            date::DATE AS date,110.0 AS close,90.0 AS low,100.0 AS high,
            1.0 AS volume,0.0 AS daily_return
            FROM (VALUES ('2020-12-15'),('2021-01-15'),('2021-01-25')) AS p(date)""")
        con.execute("""CREATE TABLE actions_flat (
            permaticker BIGINT,date DATE,action VARCHAR)""")
        con.execute("""CREATE TABLE universe AS SELECT permaticker::BIGINT AS permaticker,
            date::DATE AS date FROM (VALUES
                (1,'2020-11-30'),(2,'2021-01-29')) AS u(permaticker,date)""")
        con.execute("""CREATE TABLE calendar AS SELECT DISTINCT date FROM prices
            UNION SELECT DATE '2020-11-30' UNION SELECT DATE '2020-12-31'
            UNION SELECT DATE '2021-01-29'""")
        _flags(con, load_settings(), tmp_path, [2020, 2021], end=date(2021, 1, 20))
        result = con.execute("""SELECT date,in_universe FROM read_parquet(?) ORDER BY date""", [
            str(tmp_path / "flags.parquet")
        ]).fetchall()
        assert result == [(date(2020, 12, 15), True), (date(2021, 1, 15), False)]


def test_sep_mapping_retains_reused_suffix_and_distinct_private_acquirer_names(tmp_path: Path):
    settings = load_settings()
    settings["data"]["data_root"] = str(tmp_path)
    tickers = [
        ["table", "permaticker", "ticker", "name", "category", "siccode", "lastupdated"],
        ["SEP", 101, "SAME", "Current", "Domestic Common Stock", 3571, "2021-01-31"],
        ["SEP", 202, "SAME1", "Old", "Domestic Common Stock", 2834, "2021-01-31"],
        ["SF1", 999, "SAME1", "Other Table", "Domestic Common Stock", 2834, "2021-01-31"],
    ]
    actions = [
        ["date", "action", "ticker", "name", "value", "contraticker", "contraname"],
        ["2020-12-31", "acquisitionby", "SAME1", "Old", 100, "N/A", "Private Alpha"],
        ["2020-12-31", "acquisitionby", "SAME1", "Old", 100, "N/A", "Private Beta"],
        ["2020-12-31", "acquisitionstock", "SAME1", "Old", 0.5, "SAME", "Current"],
    ]
    for name, rows in (("tickers", tickers), ("actions", actions)):
        stream = io.StringIO()
        csv.writer(stream).writerows(rows)
        path = tmp_path / f"raw/sharadar/{name}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{name}.csv", stream.getvalue())
    with duckdb.connect() as con:
        _master_actions(con, settings, tmp_path / "curated")
        assert con.execute("SELECT permaticker,ticker FROM master ORDER BY 1").fetchall() == [
            (101, "SAME"), (202, "SAME1")
        ]
        actual = con.execute("""SELECT permaticker,action,contraname,contra_permaticker
            FROM actions_flat ORDER BY action,contraname""").fetchall()
        assert actual == [
            (202, "acquisitionby", "Private Alpha", None),
            (202, "acquisitionby", "Private Beta", None),
            (202, "acquisitionstock", "Current", 101),
        ]
        nested = con.execute("SELECT permaticker,array_length(events) FROM read_parquet(?)", [
            str(tmp_path / "curated/actions.parquet")
        ]).fetchall()
        assert nested == [(202, 3)]
