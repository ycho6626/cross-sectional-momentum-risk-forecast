"""Synthetic eligibility fixtures with independent counts for Methodology."""


from datetime import date
from pathlib import Path

import duckdb
import pytest

from momenta.settings import load_settings
from momenta.universe import build_universe


@pytest.fixture
def panel():
    con = duckdb.connect()
    con.execute("""CREATE TABLE calendar AS
        SELECT date::DATE AS date, row_number() OVER (ORDER BY date) AS day_index
        FROM generate_series(DATE '2018-01-01', DATE '2021-03-31', INTERVAL 1 DAY) AS d(date)
        WHERE isodow(date) <= 5""")
    con.execute("""CREATE TABLE master AS
        SELECT permaticker, 'S' || permaticker AS ticker,
            CASE WHEN permaticker = 14 THEN 'ETF' ELSE 'Domestic Common Stock' END AS category,
            CASE WHEN permaticker = 13 THEN NULL ELSE 3571 END::INTEGER AS siccode,
            'OTC' AS exchange, true AS isdelisted
        FROM range(1, 16) AS p(permaticker)""")
    con.execute("""CREATE TABLE prices AS
        SELECT permaticker, ticker, date,
            CASE WHEN permaticker = 4 THEN 4.99 ELSE 10.0 END AS closeunadj,
            10.0::DOUBLE AS close,1000.0::DOUBLE AS volume,
            0.01::DOUBLE AS daily_return
        FROM master CROSS JOIN calendar""")
    con.execute("DELETE FROM prices WHERE permaticker = 7 AND date = '2021-01-29'")
    # Twenty-three missing returns leave 229/252; twenty-two leave exactly 230/252.
    con.execute("""UPDATE prices SET daily_return = NULL
        WHERE (permaticker = 8 AND date IN (
                SELECT date FROM calendar WHERE date <= '2021-01-29' ORDER BY date DESC LIMIT 23))
            OR (permaticker = 15 AND date IN (
                SELECT date FROM calendar WHERE date <= '2021-01-29'
                ORDER BY date DESC LIMIT 22))""")
    con.execute("""CREATE TABLE monthly_returns AS
        SELECT permaticker, last_day(date)::DATE AS date, 0.01::DOUBLE AS monthly_return
        FROM master CROSS JOIN (
            SELECT DISTINCT date_trunc('month', date)::DATE AS date FROM calendar)""")
    con.execute("""DELETE FROM monthly_returns
        WHERE (permaticker = 9 AND date = '2020-02-29')
            OR (permaticker = 10 AND date = '2021-01-31')
            OR (permaticker = 11 AND date = '2020-01-31')
            OR (permaticker = 12 AND date < '2019-03-31')""")
    con.execute("""CREATE TABLE caps AS
        SELECT permaticker,
            CASE WHEN permaticker = 2 THEN DATE '2021-01-22'
                WHEN permaticker = 3 THEN DATE '2021-01-21'
                ELSE DATE '2021-01-29' END AS date,
            CASE WHEN permaticker IN (1, 2) THEN 100.0
                WHEN permaticker = 6 THEN 120.0
                WHEN permaticker = 12 THEN 110.0
                WHEN permaticker = 13 THEN 90.0
                WHEN permaticker = 15 THEN 80.0
                WHEN permaticker = 10 THEN 50.0
                WHEN permaticker = 11 THEN 40.0
                ELSE 1000.0 END::DOUBLE AS marketcap
        FROM master""")
    con.execute("""CREATE TABLE actions_flat (
        permaticker BIGINT, date DATE, action VARCHAR, value DOUBLE, contraticker VARCHAR)""")
    con.execute("""INSERT INTO actions_flat VALUES
        (5, '2021-02-01', 'sicchangefrom', 6770, NULL),
        (5, '2021-02-01', 'sicchangeto', 3571, NULL),
        (6, '2021-01-29', 'sicchangefrom', 6770, NULL),
        (6, '2021-01-29', 'sicchangeto', 3571, NULL)""")
    settings = load_settings()
    settings["data"]["universe_start"] = "2021-01-01"
    settings["universe"]["n_universe"] = 5
    settings["universe"]["cap_repair"] = False
    # Real master snapshots carry a date; it must never replace the observation date.
    con.execute("ALTER TABLE master ADD COLUMN date DATE DEFAULT DATE '2026-10-06'")
    yield con, settings
    con.close()


def test_as_of_membership_ties_cap_staleness_skip_month_and_res_count(panel, tmp_path: Path):
    con, settings = panel
    summary = build_universe(con, settings, tmp_path, date(2021, 1, 29))
    assert summary == {"months": 1, "universe_rows": 5, "shortfall_months": 0}
    members = con.execute("""SELECT permaticker, marketcap, res_eligible, rank
        FROM read_parquet(?) ORDER BY rank""", [str(tmp_path / "universe.parquet")]).fetchall()
    assert members == [
        (6, 120, True, 1), (12, 110, False, 2), (1, 100, True, 3),
        (2, 100, True, 4), (13, 90, True, 5),
    ]
    # All nine candidates have the same dollar volume. ID 3 has no usable cap,
    # but still precedes ID 6 in the candidate ranking; nonmembers count too.
    assert con.execute("""SELECT permaticker,dollar_volume_rank
        FROM read_parquet(?) ORDER BY rank""", [
        str(tmp_path / "universe.parquet")
    ]).fetchall() == [(6, 4), (12, 7), (1, 1), (2, 2), (13, 8)]
    audit = con.execute("""SELECT date,candidates,with_marketcap,marketcap_coverage,
        universe_size,res_eligible,shortfall FROM read_parquet(?)""", [
        str(tmp_path / "universe_audit.parquet")
    ]).fetchone()
    assert audit == (date(2021, 1, 29), 9, 8, 8 / 9, 5, 4, 0)
    sic = con.execute("""SELECT permaticker, siccode, changes_rolled_back
        FROM read_parquet(?) WHERE permaticker IN (5, 6, 13) ORDER BY permaticker""", [
        str(tmp_path / "pit_sic.parquet")
    ]).fetchall()
    assert sic == [(5, 6770, 1), (6, 3571, 0), (13, None, 0)]


def test_shortfall_keeps_every_qualifying_name_and_reports_res_subset(panel, tmp_path: Path):
    con, settings = panel
    settings["universe"]["n_universe"] = 20
    build_universe(con, settings, tmp_path, date(2021, 1, 29))
    members = con.execute("""SELECT permaticker FROM read_parquet(?) ORDER BY permaticker""", [
        str(tmp_path / "universe.parquet")
    ]).fetchall()
    assert members == [(1,), (2,), (6,), (10,), (11,), (12,), (13,), (15,)]
    assert con.execute("""SELECT universe_size, res_eligible, shortfall FROM read_parquet(?)""", [
        str(tmp_path / "universe_audit.parquet")
    ]).fetchone() == (8, 7, 12)


def test_future_prices_returns_caps_and_paired_sic_change_cannot_change_membership(
    panel, tmp_path: Path
):
    con, settings = panel
    cutoff = date(2021, 1, 29)
    build_universe(con, settings, tmp_path / "before", cutoff)
    before = con.execute("""SELECT * FROM read_parquet(?) ORDER BY rank""", [
        str(tmp_path / "before/universe.parquet")
    ]).fetchall()
    con.execute(
        "UPDATE prices SET closeunadj = 0.01, daily_return = -0.99 WHERE date > ?", [cutoff]
    )
    con.execute("UPDATE monthly_returns SET monthly_return = NULL WHERE date >= '2021-02-01'")
    con.execute("INSERT INTO caps SELECT permaticker, DATE '2021-02-01', 999999 FROM master")
    con.execute("UPDATE master SET exchange = 'NYSE', isdelisted = false")
    con.execute("UPDATE master SET siccode = 6770 WHERE permaticker = 1")
    con.execute("""INSERT INTO actions_flat VALUES
        (1, '2021-02-01', 'sicchangefrom', 3571, NULL),
        (1, '2021-02-01', 'sicchangeto', 6770, NULL)""")
    build_universe(con, settings, tmp_path / "after", cutoff)
    after = con.execute("""SELECT * FROM read_parquet(?) ORDER BY rank""", [
        str(tmp_path / "after/universe.parquet")
    ]).fetchall()
    assert after == before


def test_midmonth_through_does_not_create_a_false_month_end(panel, tmp_path: Path):
    con, settings = panel
    with pytest.raises(ValueError, match="No complete FF3 month-ends"):
        build_universe(con, settings, tmp_path, date(2021, 1, 15))


def test_sic_audit_counts_unknown_master_and_effective_pairs(panel, tmp_path: Path):
    con, settings = panel
    build_universe(con, settings, tmp_path, date(2021, 1, 29))
    audit = con.execute("SELECT * FROM read_parquet(?)", [
        str(tmp_path / "sic_audit.parquet")
    ]).fetchone()
    assert audit == (2021, 15, 14, 14 / 15, 1, 2, 2, 0)
    issues = con.execute("SELECT * FROM read_parquet(?)", [
        str(tmp_path / "sic_issues.parquet")
    ]).fetchall()
    assert issues == [(13, None, "missing_current_sic")]


def test_pinned_repairs_accept_only_approved_missing_caps_and_can_be_disabled(
    panel, tmp_path: Path
):
    con, settings = panel
    cutoff = date(2021, 1, 29)
    # S10 now has no observed cap, so its rejected repair would rank first if admitted.
    con.execute("DELETE FROM caps WHERE permaticker=10")
    con.execute("""CREATE TABLE cap_repairs (
        permaticker BIGINT,date DATE,t0 DATE,distance_trading_days BIGINT,
        cap_source VARCHAR,anchor_marketcap DOUBLE,anchor_close DOUBLE,close DOUBLE,
        repaired_marketcap DOUBLE,median_dollar_volume DOUBLE,implied_turnover DOUBLE,
        accepted BOOLEAN,exception VARCHAR)""")
    con.execute("""INSERT INTO cap_repairs VALUES
        (3,'2021-01-29','2021-01-21',6,'carried_forward',200,10,10,200,
            200000,0.001,true,NULL),
        (10,'2021-01-29','2021-02-26',20,'backfilled',2000,10,10,2000,
            400000000,0.20,false,'turnover_above_limit'),
        (1,'2021-01-29','2021-02-26',20,'backfilled',9999,10,10,9999,
            9999000,0.001,true,NULL)""")
    build_universe(con, settings, tmp_path / "disabled", cutoff)
    old_fields = con.execute("""SELECT permaticker,date,marketcap,res_eligible,rank
        FROM read_parquet(?) ORDER BY rank""", [
        str(tmp_path / "disabled/universe.parquet")
    ]).fetchall()
    assert old_fields == [
        (6, cutoff, 120, True, 1), (12, cutoff, 110, False, 2),
        (1, cutoff, 100, True, 3), (2, cutoff, 100, True, 4), (13, cutoff, 90, True, 5),
    ]

    settings["universe"]["cap_repair"] = True
    build_universe(con, settings, tmp_path / "enabled", cutoff)
    repaired = con.execute("""SELECT permaticker,date,marketcap,res_eligible,rank,cap_source
        FROM read_parquet(?) ORDER BY rank""", [
        str(tmp_path / "enabled/universe.parquet")
    ]).fetchall()
    assert repaired == [
        (3, cutoff, 200, True, 1, "carried_forward"),
        (6, cutoff, 120, True, 2, "observed"),
        (12, cutoff, 110, False, 3, "observed"),
        (1, cutoff, 100, True, 4, "observed"),
        (2, cutoff, 100, True, 5, "observed"),
    ]


@pytest.mark.parametrize("repair_enabled", [False, True])
def test_observed_turnover_floor_is_inclusive_and_never_falls_back_to_repair(
    panel, tmp_path: Path, repair_enabled: bool
):
    con, settings = panel
    cutoff = date(2021, 1, 29)
    settings["universe"]["n_universe"] = 20
    settings["universe"]["cap_repair"] = repair_enabled
    # At a $100m observed cap: $2,999/day fails; $3,000/day is exactly 0.003%.
    con.execute("UPDATE prices SET volume=299.9 WHERE permaticker=1")
    con.execute("UPDATE prices SET volume=300.0 WHERE permaticker=2")
    # Both alternate anchors would yield plausible repairs if the present cap
    # were wrongly treated as absent: $1m -> 0.2999%; $2m -> 0.14995%.
    con.execute("""INSERT INTO caps VALUES
        (1,'2021-01-20',1), (1,'2021-02-01',2)""")
    generated = tmp_path / "generated"
    build_universe(con, settings, generated, cutoff, build_repairs=True)
    members = con.execute("""SELECT permaticker FROM read_parquet(?) ORDER BY permaticker""", [
        str(generated / "universe.parquet")
    ]).fetchall()
    assert members == [(2,), (6,), (10,), (11,), (12,), (13,), (15,)]
    observed = con.execute("""SELECT marketcap,cap_source FROM read_parquet(?)
        WHERE permaticker=2""", [str(generated / "universe.parquet")]).fetchone()
    assert observed == (100, "observed")
    assert con.execute(
        "SELECT count(*) FROM cap_repairs WHERE permaticker IN (1,2)"
    ).fetchone() == (0,)
    audit = con.execute("""SELECT candidates,with_marketcap,with_usable_observed_marketcap,
        unusable_observed_caps,marketcap_coverage,usable_observed_marketcap_coverage
        FROM read_parquet(?)""", [str(generated / "universe_audit.parquet")]).fetchone()
    assert audit == (9, 8, 7, 1, 8 / 9, 7 / 9)
    exceptions = con.execute("""SELECT permaticker,date,marketcap,median_dollar_volume,
        implied_turnover,exception FROM read_parquet(?)""", [
        str(generated / "observed_cap_exceptions.parquet")
    ]).fetchall()
    assert len(exceptions) == 1
    assert exceptions[0][:4] == (1, cutoff, 100, 2999)
    assert exceptions[0][4] == pytest.approx(0.00002999)
    assert exceptions[0][5] == "observed_turnover_below_limit"

    # A stale/malformed pinned row must not override the unusable observed cap.
    con.execute("""INSERT INTO cap_repairs VALUES
        (1,'2021-01-29','2021-01-20',7,'carried_forward',1,10,10,1,
            2999,0.002999,true,NULL)""")
    fixed = tmp_path / "fixed"
    build_universe(con, settings, fixed, cutoff)
    assert con.execute("""SELECT permaticker FROM read_parquet(?) ORDER BY permaticker""", [
        str(fixed / "universe.parquet")
    ]).fetchall() == members
