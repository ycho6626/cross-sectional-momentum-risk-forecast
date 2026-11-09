"""Hand-computed cap and turnover examples for the market-cap repair rule."""


from pathlib import Path

import duckdb
import pytest

from momenta.cap_repair import build_cap_repairs
from momenta.settings import load_settings


@pytest.fixture
def repair_panel():
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE calendar AS
            SELECT date::DATE AS date, row_number() OVER (ORDER BY date) - 1 AS day_index
            FROM generate_series(DATE '2018-01-01', DATE '2021-12-31', INTERVAL 1 DAY) d(date)
            WHERE isodow(date) <= 5""")
        con.execute("""CREATE TABLE _m1_all_candidates (
            permaticker BIGINT, date DATE, marketcap DOUBLE, res_eligible BOOLEAN,
            close DOUBLE, median_dollar_volume DOUBLE)""")
        con.execute("""CREATE TABLE prices (
            permaticker BIGINT, date DATE, close DOUBLE, closeunadj DOUBLE, closeadj DOUBLE)""")
        con.execute("""CREATE TABLE caps (permaticker BIGINT, date DATE, marketcap DOUBLE)""")
        yield con


def test_backfill_across_two_for_one_split_uses_split_adjusted_close(repair_panel, tmp_path: Path):
    con = repair_panel
    # m: 1m shares × $100 unadjusted = $100m; split-adjusted close is $50.
    # After 2:1 split: 2m shares × $60 = $120m. 120 × 50 / 60 = $100m.
    con.execute("""INSERT INTO _m1_all_candidates
        VALUES (1, '2020-01-31', NULL, true, 50, 300000)""")
    con.execute("""INSERT INTO prices VALUES
        (1, '2020-01-31', 50, 100, 45), (1, '2020-02-03', 60, 60, 57)""")
    con.execute("INSERT INTO caps VALUES (1, '2020-02-03', 120)")
    build_cap_repairs(con, load_settings(), tmp_path)
    row = con.execute("""SELECT repaired_marketcap, implied_turnover, cap_source,
        distance_trading_days, accepted, exception FROM cap_repairs""").fetchone()
    assert row == (100, 0.003, "backfilled", 1, True, None)


def test_carry_forward_wins_when_later_cap_also_exists(repair_panel, tmp_path: Path):
    con = repair_panel
    con.execute("""INSERT INTO _m1_all_candidates
        VALUES (1, '2020-01-31', NULL, true, 50, 300000)""")
    con.execute("""INSERT INTO prices VALUES
        (1, '2020-01-20', 40, 40, 40), (1, '2020-02-03', 50, 50, 50)""")
    con.execute("INSERT INTO caps VALUES (1, '2020-01-20', 80), (1, '2020-02-03', 200)")
    build_cap_repairs(con, load_settings(), tmp_path)
    assert con.execute("""SELECT repaired_marketcap, cap_source, distance_trading_days,
        accepted FROM cap_repairs""").fetchone() == (100, "carried_forward", 9, True)


def test_turnover_guard_accepts_both_limits_and_rejects_just_outside(repair_panel, tmp_path: Path):
    con = repair_panel
    con.execute("""INSERT INTO _m1_all_candidates VALUES
        (1, '2020-01-31', NULL, true, 50, 10000001),
        (2, '2020-01-31', NULL, true, 50, 10000000),
        (3, '2020-01-31', NULL, true, 50, 49999),
        (4, '2020-01-31', NULL, true, 50, 50000)""")
    con.execute("""INSERT INTO prices VALUES
        (1, '2020-02-03', 60, 60, 60), (2, '2020-02-03', 60, 60, 60),
        (3, '2020-02-03', 60, 60, 60), (4, '2020-02-03', 60, 60, 60)""")
    con.execute("""INSERT INTO caps VALUES
        (1, '2020-02-03', 120), (2, '2020-02-03', 120),
        (3, '2020-02-03', 120), (4, '2020-02-03', 120)""")
    build_cap_repairs(con, load_settings(), tmp_path)
    assert con.execute("""SELECT permaticker, repaired_marketcap, accepted, exception
        FROM cap_repairs ORDER BY 1""").fetchall() == [
        (1, 100, False, "turnover_above_limit"), (2, 100, True, None),
        (3, 100, False, "turnover_below_limit"), (4, 100, True, None),
    ]
    turnover = con.execute(
        "SELECT implied_turnover FROM cap_repairs WHERE permaticker=1"
    ).fetchone()[0]
    assert turnover > 0.10
    # $50,000 / $100m = 0.0005 and $10m / $100m = 0.10; equality is accepted.
    assert con.execute("""SELECT implied_turnover FROM cap_repairs
        WHERE permaticker IN (2, 4) ORDER BY permaticker""").fetchall() == [(0.10,), (0.0005,)]
    assert con.execute("""SELECT permaticker,exception FROM read_parquet(?)
        ORDER BY permaticker""", [
        str(tmp_path / "cap_repair_exceptions.parquet")
    ]).fetchall() == [(1, "turnover_above_limit"), (3, "turnover_below_limit")]


def test_252_day_boundary_and_unrepairable_rows(repair_panel, tmp_path: Path):
    con = repair_panel
    con.execute("""INSERT INTO _m1_all_candidates VALUES
        (1, '2020-01-31', NULL, true, 50, 300000),
        (2, '2020-01-31', NULL, true, 50, 300000),
        (3, '2020-01-31', NULL, true, 50, 300000),
        (4, '2020-01-31', NULL, true, 50, 300000)""")
    con.execute("""INSERT INTO caps
        SELECT p, c.date, 100 FROM (VALUES (1,252), (2,253), (4,253)) x(p, distance)
        JOIN calendar c ON c.day_index =
            (SELECT day_index FROM calendar WHERE date='2020-01-31') - distance""")
    con.execute("INSERT INTO caps VALUES (2, '2020-02-03', 200)")
    con.execute("INSERT INTO prices SELECT permaticker, date, 50, 50, 50 FROM caps")
    build_cap_repairs(con, load_settings(), tmp_path)
    assert con.execute("""SELECT permaticker, repaired_marketcap, cap_source,
        distance_trading_days, exception FROM cap_repairs ORDER BY 1""").fetchall() == [
        (1, 100, "carried_forward", 252, None),
        (2, 200, "backfilled", 1, None),
        (3, None, None, None, "no_daily_cap"),
        (4, None, None, None, "no_qualifying_anchor"),
    ]
