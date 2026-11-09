"""Hand-calculated data-integrity and terminal-exposure audit examples."""


from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from momenta.checks import (
    _cap_checks,
    _cap_rank_checks,
    _cap_tables,
    _coverage,
    _delisting_value,
    _observed_cap_checks,
    _yearly_prices,
)


def test_price_audit_detects_same_date_duplicates_and_invalid_price(tmp_path: Path):
    partition = tmp_path / "prices" / "year=1998"
    partition.mkdir(parents=True)
    pl.DataFrame(
        {
            "permaticker": [1, 1],
            "date": [date(1998, 1, 2), date(1998, 1, 2)],
            "close": [10.0, 0.0],
            "closeadj": [10.0, 10.0],
            "closeunadj": [10.0, 10.0],
        }
    ).write_parquet(partition / "part.parquet")
    with duckdb.connect() as con:
        result = _yearly_prices(con, tmp_path)[0]
    assert result["price_rows"] == 2
    assert result["stocks_with_prices"] == 1
    assert result["duplicate_rows"] == 1
    assert result["invalid_price_rows"] == 1


def test_terminal_exposure_uses_strictly_previous_formation_and_zero_event_months():
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE universe AS SELECT * FROM (VALUES
            (1,DATE '1998-12-31',20.0),(2,DATE '1998-12-31',80.0),
            (1,DATE '1999-01-29',60.0),(2,DATE '1999-01-29',40.0),
            (1,DATE '1999-02-26',70.0),(2,DATE '1999-02-26',30.0)
        ) t(permaticker,date,marketcap)""")
        con.execute("CREATE TABLE universe_audit AS SELECT DISTINCT date FROM universe")
        con.execute("""CREATE TABLE delistings AS SELECT * FROM (VALUES
            (1,DATE '1999-01-29','performance_delisting',false),
            (2,DATE '1999-02-15','performance_delisting',false),
            (3,DATE '1999-02-15','unexplained_end',false),
            (2,DATE '1999-02-26','performance_delisting',true)
        ) t(permaticker,last_date,event_class,censored)""")
        result = {row["event_class"]: row for row in _delisting_value(con)}
    performance = result["performance_delisting"]
    assert performance["events"] == 2
    assert performance["universe_events"] == 2
    assert performance["cumulative_formation_weight"] == pytest.approx(0.2 + 0.4)
    assert performance["mean_monthly_weight_pct"] == pytest.approx(20.0)
    assert performance["max_monthly_weight_pct"] == pytest.approx(40.0)
    assert result["unexplained_end"]["events"] == 1
    assert result["unexplained_end"]["universe_events"] == 0
    assert result["unexplained_end"]["mean_monthly_weight_pct"] == 0


def test_yearly_coverage_counts_distinct_members_and_dates_events_by_last_trade(tmp_path: Path):
    with duckdb.connect(
        config={
            "memory_limit": "3GB",
            "threads": 4,
            "temp_directory": str(tmp_path),
        }
    ) as con:
        con.execute("""CREATE TABLE universe AS SELECT * FROM (VALUES
            (1,DATE '1999-01-29',true),(2,DATE '1999-01-29',false),
            (1,DATE '1999-02-26',true),(2,DATE '1999-02-26',true),
            (3,DATE '1999-02-26',false),
            (1,DATE '2000-01-31',false),(4,DATE '2000-01-31',true)
        ) t(permaticker,date,res_eligible)""")
        con.execute("""CREATE TABLE universe_audit AS
            SELECT date,universe_size,res_eligible,coverage::DOUBLE marketcap_coverage
            FROM (VALUES (DATE '1999-01-29',2,1,1.0),
                         (DATE '1999-02-26',3,2,0.995),
                         (DATE '2000-01-31',2,1,0.999))
            t(date,universe_size,res_eligible,coverage)""")
        con.execute("""CREATE TABLE flags AS SELECT * FROM (VALUES
            (1,DATE '1998-12-01','close_outside_range',false),
            (1,DATE '1999-02-01','close_outside_range',true),
            (1,DATE '1999-02-01','price_change_zero_volume',true),
            (9,DATE '1999-02-01','spike_reversal',false),
            (9,DATE '2000-02-01','spike_reversal',false)
        ) t(permaticker,date,flag,in_universe)""")
        con.execute("""CREATE TABLE delistings AS SELECT * FROM (VALUES
            (2,DATE '1999-06-01',DATE '1999-06-02','acquisition_with_terms',false),
            (3,DATE '1999-12-31',DATE '2000-01-03','performance_delisting',false),
            (4,DATE '1999-12-31',NULL,'voluntary_delisting',true),
            (5,DATE '2000-03-01',DATE '2000-03-02','unexplained_end',false)
        ) t(permaticker,last_date,date,event_class,censored)""")
        prices = [
            {"year": 1998, "stocks_with_prices": 3},
            {"year": 1999, "stocks_with_prices": 7},
            {"year": 2000, "stocks_with_prices": 4},
        ]
        result = {row["year"]: row for row in _coverage(con, prices)}

    # Five membership rows in 1999 contain three distinct names; the three
    # RES membership rows contain only names 1 and 2. Each flag is counted once.
    assert result[1999] == {
        "year": 1999,
        "stocks_with_prices": 7,
        "eligible_stocks": 3,
        "res_eligible_stocks": 2,
        "month_ends": 2,
        "min_universe": 2,
        "max_universe": 3,
        "min_res_eligible": 1,
        "max_res_eligible": 2,
        "min_marketcap_coverage": pytest.approx(0.995),
        "flag_rows": 3,
        "universe_flag_rows": 2,
        "acquisition_with_terms": 1,
        "election": 0,
        "acquisition_without_terms": 0,
        "voluntary_delisting": 0,
        "performance_delisting": 1,
        "unexplained_end": 0,
    }
    assert result[1998]["stocks_with_prices"] == 3
    assert result[1998]["eligible_stocks"] == 0
    assert result[1998]["res_eligible_stocks"] == 0
    assert result[1998]["month_ends"] == 0
    assert result[1998]["min_marketcap_coverage"] is None
    assert result[1998]["flag_rows"] == 1
    assert result[1998]["universe_flag_rows"] == 0
    assert result[2000]["stocks_with_prices"] == 4
    assert result[2000]["eligible_stocks"] == 2
    assert result[2000]["res_eligible_stocks"] == 1
    assert result[2000]["month_ends"] == 1
    assert result[2000]["flag_rows"] == 1
    assert result[2000]["universe_flag_rows"] == 0
    assert result[2000]["performance_delisting"] == 0
    assert result[2000]["unexplained_end"] == 1


@pytest.fixture
def cap_audit_data(tmp_path: Path):
    settings = {
        "universe": {
            "cap_gate_max_missing": 5,
            "cap_gate_dollar_volume_rank": 1000,
            "cap_observed_min_turnover": 0.00003,
            "cap_repair_min_turnover": 0.0005,
            "cap_repair_max_turnover": 0.10,
            "rank1_max_dollar_volume_rank": 100,
            "top10_max_dollar_volume_rank": 1000,
            "cap_carry_forward_days": 252,
            "cap_turnover_window_days": 21,
            "marketcap_lookback_days": 5,
        }
    }
    with duckdb.connect(
        config={
            "memory_limit": "3GB",
            "threads": 4,
            "temp_directory": str(tmp_path),
        }
    ) as con:
        con.execute("""CREATE TABLE cap_repairs(
            permaticker BIGINT,date DATE,t0 DATE,distance_trading_days BIGINT,
            cap_source VARCHAR,anchor_marketcap DOUBLE,anchor_close DOUBLE,close DOUBLE,
            repaired_marketcap DOUBLE,median_dollar_volume DOUBLE,implied_turnover DOUBLE,
            accepted BOOLEAN,exception VARCHAR)""")
        con.execute("""INSERT INTO cap_repairs VALUES
            (1,'1999-01-29','1999-01-01',20,'carried_forward',100,100,50,50,1000000,.02,true,NULL),
            (2,'1999-01-29','1999-02-12',10,'backfilled',40,50,100,80,2000000,.025,true,NULL),
            (3,'1999-01-29','1999-02-12',10,'backfilled',20,10,10,20,4000000,.2,
                false,'turnover_above_limit'),
            (4,'1999-01-29',NULL,NULL,NULL,NULL,NULL,10,NULL,1000,NULL,false,'no_daily_cap'),
            (1,'1999-02-26','1999-01-01',40,'carried_forward',100,100,50,50,1000000,.02,true,NULL),
            (5,'1999-02-26','1999-04-23',40,'backfilled',100,10,10,100,10000000,.1,true,NULL),
            (6,'1999-02-26','1999-04-09',30,'backfilled',100,NULL,10,NULL,1000,NULL,
                false,'missing_anchor_price')""")
        con.execute("""CREATE TABLE cap_repair_exceptions AS
            SELECT * FROM cap_repairs WHERE NOT accepted""")
        con.execute("""CREATE TABLE observed_cap_exceptions(
            permaticker BIGINT,date DATE,marketcap DOUBLE,median_dollar_volume DOUBLE,
            implied_turnover DOUBLE,exception VARCHAR)""")
        con.execute("""INSERT INTO observed_cap_exceptions VALUES
            (8,'1999-01-29',400,4000,.00001,'observed_turnover_below_limit'),
            (8,'1999-02-26',500,5000,.00001,'observed_turnover_below_limit'),
            (9,'1999-02-26',200,NULL,NULL,'observed_invalid_turnover')""")
        con.execute("""CREATE TABLE universe AS SELECT permaticker,date,
            marketcap::DOUBLE marketcap,cap_source,
            CASE permaticker WHEN 2 THEN 7 WHEN 5 THEN 31 WHEN 1 THEN 45 ELSE 492 END
                AS dollar_volume_rank,
            row_number() OVER (PARTITION BY date ORDER BY marketcap DESC,permaticker) AS rank
            FROM (VALUES
            (1,DATE '1999-01-29',50,'carried_forward'),
            (2,DATE '1999-01-29',80,'backfilled'),(7,DATE '1999-01-29',70,'observed'),
            (1,DATE '1999-02-26',50,'carried_forward'),
            (5,DATE '1999-02-26',100,'backfilled'),(7,DATE '1999-02-26',50,'observed')
            ) t(permaticker,date,marketcap,cap_source)""")
        con.execute("""CREATE TABLE universe_audit AS SELECT date,candidates,with_marketcap,
            with_usable_observed_marketcap,unusable_observed_caps,
            with_repaired_marketcap,with_marketcap::DOUBLE/candidates marketcap_coverage,
            with_usable_observed_marketcap::DOUBLE/candidates usable_observed_marketcap_coverage,
            with_repaired_marketcap::DOUBLE/candidates repaired_marketcap_coverage,
            high_dollar_volume_missing_before,high_dollar_volume_missing_after,
            repaired_members,repaired_cap_share::DOUBLE repaired_cap_share FROM (VALUES
            (DATE '1999-01-29',7,3,2,4,1,2,1,2,.65),
            (DATE '1999-02-26',7,4,2,4,2,2,0,2,.75))
            t(date,candidates,with_marketcap,with_usable_observed_marketcap,
              with_repaired_marketcap,unusable_observed_caps,
              high_dollar_volume_missing_before,high_dollar_volume_missing_after,
              repaired_members,repaired_cap_share)""")
        con.execute("""CREATE TABLE master AS SELECT * FROM (VALUES
            (1,'ALFA','Alpha Company'),(2,'BETA','Beta Company'),
            (5,'GAMMA','Gamma Company'),(7,'DELTA','Delta Company')) t(permaticker,ticker,name)""")
        yield con, settings


def test_cap_audit_accepts_consistent_repairs_and_reports_all_candidate_coverage(cap_audit_data):
    con, settings = cap_audit_data
    checks = _cap_checks(con, settings)
    checks += _observed_cap_checks(con, settings)
    assert all(row["pass"] for row in checks)
    tables = _cap_tables(con)
    year = tables["cap_coverage_by_year"][0]
    # Each month has seven candidates. Raw caps cover three then four; usable
    # observed caps cover two in each month, and accepted repairs add two each.
    assert year["min_observed_coverage"] == pytest.approx(3 / 7)
    assert year["min_usable_observed_coverage"] == pytest.approx(2 / 7)
    assert year["min_repaired_coverage"] == pytest.approx(4 / 7)
    assert year["unusable_observed_stock_months"] == 3
    assert year["max_unusable_observed_caps"] == 2
    assert year["max_high_volume_missing_before"] == 2
    assert year["max_high_volume_missing_after"] == 1
    assert year["mean_repaired_members"] == 2
    assert year["mean_repaired_cap_share_pct"] == pytest.approx(70)
    assert year["max_repaired_cap_share_pct"] == pytest.approx(75)
    status = {(row["cap_source"], row["status"]): row for row in tables["cap_repair_status"]}
    assert status[("carried_forward", "accepted")]["stock_months"] == 2
    assert status[("carried_forward", "accepted")]["stocks"] == 1
    assert status[("backfilled", "accepted")]["stock_months"] == 2
    assert status[("backfilled", "turnover_above_limit")]["max_implied_turnover_pct"] == 20
    assert status[("none", "no_daily_cap")]["stocks"] == 1
    distances = {row["accepted"]: row for row in tables["cap_backfill_distances"]}
    # Accepted distances are [10,40], so interpolated quartiles are 17.5 and 32.5.
    assert distances[True] == {
        "accepted": True,
        "stock_months": 2,
        "stocks": 2,
        "min_trading_days": 10,
        "p25_trading_days": 17.5,
        "median_trading_days": 25,
        "p75_trading_days": 32.5,
        "p95_trading_days": pytest.approx(38.5),
        "max_trading_days": 40,
    }
    assert distances[False]["stock_months"] == 2
    assert distances[False]["median_trading_days"] == 20
    observed = {row["exception"]: row for row in tables["observed_cap_exception_status"]}
    assert observed["observed_turnover_below_limit"]["stock_months"] == 2
    assert observed["observed_turnover_below_limit"]["stocks"] == 1
    assert observed["observed_turnover_below_limit"]["median_implied_turnover_pct"] == 0.001
    assert observed["observed_invalid_turnover"]["stock_months"] == 1


@pytest.mark.parametrize(("missing", "passes"), [(5, True), (6, False)])
def test_high_dollar_volume_gate_has_inclusive_five_name_boundary(cap_audit_data, missing, passes):
    con, settings = cap_audit_data
    con.execute(
        """UPDATE universe_audit SET high_dollar_volume_missing_before=6,
        high_dollar_volume_missing_after=? WHERE date=DATE '1999-01-29'""",
        [missing],
    )
    gate = next(
        row for row in _cap_checks(con, settings) if row["check"] == "high_dollar_volume_cap_gate"
    )
    assert gate["pass"] is passes
    assert gate["value"] == missing


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            "UPDATE cap_repairs SET implied_turnover=median_dollar_volume/repaired_marketcap "
            "WHERE permaticker=2",
            "cap_repair_arithmetic",
        ),
        (
            "UPDATE cap_repairs SET t0=DATE '1999-02-12' WHERE permaticker=1 "
            "AND date=DATE '1999-01-29'",
            "cap_repair_terms",
        ),
        (
            "UPDATE universe SET permaticker=3,marketcap=20 WHERE permaticker=2",
            "selected_cap_repairs",
        ),
        ("DELETE FROM cap_repair_exceptions WHERE permaticker=4", "cap_repair_exception_table"),
        ("UPDATE universe_audit SET repaired_cap_share=.5", "cap_repair_monthly_audit"),
    ],
)
def test_cap_audit_rejects_corrupt_terms_selection_and_summary(
    cap_audit_data, mutation, failed_check
):
    con, settings = cap_audit_data
    con.execute(mutation)
    checks = {row["check"]: row for row in _cap_checks(con, settings)}
    assert checks[failed_check]["pass"] is False
    assert checks[failed_check]["value"] > 0


@pytest.mark.parametrize(("turnover", "passes"), [(0.0004999, False), (0.0005, True)])
def test_accepted_repair_turnover_floor_is_inclusive(cap_audit_data, turnover, passes):
    con, settings = cap_audit_data
    con.execute(
        """UPDATE cap_repairs SET implied_turnover=?,
        median_dollar_volume=?*repaired_marketcap*1e6 WHERE permaticker=2""",
        [turnover, turnover],
    )
    checks = {row["check"]: row for row in _cap_checks(con, settings)}
    assert checks["cap_repair_terms"]["pass"] is passes
    assert checks["cap_repair_arithmetic"]["pass"] is True


@pytest.mark.parametrize(
    ("turnover", "exception", "passes"),
    [
        (0.00002999, "observed_turnover_below_limit", True),
        (0.00003, "observed_turnover_below_limit", False),
        (-0.01, "observed_invalid_turnover", True),
        (-0.01, "observed_turnover_below_limit", False),
    ],
)
def test_observed_exceptions_preserve_floor_boundary_and_invalid_reason(
    cap_audit_data,
    turnover,
    exception,
    passes,
):
    con, settings = cap_audit_data
    con.execute(
        """UPDATE observed_cap_exceptions SET implied_turnover=?,
        median_dollar_volume=?*marketcap*1e6,exception=?
        WHERE permaticker=8 AND date=DATE '1999-01-29'""",
        [turnover, turnover, exception],
    )
    checks = {row["check"]: row for row in _observed_cap_checks(con, settings)}
    assert checks["observed_cap_exception_terms"]["pass"] is passes


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE universe SET permaticker=8 WHERE permaticker=7 AND date=DATE '1999-01-29'",
        "INSERT INTO cap_repairs SELECT * REPLACE (8 AS permaticker) FROM cap_repairs "
        "WHERE permaticker=1 AND date=DATE '1999-01-29'",
    ],
)
def test_unusable_observed_caps_are_neither_selected_nor_repaired(cap_audit_data, mutation):
    con, settings = cap_audit_data
    con.execute(mutation)
    checks = {row["check"]: row for row in _observed_cap_checks(con, settings)}
    assert checks["unusable_observed_caps_excluded"]["pass"] is False


def test_rank_audit_reports_distinct_largest_companies_and_all_months(cap_audit_data):
    con, settings = cap_audit_data
    checks, tables = _cap_rank_checks(con, settings)
    assert all(row["pass"] for row in checks)
    assert tables["cap_rank_summary"] == [
        {
            "months": 2,
            "invalid_rank_months": 0,
            "rank1_months_failing": 0,
            "top10_months_failing": 0,
            "max_rank1_dollar_volume_rank": 31,
            "max_top10_dollar_volume_rank": 492,
            "invalid_rank_rows": 0,
            "invalid_dollar_volume_rows": 0,
        }
    ]
    largest = {row["ticker"]: row for row in tables["largest_universe_members"]}
    assert set(largest) == {"BETA", "GAMMA"}
    assert largest["BETA"]["name"] == "Beta Company"
    assert largest["BETA"]["months_at_rank1"] == 1
    assert largest["BETA"]["max_dollar_volume_rank"] == 7
    assert largest["GAMMA"]["months_at_rank1"] == 1
    assert largest["GAMMA"]["max_dollar_volume_rank"] == 31
    assert tables["cap_dollar_volume_by_month"] == [
        {
            "date": date(1999, 1, 29),
            "rank1_dollar_volume_rank": 7,
            "top10_max_dollar_volume_rank": 492,
            "top10_count": 3,
            "invalid_top10_dollar_volume_ranks": 0,
            "rank1_pass": True,
            "top10_pass": True,
        },
        {
            "date": date(1999, 2, 26),
            "rank1_dollar_volume_rank": 31,
            "top10_max_dollar_volume_rank": 492,
            "top10_count": 3,
            "invalid_top10_dollar_volume_ranks": 0,
            "rank1_pass": True,
            "top10_pass": True,
        },
    ]
    assert con.execute("SELECT count(*) FROM _cap_rank_months").fetchone()[0] == 2


@pytest.mark.parametrize(
    ("cap_rank", "dv_rank", "gate", "passes"),
    [
        (1, 100, "rank1_dollar_volume_rank", True),
        (1, 101, "rank1_dollar_volume_rank", False),
        (2, 1000, "top10_dollar_volume_rank", True),
        (2, 1001, "top10_dollar_volume_rank", False),
    ],
)
def test_dollar_volume_gates_have_inclusive_rank_boundaries(
    cap_audit_data,
    cap_rank,
    dv_rank,
    gate,
    passes,
):
    con, settings = cap_audit_data
    con.execute(
        """UPDATE universe SET dollar_volume_rank=?
        WHERE rank=? AND date=DATE '1999-01-29'""",
        [dv_rank, cap_rank],
    )
    checks, tables = _cap_rank_checks(con, settings)
    assert next(row["pass"] for row in checks if row["check"] == gate) is passes
    monthly_key = "rank1_pass" if cap_rank == 1 else "top10_pass"
    assert tables["cap_dollar_volume_by_month"][0][monthly_key] is passes


def test_cap_ratio_does_not_gate_dollar_volume_sanity(cap_audit_data):
    con, settings = cap_audit_data
    con.execute("UPDATE universe SET marketcap=1000 WHERE permaticker=2")
    checks, _ = _cap_rank_checks(con, settings)
    # 1000/70 exceeds the obsolete ratio limit, with valid supplied DV ranks.
    assert all(row["pass"] for row in checks)
    assert {row["check"] for row in checks} == {
        "universe_cap_ranks",
        "rank1_dollar_volume_rank",
        "top10_dollar_volume_rank",
    }


def test_top10_gate_includes_rank_ten_and_excludes_rank_eleven(cap_audit_data):
    con, settings = cap_audit_data
    con.execute("""INSERT INTO universe
        SELECT 100+r,DATE '1999-01-29',44-r,'observed',
            CASE WHEN r=11 THEN 2001 ELSE 200+r END,r
        FROM range(4,12) t(r)""")
    checks, tables = _cap_rank_checks(con, settings)
    assert all(row["pass"] for row in checks)
    assert tables["cap_dollar_volume_by_month"][0]["top10_count"] == 10
    assert tables["cap_rank_summary"][0]["max_top10_dollar_volume_rank"] == 492
    con.execute("UPDATE universe SET dollar_volume_rank=1001 WHERE rank=10")
    checks, _ = _cap_rank_checks(con, settings)
    assert (
        next(row["pass"] for row in checks if row["check"] == "top10_dollar_volume_rank") is False
    )


@pytest.mark.parametrize("dv_rank", [None, 0, -1])
def test_invalid_top10_dollar_volume_ranks_cannot_pass_by_aggregate_null_omission(
    cap_audit_data,
    dv_rank,
):
    con, settings = cap_audit_data
    con.execute(
        """UPDATE universe SET dollar_volume_rank=?
        WHERE rank=2 AND date=DATE '1999-01-29'""",
        [dv_rank],
    )
    checks, tables = _cap_rank_checks(con, settings)
    passed = {row["check"]: row["pass"] for row in checks}
    assert passed["universe_cap_ranks"] is False
    assert passed["rank1_dollar_volume_rank"] is True
    assert passed["top10_dollar_volume_rank"] is False
    assert tables["cap_rank_summary"][0]["invalid_dollar_volume_rows"] == 1
    assert tables["cap_dollar_volume_by_month"][0]["top10_pass"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM universe WHERE rank=2 AND date=DATE '1999-01-29'",
        "UPDATE universe SET rank=1 WHERE rank=2 AND date=DATE '1999-01-29'",
        "UPDATE universe SET marketcap=NULL WHERE rank=1 AND date=DATE '1999-01-29'",
        "UPDATE universe SET marketcap=0 WHERE rank=2 AND date=DATE '1999-01-29'",
        "UPDATE universe SET marketcap='Infinity'::DOUBLE WHERE rank=1 AND date=DATE '1999-01-29'",
    ],
)
def test_rank_audit_fails_missing_duplicate_or_invalid_top_ranks(cap_audit_data, mutation):
    con, settings = cap_audit_data
    con.execute(mutation)
    checks, tables = _cap_rank_checks(con, settings)
    assert all(not row["pass"] for row in checks)
    assert tables["cap_rank_summary"][0]["invalid_rank_months"] == 1
