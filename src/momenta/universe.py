"""Monthly as-of universe construction from bounded DuckDB scans (Methodology)."""


import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from momenta.cap_repair import build_cap_repairs
from momenta.events import Action, rollback_sic
from momenta.settings import sql_path


def _build_sic_intervals(con: duckdb.DuckDBPyConnection, output: Path) -> None:
    """Compress point-in-time SIC rollbacks into intervals between effective SIC changes."""
    actions: dict[int, list[Action]] = defaultdict(list)
    for permaticker, when, kind, value in con.execute("""
        SELECT permaticker, date, action, value FROM actions_flat
        WHERE action IN ('sicchangefrom', 'sicchangeto')
        ORDER BY permaticker, date, action
    """).fetchall():
        actions[permaticker].append(Action(when, kind, value))

    interval_path = output / ".sic_intervals.csv"
    issue_path = output / ".sic_issues.csv"
    with (
        interval_path.open("w", newline="") as intervals,
        issue_path.open("w", newline="") as issues,
    ):
        interval_writer, issue_writer = csv.writer(intervals), csv.writer(issues)
        interval_writer.writerow(
            ["permaticker", "valid_from", "valid_to", "siccode", "changes_rolled_back"]
        )
        issue_writer.writerow(["permaticker", "date", "issue"])
        for permaticker, current_sic in con.execute(
            "SELECT permaticker, siccode FROM master ORDER BY permaticker"
        ).fetchall():
            rows = actions.get(permaticker, [])
            boundaries = [date.min, *sorted({a.date for a in rows}), date.max]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                result = rollback_sic(current_sic, rows, start)
                interval_writer.writerow(
                    [permaticker, start, end, result.siccode, result.changes_rolled_back]
                )
                if start == date.min:
                    for issue in result.issues:
                        issue_writer.writerow([permaticker, issue.date, issue.issue])

    con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_sic_intervals AS
        SELECT * FROM read_csv({sql_path(interval_path)}, header=true,
            columns={{'permaticker':'BIGINT','valid_from':'DATE','valid_to':'DATE',
                     'siccode':'INTEGER','changes_rolled_back':'INTEGER'}})""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_sic_issues AS
        SELECT * FROM read_csv({sql_path(issue_path)}, header=true,
            columns={{'permaticker':'BIGINT','date':'DATE','issue':'VARCHAR'}})""")
    interval_path.unlink()
    issue_path.unlink()


def build_universe(
    con: duckdb.DuckDBPyConnection,
    settings: dict[str, Any],
    output: Path,
    through: date | None = None,
    *,
    build_repairs: bool = False,
) -> dict[str, int]:
    """Build U_m = top N caps among as-of eligible names (Methodology).

    Daily availability counts non-null returns over calendar indices m−251..m.
    Formation requires every month m−11..m−1; RES counts months m−35..m.
    Price must exist at the exact trading month-end. Caps use the latest value
    no more than the configured five trading days earlier. Observed caps must
    meet their turnover floor; unusable observed caps cannot be repaired. Missing
    caps use accepted pinned repairs if enabled (Methodology). Only curation sets
    build_repairs=True; subsequent
    builds consume the fixed cap_repairs relation. Unknown SIC is kept
    and audited; only the specified blank-check SIC is excluded. The monthly
    French file's last month is complete by the source's publication convention.
    """
    output.mkdir(parents=True, exist_ok=True)
    universe, signals = settings["universe"], settings["signals"]
    daily_window = int(universe["daily_window"])
    cap_lookback = int(universe["marketcap_lookback_days"])
    turnover_window = int(universe["cap_turnover_window_days"])
    formation = int(signals["formation_months"])
    res_window = int(signals["res_window_months"])
    min_daily = int(universe["min_daily_returns_252"])
    min_res = int(signals["res_min_months"])
    n_universe = int(universe["n_universe"])
    compression = settings["storage"]["compression"]
    bound = through or con.execute("SELECT max(date) FROM calendar").fetchone()[0]
    start = date.fromisoformat(settings["data"]["universe_start"])
    con.execute("""CREATE OR REPLACE TEMP TABLE _m1_months AS
        SELECT date, day_index FROM (
            SELECT date, day_index,
                row_number() OVER (PARTITION BY date_trunc('month', date) ORDER BY date DESC) AS n
            FROM calendar
        ) WHERE n = 1 AND date BETWEEN ? AND ?
        ORDER BY date""", [start, bound])
    months = con.execute("SELECT count(*) FROM _m1_months").fetchone()[0]
    if not months:
        raise ValueError("No complete FF3 month-ends fall in the requested universe period")

    _build_sic_intervals(con, output)
    con.execute("""CREATE OR REPLACE TEMP TABLE _m1_all_candidates (
        permaticker BIGINT, date DATE, marketcap DOUBLE, res_eligible BOOLEAN,
        close DOUBLE, median_dollar_volume DOUBLE)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _m1_pit_sic (
        permaticker BIGINT, date DATE, siccode INTEGER, changes_rolled_back INTEGER)""")

    for (year,) in con.execute("SELECT DISTINCT year(date) FROM _m1_months ORDER BY 1").fetchall():
        con.execute("""CREATE OR REPLACE TEMP TABLE _m1_year_months AS
            SELECT * FROM _m1_months WHERE year(date) = ?""", [year])
        first_index, last_day = con.execute(
            "SELECT min(day_index), max(date) FROM _m1_year_months"
        ).fetchone()
        first_day = con.execute("SELECT min(date) FROM calendar WHERE day_index >= ?", [
            first_index - max(daily_window - 1, cap_lookback, turnover_window - 1)
        ]).fetchone()[0]
        con.execute("""CREATE OR REPLACE TEMP TABLE _m1_year_sic AS
            SELECT s.permaticker, m.date, s.siccode, s.changes_rolled_back
            FROM _m1_sic_intervals s JOIN _m1_year_months m
                ON m.date >= s.valid_from AND m.date < s.valid_to""")
        con.execute("INSERT INTO _m1_pit_sic SELECT * FROM _m1_year_sic")
        # The date predicate limits the price scan to the year plus its warm-up.
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_price_snapshots AS
            WITH counted AS (
                SELECT p.permaticker, p.date, p.closeunadj, p.close,
                    median(p.close * p.volume) OVER (PARTITION BY p.permaticker
                        ORDER BY c.day_index RANGE BETWEEN {turnover_window - 1} PRECEDING
                        AND CURRENT ROW) AS median_dollar_volume,
                    count(p.daily_return) OVER (PARTITION BY p.permaticker
                        ORDER BY c.day_index RANGE BETWEEN {daily_window - 1} PRECEDING
                        AND CURRENT ROW) AS n_daily
                FROM prices p JOIN calendar c USING (date)
                WHERE p.date BETWEEN ? AND ?
            ) SELECT p.* FROM counted p JOIN _m1_year_months m USING (date)""",
            [first_day, last_day])
        history_months = max(formation, res_window - 1)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_monthly_counts AS
            SELECT r.permaticker, m.date,
                count(DISTINCT r.date) FILTER (WHERE r.monthly_return IS NOT NULL
                    AND r.date >= date_trunc('month', m.date) - INTERVAL '{formation} months'
                    AND r.date < date_trunc('month', m.date)) AS n_formation,
                count(DISTINCT r.date) FILTER (WHERE r.monthly_return IS NOT NULL
                    AND r.date >= date_trunc('month', m.date)
                        - INTERVAL '{res_window - 1} months') AS n_res
            FROM monthly_returns r JOIN _m1_year_months m
                ON r.date >= date_trunc('month', m.date) - INTERVAL '{history_months} months'
                AND r.date <= last_day(m.date)
            WHERE r.date >= make_date({year}, 1, 1) - INTERVAL '{history_months} months'
                AND r.date <= last_day(?)
            GROUP BY r.permaticker, m.date""", [last_day])
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_cap_snapshots AS
            SELECT caps.permaticker, m.date, caps.marketcap
            FROM caps JOIN calendar c USING (date)
            JOIN _m1_year_months m
                ON c.day_index BETWEEN m.day_index - {cap_lookback} AND m.day_index
            WHERE caps.date BETWEEN ? AND ? AND caps.marketcap IS NOT NULL
                AND isfinite(caps.marketcap) AND caps.marketcap > 0
            QUALIFY row_number() OVER (PARTITION BY caps.permaticker, m.date
                ORDER BY caps.date DESC) = 1""", [first_day, last_day])
        con.execute(f"""INSERT INTO _m1_all_candidates
            SELECT p.permaticker, p.date, c.marketcap, h.n_res >= {min_res} AS res_eligible,
                p.close, p.median_dollar_volume
            FROM _m1_price_snapshots p
            JOIN (SELECT permaticker,category FROM master) s USING (permaticker)
            JOIN _m1_year_sic sic USING (permaticker, date)
            JOIN _m1_monthly_counts h USING (permaticker, date)
            LEFT JOIN _m1_cap_snapshots c USING (permaticker, date)
            WHERE s.category IN (SELECT unnest(?))
                AND (sic.siccode IS NULL OR sic.siccode <> ?)
                AND p.closeunadj >= ? AND p.n_daily >= {min_daily}
                AND h.n_formation = {formation}""",
            [universe["categories"], universe["blank_check_sic"], universe["min_price"]])
    if build_repairs:
        build_cap_repairs(con, settings, output)
    if universe["cap_repair"]:
        # This join uses a pinned input. It must never query future prices or caps.
        repair_join = "LEFT JOIN cap_repairs r USING (permaticker, date)"
        repaired = "CASE WHEN r.accepted THEN r.repaired_marketcap END"
        source = "CASE WHEN r.accepted THEN r.cap_source END"
    else:
        repair_join, repaired, source = "", "NULL::DOUBLE", "NULL::VARCHAR"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_candidates AS
        WITH measured AS (
            SELECT *, median_dollar_volume / (marketcap * 1e6) AS observed_turnover
            FROM _m1_all_candidates
        ), observed AS (
            SELECT *, coalesce(isfinite(observed_turnover) AND observed_turnover >=
                {float(universe['cap_observed_min_turnover'])}, false) AS observed_usable
            FROM measured
        ) SELECT c.*,
            CASE WHEN c.marketcap IS NOT NULL THEN
                CASE WHEN c.observed_usable THEN c.marketcap END
                ELSE {repaired} END AS effective_marketcap,
            CASE WHEN c.marketcap IS NOT NULL THEN
                CASE WHEN c.observed_usable THEN 'observed' END
                ELSE {source} END AS cap_source,
            row_number() OVER (PARTITION BY c.date
                ORDER BY c.median_dollar_volume DESC NULLS LAST, c.permaticker)
                AS dollar_volume_rank
        FROM observed c {repair_join}""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _m1_observed_cap_exceptions AS
        SELECT permaticker, date, marketcap, median_dollar_volume,
            observed_turnover AS implied_turnover,
            CASE WHEN observed_turnover IS NULL OR NOT isfinite(observed_turnover)
                OR observed_turnover < 0 THEN 'observed_invalid_turnover'
                ELSE 'observed_turnover_below_limit' END AS exception
        FROM _m1_candidates WHERE marketcap IS NOT NULL AND NOT observed_usable""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_universe AS
        SELECT permaticker, date, effective_marketcap AS marketcap, res_eligible,
            row_number() OVER (PARTITION BY date
                ORDER BY effective_marketcap DESC, permaticker) AS rank, cap_source,
            dollar_volume_rank
        FROM _m1_candidates WHERE effective_marketcap IS NOT NULL
        QUALIFY rank <= {n_universe}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _m1_universe_audit AS
        WITH candidates AS (
            SELECT date, count(*) AS candidates, count(marketcap) AS with_marketcap,
                count(*) FILTER (WHERE observed_usable) AS with_usable_observed_marketcap,
                count(effective_marketcap) AS with_repaired_marketcap,
                count(*) FILTER (WHERE NOT observed_usable AND dollar_volume_rank <=
                    {int(universe['cap_gate_dollar_volume_rank'])})
                    AS high_dollar_volume_missing_before,
                count(*) FILTER (WHERE effective_marketcap IS NULL AND dollar_volume_rank <=
                    {int(universe['cap_gate_dollar_volume_rank'])})
                    AS high_dollar_volume_missing_after
            FROM _m1_candidates GROUP BY date
        ), selected AS (
            SELECT date, count(*) AS universe_size,
                count(*) FILTER (WHERE res_eligible) AS res_eligible,
                count(*) FILTER (WHERE cap_source <> 'observed') AS repaired_members,
                coalesce(sum(marketcap ORDER BY permaticker)
                    FILTER (WHERE cap_source <> 'observed'), 0)
                    / sum(marketcap ORDER BY permaticker) AS repaired_cap_share
            FROM _m1_universe GROUP BY date
        ) SELECT m.date, coalesce(candidates, 0) AS candidates,
            coalesce(with_marketcap, 0) AS with_marketcap,
            CASE WHEN coalesce(candidates, 0) = 0 THEN 1.0
                ELSE with_marketcap::DOUBLE / candidates END AS marketcap_coverage,
            coalesce(universe_size, 0) AS universe_size, coalesce(res_eligible, 0) AS res_eligible,
            {n_universe} - coalesce(universe_size, 0) AS shortfall,
            coalesce(with_repaired_marketcap, 0) AS with_repaired_marketcap,
            CASE WHEN coalesce(candidates, 0) = 0 THEN 1.0
                ELSE with_repaired_marketcap::DOUBLE / candidates END
                AS repaired_marketcap_coverage,
            coalesce(high_dollar_volume_missing_before, 0) AS high_dollar_volume_missing_before,
            coalesce(high_dollar_volume_missing_after, 0) AS high_dollar_volume_missing_after,
            coalesce(repaired_members, 0) AS repaired_members,
            coalesce(repaired_cap_share, 0) AS repaired_cap_share,
            coalesce(with_usable_observed_marketcap, 0) AS with_usable_observed_marketcap,
            CASE WHEN coalesce(candidates, 0) = 0 THEN 1.0
                ELSE with_usable_observed_marketcap::DOUBLE / candidates END
                AS usable_observed_marketcap_coverage,
            coalesce(with_marketcap - with_usable_observed_marketcap, 0)
                AS unusable_observed_caps
        FROM _m1_months m LEFT JOIN candidates USING (date) LEFT JOIN selected USING (date)""")

    con.execute("""CREATE OR REPLACE TEMP TABLE _m1_sic_audit AS
        WITH codes AS (
            SELECT year(date) AS year, count(*) AS observations, count(siccode) AS known_sic,
                count(siccode)::DOUBLE / count(*) AS sic_coverage,
                count(*) FILTER (WHERE changes_rolled_back > 0) AS rolled_back_observations
            FROM _m1_pit_sic GROUP BY 1
        ), events AS (
            SELECT permaticker, date,
                count(*) FILTER (WHERE action = 'sicchangefrom') AS n_from,
                count(*) FILTER (WHERE action = 'sicchangeto') AS n_to,
                count(*) FILTER (WHERE value > 0 AND value < 10000
                    AND value = floor(value)) AS n_valid
            FROM actions_flat WHERE action IN ('sicchangefrom', 'sicchangeto')
            GROUP BY permaticker, date
        ), event_counts AS (
            SELECT year(date) AS year, count(*) AS sic_change_events,
                count(*) FILTER (WHERE n_from = 1 AND n_to = 1 AND n_valid = 2)
                    AS paired_sic_change_events FROM events GROUP BY 1
        ), issue_counts AS (
            SELECT year(date) AS year, count(*) AS sic_change_issues
            FROM _m1_sic_issues WHERE date IS NOT NULL GROUP BY 1
        ) SELECT codes.*, coalesce(sic_change_events, 0) AS sic_change_events,
            coalesce(paired_sic_change_events, 0) AS paired_sic_change_events,
            coalesce(sic_change_issues, 0) AS sic_change_issues
        FROM codes LEFT JOIN event_counts USING (year) LEFT JOIN issue_counts USING (year)""")
    for table, order in (
        ("universe", "date, rank"),
        ("universe_audit", "date"),
        ("observed_cap_exceptions", "date, permaticker"),
        ("pit_sic", "date, permaticker"),
        ("sic_audit", "year"),
        ("sic_issues", "date, permaticker, issue"),
    ):
        con.execute(f"""COPY (SELECT * FROM _m1_{table} ORDER BY {order})
            TO {sql_path(output / f'{table}.parquet')}
            (FORMAT PARQUET, COMPRESSION {sql_path(compression)})""")
    return {
        "months": months,
        "universe_rows": con.execute("SELECT count(*) FROM _m1_universe").fetchone()[0],
        "shortfall_months": con.execute(
            "SELECT count(*) FROM _m1_universe_audit WHERE shortfall > 0"
        ).fetchone()[0],
    }
