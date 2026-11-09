"""Pinned size-only market-cap repairs, in DAILY's USD millions (Methodology)."""


from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from momenta.settings import sql_path


def build_cap_repairs(
    con: duckdb.DuckDBPyConnection, settings: dict[str, Any], output: Path
) -> None:
    """Pin cap(t0) × split-adjusted close(m) / close(t0), per Methodology.

    Input candidates are otherwise eligible stock-months. Prefer the latest
    earlier positive cap within 252 trading days; otherwise use the first later
    positive cap. Turnover is median(close × volume, 21 days) / repaired cap in
    dollars, accepted within the inclusive configured turnover bounds. Rejections
    retain proposed values for audit. Prices scan by year. A present observed cap
    is never repaired, including one rejected by its own turnover floor.
    """
    universe = settings["universe"]
    con.execute("""CREATE OR REPLACE TEMP TABLE _repair_missing AS
        SELECT * FROM _m1_all_candidates WHERE marketcap IS NULL""")
    # DAILY extends beyond the latest published French month. Extend only that
    # tail for anchor distances; membership still uses the original FF calendar.
    con.execute("""CREATE OR REPLACE TEMP TABLE _repair_calendar AS
        SELECT date, row_number() OVER (ORDER BY date) - 1 AS day_index FROM (
            SELECT date FROM calendar UNION
            SELECT DISTINCT date FROM caps WHERE date > (SELECT max(date) FROM calendar))""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _repair_anchors AS
        SELECT c.permaticker, c.date, c.marketcap, d.day_index
        FROM caps c JOIN _repair_calendar d USING (date)
        SEMI JOIN _repair_missing m USING (permaticker)
        WHERE c.marketcap > 0 AND isfinite(c.marketcap)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _repair_targets AS
        WITH earlier AS (
            SELECT m.*, d.day_index, a.date AS earlier_date,
                a.marketcap AS earlier_cap, a.day_index AS earlier_index
            FROM _repair_missing m JOIN _repair_calendar d USING (date)
            ASOF LEFT JOIN _repair_anchors a
                ON m.permaticker = a.permaticker AND m.date > a.date
        ), alternatives AS (
            SELECT m.*, a.date AS later_date, a.marketcap AS later_cap,
                a.day_index AS later_index,
                coalesce(m.day_index - m.earlier_index <=
                    {int(universe['cap_carry_forward_days'])}, false) AS carry
            FROM earlier m ASOF LEFT JOIN _repair_anchors a
                ON m.permaticker = a.permaticker AND m.date < a.date
        ) SELECT *,
            CASE WHEN carry THEN earlier_date ELSE later_date END AS t0,
            CASE WHEN carry THEN earlier_cap ELSE later_cap END AS anchor_marketcap,
            CASE WHEN carry THEN day_index - earlier_index
                ELSE later_index - day_index END AS distance_trading_days,
            CASE WHEN carry THEN 'carried_forward'
                WHEN later_date IS NOT NULL THEN 'backfilled' END AS cap_source
        FROM alternatives""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _repair_anchor_prices (
        permaticker BIGINT, date DATE, close DOUBLE)""")
    years = con.execute("""SELECT DISTINCT year(t0) FROM _repair_targets
        WHERE t0 IS NOT NULL ORDER BY 1""").fetchall()
    for (year,) in years:
        con.execute("""INSERT INTO _repair_anchor_prices
            SELECT p.permaticker, p.date, p.close FROM prices p
            SEMI JOIN _repair_targets t ON p.permaticker = t.permaticker AND p.date = t.t0
            WHERE p.date BETWEEN ? AND ?""", [date(year, 1, 1), date(year, 12, 31)])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cap_repairs AS
        WITH proposed AS (
            SELECT t.permaticker, t.date, t.t0, t.distance_trading_days, t.cap_source,
                t.anchor_marketcap, p.close AS anchor_close, t.close,
                CASE WHEN p.close > 0 AND isfinite(p.close) THEN
                    t.anchor_marketcap * t.close / p.close END AS repaired_marketcap,
                t.median_dollar_volume,
                t.earlier_date, t.later_date
            FROM _repair_targets t LEFT JOIN _repair_anchor_prices p
                ON t.permaticker = p.permaticker AND t.t0 = p.date
        ), measured AS (
            SELECT *, median_dollar_volume / (repaired_marketcap * 1e6) AS implied_turnover
            FROM proposed
        ), classified AS (
            SELECT *, CASE
                WHEN earlier_date IS NULL AND later_date IS NULL THEN 'no_daily_cap'
                WHEN t0 IS NULL THEN 'no_qualifying_anchor'
                WHEN repaired_marketcap IS NULL OR NOT isfinite(repaired_marketcap)
                    OR repaired_marketcap <= 0 THEN 'missing_anchor_price'
                WHEN implied_turnover IS NULL OR NOT isfinite(implied_turnover)
                    OR implied_turnover < 0 THEN 'invalid_turnover'
                WHEN implied_turnover < {float(universe['cap_repair_min_turnover'])}
                    THEN 'turnover_below_limit'
                WHEN implied_turnover > {float(universe['cap_repair_max_turnover'])}
                    THEN 'turnover_above_limit' END AS exception
            FROM measured
        ) SELECT permaticker, date, t0, distance_trading_days, cap_source,
            anchor_marketcap, anchor_close, close, repaired_marketcap,
            median_dollar_volume, implied_turnover, exception IS NULL AS accepted, exception
        FROM classified""")
    for name, predicate in (("cap_repairs", "true"), ("cap_repair_exceptions", "NOT accepted")):
        con.execute(f"""COPY (SELECT * FROM cap_repairs WHERE {predicate}
            ORDER BY date, permaticker) TO {sql_path(output / f'{name}.parquet')}
            (FORMAT PARQUET, COMPRESSION {sql_path(settings['storage']['compression'])})""")
