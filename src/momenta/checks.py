"""research integrity checks and aggregate data audits (Methodology)."""


from __future__ import annotations

import csv
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from momenta.manifest import Manifest, provenance, sha256_file
from momenta.reference_data import FRENCH_FILES, JKP_FACTORS
from momenta.settings import REPO, connect, sql_path

EVENT_CLASSES = (
    "acquisition_with_terms",
    "election",
    "acquisition_without_terms",
    "voluntary_delisting",
    "performance_delisting",
    "unexplained_end",
)


_CURATED_TABLES = (
    "master",
    "actions",
    "monthly_returns",
    "pit_sic",
    "universe",
    "universe_audit",
    "delistings",
    "exceptions",
    "flags",
    "sic_audit",
    "trading_calendar",
    "cap_repairs",
    "cap_repair_exceptions",
    "observed_cap_exceptions",
)


def rows(con: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    """Fetch only a small aggregate query, never a stock-price panel."""
    result = con.execute(query)
    names = [column[0] for column in result.description]
    return [dict(zip(names, values, strict=True)) for values in result.fetchall()]


def write_table(name: str, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Write a derived aggregate CSV with a complete provenance sidecar."""
    path = REPO / "results" / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        if records:
            writer = csv.DictWriter(output, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    path.with_suffix(".provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )


def _yearly_prices(
    con: duckdb.DuckDBPyConnection,
    curated: Path,
) -> list[dict[str, Any]]:
    """Audit one year at a time within the Methodology DuckDB resource limits."""
    summaries = []
    for partition in sorted((curated / "prices").glob("year=*")):
        year = int(partition.name.split("=", 1)[1])
        con.execute(
            f"CREATE OR REPLACE VIEW price_year AS SELECT * FROM read_parquet("
            f"{sql_path(partition / '*.parquet')}, hive_partitioning=false)"
        )
        columns = {item[0] for item in con.execute("DESCRIBE price_year").fetchall()}
        fields = [
            field
            for field in ("open", "high", "low", "close", "closeadj", "closeunadj")
            if field in columns
        ]
        invalid = " OR ".join(
            f"{field} IS NULL OR NOT isfinite({field}) OR {field} <= 0" for field in fields
        )
        if not {"close", "closeadj", "closeunadj"}.issubset(columns):
            raise ValueError(f"Missing required price columns in year {year}")
        summary = rows(
            con,
            f"""SELECT {year} AS year, count(*) AS price_rows,
            count(DISTINCT permaticker) AS stocks_with_prices,
            min(date) AS first_date, max(date) AS last_date,
            count(*) FILTER (WHERE {invalid}) AS invalid_price_rows,
            count(*) FILTER (WHERE permaticker IS NULL OR date IS NULL) AS missing_keys,
            count(*) FILTER (WHERE year(date) != {year}) AS wrong_partition_rows
            FROM price_year""",
        )[0]
        summary["duplicate_rows"] = con.execute("""SELECT coalesce(sum(n - 1), 0) FROM
            (SELECT count(*) n FROM price_year GROUP BY permaticker, date HAVING count(*) > 1)
            """).fetchone()[0]
        summaries.append(summary)
        print(f"Checked prices {year}: {summary['price_rows']:,} rows", flush=True)
    if not summaries:
        raise ValueError("No curated yearly price partitions; run data curate")
    return summaries


def _yearly_caps(con: duckdb.DuckDBPyConnection, curated: Path) -> list[dict[str, Any]]:
    summaries = []
    for partition in sorted((curated / "caps").glob("year=*")):
        year = int(partition.name.split("=", 1)[1])
        con.execute(
            f"CREATE OR REPLACE VIEW cap_year AS SELECT * FROM read_parquet("
            f"{sql_path(partition / '*.parquet')}, hive_partitioning=false)"
        )
        summaries.extend(
            rows(
                con,
                f"""SELECT {year} AS year, count(*) AS cap_rows,
            min(date) first_date, max(date) last_date,
            count(*) - count(DISTINCT (permaticker, date)) AS duplicate_rows,
            count(*) FILTER (WHERE permaticker IS NULL OR date IS NULL) AS missing_keys,
            count(*) FILTER (WHERE year(date) != {year}) AS wrong_partition_rows
            FROM cap_year""",
            )
        )
    if not summaries:
        raise ValueError("No curated yearly market-cap partitions; run data curate")
    return summaries


def _availability(
    manifest: Manifest,
    con: duckdb.DuckDBPyConnection,
    prices: list[dict[str, Any]],
    caps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranges = {
        "sharadar/stocks": (prices[0]["first_date"], prices[-1]["last_date"]),
        "sharadar/daily": (caps[0]["first_date"], caps[-1]["last_date"]),
    }
    for name, table in (("tickers", "master"), ("actions", "actions")):
        ranges[f"sharadar/{name}"] = con.execute(
            f"SELECT min(date), max(date) FROM {table}"
        ).fetchone()
    output = []
    for name, entry in sorted(manifest.entries.items()):
        first, last = ranges.get(name, (None, None))
        first = entry.get("date_min", entry.get("first_date", first))
        last = entry.get("date_max", entry.get("last_date", last))
        timing = entry.get("publication_timing")
        if timing is None:
            timing = {
                "sharadar/stocks": "After trading-day close; executed no earlier than next close.",
                "sharadar/daily": "After the close; market cap used as of observation date.",
                "sharadar/tickers": "Pinned master; historical SIC rolled back via actions.",
                "sharadar/actions": "Event dates are last-trade dates; terms may be learned later.",
                "sharadar/descriptions": "Snapshot of source definitions, not a dated predictor.",
            }.get(name, "See source documentation; final vintage aligned by observation date.")
        output.append(
            {
                "input": name,
                "first_observation": first or "not date-indexed",
                "last_observation": last or "not date-indexed",
                "publication_timing": timing,
                "vintage": entry["vintage"],
                "downloaded_at": entry["downloaded_at"],
                "source_url": entry["source_url"],
                "sha256": entry["sha256"],
            }
        )
    return output


def _coverage(
    con: duckdb.DuckDBPyConnection,
    prices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    membership = {
        item["year"]: item
        for item in rows(
            con,
            """SELECT year(date)::INTEGER AS year,
        count(DISTINCT permaticker) eligible_stocks,
        count(DISTINCT permaticker) FILTER (WHERE res_eligible) res_eligible_stocks
        FROM universe GROUP BY year(date)""",
        )
    }
    months = {
        item["year"]: item
        for item in rows(
            con,
            """SELECT year(date)::INTEGER AS year,
        count(*) month_ends, min(universe_size) min_universe, max(universe_size) max_universe,
        min(res_eligible) min_res_eligible, max(res_eligible) max_res_eligible,
        min(marketcap_coverage) min_marketcap_coverage
        FROM universe_audit GROUP BY year(date)""",
        )
    }
    flags = {
        item["year"]: item
        for item in rows(
            con,
            """SELECT year(date)::INTEGER AS year,
        count(*) flag_rows, count(*) FILTER (WHERE in_universe) universe_flag_rows
        FROM flags GROUP BY year(date)""",
        )
    }
    events = {
        (item["year"], item["event_class"]): item["n"]
        for item in rows(
            con,
            """
        SELECT year(last_date)::INTEGER AS year, event_class, count(*) n
        FROM delistings WHERE NOT censored GROUP BY year(last_date), event_class""",
        )
    }
    output = []
    for price in prices:
        year = price["year"]
        row = {
            "year": year,
            "stocks_with_prices": price["stocks_with_prices"],
            "eligible_stocks": 0,
            "res_eligible_stocks": 0,
            "month_ends": 0,
            "min_universe": 0,
            "max_universe": 0,
            "min_res_eligible": 0,
            "max_res_eligible": 0,
            "min_marketcap_coverage": None,
            "flag_rows": 0,
            "universe_flag_rows": 0,
        }
        for section in (membership, months, flags):
            row.update(section.get(year, {}))
        row.update({kind: events.get((year, kind), 0) for kind in EVENT_CLASSES})
        output.append(row)
    return output


def _delisting_value(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Use previous formation market-cap weights as the value-exposure proxy."""
    con.execute("""CREATE OR REPLACE TEMP TABLE affected_value AS
        WITH months AS (SELECT date, sum(marketcap ORDER BY permaticker) total_cap
                        FROM universe GROUP BY date),
        previous AS (SELECT d.*, (SELECT max(u.date) FROM months u WHERE u.date < d.last_date)
                    formation_date FROM delistings d WHERE NOT censored)
        SELECT d.event_class, d.formation_date, d.permaticker,
               coalesce(u.marketcap / m.total_cap, 0) formation_weight,
               u.permaticker IS NOT NULL in_universe
        FROM previous d LEFT JOIN universe u
          ON u.permaticker=d.permaticker AND u.date=d.formation_date
        LEFT JOIN months m ON m.date=d.formation_date""")
    grouped = {
        row["event_class"]: row
        for row in rows(
            con,
            """SELECT event_class,
        count(*) events, count(*) FILTER (WHERE in_universe) universe_events,
        sum(formation_weight ORDER BY permaticker) cumulative_formation_weight
        FROM affected_value GROUP BY event_class""",
        )
    }
    peaks = {
        row["event_class"]: row["max_monthly_weight"]
        for row in rows(
            con,
            """
        SELECT event_class, max(weight) max_monthly_weight FROM (
          SELECT event_class, formation_date, sum(formation_weight ORDER BY permaticker) weight
          FROM affected_value GROUP BY event_class, formation_date) GROUP BY event_class""",
        )
    }
    n_months = con.execute("SELECT count(*) FROM universe_audit").fetchone()[0]
    output = []
    for kind in EVENT_CLASSES:
        row = grouped.get(
            kind,
            {
                "event_class": kind,
                "events": 0,
                "universe_events": 0,
                "cumulative_formation_weight": 0.0,
            },
        )
        row["mean_monthly_weight_pct"] = 100 * row["cumulative_formation_weight"] / max(n_months, 1)
        row["max_monthly_weight_pct"] = 100 * peaks.get(kind, 0.0)
        output.append(row)
    return output


def _cap_checks(con: duckdb.DuckDBPyConnection, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Audit pinned cap repairs and the amended market-cap coverage gate.

    Methodology defines cap_m = cap_t0 × close_m / close_t0, with caps in
    USD millions. Implied turnover = median dollar volume / (cap_m × 10^6).
    """
    universe = settings["universe"]
    limit = int(universe["cap_gate_max_missing"])
    rank = int(universe["cap_gate_dollar_volume_rank"])
    turnover_limit = float(universe["cap_repair_max_turnover"])
    turnover_minimum = float(universe["cap_repair_min_turnover"])
    carry_days = int(universe["cap_carry_forward_days"])
    observed_days = int(universe["marketcap_lookback_days"])
    maximum, invalid_gate = con.execute("""SELECT max(high_dollar_volume_missing_after),
        count(*) FILTER (WHERE high_dollar_volume_missing_after IS NULL
            OR high_dollar_volume_missing_before IS NULL
            OR high_dollar_volume_missing_after < 0
            OR high_dollar_volume_missing_after > high_dollar_volume_missing_before)
        FROM universe_audit""").fetchone()
    result = [
        {
            "check": "high_dollar_volume_cap_gate",
            "pass": maximum is not None and maximum <= limit and invalid_gate == 0,
            "value": maximum,
            "criterion": f"At most {limit} top-{rank} dollar-volume candidates lack caps/month",
        }
    ]
    invalid = con.execute(f"""SELECT count(*) FROM cap_repairs WHERE accepted IS NULL
        OR (cap_source IS NOT NULL AND cap_source NOT IN ('carried_forward','backfilled'))
        OR (accepted AND (
            exception IS NOT NULL OR cap_source IS NULL OR t0 IS NULL
            OR distance_trading_days IS NULL OR distance_trading_days <= 0
            OR repaired_marketcap IS NULL OR NOT isfinite(repaired_marketcap)
            OR repaired_marketcap <= 0 OR anchor_marketcap IS NULL
            OR NOT isfinite(anchor_marketcap) OR anchor_marketcap <= 0
            OR anchor_close IS NULL OR NOT isfinite(anchor_close) OR anchor_close <= 0
            OR close IS NULL OR NOT isfinite(close) OR close <= 0
            OR median_dollar_volume IS NULL OR NOT isfinite(median_dollar_volume)
            OR median_dollar_volume < 0 OR implied_turnover IS NULL
            OR NOT isfinite(implied_turnover) OR implied_turnover < {turnover_minimum}
            OR implied_turnover > {turnover_limit}
            OR (cap_source='carried_forward' AND (t0 >= date
                OR distance_trading_days <= {observed_days}
                OR distance_trading_days > {carry_days}))
            OR (cap_source='backfilled' AND t0 <= date)))
        OR (NOT accepted AND (exception IS NULL OR exception=''))
        OR (exception='turnover_above_limit'
            AND (implied_turnover IS NULL OR implied_turnover <= {turnover_limit}))
        OR (exception='turnover_below_limit'
            AND (implied_turnover IS NULL OR NOT isfinite(implied_turnover)
                 OR implied_turnover < 0 OR implied_turnover >= {turnover_minimum}))
        OR (exception='no_daily_cap' AND (t0 IS NOT NULL OR cap_source IS NOT NULL))
        """).fetchone()[0]
    result.append(
        {
            "check": "cap_repair_terms",
            "pass": invalid == 0,
            "value": invalid,
            "criterion": "0 invalid repair sources, anchor dates, guards or exceptions",
        }
    )
    arithmetic = con.execute("""SELECT count(*) FROM cap_repairs WHERE accepted AND (
        abs(repaired_marketcap - anchor_marketcap * close / anchor_close)
            > 1e-10 * greatest(1.0, abs(repaired_marketcap))
        OR abs(implied_turnover - median_dollar_volume / (repaired_marketcap * 1e6))
            > 1e-10 * greatest(1.0, abs(implied_turnover)))""").fetchone()[0]
    result.append(
        {
            "check": "cap_repair_arithmetic",
            "pass": arithmetic == 0,
            "value": arithmetic,
            "criterion": "0 cap-ratio or turnover-unit mismatches",
        }
    )
    exception_mismatches = con.execute("""SELECT count(*) FROM (
        (SELECT permaticker,date FROM cap_repairs WHERE NOT accepted
         EXCEPT ALL SELECT permaticker,date FROM cap_repair_exceptions)
        UNION ALL
        (SELECT permaticker,date FROM cap_repair_exceptions EXCEPT ALL
         SELECT permaticker,date FROM cap_repairs WHERE NOT accepted))""").fetchone()[0]
    result.append(
        {
            "check": "cap_repair_exception_table",
            "pass": exception_mismatches == 0,
            "value": exception_mismatches,
            "criterion": "Every rejected repair is retained once outside git",
        }
    )
    selected_mismatches = con.execute("""SELECT count(*) FROM universe u
        LEFT JOIN cap_repairs r USING(permaticker,date)
        WHERE u.cap_source IS NULL
           OR u.cap_source NOT IN ('observed','carried_forward','backfilled')
           OR (u.cap_source != 'observed' AND (r.permaticker IS NULL OR NOT r.accepted
               OR u.cap_source != r.cap_source
               OR abs(u.marketcap-r.repaired_marketcap)
                   > 1e-10 * greatest(1.0,abs(u.marketcap))))
           OR (u.cap_source='observed' AND r.accepted)""").fetchone()[0]
    result.append(
        {
            "check": "selected_cap_repairs",
            "pass": selected_mismatches == 0,
            "value": selected_mismatches,
            "criterion": "Selected repaired caps match accepted pinned rows",
        }
    )
    repairs_enabled = "true" if universe.get("cap_repair", True) else "false"
    audit_mismatches = con.execute(f"""WITH selected AS (
        SELECT date, count(*) FILTER (WHERE cap_source != 'observed') repaired_members,
            coalesce(sum(marketcap ORDER BY permaticker)
                FILTER (WHERE cap_source != 'observed'),0)
                / sum(marketcap ORDER BY permaticker) repaired_share
        FROM universe GROUP BY date), repaired AS (
        SELECT date,count(*) FILTER (WHERE accepted AND {repairs_enabled}) accepted_repairs
        FROM cap_repairs GROUP BY date), observed_exceptions AS (
        SELECT date,count(*) unusable FROM observed_cap_exceptions GROUP BY date)
        SELECT count(*) FROM universe_audit a LEFT JOIN selected s USING(date)
        LEFT JOIN repaired r USING(date)
        LEFT JOIN observed_exceptions o USING(date)
        WHERE a.repaired_members IS DISTINCT FROM coalesce(s.repaired_members,0)
           OR a.repaired_cap_share IS NULL
           OR abs(a.repaired_cap_share-coalesce(s.repaired_share,0)) > 1e-10
           OR a.with_marketcap > a.candidates OR a.with_usable_observed_marketcap < 0
           OR a.with_usable_observed_marketcap > a.with_repaired_marketcap
           OR a.with_repaired_marketcap > a.candidates
           OR a.unusable_observed_caps IS DISTINCT FROM coalesce(o.unusable,0)
           OR a.with_usable_observed_marketcap
                != a.with_marketcap - coalesce(o.unusable,0)
           OR a.with_repaired_marketcap
                != a.with_usable_observed_marketcap + coalesce(r.accepted_repairs,0)
           OR a.marketcap_coverage IS NULL OR a.repaired_marketcap_coverage IS NULL
           OR a.usable_observed_marketcap_coverage IS NULL
           OR abs(a.marketcap_coverage - CASE WHEN a.candidates=0 THEN 1.0
                    ELSE a.with_marketcap::DOUBLE/a.candidates END) > 1e-10
           OR abs(a.usable_observed_marketcap_coverage - CASE WHEN a.candidates=0 THEN 1.0
                    ELSE a.with_usable_observed_marketcap::DOUBLE/a.candidates END) > 1e-10
           OR abs(a.repaired_marketcap_coverage - CASE WHEN a.candidates=0 THEN 1.0
                    ELSE a.with_repaired_marketcap::DOUBLE/a.candidates END) > 1e-10
        """).fetchone()[0]
    result.append(
        {
            "check": "cap_repair_monthly_audit",
            "pass": audit_mismatches == 0,
            "value": audit_mismatches,
            "criterion": "Monthly coverage, repaired counts and value shares reconcile",
        }
    )
    return result


def _observed_cap_checks(
    con: duckdb.DuckDBPyConnection,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Verify the observed-turnover floor and exclusion without repair (Methodology)."""
    floor = float(settings["universe"]["cap_observed_min_turnover"])
    invalid = con.execute(f"""SELECT count(*) FROM observed_cap_exceptions
        WHERE marketcap IS NULL OR NOT isfinite(marketcap) OR marketcap <= 0
           OR exception IS NULL OR exception NOT IN
                ('observed_turnover_below_limit','observed_invalid_turnover')
           OR (exception='observed_turnover_below_limit' AND (implied_turnover IS NULL
                OR NOT isfinite(implied_turnover) OR implied_turnover < 0
                OR implied_turnover >= {floor}))
           OR (exception='observed_invalid_turnover'
                AND isfinite(implied_turnover) AND implied_turnover >= 0)
        """).fetchone()[0]
    arithmetic = con.execute("""WITH expected AS (
        SELECT *,median_dollar_volume/(marketcap*1e6) AS expected_turnover
        FROM observed_cap_exceptions)
        SELECT count(*) FROM expected WHERE CASE
            WHEN expected_turnover IS NULL OR NOT isfinite(expected_turnover)
                THEN coalesce(isfinite(implied_turnover),false)
            ELSE implied_turnover IS NULL OR NOT isfinite(implied_turnover)
                OR abs(implied_turnover-expected_turnover)
                    > 1e-10*greatest(1.0,abs(expected_turnover)) END
        """).fetchone()[0]
    selected, repaired = con.execute("""SELECT
        (SELECT count(*) FROM observed_cap_exceptions e
            JOIN universe u USING(permaticker,date)),
        (SELECT count(*) FROM observed_cap_exceptions e
            JOIN cap_repairs r USING(permaticker,date))""").fetchone()
    return [
        {
            "check": "observed_cap_exception_terms",
            "pass": invalid == 0 and arithmetic == 0,
            "value": f"{invalid} invalid reasons; {arithmetic} turnover-unit mismatches",
            "criterion": f"Observed exceptions match the {floor:.3%} daily turnover floor",
        },
        {
            "check": "unusable_observed_caps_excluded",
            "pass": selected == 0 and repaired == 0,
            "value": f"{selected} universe members; {repaired} repair attempts",
            "criterion": "Unusable observed caps are excluded and never repaired",
        },
    ]


def _cap_rank_checks(
    con: duckdb.DuckDBPyConnection,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Audit cap order and candidate-wide dollar-volume ranks (Methodology)."""
    rank1_limit = int(settings["universe"]["rank1_max_dollar_volume_rank"])
    top10_limit = int(settings["universe"]["top10_max_dollar_volume_rank"])
    con.execute("""CREATE OR REPLACE TEMP VIEW _cap_rank_rows AS SELECT *,
        coalesce(isfinite(marketcap) AND marketcap>0 AND rank=expected_rank,false)
            AS valid_cap_rank,
        coalesce(isfinite(dollar_volume_rank) AND dollar_volume_rank>0
            AND dollar_volume_rank=floor(dollar_volume_rank),false) AS valid_dollar_volume_rank
        FROM (
        SELECT *,row_number() OVER (PARTITION BY date ORDER BY marketcap DESC,permaticker)
            AS expected_rank FROM universe)""")
    invalid_rows, invalid_dv_rows = con.execute("""SELECT
        count(*) FILTER (WHERE NOT valid_cap_rank),
        count(*) FILTER (WHERE NOT valid_dollar_volume_rank) FROM _cap_rank_rows""").fetchone()
    con.execute(f"""CREATE OR REPLACE TEMP VIEW _cap_rank_months AS
        WITH ranked AS (SELECT date,
            count(*) AS member_count,
            count(*) FILTER (WHERE rank=1) AS rank1_count,
            count(*) FILTER (WHERE rank BETWEEN 1 AND 10) AS top10_count,
            count(*) FILTER (WHERE NOT valid_cap_rank) AS invalid_cap_ranks,
            count(*) FILTER (WHERE rank BETWEEN 1 AND 10 AND NOT valid_dollar_volume_rank)
                AS invalid_top10_dollar_volume_ranks,
            max(permaticker) FILTER (WHERE rank=1) AS rank1_permaticker,
            max(dollar_volume_rank) FILTER (WHERE rank=1 AND valid_dollar_volume_rank)
                AS rank1_dollar_volume_rank,
            max(dollar_volume_rank) FILTER (WHERE rank BETWEEN 1 AND 10
                AND valid_dollar_volume_rank) AS top10_max_dollar_volume_rank
            FROM _cap_rank_rows GROUP BY date), named AS (
        SELECT a.date,r.* EXCLUDE(date),m.ticker,m.name
        FROM universe_audit a LEFT JOIN ranked r USING(date)
        LEFT JOIN master m ON m.permaticker=r.rank1_permaticker)
        SELECT *,coalesce(valid_cap_ranks AND rank1_dollar_volume_rank<={rank1_limit},false)
            AS rank1_pass,
            coalesce(valid_cap_ranks AND invalid_top10_dollar_volume_ranks=0
                AND top10_max_dollar_volume_rank<={top10_limit},false) AS top10_pass FROM (
        SELECT *,coalesce(rank1_count=1 AND invalid_cap_ranks=0
            AND top10_count=least(10,member_count),false) AS valid_cap_ranks FROM named)""")
    summary = rows(
        con,
        f"""SELECT count(*) AS months,
        count(*) FILTER (WHERE NOT valid_cap_ranks) AS invalid_rank_months,
        count(*) FILTER (WHERE NOT rank1_pass) AS rank1_months_failing,
        count(*) FILTER (WHERE NOT top10_pass) AS top10_months_failing,
        max(rank1_dollar_volume_rank) AS max_rank1_dollar_volume_rank,
        max(top10_max_dollar_volume_rank) AS max_top10_dollar_volume_rank,
        {invalid_rows} AS invalid_rank_rows,{invalid_dv_rows} AS invalid_dollar_volume_rows
        FROM _cap_rank_months""",
    )[0]
    complete = (
        summary["months"] > 0
        and summary["invalid_rank_months"] == 0
        and invalid_rows == 0
        and invalid_dv_rows == 0
    )
    checks = [
        {
            "check": "universe_cap_ranks",
            "pass": complete,
            "value": f"{invalid_rows} invalid cap ranks; {invalid_dv_rows} invalid dollar-volume "
            f"ranks; {summary['invalid_rank_months']} invalid months",
            "criterion": "Finite positive caps in rank order and valid dollar-volume ranks",
        },
        {
            "check": "rank1_dollar_volume_rank",
            "pass": summary["months"] > 0 and summary["rank1_months_failing"] == 0,
            "value": summary["max_rank1_dollar_volume_rank"],
            "criterion": f"Largest member's dollar-volume rank <= {rank1_limit} each month",
        },
        {
            "check": "top10_dollar_volume_rank",
            "pass": summary["months"] > 0 and summary["top10_months_failing"] == 0,
            "value": summary["max_top10_dollar_volume_rank"],
            "criterion": f"Every top-10 member's dollar-volume rank <= {top10_limit} each month",
        },
    ]
    largest = rows(
        con,
        """SELECT ticker,name,count(*) AS months_at_rank1,
        min(date) AS first_month,max(date) AS last_month,
        max(rank1_dollar_volume_rank) AS max_dollar_volume_rank
        FROM _cap_rank_months WHERE rank1_count=1
        GROUP BY rank1_permaticker,ticker,name ORDER BY first_month,ticker""",
    )
    monthly = rows(
        con,
        """SELECT date,rank1_dollar_volume_rank,top10_max_dollar_volume_rank,
        top10_count,coalesce(invalid_top10_dollar_volume_ranks,0)
            AS invalid_top10_dollar_volume_ranks,
        rank1_pass,top10_pass
        FROM _cap_rank_months ORDER BY date""",
    )
    return checks, {
        "cap_rank_summary": [summary],
        "largest_universe_members": largest,
        "cap_dollar_volume_by_month": monthly,
    }


def _cap_tables(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict[str, Any]]]:
    """Describe cap repairs using only aggregates suitable for the repository."""
    return {
        "cap_coverage_by_year": rows(
            con,
            """SELECT year(date)::INTEGER AS year,
            count(*) AS months, min(marketcap_coverage) min_observed_coverage,
            min(usable_observed_marketcap_coverage) min_usable_observed_coverage,
            min(repaired_marketcap_coverage) min_repaired_coverage,
            sum(unusable_observed_caps) AS unusable_observed_stock_months,
            max(unusable_observed_caps) AS max_unusable_observed_caps,
            max(high_dollar_volume_missing_before) max_high_volume_missing_before,
            max(high_dollar_volume_missing_after) max_high_volume_missing_after,
            sum(repaired_members ORDER BY date)::DOUBLE/count(*) mean_repaired_members,
            max(repaired_members) max_repaired_members,
            100*sum(repaired_cap_share ORDER BY date)/count(*) mean_repaired_cap_share_pct,
            100*max(repaired_cap_share) max_repaired_cap_share_pct
            FROM universe_audit GROUP BY year(date) ORDER BY year""",
        ),
        "cap_repair_status": rows(
            con,
            """SELECT coalesce(cap_source,'none') cap_source,
            CASE WHEN accepted THEN 'accepted' ELSE exception END status,
            count(*) stock_months, count(DISTINCT permaticker) stocks,
            100*median(implied_turnover) median_implied_turnover_pct,
            100*max(implied_turnover) max_implied_turnover_pct
            FROM cap_repairs GROUP BY cap_source,status ORDER BY cap_source,status""",
        ),
        "cap_backfill_distances": rows(
            con,
            """SELECT accepted,
            count(*) stock_months, count(DISTINCT permaticker) stocks,
            min(distance_trading_days) min_trading_days,
            quantile_cont(distance_trading_days,0.25) p25_trading_days,
            median(distance_trading_days) median_trading_days,
            quantile_cont(distance_trading_days,0.75) p75_trading_days,
            quantile_cont(distance_trading_days,0.95) p95_trading_days,
            max(distance_trading_days) max_trading_days
            FROM cap_repairs WHERE cap_source='backfilled'
            GROUP BY accepted ORDER BY accepted DESC""",
        ),
        "observed_cap_exception_status": rows(
            con,
            """SELECT exception,
            count(*) AS stock_months,count(DISTINCT permaticker) AS stocks,
            100*min(implied_turnover) FILTER (WHERE isfinite(implied_turnover))
                AS min_implied_turnover_pct,
            100*median(implied_turnover) FILTER (WHERE isfinite(implied_turnover))
                AS median_implied_turnover_pct,
            100*max(implied_turnover) FILTER (WHERE isfinite(implied_turnover))
                AS max_implied_turnover_pct
            FROM observed_cap_exceptions GROUP BY exception ORDER BY exception""",
        ),
    }


def check(settings: dict[str, Any]) -> bool:
    """Run the data checks, save aggregate audits, and return whether all gates pass."""
    root = Path(settings["data"]["data_root"])
    curated = root / "curated"
    metadata_path = curated / "_metadata.json"
    if not metadata_path.exists():
        raise ValueError("Missing curation metadata; run data curate")
    curation = json.loads(metadata_path.read_text())
    manifest = Manifest(root)
    corrections = REPO / "config" / "price_corrections.csv"
    if not corrections.exists():
        raise ValueError("Missing config/price_corrections.csv")
    for key, path in (
        ("config_hash", Path(settings["config_path"])),
        ("data_manifest_hash", manifest.path),
        ("price_corrections_hash", corrections),
        ("cap_repairs_hash", curated / "cap_repairs.parquet"),
        ("observed_cap_exceptions_hash", curated / "observed_cap_exceptions.parquet"),
    ):
        if curation.get(key) != sha256_file(path):
            raise ValueError(f"Stale curation ({key} changed); rerun momenta data curate")
    required = [f"sharadar/{name}" for name in settings["data"]["sharadar_tables"]]
    required += [f"french/{name}" for name in FRENCH_FILES]
    required += [
        f"jkp/{name}_{frequency}" for name in JKP_FACTORS for frequency in ("monthly", "daily")
    ]
    required += [f"fred/{name}" for name in settings["data"]["fred_series"]]
    for name in required:
        if not manifest.verify(name):
            raise ValueError(f"Missing pinned source {name}; run data pull")
    metadata = provenance(REPO, Path(settings["config_path"]), manifest)
    metadata["curated_metadata_hash"] = sha256_file(metadata_path)
    metadata["price_corrections_hash"] = sha256_file(corrections)
    metadata["cap_repairs_hash"] = curation["cap_repairs_hash"]
    metadata["observed_cap_exceptions_hash"] = curation["observed_cap_exceptions_hash"]
    metadata["mode"] = curation["mode"]
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any, criterion: str) -> None:
        checks.append({"check": name, "pass": bool(passed), "value": value, "criterion": criterion})

    add(
        "raw_snapshots",
        True,
        len(required),
        "All required sources have verified SHA-256 and vintage",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    forbidden = [
        name
        for name in tracked
        if name == ".env"
        or name.startswith(("data/", "raw/", "curated/"))
        or Path(name).suffix in (".parquet", ".zip", ".gz", ".duckdb")
    ]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", ".env", "data/probe.csv", "probe.parquet"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    add(
        "repository_data_exclusion",
        not forbidden and len(ignored) == 3,
        f"{len(forbidden)} tracked data/secret files; {len(ignored)}/3 ignore probes",
        "No tracked licensed data or .env; ignore rules cover stray data",
    )
    with connect(settings) as con:
        for name in _CURATED_TABLES:
            path = curated / f"{name}.parquet"
            if not path.exists():
                raise ValueError(f"Missing curated table {name}; run data curate")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})")
        prices = _yearly_prices(con, curated)
        caps = _yearly_caps(con, curated)
        first_date = prices[0]["first_date"]
        add(
            "sep_full_history",
            first_date <= date.fromisoformat(settings["data"]["sep_required_start"]),
            first_date,
            f"First SEP date <= {settings['data']['sep_required_start']}",
        )
        for label, summaries in (("prices", prices), ("caps", caps)):
            for field in ("duplicate_rows", "missing_keys", "wrong_partition_rows"):
                value = sum(item[field] for item in summaries)
                add(f"{label}_{field}", value == 0, value, "0 rows")
        invalid = sum(item["invalid_price_rows"] for item in prices)
        add("invalid_prices", invalid == 0, invalid, "0 null, non-finite or non-positive prices")
        for name in (
            "master",
            "actions",
            "monthly_returns",
            "pit_sic",
            "universe",
            "cap_repairs",
            "cap_repair_exceptions",
            "observed_cap_exceptions",
        ):
            duplicates, missing = con.execute(f"""SELECT
                count(*) - count(DISTINCT (permaticker,date)),
                count(*) FILTER (WHERE permaticker IS NULL OR date IS NULL)
                FROM {name}""").fetchone()
            add(
                f"{name}_keys",
                duplicates == 0 and missing == 0,
                f"{duplicates} duplicate; {missing} missing",
                "Unique, non-null (permaticker, date)",
            )
        common_end = str(curation["common_end"])
        bound = min(common_end, curation.get("through") or common_end)
        universe_start = sql_path(settings["data"]["universe_start"])
        con.execute(f"""CREATE TEMP VIEW expected_months AS SELECT date FROM
            (SELECT max(date) date FROM trading_calendar GROUP BY year(date), month(date))
            WHERE date >= DATE {universe_start} AND date <= DATE {sql_path(bound)}""")
        missing_months = con.execute("""SELECT count(*) FROM expected_months e
            LEFT JOIN universe_audit a USING(date) WHERE a.date IS NULL""").fetchone()[0]
        extra_months = con.execute("""SELECT count(*) FROM universe_audit a
            LEFT JOIN expected_months e USING(date) WHERE e.date IS NULL""").fetchone()[0]
        add(
            "universe_month_ends",
            missing_months == 0 and extra_months == 0,
            f"{missing_months} missing; {extra_months} extra",
            "Every expected month-end",
        )
        months, shortfalls, invalid_sizes = con.execute(f"""SELECT
            count(*), count(*) FILTER (WHERE shortfall > 0),
            count(*) FILTER (WHERE universe_size > {int(settings["universe"]["n_universe"])}
                OR universe_size + shortfall != {int(settings["universe"]["n_universe"])}
                OR res_eligible > universe_size OR res_eligible < 0)
            FROM universe_audit""").fetchone()
        add(
            "universe_sizes",
            months > 0 and invalid_sizes == 0,
            f"{months} month-ends; {shortfalls} documented shortfalls",
            "Configured size or recorded shortfall",
        )
        checks.extend(_cap_checks(con, settings))
        checks.extend(_observed_cap_checks(con, settings))
        rank_checks, rank_tables = _cap_rank_checks(con, settings)
        checks.extend(rank_checks)
        rank_details = rows(con, "SELECT * FROM _cap_rank_months ORDER BY date")
        mismatches = con.execute("""SELECT count(*) FROM universe_audit a LEFT JOIN
            (SELECT date,count(*) n,count(*) FILTER (WHERE res_eligible) nr
             FROM universe GROUP BY date) u USING(date)
            WHERE a.universe_size != coalesce(u.n,0)
               OR a.res_eligible != coalesce(u.nr,0)""").fetchone()[0]
        add(
            "universe_audit_matches_members",
            mismatches == 0,
            mismatches,
            "0 mismatching month-ends",
        )
        sic = rows(con, "SELECT * FROM sic_audit ORDER BY year")
        sic_total, sic_known = con.execute("SELECT count(*),count(siccode) FROM pit_sic").fetchone()
        add(
            "point_in_time_sic",
            bool(sic) and sic_total > 0,
            f"{sic_known}/{sic_total} known SIC observations",
            "SIC table and annual action-coverage audit exist",
        )
        unknown_classes = con.execute("""SELECT count(*) FROM delistings
            WHERE NOT censored AND (event_class IS NULL OR event_class NOT IN (
            'acquisition_with_terms','election','acquisition_without_terms',
            'voluntary_delisting','performance_delisting','unexplained_end'))""").fetchone()[0]
        add(
            "delisting_classes",
            unknown_classes == 0,
            unknown_classes,
            "0 unclassified uncensored endings",
        )
        expected_exceptions = con.execute("""SELECT count(*) FROM delistings
            WHERE exception IS NOT NULL AND exception != ''""").fetchone()[0]
        exceptions = con.execute("SELECT count(*) FROM exceptions").fetchone()[0]
        censored = con.execute("SELECT count(*) FROM delistings WHERE censored").fetchone()[0]
        add(
            "delisting_exceptions",
            exceptions == expected_exceptions,
            exceptions,
            f"{expected_exceptions} classified exceptions retained outside git",
        )
        with corrections.open(newline="") as source:
            corrected = list(csv.DictReader(source))
        evidence = all(str(row.get("evidence", "")).strip() for row in corrected)
        add(
            "price_corrections",
            evidence,
            len(corrected),
            "Every correction has independent evidence",
        )
        flag_counts = rows(
            con,
            """SELECT flag,count(*) all_rows,
            count(*) FILTER (WHERE in_universe) universe_rows
            FROM flags GROUP BY flag ORDER BY flag""",
        )
        flags_by_kind = {row["flag"]: row for row in flag_counts}
        flag_counts = [
            flags_by_kind.get(
                kind,
                {
                    "flag": kind,
                    "all_rows": 0,
                    "universe_rows": 0,
                },
            )
            for kind in ("close_outside_range", "price_change_zero_volume", "spike_reversal")
        ]
        add(
            "data_flags",
            True,
            sum(row["universe_rows"] for row in flag_counts),
            "Universe flag rows reported diagnostically; no price rewriting",
        )
        tables = {
            **_cap_tables(con),
            **rank_tables,
            "coverage_by_year": _coverage(con, prices),
            "availability": _availability(manifest, con, prices, caps),
            "universe_by_month": rows(con, "SELECT * FROM universe_audit ORDER BY date"),
            "sic_by_year": sic,
            "delisting_value": _delisting_value(con),
            "data_flags": flag_counts,
            "price_integrity_by_year": prices,
        }
        mapping_path = curated / "mapping_audit.parquet"
        if mapping_path.exists():
            tables["security_mapping"] = rows(
                con,
                f"""SELECT * FROM
                read_parquet({sql_path(mapping_path)}) ORDER BY source,year""",
            )
            unmapped_prices = sum(
                row["unmapped"] for row in tables["security_mapping"] if row["source"] == "stocks"
            )
            add(
                "permanent_id_price_mapping",
                unmapped_prices == 0,
                unmapped_prices,
                "0 price rows without an unambiguous SEP permanent ID",
            )
    tables["data_checks"] = checks
    for name, records in tables.items():
        write_table(name, records, metadata)
    passed = all(item["pass"] for item in checks)
    summary = {
        "passed": passed,
        "checks": checks,
        "tables": tables,
        "provenance": metadata,
        "curation": curation,
        "exceptions": exceptions,
        "corrections": len(corrected),
        "censored_endings": censored,
    }
    destination = root / "derived" / "m1_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / "cap_rank_by_month.json").write_text(
        json.dumps({"provenance": metadata, "months": rank_details}, indent=2, default=str) + "\n"
    )
    destination.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    for item in checks:
        print(f"{'PASS' if item['pass'] else 'FAIL'} {item['check']}: {item['value']}")
    return passed
