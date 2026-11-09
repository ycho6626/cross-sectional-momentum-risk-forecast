"""Bounded, permanent-ID data curation for research (Methodology)."""


from __future__ import annotations

import bisect
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from momenta.data import extract_csv, stage_years
from momenta.events import Action, classify_delisting
from momenta.manifest import Manifest, provenance, sha256_file
from momenta.reference_data import curate_references
from momenta.settings import REPO, connect, sql_path

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "closeadj", "closeunadj")


def save_query(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    """Write a deterministic Parquet result from a bounded SQL query."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.part")
    con.execute(f"COPY ({query}) TO {sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    temporary.replace(path)


def _view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})")


def _corrections(con: duckdb.DuckDBPyConnection) -> int:
    """Load evidence-backed, vintage-bound correction instructions (Methodology)."""
    path = REPO / "config/price_corrections.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    con.execute("""CREATE TABLE corrections(permaticker BIGINT, date DATE, field VARCHAR,
                   old_value DOUBLE, new_value DOUBLE, evidence VARCHAR)""")
    seen = set()
    for row in rows:
        key = (int(row["permaticker"]), date.fromisoformat(row["date"]), row["field"])
        if key in seen or row["field"] not in PRICE_FIELDS or not row["evidence"].strip():
            raise ValueError("Corrections require unique keys, an allowed field and evidence")
        seen.add(key)
        old, new = float(row["old_value"]), float(row["new_value"])
        if not isfinite(old) or not isfinite(new) or (row["field"] == "volume" and new < 0):
            raise ValueError("Corrections require finite values and nonnegative volume")
        if new <= 0 and row["field"] != "volume":
            raise ValueError("Corrected prices must be positive")
        con.execute(
            "INSERT INTO corrections VALUES (?, ?, ?, ?, ?, ?)", [*key, old, new, row["evidence"]]
        )
    return len(rows)


def _master_actions(con: duckdb.DuckDBPyConnection, settings: dict, output: Path) -> None:
    root = Path(settings["data"]["data_root"])
    for name in ["tickers", "actions"]:
        path = extract_csv(root / f"raw/sharadar/{name}.zip", root / "raw/extracted")
        con.execute(f"""CREATE VIEW raw_{name} AS
                        SELECT * FROM read_csv({sql_path(path)}, all_varchar=true)""")
    # The source uses SEP even though the HTTP bulk endpoint is named stocks.
    con.execute("""CREATE TABLE master AS SELECT permaticker::BIGINT AS permaticker,
        ticker, name, category, try_cast(siccode AS INTEGER) AS siccode,
        lastupdated::DATE AS date FROM raw_tickers WHERE "table" = 'SEP' """)
    count = con.execute("SELECT count(*) FROM master").fetchone()[0]
    if count == 0:
        raise ValueError("No table='SEP' master records: refuse an inferred ticker mapping")
    for key in ["ticker", "permaticker"]:
        if con.execute(f"SELECT count(*)-count(DISTINCT {key}) FROM master").fetchone()[0]:
            raise ValueError(f"Ambiguous SEP security-master {key} mapping")
    save_query(con, "SELECT * FROM master ORDER BY permaticker", output / "master.parquet")
    con.execute("""CREATE TABLE actions_flat AS SELECT m.permaticker, a.date::DATE AS date,
        a.action, try_cast(a.value AS DOUBLE) AS value, a.contraticker,a.name,a.contraname,
        c.permaticker AS contra_permaticker
        FROM raw_actions a JOIN master m USING(ticker)
        LEFT JOIN master c ON c.ticker=a.contraticker""")
    duplicate = con.execute("""SELECT count(*) FROM (
        SELECT permaticker,date,action,contraticker,name,contraname,count(*) n FROM actions_flat
        GROUP BY ALL HAVING n>1)""").fetchone()[0]
    if duplicate:
        raise ValueError(f"{duplicate} duplicate corporate-action keys")
    save_query(
        con,
        """SELECT permaticker,date,
        list(struct_pack(action:=action,value:=value,contraticker:=contraticker,
                         contra_permaticker:=contra_permaticker,name:=name,contraname:=contraname)
             ORDER BY action,contraticker,name,contraname) AS events
        FROM actions_flat GROUP BY permaticker,date ORDER BY permaticker,date""",
        output / "actions.parquet",
    )
    con.execute("""CREATE TABLE mapping_audit(source VARCHAR,year INTEGER,
                   total BIGINT,unmapped BIGINT)""")
    con.execute("""INSERT INTO mapping_audit SELECT 'actions',year(a.date::DATE),count(*),
        count(*) FILTER(WHERE m.permaticker IS NULL)
        FROM raw_actions a LEFT JOIN master m USING(ticker) GROUP BY year(a.date::DATE)""")


def _yearly_prices(
    con: duckdb.DuckDBPyConnection, settings: dict, output: Path, through: date | None
) -> list[int]:
    root = Path(settings["data"]["data_root"])
    folders = sorted((root / "raw/staged/stocks").glob("source_year=*"))
    years = [int(f.name.split("=")[1]) for f in folders]
    if through:
        years = [y for y in years if y <= through.year]
    con.execute("""CREATE TABLE previous_price(permaticker BIGINT,ticker VARCHAR,date DATE,
        open DOUBLE,high DOUBLE,low DOUBLE,close DOUBLE,volume DOUBLE,
        closeadj DOUBLE,closeunadj DOUBLE)""")
    con.execute(
        "CREATE TABLE IF NOT EXISTS mapping_audit"
        "(source VARCHAR,year INTEGER,total BIGINT,unmapped BIGINT)"
    )
    for year in years:
        print(f"Curate prices {year}", flush=True)
        source = root / f"raw/staged/stocks/source_year={year}/*.parquet"
        con.execute(f"""CREATE OR REPLACE VIEW raw_year AS SELECT *
                        FROM read_parquet({sql_path(source)})""")
        total, unmapped = (
            con.execute("""SELECT count(*),count(*) FILTER(WHERE m.permaticker IS NULL)
            FROM raw_year p LEFT JOIN master m USING(ticker)""").fetchone()
        )
        con.execute("INSERT INTO mapping_audit VALUES ('stocks',?,?,?)", [year, total, unmapped])
        if unmapped:
            raise ValueError(
                f"{unmapped} stocks rows have no unambiguous SEP permanent ID in {year}"
            )
        fields = ",".join(f"p.{f}::DOUBLE AS {f}" for f in PRICE_FIELDS)
        con.execute(f"""CREATE OR REPLACE TABLE year_prices AS
            SELECT m.permaticker,p.ticker,p.date::DATE AS date,{fields}
            FROM raw_year p JOIN master m USING(ticker)""")
        for field in PRICE_FIELDS:
            mismatch = con.execute(f"""SELECT count(*) FROM corrections c
                LEFT JOIN year_prices p ON p.permaticker=c.permaticker AND p.date=c.date
                WHERE year(c.date)={year} AND c.field='{field}'
                AND (p.permaticker IS NULL OR
                     p.{field} IS DISTINCT FROM c.old_value)""").fetchone()[0]
            if mismatch:
                raise ValueError(f"{mismatch} correction old values do not match pinned prices")
            con.execute(f"""UPDATE year_prices AS p SET {field}=c.new_value FROM corrections c
                WHERE c.permaticker=p.permaticker AND c.date=p.date AND c.field='{field}' """)
        if con.execute("""SELECT count(*) FROM (SELECT permaticker,date,count(*) n
                          FROM year_prices GROUP BY ALL HAVING n>1)""").fetchone()[0]:
            raise ValueError(f"Duplicate price keys in {year}")
        if con.execute("""SELECT count(*) FROM year_prices WHERE close<=0 OR closeadj<=0
                          OR closeunadj<=0 OR close IS NULL OR closeadj IS NULL
                          OR closeunadj IS NULL""").fetchone()[0]:
            raise ValueError(
                f"Non-positive or missing closes in {year}; require evidenced correction"
            )
        # Calendar adjacency prevents a multi-day missing interval counting as one daily return.
        query = f"""WITH history AS (
            SELECT * FROM previous_price UNION ALL SELECT * FROM year_prices),
          lagged AS (SELECT *,lag(closeadj) OVER w AS prev_adj,
              lag(date) OVER w AS prev_date FROM history
              WINDOW w AS (PARTITION BY permaticker ORDER BY date))
          SELECT p.* EXCLUDE(prev_adj,prev_date),
            CASE WHEN c.day_index=pc.day_index+1 THEN closeadj/prev_adj-1 END AS daily_return
          FROM lagged p LEFT JOIN calendar c ON p.date=c.date
          LEFT JOIN calendar pc ON p.prev_date=pc.date WHERE year(p.date)={year}
          ORDER BY permaticker,date"""
        save_query(con, query, output / f"prices/year={year}/part.parquet")
        con.execute("""CREATE OR REPLACE TABLE previous_price AS SELECT * FROM year_prices
            QUALIFY row_number() OVER(PARTITION BY permaticker ORDER BY date DESC)=1""")
    _view(con, "prices", output / "prices/year=*/part.parquet")
    return years


def _caps(
    con: duckdb.DuckDBPyConnection,
    settings: dict,
    manifest: Manifest,
    output: Path,
    years: list[int],
) -> None:
    stage_years(settings, manifest, "daily")
    root = Path(settings["data"]["data_root"])
    for folder in sorted((root / "raw/staged/daily").glob("source_year=*")):
        year = int(folder.name.split("=")[1])
        if year not in years:
            continue
        print(f"Curate market caps {year}", flush=True)
        source = folder / "*.parquet"
        con.execute(f"""CREATE OR REPLACE VIEW raw_caps AS
                        SELECT * FROM read_parquet({sql_path(source)})""")
        total, unmapped = (
            con.execute("""SELECT count(*),count(*) FILTER(WHERE m.permaticker IS NULL)
            FROM raw_caps p LEFT JOIN master m USING(ticker)""").fetchone()
        )
        con.execute("INSERT INTO mapping_audit VALUES ('daily',?,?,?)", [year, total, unmapped])
        save_query(
            con,
            """SELECT m.permaticker,p.date::DATE AS date,
                p.marketcap::DOUBLE AS marketcap
            FROM raw_caps p JOIN master m USING(ticker) ORDER BY permaticker,date""",
            output / f"caps/year={year}/part.parquet",
        )
    _view(con, "caps", output / "caps/year=*/part.parquet")
    save_query(
        con, "SELECT * FROM mapping_audit ORDER BY source,year", output / "mapping_audit.parquet"
    )


def _delistings(
    con: duckdb.DuckDBPyConnection, settings: dict, output: Path, through: date | None
) -> tuple[date, date]:
    root = Path(settings["data"]["data_root"])
    raw = root / "raw/staged/stocks/source_year=*/*.parquet"
    # Full-series ends are used even in the thin run; truncation must never invent a delisting.
    con.execute(f"""CREATE TABLE stock_ends AS SELECT m.permaticker,p.ticker,
        max(p.date::DATE) AS last_date,arg_max(p.closeunadj::DOUBLE,p.date::DATE) AS last_close
        FROM read_parquet({sql_path(raw)}) p JOIN master m USING(ticker)
        GROUP BY m.permaticker,p.ticker""")
    panel_end = con.execute("SELECT max(last_date) FROM stock_ends").fetchone()[0]
    common_end = min(panel_end, con.execute("SELECT max(date) FROM calendar").fetchone()[0])
    limit = min(through, common_end) if through else panel_end
    candidates = con.execute(
        """SELECT permaticker,ticker,last_date,last_close FROM stock_ends
                               WHERE last_date<? AND last_date<=? ORDER BY permaticker""",
        [panel_end, limit],
    ).fetchall()
    days = [r[0] for r in con.execute("SELECT date FROM calendar ORDER BY date").fetchall()]
    # SEP-observed market dates extend event matching only, never the trading calendar.
    late_days = con.execute(
        f"""SELECT DISTINCT date::DATE AS d
        FROM read_parquet({sql_path(raw)}) WHERE date::DATE>?
        ORDER BY d""",
        [days[-1]],
    ).fetchall()
    event_days = days + [r[0] for r in late_days]
    # Late actions are counted/censored; no returns are invented beyond the French calendar.
    action_rows = con.execute("""SELECT permaticker,date,action,value,contraticker
        FROM actions_flat WHERE action IN ('acquisitioncash','acquisitionstock',
        'acquisitionelectcash','acquisitionelectstock','acquisitionby','mergerto',
        'voluntarydelisting','bankruptcyliquidation','regulatorydelisting','delisted')
        ORDER BY permaticker,date,action,contraticker""").fetchall()
    by_security: dict[int, list[Action]] = defaultdict(list)
    for p, d, a, v, c in action_rows:
        by_security[p].append(Action(d, a, v, c))
    requests = []
    matched = {}
    window = settings["delisting"]["match_window_days"]
    for p, _ticker, last, _close in candidates:
        index = bisect.bisect_left(days, last)
        credit = days[index + 1] if index + 1 < len(days) and days[index] == last else None
        event_index = bisect.bisect_left(event_days, last)
        selected = [
            a
            for a in by_security[p]
            if abs(bisect.bisect_left(event_days, a.date) - event_index) <= window
        ]
        matched[p] = (credit, selected)
        if credit:
            requests.extend((p, a.contraticker, credit) for a in selected if a.contraticker)
    prices: dict[int, dict[str, float]] = defaultdict(dict)
    if requests:
        con.execute("CREATE TEMP TABLE term_requests(permaticker BIGINT,ticker VARCHAR,date DATE)")
        con.executemany("INSERT INTO term_requests VALUES (?,?,?)", sorted(set(requests)))
        for p, ticker, price in con.execute("""SELECT r.permaticker,r.ticker,p.closeunadj
            FROM term_requests r JOIN master m ON m.ticker=r.ticker
            JOIN prices p ON p.permaticker=m.permaticker AND p.date=r.date""").fetchall():
            prices[p][ticker] = price
    records = []
    for p, ticker, last, close in candidates:
        credit, selected = matched[p]
        corrected = con.execute(
            """SELECT new_value FROM corrections WHERE permaticker=?
                                   AND date=? AND field='closeunadj' """,
            [p, last],
        ).fetchone()
        if corrected:
            close = corrected[0]
        result = classify_delisting(
            close,
            selected,
            prices[p],
            performance_return=settings["delisting"]["performance_return"],
            unexplained_return=settings["delisting"]["unexplained_return"],
            election_choice=settings["delisting"]["election_choice"],
        )
        record = {
            "permaticker": p,
            "ticker": ticker,
            "last_date": last,
            "date": credit,
            **asdict(result),
            "censored": credit is None,
        }
        if credit is None:
            record.update(
                terminal_return=None, terminal_value=None, exception="outside_french_calendar"
            )
        records.append(record)
    schema = {
        "permaticker": pl.Int64,
        "ticker": pl.String,
        "last_date": pl.Date,
        "date": pl.Date,
        "event_class": pl.String,
        "terminal_return": pl.Float64,
        "terminal_value": pl.Float64,
        "exception": pl.String,
        "election_choice": pl.String,
        "censored": pl.Boolean,
    }
    frame = pl.DataFrame(records, schema=schema)
    frame.write_parquet(output / "delistings.parquet")
    frame.filter(pl.col("exception").is_not_null()).write_parquet(output / "exceptions.parquet")
    _view(con, "delistings", output / "delistings.parquet")
    return panel_end, common_end


def _monthly(
    con: duckdb.DuckDBPyConnection, settings: dict, output: Path, years: list[int], common_end: date
) -> None:
    con.execute("""CREATE TABLE monthly_returns(permaticker BIGINT,date DATE,
                   monthly_return DOUBLE,observed_days BIGINT,expected_days BIGINT)""")
    for year in years:
        print(f"Compound monthly returns {year}", flush=True)
        # A terminal month is complete through the credit date; proceeds then remain flat.
        con.execute(
            """INSERT INTO monthly_returns WITH
          daily AS (SELECT permaticker,date,daily_return AS r FROM prices
                    WHERE year(date)=? AND date<=?
                    UNION ALL SELECT permaticker,date,terminal_return FROM delistings
                    WHERE NOT censored AND year(date)=? AND date<=?),
          counts AS (SELECT last_day(date) AS month,count(*) AS n FROM calendar
                     WHERE year(date)=? GROUP BY ALL HAVING max(date)<=?),
          aggregated AS (SELECT permaticker,last_day(date) AS month,
              product(1+r ORDER BY date)-1 AS compounded,count(r) AS n FROM daily GROUP BY ALL)
          SELECT a.permaticker,a.month,
              CASE WHEN a.n=c.n OR (d.permaticker IS NOT NULL AND NOT d.censored
                   AND a.n=(SELECT count(*) FROM calendar cal
                      WHERE last_day(cal.date)=a.month AND cal.date<=d.date))
                   THEN a.compounded END,a.n,c.n
          FROM aggregated a JOIN counts c ON a.month=c.month
          LEFT JOIN delistings d ON d.permaticker=a.permaticker
            AND last_day(d.date)=a.month""",
            [year, common_end, year, common_end, year, common_end],
        )
    save_query(
        con,
        "SELECT * FROM monthly_returns ORDER BY permaticker,date",
        output / "monthly_returns.parquet",
    )


def _flags(
    con: duckdb.DuckDBPyConnection,
    settings: dict,
    output: Path,
    years: list[int],
    end: date | None = None,
) -> None:
    con.execute("CREATE TABLE flags_raw(permaticker BIGINT,date DATE,flag VARCHAR)")
    threshold = settings["flags"]["spike_abs_return"]
    net = settings["flags"]["spike_net_two_day"]
    for year in years:
        print(f"Flag prices {year}", flush=True)
        con.execute(f"""INSERT INTO flags_raw WITH windowed AS (
          SELECT *,lead(daily_return) OVER w AS next_return,lead(date) OVER w AS next_date,
                 lag(close) OVER w AS prev_close
          FROM prices WHERE date BETWEEN DATE '{year - 1}-12-01' AND DATE '{year + 1}-01-31'
          WINDOW w AS (PARTITION BY permaticker ORDER BY date)),
          flags AS (
           SELECT permaticker,date,'close_outside_range' AS flag FROM windowed
            WHERE close<low OR close>high
           UNION ALL SELECT permaticker,date,'price_change_zero_volume' FROM windowed
            WHERE volume=0 AND close!=prev_close
           UNION ALL SELECT p.permaticker,p.date,'spike_reversal' FROM windowed p
            WHERE abs(daily_return)>{threshold} AND daily_return*next_return<0
            AND abs((1+daily_return)*(1+next_return)-1)<{net}
            AND NOT EXISTS(SELECT 1 FROM actions_flat a WHERE a.permaticker=p.permaticker
                           AND a.date IN (p.date,p.next_date)))
          SELECT * FROM flags WHERE year(date)={year}""")
    if end is not None:
        con.execute("DELETE FROM flags_raw WHERE date>?", [end])
    # Include empty month-end formations when ending prior membership.
    save_query(
        con,
        """WITH origins AS (SELECT date,lead(date) OVER(ORDER BY date) AS next_date
            FROM (SELECT max(date) AS date FROM calendar GROUP BY year(date),month(date)))
        SELECT f.*,EXISTS(SELECT 1 FROM universe u JOIN origins o ON u.date=o.date
            WHERE u.permaticker=f.permaticker AND f.date>u.date
            AND (f.date<=o.next_date OR o.next_date IS NULL)) AS in_universe
        FROM flags_raw f ORDER BY permaticker,date,flag""",
        output / "flags.parquet",
    )


def curate(settings: dict[str, Any], through: date | None = None) -> None:
    """Build the thin path or full curated data layer, then record its pinned provenance."""
    from momenta.universe import build_universe

    root = Path(settings["data"]["data_root"])
    manifest = Manifest(root)
    for source in manifest.entries:
        manifest.verify(source)
    output = root / "curated"
    output.mkdir(parents=True, exist_ok=True)
    (output / "_metadata.json").unlink(missing_ok=True)
    # These directories contain only our deterministic generated outputs.
    for name in ["prices", "caps"]:
        if (output / name).exists():
            shutil.rmtree(output / name)
    curate_references(settings, manifest)
    with connect(settings) as con:
        con.execute(f"""CREATE VIEW calendar AS SELECT date,
            row_number() OVER(ORDER BY date)-1 AS day_index
            FROM read_parquet({sql_path(output / "trading_calendar.parquet")})""")
        corrections = _corrections(con)
        _master_actions(con, settings, output)
        # Backfilled cap anchors can follow a thin build's formation dates.
        # Pin the full input history; --through limits monthly outputs only.
        years = _yearly_prices(con, settings, output, None)
        _caps(con, settings, manifest, output, years)
        panel_end, common_end = _delistings(con, settings, output, through)
        _monthly(con, settings, output, years, min(through, common_end) if through else common_end)
        build_universe(
            con, settings, output, through=min(through, common_end) if through else common_end,
            build_repairs=True,
        )
        _view(con, "universe", output / "universe.parquet")
        _flags(con, settings, output, years, min(through, common_end) if through else common_end)
        dates = [
            str(row[0])
            for row in con.execute("SELECT DISTINCT date FROM universe ORDER BY date").fetchall()
        ]
    metadata = {
        **provenance(REPO, Path(settings["config_path"]), manifest),
        "mode": "thin" if through else "full",
        "through": str(through) if through else None,
        "panel_end": str(panel_end),
        "common_end": str(common_end),
        "corrections": corrections,
        "price_corrections_hash": sha256_file(REPO / "config/price_corrections.csv"),
        "cap_repairs_hash": sha256_file(output / "cap_repairs.parquet"),
        "observed_cap_exceptions_hash": sha256_file(output / "observed_cap_exceptions.parquet"),
        "universe_dates": dates,
    }
    (output / "_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    for path in output.rglob("*.parquet"):
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    print(f"Curation complete: {len(dates)} month-end universes ({metadata['mode']})", flush=True)
