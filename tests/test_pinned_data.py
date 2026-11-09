"""Real-vintage universe leakage test; requires completed curation."""


from datetime import date
from pathlib import Path

import pytest

from momenta.settings import connect, load_settings, sql_path
from momenta.universe import build_universe

pytestmark = pytest.mark.slow


def test_pinned_universe_ignores_future_prices_caps_returns_and_terminal_actions(
    tmp_path: Path,
) -> None:
    """At t, U_m and RES eligibility are invariant to later observations (Methodology).

    This checks the actual pinned vintage against a rebuilt historical month,
    then damages all later price/cap/return rows across the full pinned vintage.
    Views avoid materializing that history; universe price scans stay bounded
    to the tested month and its required warm-up.
    Accepted cap repairs remain fixed input data under Methodology, including
    backfills whose anchor date follows the tested month.
    November 2000 has the same trading and calendar month-end, so changing
    monthly rows after t cannot change a month already fully observed at t.
    SIC history is preserved: its later change-from rows describe earlier SIC.
    Synthetic future delisting actions must never exclude a historical member.
    """
    settings = load_settings()
    settings["universe"]["cap_repair"] = True
    curated = Path(settings["data"]["data_root"]) / "curated"
    cutoff = date(2000, 11, 30)
    next_day = date(2000, 12, 1)
    required = [
        curated / name for name in (
            "trading_calendar.parquet", "master.parquet", "actions.parquet",
            "monthly_returns.parquet", "universe.parquet", "cap_repairs.parquet",
            "prices/year=1999/part.parquet", "prices/year=2000/part.parquet",
            "caps/year=1999/part.parquet", "caps/year=2000/part.parquet",
        )
    ]
    missing = [str(path.relative_to(curated)) for path in required if not path.exists()]
    if missing:
        pytest.skip('Pinned curation is absent: ' + ", ".join(missing))

    settings["data"]["universe_start"] = "2000-11-01"
    with connect(settings) as con:
        for name in ("master", "monthly_returns"):
            con.execute(f"""CREATE VIEW source_{name} AS SELECT *
                FROM read_parquet({sql_path(curated / f'{name}.parquet')})""")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM source_{name}")
        con.execute(f"""CREATE VIEW cap_repairs AS
            SELECT * FROM read_parquet({sql_path(curated / 'cap_repairs.parquet')})""")
        con.execute(f"""CREATE VIEW calendar AS SELECT date,
            row_number() OVER (ORDER BY date) - 1 AS day_index
            FROM read_parquet({sql_path(curated / 'trading_calendar.parquet')})""")
        con.execute(f"""CREATE VIEW source_actions AS
            SELECT a.permaticker,a.date,e.action,e.value,e.contraticker
            FROM read_parquet({sql_path(curated / 'actions.parquet')}) a,
                UNNEST(a.events) AS event(e)""")
        con.execute("CREATE VIEW actions_flat AS SELECT * FROM source_actions")
        # Include actual later anchors, even years after t, without materializing prices.
        for name in ("prices", "caps"):
            con.execute(f"""CREATE VIEW source_{name} AS
                SELECT * FROM read_parquet({sql_path(curated / f'{name}/year=*/part.parquet')})""")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM source_{name}")

        pinned = con.execute("""SELECT * FROM read_parquet(?)
            WHERE date = ? ORDER BY rank""", [
            str(curated / "universe.parquet"), cutoff,
        ]).fetchall()
        assert pinned, 'Pinned universe lacks 2000-11-30; complete the curation first'
        backfilled_members = con.execute("""SELECT count(*)
            FROM read_parquet(?) u JOIN cap_repairs r USING (permaticker,date)
            WHERE u.date=? AND u.cap_source='backfilled' AND r.accepted AND r.t0>u.date""", [
            str(curated / "universe.parquet"), cutoff,
        ]).fetchone()[0]
        assert backfilled_members > 0, "Leakage fixture must exercise later-anchor pinned repairs"
        available_later_anchors = con.execute("""SELECT count(*)
            FROM read_parquet(?) u JOIN cap_repairs r USING (permaticker,date)
            JOIN source_caps c ON c.permaticker=r.permaticker AND c.date=r.t0
            WHERE u.date=? AND u.cap_source='backfilled' AND r.accepted
                AND r.t0>u.date AND c.marketcap=r.anchor_marketcap""", [
            str(curated / "universe.parquet"), cutoff,
        ]).fetchone()[0]
        assert available_later_anchors == backfilled_members
        baseline = tmp_path / "baseline"
        build_universe(con, settings, baseline, through=cutoff)
        before = con.execute("SELECT * FROM read_parquet(?) ORDER BY rank", [
            str(baseline / "universe.parquet"),
        ]).fetchall()
        assert before == pinned

        # Nonempty changed sets keep this invariance test from passing vacuously.
        for query in (
            "SELECT count(*) FROM source_prices WHERE date > ? AND daily_return IS NOT NULL",
            "SELECT count(*) FROM source_caps WHERE date > ? AND marketcap > 0",
            "SELECT count(*) FROM source_monthly_returns "
            "WHERE date > ? AND monthly_return IS NOT NULL",
        ):
            assert con.execute(query, [cutoff]).fetchone()[0] > 0
        con.execute(f"""CREATE OR REPLACE VIEW prices AS
            SELECT * REPLACE (
                CASE WHEN date > DATE '{cutoff}' THEN 0.01 ELSE closeunadj END AS closeunadj,
                CASE WHEN date > DATE '{cutoff}' THEN 1e10 ELSE close END AS close,
                CASE WHEN date > DATE '{cutoff}' THEN 1e12 ELSE volume END AS volume,
                CASE WHEN date > DATE '{cutoff}' THEN NULL ELSE daily_return END AS daily_return)
            FROM source_prices""")
        con.execute(f"""CREATE OR REPLACE VIEW caps AS
            SELECT * REPLACE (CASE WHEN date > DATE '{cutoff}'
                THEN 1e15 / permaticker ELSE marketcap END AS marketcap)
            FROM source_caps""")
        con.execute(f"""CREATE OR REPLACE VIEW monthly_returns AS
            SELECT * REPLACE (CASE WHEN date > DATE '{cutoff}'
                THEN NULL ELSE monthly_return END AS monthly_return)
            FROM source_monthly_returns""")
        con.execute(f"""CREATE OR REPLACE VIEW actions_flat AS
            SELECT * FROM source_actions UNION ALL
            SELECT permaticker,DATE '{next_day}',action,0.0,NULL::VARCHAR
            FROM master CROSS JOIN (VALUES ('delisted'),('bankruptcyliquidation'),
                ('acquisitioncash')) AS noise(action)""")

        perturbed = tmp_path / "perturbed"
        build_universe(con, settings, perturbed, through=cutoff)
        after = con.execute("SELECT * FROM read_parquet(?) ORDER BY rank", [
            str(perturbed / "universe.parquet"),
        ]).fetchall()
        # Full rows include the cap, rank, and RES flag, not just membership.
        assert after == before
        for name, order in (("universe_audit", "date"), ("pit_sic", "permaticker,date")):
            original = con.execute(f"SELECT * FROM read_parquet(?) ORDER BY {order}", [
                str(baseline / f"{name}.parquet"),
            ]).fetchall()
            actual = con.execute(f"SELECT * FROM read_parquet(?) ORDER BY {order}", [
                str(perturbed / f"{name}.parquet"),
            ]).fetchall()
            assert actual == original
