"""Real-vintage signal, formation, and ledger leakage checks (Methodology)."""


from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from momenta.manifest import sha256_file
from momenta.portfolios import form_portfolios, run_portfolios
from momenta.settings import load_settings
from momenta.signals import build_signals

pytestmark = pytest.mark.slow


def test_pinned_signals_weights_and_book_ignore_future_observations(tmp_path: Path) -> None:
    """Later observations cannot affect signals, weights, or holdings at t.

    Use actual pinned October/November 2000 formations and hold through November
    30. Monthly returns and factors include later history; selected-ID prices
    include the whole of 2000, including December. Delisting copies retain all
    selected securities' later events. Damage every supplied observation after
    t without changing any earlier observation or the pinned cap repairs.

    Universe rebuilding under later price/cap/return/action perturbations is
    covered by test_pinned_data.py's
    test_pinned_universe_ignores_future_prices_caps_returns_and_terminal_actions.
    Here U_m and its accepted repaired caps remain fixed, as Methodology requires.
    Both t and the preceding formation are calendar and trading month-ends.
    """
    settings = load_settings()
    curated = Path(settings["data"]["data_root"]) / "curated"
    cutoff, previous_formation = date(2000, 11, 30), date(2000, 10, 31)
    required = [
        curated / name for name in (
            "universe.parquet", "monthly_returns.parquet", "french/ff3_monthly.parquet",
            "trading_calendar.parquet", "cap_repairs.parquet", "delistings.parquet",
            "prices/year=2000/part.parquet",
        )
    ]
    missing = [str(path.relative_to(curated)) for path in required if not path.exists()]
    if missing:
        pytest.skip('Pinned curation is absent: ' + ", ".join(missing))
    repair_path = curated / "cap_repairs.parquet"
    repair_hash = sha256_file(repair_path)
    universe = pl.scan_parquet(curated / "universe.parquet").filter(
        pl.col("date").is_in([previous_formation, cutoff]),
    ).collect().sort("date", "permaticker")
    assert set(universe["date"].to_list()) == {previous_formation, cutoff}
    assert universe.filter("res_eligible").height > 0
    assert universe.filter(pl.col("cap_source") == "backfilled").height > 0

    ids = universe["permaticker"].unique().implode()
    monthly = pl.scan_parquet(curated / "monthly_returns.parquet").filter(
        pl.col("permaticker").is_in(ids),
    ).select("permaticker", "date", "monthly_return").collect()
    factors = pl.read_parquet(curated / "french/ff3_monthly.parquet")
    future = pl.col("date") > cutoff
    assert monthly.filter(future & pl.col("monthly_return").is_not_null()).height > 0
    assert factors.filter(future).height > 0
    damaged_monthly = monthly.with_columns(
        pl.when(future).then(1000 + pl.col("monthly_return").abs().fill_null(0))
        .otherwise(pl.col("monthly_return")).alias("monthly_return"),
    )
    damaged_factors = factors.with_columns([
        pl.when(future).then(10000 + pl.col(column).abs().fill_null(0))
        .otherwise(pl.col(column)).alias(column)
        for column in ("mkt_rf", "smb", "hml", "rf")
    ])
    assert not damaged_monthly.filter(future).equals(monthly.filter(future))
    assert not damaged_factors.filter(future).equals(factors.filter(future))

    signals = build_signals(monthly, factors, universe, settings)
    damaged_signals = build_signals(damaged_monthly, damaged_factors, universe, settings)
    assert_frame_equal(signals, damaged_signals, check_exact=True)
    assert_frame_equal(signals.select(universe.columns), universe, check_exact=True)
    weights = form_portfolios(signals, settings)
    damaged_weights = form_portfolios(damaged_signals, settings)
    assert_frame_equal(weights, damaged_weights, check_exact=True)
    assert set(weights["formation"].to_list()) == {previous_formation, cutoff}
    assert set(weights["series"].to_list()) == {"RAW", "RAW_c", "RES"}

    held_ids = weights["permaticker"].unique().implode()
    prices = pl.scan_parquet(curated / "prices/year=2000/part.parquet").filter(
        pl.col("permaticker").is_in(held_ids),
    ).select("permaticker", "date", "daily_return").collect()
    delistings = pl.scan_parquet(curated / "delistings.parquet").filter(
        pl.col("permaticker").is_in(held_ids),
    ).select("permaticker", "date", "terminal_return").collect()
    assert prices.filter(future & pl.col("daily_return").is_not_null()).height > 0
    assert prices["date"].max() >= date(2000, 12, 29)
    assert delistings.filter(future & pl.col("terminal_return").is_not_null()).height > 0
    damaged_prices = prices.with_columns(
        pl.when(future).then(100 + pl.col("daily_return").abs().fill_null(0))
        .otherwise(pl.col("daily_return")).alias("daily_return"),
    )
    damaged_delistings = delistings.with_columns(
        pl.when(future).then(100 + pl.col("terminal_return").abs().fill_null(0))
        .otherwise(pl.col("terminal_return")).alias("terminal_return"),
        # Future events stay future; no historical terminal credit is moved.
        pl.when(future).then(pl.col("date") + pl.duration(days=31))
        .otherwise(pl.col("date")).alias("date"),
    )
    assert not damaged_prices.filter(future).equals(prices.filter(future))
    assert not damaged_delistings.filter(future).equals(delistings.filter(future))
    for original, changed in (
        (monthly, damaged_monthly), (factors, damaged_factors),
        (prices, damaged_prices), (delistings, damaged_delistings),
    ):
        assert_frame_equal(original.filter(~future), changed.filter(~future), check_exact=True)

    calendar = pl.read_parquet(curated / "trading_calendar.parquet").filter(
        pl.col("date") > previous_formation, pl.col("date") <= cutoff,
    )["date"].to_list()
    assert calendar[0] == date(2000, 11, 1)
    assert calendar[-1] == cutoff
    outputs: list[Path] = []
    audits = []
    for name, formation_weights, daily_data, terminal_data in (
        ("baseline", weights, prices, delistings),
        ("perturbed", damaged_weights, damaged_prices, damaged_delistings),
    ):
        copied = tmp_path / name / "curated"
        price_dir = copied / "prices/year=2000"
        price_dir.mkdir(parents=True)
        daily_data.write_parquet(price_dir / "part.parquet")
        terminal_data.write_parquet(copied / "delistings.parquet")
        output = tmp_path / name / "output"
        audits.append(run_portfolios(formation_weights, calendar, copied, output, settings))
        outputs.append(output)
    assert audits[0] == audits[1]
    assert audits[0]["held_position_days"] > 0
    tables = (
        "academic_daily.parquet", "academic_monthly.parquet", "book_daily.parquet",
        "book_monthly.parquet", "rebalance_summary.parquet", "final_positions.parquet",
        "ledger/year=2000/part.parquet",
    )
    for relative in tables:
        before = pl.read_parquet(outputs[0] / relative)
        after = pl.read_parquet(outputs[1] / relative)
        assert before.height > 0, f"Leakage fixture must exercise {relative}"
        keys = [column for column in ("date", "series", "construction", "permaticker")
                if column in before.columns]
        assert_frame_equal(before.sort(keys), after.sort(keys), check_exact=True)
        assert before["date"].max() <= cutoff
        if relative == "final_positions.parquet":
            assert before["date"].min() == cutoff
        if relative.startswith("ledger/"):
            assert set(before["formation"].to_list()) == {previous_formation}
    assert sha256_file(repair_path) == repair_hash, "Pinned cap repairs must remain unchanged"
