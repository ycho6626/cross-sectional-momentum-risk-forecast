"""Independent formation and dated-accounting fixtures for Methodology."""


from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from momenta.portfolios import form_portfolios, run_portfolios, tail_weights
from momenta.settings import load_settings


def test_full_universe_cap_percentile_is_shared_by_res_and_raw_control():
    settings = load_settings()
    signals = pl.DataFrame({
        "date": [date(2000, 1, 31)] * 30, "permaticker": list(range(1, 31)),
        "marketcap": [float(i) for i in range(1, 30)] + [100.0],
        "raw": [float(i) for i in range(30)], "res": [float(i) for i in range(30)],
        "res_eligible": [i < 10 or i == 30 for i in range(1, 31)],
        "cap_source": ["observed"] * 29 + ["backfilled"],
    })
    formed = form_portfolios(signals, settings)
    # Linear p80 of all 30 caps is 24 + 0.2*(25-24) = 24.2.
    assert formed["cap_p80"].to_numpy() == pytest.approx(24.2)
    for series in ("RES", "RAW_c"):
        leg = formed.filter(
            pl.col("series") == series, pl.col("construction") == "tercile_capped_vw",
            pl.col("weight") > 0,
        ).sort("permaticker")
        assert leg["permaticker"].to_list() == [7, 8, 9, 30]
        assert leg["weight"].to_list() == pytest.approx([35/241, 40/241, 45/241, 121/241])
    decile = formed.filter(
        pl.col("series") == "RAW", pl.col("construction") == "decile_vw",
        pl.col("weight") > 0,
    ).sort("permaticker")
    assert decile["weight"].to_list() == pytest.approx([28/157, 29/157, 100/157])
    assert not formed.filter(
        pl.col("series") == "RAW_c", pl.col("construction") == "decile_vw"
    ).height


def test_equal_signal_values_share_the_upper_ecdf_bin():
    signal = np.array([0, 0, 1, 2, 3, 4.0])
    weights = tail_weights(signal, np.ones(6), bins=3, cap_ceiling=None)
    assert weights == pytest.approx([-0.5, -0.5, 0, 0, 0.5, 0.5])
    with pytest.raises(ValueError, match="empty extreme"):
        tail_weights(np.ones(6), np.ones(6), bins=3, cap_ceiling=None)


def _inputs(tmp_path, days, returns, terminals=()):
    curated = tmp_path / "curated"
    prices = curated / "prices/year=2000"
    prices.mkdir(parents=True)
    pl.DataFrame([
        {"permaticker": p, "date": d, "daily_return": r}
        for d, row in zip(days, returns, strict=True)
        for p, r in enumerate(row, 1) if r is not None
    ]).write_parquet(prices / "part.parquet")
    pl.DataFrame(
        terminals, schema={"permaticker": pl.Int64, "date": pl.Date, "terminal_return": pl.Float64},
        orient="row",
    ).write_parquet(curated / "delistings.parquet")
    return curated


def _weights(formations):
    return pl.DataFrame([
        {"formation": day, "series": "RAW", "construction": "tercile_capped_vw",
         "permaticker": p, "weight": float(w)}
        for day, row in formations for p, w in enumerate(row, 1) if w
    ])


def test_first_holding_day_earns_previous_book_before_execution(tmp_path: Path):
    days = [date(2000, 2, 1), date(2000, 2, 29), date(2000, 3, 1), date(2000, 3, 2)]
    curated = _inputs(tmp_path, days, [[0]*4, [0]*4, [.1, 0, .5, -.5], [0, 0, 0, .1]])
    weights = _weights([(date(2000, 1, 31), [1, -1, 0, 0]),
                        (date(2000, 2, 29), [0, 0, 1, -1])])
    output = tmp_path / "result"
    run_portfolios(weights, days, curated, output, load_settings())
    book = pl.read_parquet(output / "book_daily.parquet")
    assert book["daily_return"].to_list() == pytest.approx([0, 0, .1, -.1])
    assert book["nav"].to_list() == pytest.approx([1, 1, 1.1, .99])
    academic = pl.read_parquet(output / "academic_daily.parquet")
    assert academic["wml"].to_list() == pytest.approx([0, 0, 1, -.1])
    rebalance = pl.read_parquet(output / "rebalance_summary.parquet")
    assert rebalance["turnover"].to_list() == pytest.approx([2, 43/11])
    march = pl.read_parquet(output / "academic_monthly.parquet").row(1, named=True)
    # Winner grows 1.5; loser grows 0.5*1.1=0.55. Difference = 0.95.
    assert march["wml"] == pytest.approx(.95)
    ledger = pl.read_parquet(output / "ledger/year=2000/part.parquet")
    assert ledger.filter(pl.col("date") == days[2])["turnover"].sum() == pytest.approx(43/11)


def test_terminal_credit_renormalizes_academic_but_book_holds_cash(tmp_path: Path):
    days = [date(2000, 2, 1), date(2000, 2, 2), date(2000, 2, 3)]
    curated = _inputs(tmp_path, days, [[0, 0, 0], [None, .2, 0], [None, .1, 0]],
                      [(1, days[1], -.4)])
    output = tmp_path / "result"
    audit = run_portfolios(_weights([(date(2000, 1, 31), [.5, .5, -1])]),
                           days, curated, output, load_settings())
    book = pl.read_parquet(output / "book_daily.parquet")
    assert book["nav"].to_list() == pytest.approx([1, .9, .96])
    assert book["cash"].to_list() == pytest.approx([1, 1.3, 1.3])
    assert book["terminal_cash"].to_list() == pytest.approx([0, .3, 0])
    monthly = pl.read_parquet(output / "academic_monthly.parquet")
    assert monthly["wml"].item() == pytest.approx(-.01)
    assert audit["terminal_credits"] == 1
    assert audit["missing_held_returns"] == 0


def test_terminal_target_on_execution_day_stays_in_cash_without_reallocation(tmp_path: Path):
    days = [date(2000, 2, 1), date(2000, 2, 2)]
    curated = _inputs(tmp_path, days, [[None, 0, 0], [None, .1, 0]], [(1, days[0], -.4)])
    output = tmp_path / "result"
    audit = run_portfolios(_weights([(date(2000, 1, 31), [.5, .5, -1])]),
                           days, curated, output, load_settings())
    book = pl.read_parquet(output / "book_daily.parquet")
    assert book["nav"].to_list() == pytest.approx([1, 1.05])
    assert book["gross_exposure"][0] == pytest.approx(1.5)
    assert audit["unavailable_execution_targets"] == 1
    ledger = pl.read_parquet(output / "ledger/year=2000/part.parquet")
    assert 1 not in ledger["permaticker"].to_list()


def test_missing_return_on_held_day_fails_loudly(tmp_path: Path):
    days = [date(2000, 2, 1), date(2000, 2, 2)]
    curated = _inputs(tmp_path, days, [[0, 0], [None, 0]])
    with pytest.raises(ValueError, match="Missing held return on 2000-02-02.*permatickers \\[1\\]"):
        run_portfolios(_weights([(date(2000, 1, 31), [1, -1])]), days,
                       curated, tmp_path / "result", load_settings())
