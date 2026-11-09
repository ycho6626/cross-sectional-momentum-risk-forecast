"""Bounded real-vintage feature leakage test; the data and portfolio tests cover U and holdings
(Methodology)."""


from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from momenta.features import (
    _stock_panel,
    cross_features,
    macro_features,
    market_features,
    own_features,
)
from momenta.manifest import sha256_file
from momenta.settings import load_settings

pytestmark = pytest.mark.slow


def test_pinned_features_ignore_future_stock_book_factor_signal_weight_and_macro_data() -> None:
    """Damage December 2005 inputs; all features through November must remain identical.

    U and repaired caps are fixed; test_pinned_data.py separately rebuilds U
    under future input damage, and test_pinned_portfolios.py tests its holdings.
    This test never fits a forecasting model or evaluates a holdout origin.
    """
    settings = load_settings()
    root = Path(settings["data"]["data_root"])
    curated, portfolio_dir = root/"curated", root/'derived/portfolios'
    required = [portfolio_dir/"signals.parquet", portfolio_dir/"formation_weights.parquet", portfolio_dir/"book_daily.parquet",  # noqa: E501
                curated/"cap_repairs.parquet", curated/"prices/year=2005/part.parquet"]
    if any(not path.exists() for path in required):
        pytest.skip('Pinned curation and portfolio artifacts are absent')
    cutoff, last = date(2005, 11, 30), date(2005, 12, 30)
    start = date(2004, 6, 1)
    repair = sha256_file(curated/"cap_repairs.parquet")
    factors = pl.read_parquet(curated/"french/ff3_daily.parquet").filter(
        pl.col("date") <= last,
    )
    calendar = factors["date"].to_list()
    days = [day for day in calendar if start <= day <= last]
    universe = pl.read_parquet(curated/"universe.parquet").filter(
        pl.col("date").is_between(date(2004, 5, 1), last),
    ).select("date", "permaticker")
    signals = pl.read_parquet(portfolio_dir/"signals.parquet").filter(
        pl.col("date").is_between(date(2004, 5, 1), last),
    )
    weights = pl.read_parquet(portfolio_dir/"formation_weights.parquet").filter(
        pl.col("formation").is_between(date(2004, 5, 1), last),
    )
    books = pl.read_parquet(portfolio_dir/"book_daily.parquet").filter(
        pl.col("construction") == settings["portfolios"]["primary"], pl.col("date") <= last,
    )
    terminal = pl.read_parquet(curated/"delistings.parquet").select(
        "date", "permaticker", "terminal_return",
    )
    ids = np.sort(universe["permaticker"].unique().to_numpy())
    returns, dv, _ = _stock_panel(curated, days, ids, terminal)
    future_rows = np.array([day > cutoff for day in days])
    assert np.isfinite(returns[future_rows]).sum() > 1000
    assert np.isfinite(dv[future_rows]).sum() > 1000
    damaged_returns, damaged_dv = returns.copy(), dv.copy()
    damaged_returns[future_rows] = 1 + np.abs(damaged_returns[future_rows])
    damaged_dv[future_rows] = 1 + np.abs(damaged_dv[future_rows])*100
    future = pl.col("date") > cutoff
    damaged_factors = factors.with_columns([
        pl.when(future).then(1+pl.col(column).abs()).otherwise(pl.col(column)).alias(column)
        for column in ("mkt_rf", "smb", "hml", "rf")
    ])
    damaged_signals = signals.with_columns(
        pl.when(future).then(-100-pl.col("raw")).otherwise(pl.col("raw")).alias("raw"),
    )
    damaged_weights = weights.with_columns(
        pl.when(pl.col("formation") > cutoff).then(-pl.col("weight"))
        .otherwise(pl.col("weight")).alias("weight"),
    )
    assert damaged_signals.filter(future).height > 0
    assert not damaged_weights.equals(weights)
    args = (days, ids, returns, dv, universe, signals, weights, factors, calendar, settings)
    before = cross_features(*args).filter(pl.col("date") <= cutoff)
    after = cross_features(days, ids, damaged_returns, damaged_dv, universe, damaged_signals,
                           damaged_weights, damaged_factors, calendar, settings)
    after = after.filter(pl.col("date") <= cutoff)
    assert before.filter(pl.col("comom").is_finite()).height > 0
    assert before.filter(pl.col("book_rv_126").is_finite()).height > 0
    assert_frame_equal(before, after, check_exact=True)
    changed_books = books.with_columns(
        pl.when(future).then(2+pl.col("daily_return").abs())
        .otherwise(pl.col("daily_return")).alias("daily_return"),
    )
    assert books.filter(future).height > 0
    assert_frame_equal(own_features(books, factors, settings).filter(pl.col("date") <= cutoff),
                       own_features(changed_books, damaged_factors, settings).filter(
                           pl.col("date") <= cutoff), check_exact=True)
    assert_frame_equal(market_features(factors, settings).filter(pl.col("date") <= cutoff),
                       market_features(damaged_factors, settings).filter(pl.col("date") <= cutoff),
                       check_exact=True)
    fred = {name: pl.read_parquet(curated/f"fred/{name}.parquet")
            for name in ("VIXCLS", "BAA10Y", "T10Y3M")}
    changed_fred = {name: frame.with_columns(
        pl.when(future).then(10000+pl.col("value").abs())
        .otherwise(pl.col("value")).alias("value"),
    ) for name, frame in fred.items()}
    assert all(frame.filter(future & pl.col("value").is_not_null()).height
               for frame in fred.values())
    dates = pl.DataFrame({"date": days})
    assert_frame_equal(macro_features(dates, fred).filter(pl.col("date") <= cutoff),
                       macro_features(dates, changed_fred).filter(pl.col("date") <= cutoff),
                       check_exact=True)
    assert sha256_file(curated/"cap_repairs.parquet") == repair
