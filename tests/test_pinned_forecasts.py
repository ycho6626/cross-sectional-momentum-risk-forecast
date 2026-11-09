"""Real-vintage forecasting and overlay leakage, completing's chain."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from momenta import baseline_study
from momenta.features import FEATURE_SETS
from momenta.manifest import sha256_file
from momenta.settings import load_settings
from momenta.strategies import simulate_strategy, volatility_multiples
from momenta.walkforward import walk_forward

pytestmark = pytest.mark.slow


def _damage_after(
    frame: pl.DataFrame, time_column: str, cutoff: date, columns: list[str]
) -> pl.DataFrame:
    return frame.with_columns(
        [
            pl.when(pl.col(time_column) > cutoff)
            .then(10 + pl.col(column).abs().fill_null(0))
            .otherwise(pl.col(column))
            .alias(column)
            for column in columns
        ]
    )


def _causal_bdp_transform(settings, books, monkeypatch):
    """Use production transform and cached pre-cutoff fits; do not refit/log in tests."""
    from momenta.bdp import BDPFit, filter_bdp

    output = Path(settings["data"]["data_root"]) / "derived/baseline"

    def pinned_paths(current_settings, cutoff):
        path = output / "bdp_paths" / f"{cutoff}.json"
        assert path.exists(), f"Run the baseline study to create its dated BDP fit: {path.name}"
        record = json.loads(path.read_text())
        assert date.fromisoformat(record["fit_time"]) == cutoff
        assert date.fromisoformat(record["french_last_observation"]) < cutoff
        fit = BDPFit(**record["fit"])
        states = filter_bdp(100 * books["daily_return"].to_numpy(), fit)
        paths = books.select("date", "series").with_columns(
            pl.Series("bdp_ci", np.exp(states[:, 1]) * np.tanh(states[:, 2]) / 100)
        )
        return paths, fit, record

    monkeypatch.setattr(baseline_study, "_bdp_paths", pinned_paths)
    return baseline_study._structural_transform(settings, True)


def test_pinned_forecasts_weights_and_positions_ignore_future_data(monkeypatch):
    """Future damage cannot change any learned fit, decision or executed holding.

    The data, portfolio and feature tests independently cover raw inputs → U/signals/features.
    This composition test holds the pinned repair table fixed, changes every
    supplied future feature and unresolved outcome, then propagates the same
    forecasts into the actual ledger. January's origin executes in February,
    so only the December decision may affect January's positions.
    """
    settings = load_settings()
    root = Path(settings["data"]["data_root"])
    curated, portfolio_dir = root / "curated", root / "derived/portfolios"
    baseline_dir = root / "derived/baseline"
    cutoff, future_end = date(2006, 1, 31), date(2006, 2, 28)
    required = [baseline_dir / name
                for name in ("features.parquet", "targets.parquet", "bdp_decision.json")]
    required += [
        portfolio_dir / "book_daily.parquet",
        portfolio_dir / "ledger/year=2006/part.parquet",
        curated / "trading_calendar.parquet",
        curated / "cap_repairs.parquet",
    ]
    if any(not path.exists() for path in required):
        pytest.skip("Pinned baseline inputs and portfolio books are required")
    repair_hash = sha256_file(curated / "cap_repairs.parquet")
    calendar = pl.read_parquet(curated / "trading_calendar.parquet")["date"].to_list()
    primary = settings["portfolios"]["primary"]
    features = (
        pl.scan_parquet(baseline_dir / "features.parquet")
        .filter(
            pl.col("series") == "RAW",
            pl.col("date") <= future_end,
        )
        .collect()
        .sort("date")
    )
    targets = (
        pl.scan_parquet(baseline_dir / "targets.parquet")
        .filter(
            pl.col("series") == "RAW",
            pl.col("origin") <= future_end,
            pl.col("construction") == primary,
        )
        .collect()
        .sort("origin")
    )
    books = (
        pl.scan_parquet(portfolio_dir / "book_daily.parquet")
        .filter(
            pl.col("series") == "RAW",
            pl.col("construction") == primary,
            pl.col("date") <= future_end,
        )
        .collect()
        .sort("date")
    )
    ledger = (
        pl.scan_parquet(portfolio_dir / "ledger/year=2006/part.parquet")
        .filter(
            pl.col("series") == "RAW",
            pl.col("construction") == primary,
            pl.col("date") <= future_end,
        )
        .collect()
        .sort("date", "permaticker")
    )
    assert features["date"].max() == future_end, "Build baseline inputs through February 2006"
    assert targets.filter(pl.col("origin") == cutoff).height == 1
    assert ledger.filter(pl.col("date") > cutoff).height > 0
    feature_columns = sorted(set(name for columns in FEATURE_SETS.values() for name in columns))
    changed_features = _damage_after(features, "date", cutoff, feature_columns)
    changed_targets = _damage_after(targets, "target_end", cutoff, ["y1", "y2"])
    changed_books = _damage_after(
        books, "date", cutoff, ["nav", "cash", "gross_exposure", "terminal_cash"]
    ).with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(0.05 + pl.col("daily_return").abs())
        .otherwise(pl.col("daily_return"))
        .alias("daily_return")
    )
    ledger_columns = [
        name for name, dtype in ledger.schema.items() if dtype in (pl.Float32, pl.Float64)
    ]
    changed_ledger = _damage_after(ledger, "date", cutoff, ledger_columns)
    for original, changed, time_column in (
        (features, changed_features, "date"),
        (targets, changed_targets, "target_end"),
        (books, changed_books, "date"),
        (ledger, changed_ledger, "date"),
    ):
        assert not original.equals(changed), "Every future-input perturbation must take effect"
        assert_frame_equal(
            original.filter(pl.col(time_column) <= cutoff),
            changed.filter(pl.col(time_column) <= cutoff),
            check_exact=True,
        )
    decision = json.loads((baseline_dir / "bdp_decision.json").read_text())
    outputs = []
    for data, outcomes, book, trades in (
        (features, targets, books, ledger),
        (changed_features, changed_targets, changed_books, changed_ledger),
    ):
        transform = (
            _causal_bdp_transform(settings, book, monkeypatch) if decision["full_model"] else None
        )
        runs = {}
        for target, feature_set in (("y1", "L1-own"), ("y1", "L1"), ("y3", "L1-own")):
            result = walk_forward(
                data,
                outcomes,
                calendar,
                settings,
                series="RAW",
                target=target,
                feature_set=feature_set,
                columns=FEATURE_SETS[feature_set],
                through=cutoff,
                transform=transform if feature_set == "L1" else None,
            )
            assert result.forecasts["origin"].to_list() == [date(2005, 12, 30), cutoff]
            assert result.forecasts["refit"].to_list() == [date(2005, 1, 3), date(2006, 1, 3)]
            assert result.fits.filter(
                pl.col("training_target_end_max") >= pl.col("refit")
            ).is_empty()
            runs[(target, feature_set)] = result
        predictions = runs[("y1", "L1")].forecasts
        variance = np.exp(
            predictions["prediction"].to_numpy() + predictions["residual_variance"].to_numpy() / 2
        )
        multiples = volatility_multiples(
            variance,
            settings["strategy"]["vol_target_annual"],
            settings["targets"]["horizon"],
            settings["replication"]["months_per_year"] * settings["replication"]["monthly_days"],
            settings["strategy"]["max_multiple"],
        )
        origins = predictions["origin"].to_list()
        executions = [
            calendar[calendar.index(origin) + settings["targets"]["lag"]] for origin in origins
        ]
        overlay = predictions.select("origin", pl.col("refit").alias("fit_date")).with_columns(
            pl.Series("execution_date", executions),
            pl.Series("multiple", multiples),
        )
        assert executions == [date(2006, 1, 3), date(2006, 2, 1)]
        result = simulate_strategy(
            book,
            trades,
            overlay,
            cost_bp=settings["strategy"]["cost_bp"],
            initial_nav=settings["portfolios"]["initial_nav"],
            last_date=cutoff,
        )
        assert result.daily["date"].max() == cutoff
        assert result.rebalances["origin"].to_list() == [date(2005, 12, 30)]
        assert result.ledger.height > 100
        assert result.ledger.filter(pl.col("posttrade") > 0).height > 0
        assert result.ledger.filter(pl.col("posttrade") < 0).height > 0
        outputs.append((runs, overlay, result))
    before_runs, before_overlay, before = outputs[0]
    after_runs, after_overlay, after = outputs[1]
    for key in before_runs:
        original, changed = before_runs[key], after_runs[key]
        assert_frame_equal(original.fits, changed.fits, check_exact=True)
        # Future realized outcomes may change; every learned/predicted field must not.
        assert_frame_equal(
            original.forecasts.drop("actual", "y2"),
            changed.forecasts.drop("actual", "y2"),
            check_exact=True,
        )
        assert_frame_equal(
            original.training_predictions, changed.training_predictions, check_exact=True
        )
    assert before_runs[("y3", "L1-own")].forecasts["tail_threshold"].null_count() == 0
    assert (
        not before_runs[("y1", "L1")]
        .forecasts.select("actual")
        .equals(after_runs[("y1", "L1")].forecasts.select("actual"))
    )
    assert_frame_equal(before_overlay, after_overlay, check_exact=True)
    assert_frame_equal(before.daily, after.daily, check_exact=True)
    assert_frame_equal(before.rebalances, after.rebalances, check_exact=True)
    assert_frame_equal(before.ledger, after.ledger, check_exact=True)
    # At every day, x_strategy=base_position_ratio*x_base. The ratio and the
    # actual executed per-name holdings are identical; the pinned portfolio test proves
    # the underlying position path itself ignores later stock observations.
    assert before.daily["base_position_ratio"][-1] == after.daily["base_position_ratio"][-1]
    assert sha256_file(curated / "cap_repairs.parquet") == repair_hash
