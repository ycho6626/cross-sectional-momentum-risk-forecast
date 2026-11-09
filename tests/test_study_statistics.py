"""Independent hand calculations for inference and calendar resampling."""


from datetime import date
from math import erfc, log, sqrt

import numpy as np
import polars as pl
import pytest

from momenta.study_statistics import (
    auc,
    brier,
    clark_west,
    deflated_sharpe,
    diebold_mariano,
    forecast_metrics,
    h2_contrast,
    holm_adjust,
    lo_sharpe,
    lo_sharpe_draws,
    nw_test,
    qlike,
    stationary_indices,
)


def test_holm_adjustment_uses_sorted_cumulative_max_and_preserves_order():
    # Sorted raw .01,.03,.04 => scaled .03,.06,.04 => cumulative .03,.06,.06.
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert holm_adjust([0.025, 0.025]) == [0.05, 0.05]
    assert holm_adjust([0.6, 0.8]) == [1.0, 1.0]
    assert holm_adjust([0.01, None]) == [None, None]
    with pytest.raises(ValueError, match="probabilities"):
        holm_adjust([1.1, 0.01])


def test_newey_west_hand_covariances_and_one_sided_tail():
    # mean=2.5, gamma0=1.25, gamma1=.3125; Bartlett L=1 gives SE=.625.
    result = nw_test([1, 2, 3, 4], 1)
    assert result["standard_error"] == pytest.approx(0.625)
    assert result["statistic"] == pytest.approx(4)
    assert result["pvalue"] == pytest.approx(erfc(4 / sqrt(2)) / 2)
    assert nw_test([-1, -2, -3, -4], 1)["pvalue"] > 0.99
    assert nw_test([0, 0, 0], 3)["pvalue"] is None
    assert nw_test([0.1, 0.1, 0.1], 3)["pvalue"] is None
    assert nw_test([1, 2], 3)["lags"] == 1


def test_cw_includes_adjustment_while_dm_is_unadjusted():
    # CW differences are 1,4,9,16; ordinary loss improvements .75,3,6.75,12.
    actual, base, model = [1, 2, 3, 4], [0, 0, 0, 0], [0.5, 1, 1.5, 2]
    assert clark_west(actual, base, model, 1)["mean_difference"] == 7.5
    assert diebold_mariano(actual, base, model, 1)["mean_difference"] == 5.625
    assert clark_west(actual, model, model, 1)["pvalue"] is None


def test_stationary_bootstrap_preserves_calendar_and_is_repeatable():
    months = [date(2006, month, 1) for month in range(1, 13)]
    first = stationary_indices(months, 30, 1e100, 17)
    assert first.shape == (30, 12)
    assert np.array_equal(first, stationary_indices(months, 30, 1e100, 17))
    # With no restarts, the path must advance around the circular calendar.
    assert np.all(np.diff(first, axis=1) % 12 == 1)
    assert len(np.unique(first[:, 0])) > 1
    with pytest.raises(ValueError, match="consecutive"):
        stationary_indices([date(2006, 1, 1), date(2006, 3, 1)], 20, 12, 17)


def test_qlike_uses_training_residual_variance_bias_correction():
    # Predictions become 2,2; realized variances are 4,1, ratios 2,.5.
    losses = qlike([log(4), 0], [0, 0], [2 * log(2)] * 2)
    np.testing.assert_allclose(losses, [1 - log(2), -0.5 + log(2)])
    assert losses.mean() == pytest.approx(0.25)


def test_auc_ties_brier_and_single_class_are_explicit():
    # Four positive-negative pairs: win,tie,win,win => 3.5/4.
    assert auc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == 0.875
    assert brier([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.13)
    assert auc([0, 0], [0.1, 0.5]) is None


def test_h2_recomputes_sse_ratios_on_paired_draws():
    # Full A errors1,2 / reduced2,2 => ΔA=3/8. B errors1,1 / reduced2,2 =>3/4.
    indices = np.array([[0, 0], [1, 1]])
    result = h2_contrast([0, 0], [1, 2], [2, 2], [0, 0], [1, 1], [2, 2], indices)
    assert result["delta_a"] == 0.375
    assert result["delta_b"] == 0.75
    assert result["contrast"] == -0.375
    assert result["interval"]["lower"] == pytest.approx(-0.73125)
    assert result["interval"]["upper"] == pytest.approx(-0.01875)
    identical = h2_contrast([0, 0], [1, 2], [2, 2], [0, 0], [1, 2], [2, 2], indices)
    assert identical["interval"]["lower"] == identical["interval"]["upper"] == 0


def test_lo_sharpe_hand_two_period_autocovariance():
    # q=2, mean=2.5, gamma0=1.25, gamma1=.3125 => aggregate variance3.125.
    assert lo_sharpe([1, 2, 3, 4], 2) == pytest.approx(5 / sqrt(3.125))
    assert lo_sharpe([1, 2, 3], 12) is None
    np.testing.assert_allclose(
        lo_sharpe_draws(np.array([[1, 2, 3, 4], [-1, -2, -3, -4]]), 2),
        [5 / sqrt(3.125), -5 / sqrt(3.125)],
    )


def test_dsr_zero_sharpe_and_failed_strategies_remain_in_count():
    # Symmetric returns have SR=0; one specification gives threshold0 and PSR=.5.
    assert deflated_sharpe([-2, -1, 1, 2], [0], 1)["dsr"] == pytest.approx(0.5)
    complete = deflated_sharpe([-2, -1, 1, 2], [-0.1, 0.1], 2)
    failed = deflated_sharpe([-2, -1, 1, 2], [-0.1, 0.1], 3)
    assert failed["distinct_trading_strategies"] == 3
    assert failed["strategies_with_sharpe"] == 2
    assert failed["descriptive"] is True
    assert failed["dsr"] < complete["dsr"] < 0.5


def test_forecast_loss_table_uses_paired_baseline_and_degenerate_own_test():
    frame = pl.DataFrame(
        {
            "origin": [date(2006, m, 1) for m in range(1, 4)],
            "actual": [1.0, 2.0, 3.0],
            "prediction": [0.0, 1.0, 2.0],
            "base_prediction": [0.0, 0.0, 0.0],
            "residual_variance": [0.0] * 3,
        }
    )
    result = forecast_metrics(frame, target="y1", reps=10, block_months=12, seed=1, nw_lags=3)
    assert result["mse"] == 1
    assert result["r2_oos"] == pytest.approx(1 - 3 / 14)
    assert result["mse_interval"]["lower"] == result["mse_interval"]["upper"] == 1
    own = frame.with_columns(pl.col("prediction").alias("base_prediction"))
    result = forecast_metrics(own, target="y1", reps=10, block_months=12, seed=1, nw_lags=3)
    assert result["r2_oos"] == 0
    assert result["cw"]["pvalue"] is None
