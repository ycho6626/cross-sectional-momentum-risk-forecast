"""Independent holding-window arithmetic and lag-zero BDP compatibility."""


import numpy as np
import pytest

from momenta.bdp import BDPFit, forecast_moments
from momenta.bdp_horizon import holding_moments


def test_execution_day_updates_the_state_but_does_not_earn_a_return(monkeypatch) -> None:
    class FixedRandom:
        day = 0

        def random(self, n):
            return np.zeros(n)

        def standard_t(self, nu, n):
            self.day += 1
            return np.ones(n) if self.day == 1 else np.zeros(n)

    monkeypatch.setattr(np.random, "default_rng", lambda _: FixedRandom())
    settings = {"targets": {"lag": 1, "horizon": 1},
                "bdp": {"horizon_simulations": 3, "seed": 1}}
    state = np.zeros(3)
    fit = BDPFit((.75, 0., 0.), 5, tuple(state), True, 0, 0, ())
    # Burn draw=1, w=1, location update=.75*(8/6)*1=1.
    # Next half-t draw=0 gives return=m=1 percentage point. Only that day earns.
    result = holding_moments(state, fit, settings)
    assert result["mean"] == pytest.approx(.01)
    assert result["variance"] == 0
    assert result["horizon_start"] == result["horizon_end"] == 2
    np.testing.assert_array_equal(state, np.zeros(3))


@pytest.mark.parametrize("negative_earned, expected_mean, expected_variance, negative_count",
                         [(False, .02, .0001, 0), (True, -.65, 1.3669, 1)])
def test_sample_variance_and_negative_wealth_are_measured_only_on_earned_days(
    monkeypatch, negative_earned, expected_mean, expected_variance, negative_count,
) -> None:
    class FixedRandom:
        day = 0

        def random(self, n):
            return np.array([.9, .1, .1]) if self.day == 0 or negative_earned else np.zeros(n)

        def standard_t(self, nu, n):
            self.day += 1
            return (np.array([200., 1., 1.]) if self.day == 1
                    else np.array([200. if negative_earned else 1., 2., 3.]))

    monkeypatch.setattr(np.random, "default_rng", lambda _: FixedRandom())
    settings = {"targets": {"lag": 1, "horizon": 1},
                "bdp": {"horizon_simulations": 3, "seed": 1}}
    state = np.zeros(3)
    fit = BDPFit((0., 0., 0.), 5, tuple(state), True, 0, 0, ())
    result = holding_moments(state, fit, settings)
    # Burn returns include −200%; earned returns are 1%,2%,3% or −200%,2%,3%.
    assert result["mean"] == pytest.approx(expected_mean)
    assert result["variance"] == pytest.approx(expected_variance)
    assert result["nonpositive_wealth_paths"] == negative_count
    assert result["nonpositive_earned_wealth_paths"] == negative_count


def test_zero_lag_reproduces_the_existing_seeded_next_day_horizon_exactly() -> None:
    settings = {"targets": {"lag": 0, "horizon": 21},
                "bdp": {"horizon_simulations": 4096, "seed": 20261007}}
    state = np.array([.2, np.log(1.3), np.arctanh(-.15)])
    fit = BDPFit((.01, .001, .025), 8, tuple(state), True, 0, 0, ())
    existing = forecast_moments(state, fit, settings)
    actual = holding_moments(state, fit, settings)
    assert {key: actual[key] for key in existing} == existing
    assert actual["horizon_start"] == 1
    assert actual["horizon_end"] == 21
