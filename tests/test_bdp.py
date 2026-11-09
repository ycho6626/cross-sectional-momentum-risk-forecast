"""Independent density, score, sign, and filter-timing fixtures for BDP Methodology."""


import math
from dataclasses import replace

import numpy as np
import pytest

from momenta.bdp import (
    BDPFit,
    conditional_moments,
    filter_bdp,
    filter_step,
    forecast_moments,
    likelihood_score,
    log_density,
    published_information,
    skew_t_draw,
)


def test_symmetric_density_is_student_t_and_information_normalization_is_explicit() -> None:
    # t_5(0) = Gamma(3)/(sqrt(5pi)*Gamma(2.5)). Location 2, scale 3.
    expected = math.log(2/(math.sqrt(5*math.pi)*(.75*math.sqrt(math.pi)))/3)
    state = np.array([2., math.log(3), 0.])
    assert log_density(2., state, 5) == pytest.approx(expected)
    # Printed inverse-information normalization: scale diagonal is 2/(nu+3).
    assert published_information(state, 5)[1] == pytest.approx(.25)


@pytest.mark.parametrize("value", [-2., -.1, .7, 3.])
def test_likelihood_scores_equal_independent_finite_differences(value: float) -> None:
    state = np.array([.2, math.log(1.3), np.arctanh(-.35)])
    gradient = []
    for index in range(3):
        step = np.eye(3)[index]*1e-6
        gradient.append((log_density(value, state+step, 7)-log_density(value, state-step, 7))
                        / 2e-6)
    np.testing.assert_allclose(likelihood_score(value, state, 7), gradient, atol=2e-9)


def test_density_integral_and_moments_match_closed_form() -> None:
    # Deterministic integration of both half-lines, independent of the sampler.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    u = (nodes+1)/2
    z = np.tan(np.pi*u/2)
    jac = np.pi/2*(1+z*z)
    state = np.array([.1, math.log(.8), np.arctanh(-.3)])
    nu = 9.
    integrals = np.zeros(3)
    for sign in [-1, 1]:
        observations = state[0]+sign*z
        density = np.exp([log_density(y, state, nu) for y in observations])
        integrals += np.array([np.ones(len(z)), observations, observations**2]) @ (
            density*jac*weights/2)
    mean, variance = conditional_moments(state, nu)
    assert integrals[0] == pytest.approx(1, abs=1e-11)
    assert integrals[1] == pytest.approx(mean, abs=1e-10)
    assert integrals[2]-integrals[1]**2 == pytest.approx(variance, abs=1e-10)
    assert mean < state[0]  # Published negative rho is negative skew.


def test_draw_has_correct_half_t_side_weights_and_scales() -> None:
    class FixedRandom:
        def __init__(self, uniform: float):
            self.uniform = uniform

        def random(self) -> float:
            return self.uniform

        def standard_t(self, nu: float) -> float:
            assert nu == 8
            return -2.

    state = np.array([1., math.log(3), np.arctanh(.5)])
    # Right mass .75; right scale 4.5, left scale 1.5.
    assert skew_t_draw(state, 8, FixedRandom(.74)) == pytest.approx(10)
    assert skew_t_draw(state, 8, FixedRandom(.76)) == pytest.approx(-2)


def test_published_update_hand_fixture_and_post_close_timing() -> None:
    # nu=5, state 0, sigma1: r=1 gives w=1; scores=(1,0,1).
    # Transformed information=(.75,.25,2.25); loadings .03 give (.04,0,1/75).
    state, loadings = np.zeros(3), np.full(3, .03)
    expected = np.array([.04, 0., 1/75])
    np.testing.assert_allclose(filter_step(1., state, loadings, 5), expected)
    fit = BDPFit(tuple(loadings), 5, tuple(state), True, 0, 1, ())
    paths = filter_bdp(np.array([1., -1., 5.]), fit)
    np.testing.assert_allclose(paths[0], expected)
    np.testing.assert_allclose(paths[:2], filter_bdp(np.array([1., -1.]), fit))
    changed = filter_bdp(np.array([1., -1., -100.]), fit)
    np.testing.assert_array_equal(changed[:2], paths[:2])
    with pytest.raises(ValueError):
        filter_bdp(np.array([np.nan]), replace(fit, converged=False))


def test_horizon_moments_compound_percentage_point_draws_and_count_negative_wealth() -> None:
    settings = {"targets": {"horizon": 3}, "bdp": {"horizon_simulations": 64, "seed": 1}}
    state = np.array([1., math.log(1e-12), 0.])
    fit = BDPFit((0., 0., 0.), 10, tuple(state), True, 0, 0, ())
    result = forecast_moments(state, fit, settings)
    assert result["mean"] == pytest.approx(1.01**3-1, abs=1e-13)
    assert result["variance"] < 1e-24
    assert result["nonpositive_wealth_paths"] == 0
    settings["targets"]["horizon"] = 2
    state[0] = -200  # −200% daily; negative wealth is counted, never clipped.
    result = forecast_moments(state, fit, settings)
    assert result["nonpositive_wealth_paths"] == 64
    assert result["mean"] == pytest.approx(0., abs=1e-12)
