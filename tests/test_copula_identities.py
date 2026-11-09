"""Rank-weighted return and beta identities, on finite complete panels (Methodology)."""


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import rankdata

from momenta.copula import rank_weighted_value


@settings(max_examples=32, deadline=None, derandomize=True)
@given(
    st.lists(st.tuples(st.integers(-20, 20), st.integers(-40, 40), st.integers(-20, 20)),
             min_size=2, max_size=30),
    st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False),
)
def test_rank_weighted_return_and_beta_equal_n_scale_population_covariance(rows, scale):
    data = np.asarray(rows, dtype=float)
    ranks = rankdata(data[:, 0], method='average')
    for values in (data[:, 1]/100, data[:, 2]/10):
        population_covariance = np.mean((ranks-ranks.mean())*(values-values.mean()))
        expected = len(rows)*scale*population_covariance
        assert rank_weighted_value(data[:, 0], values, scale) == pytest.approx(
            expected, rel=1e-12, abs=1e-12,
        )


def test_rank_weighted_stock_betas_equal_the_beta_of_the_raw_return_portfolio():
    signals = np.array([2., 1., 2., 4.])
    beta, intercept = np.array([1., -2., .5, 3.]), np.array([.01, -.02, .03, .04])
    market = np.array([-1., -1., 1., 1.])*.01
    orthogonal = np.array([-1., 1., -1., 1.])*.002
    stocks = intercept + market[:, None]*beta + orthogonal[:, None]*np.arange(1, 5)
    ranks, scale = np.array([2.5, 1., 2.5, 4.]), .1
    portfolio = stocks @ (scale*(ranks-ranks.mean()))
    actual_beta = np.mean((portfolio-portfolio.mean())*market)/np.mean(market**2)
    # Hand weights=(0,-.15,0,.15), giving .3+.45=.75.
    assert actual_beta == pytest.approx(.75)
    assert rank_weighted_value(signals, beta, scale) == pytest.approx(.75)
