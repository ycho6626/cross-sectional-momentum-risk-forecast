"""Mathematical identities on their stated domains (Methodology).

Finite balanced panels use population (1/T) moments. Correlation panels have
positive variance in every column. Transport panels have equal empirical
weights; the general identity permits ties, while the Spearman specialization
uses distinct ranks. Entropy and transport remain mathematical tests, not
additional production features.
"""


import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import spearmanr

from momenta.l2 import (
    balanced_standardize,
    breadth,
    cross_correlation,
    rank_correlation,
    within_correlation,
)

property_settings = settings(max_examples=32, derandomize=True, database=None, deadline=None)


@st.composite
def finite_panel(draw, *, nonconstant=False, min_columns=2, max_columns=10):
    """Small balanced T×N panels; two anchored rows ensure positive column variance."""
    rows = draw(st.integers(min_value=3, max_value=15))
    columns = draw(st.integers(min_value=min_columns, max_value=max_columns))
    cells = draw(st.lists(st.integers(-32, 32), min_size=rows*columns,
                         max_size=rows*columns))
    panel = np.array(cells, dtype=float).reshape(rows, columns)/16
    if nonconstant:
        offsets = draw(st.lists(st.integers(-16, 16), min_size=columns, max_size=columns))
        scales = draw(st.lists(st.integers(1, 16), min_size=columns, max_size=columns))
        panel[0] = np.asarray(offsets)-np.asarray(scales)
        panel[1] = np.asarray(offsets)+np.asarray(scales)
    return panel


@property_settings
@given(first=st.one_of(st.integers(-32, -1), st.integers(1, 32)),
       rest=st.lists(st.integers(-32, 32), max_size=40))
def test_binary_sign_entropy_is_determined_by_production_breadth(first, rest):
    """For nonempty nonzero signs, Shannon entropy equals H((1+M)/2)."""
    values = np.array([first, *rest, 0., 0.])
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    counts = [count for count in (positive, negative) if count]
    probabilities = [count/(positive+negative) for count in counts]
    direct_entropy = -sum(p*math.log(p) for p in probabilities)
    magnetization = breadth(values)
    assert magnetization == pytest.approx((positive-negative)/(positive+negative))
    p = (1+magnetization)/2
    binary_entropy = -sum(q*math.log(q) for q in (p, 1-p) if q > 0)
    assert binary_entropy == pytest.approx(direct_entropy, abs=2e-15)


@property_settings
@given(values=st.lists(st.integers(-1000, 1000), min_size=2, max_size=25, unique=True))
def test_median_relative_breadth_is_zero_for_distinct_observations(values):
    """Distinct observations balance signs about their median; an odd median is excluded."""
    centered = np.asarray(values, dtype=float)-np.median(values)
    assert breadth(centered) == 0


@property_settings
@given(panel=finite_panel())
def test_balanced_window_dispersion_includes_dispersion_of_stock_means(panel):
    """Mean daily dispersion = (N−1)/N(avg var−avg offdiag cov)+Var(stock means)."""
    n = panel.shape[1]
    covariance = np.cov(panel, rowvar=False, ddof=0)
    average_variance = np.trace(covariance)/n
    average_covariance = covariance[~np.eye(n, dtype=bool)].mean()
    stock_mean_dispersion = np.var(panel.mean(axis=0), ddof=0)
    expected = (n-1)/n*(average_variance-average_covariance)+stock_mean_dispersion
    observed = np.var(panel, axis=1, ddof=0).mean()
    assert observed == pytest.approx(expected, rel=2e-12, abs=2e-13)


@property_settings
@given(panel=finite_panel(nonconstant=True))
def test_average_correlation_identity_matches_the_independent_correlation_matrix(panel):
    """Balanced positive-variance columns, standardized by 1/T, give the within-leg identity."""
    balanced = balanced_standardize(panel)
    np.testing.assert_array_equal(balanced.columns, np.arange(panel.shape[1]))
    np.testing.assert_allclose(balanced.z.mean(axis=0), 0, atol=2e-15)
    np.testing.assert_allclose(np.mean(balanced.z**2, axis=0), 1, atol=2e-15)
    matrix = np.corrcoef(panel, rowvar=False)
    expected = matrix[~np.eye(panel.shape[1], dtype=bool)].mean()
    assert within_correlation(balanced.z) == pytest.approx(expected, abs=2e-12)


@property_settings
@given(panel=finite_panel(nonconstant=True, min_columns=2, max_columns=12),
       fraction=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
def test_cross_mean_covariance_is_the_average_cross_leg_correlation(panel, fraction):
    """Nonempty legs share all T dates; Cov(mean z_W,mean z_L) averages every cross pair."""
    split = 1+min(int(fraction*(panel.shape[1]-1)), panel.shape[1]-2)
    winners, losers = panel[:, :split], panel[:, split:]
    standardized_w = balanced_standardize(winners).z
    standardized_l = balanced_standardize(losers).z
    matrix = np.corrcoef(panel, rowvar=False)
    expected = matrix[:split, split:].mean()
    observed = cross_correlation(standardized_w, standardized_l)
    assert observed == pytest.approx(expected, abs=2e-12)
    assert cross_correlation(standardized_l, standardized_w) == pytest.approx(expected, abs=2e-12)


@property_settings
@given(pairs=st.lists(st.tuples(st.integers(-16, 16), st.integers(-16, 16)),
                      min_size=2, max_size=25))
def test_general_transport_gap_identity_permits_ties_and_unequal_marginal_scales(pairs):
    """Equal-mass empirical W2 uses sorted matching; labels retain the original joint pairing."""
    # Repeat one labeled pair to guarantee ties in both marginals.
    panel = np.asarray([*pairs, pairs[0]], dtype=float)
    x, y = panel[:, 0]/4, panel[:, 1]/8+2
    labeled_cost = np.mean((x-y)**2)
    empirical_w2 = np.mean((np.sort(x)-np.sort(y))**2)
    gap = labeled_cost-empirical_w2
    comonotone_product = np.mean(np.sort(x)*np.sort(y))
    realized_product = np.mean(x*y)
    assert gap == pytest.approx(2*(comonotone_product-realized_product), abs=2e-13)
    assert gap >= -2e-13


@st.composite
def rank_permutations(draw):
    """Two distinct-rank vectors of equal length, with no ties in either marginal."""
    n = draw(st.integers(min_value=2, max_value=20))
    left = draw(st.permutations(tuple(range(1, n+1))))
    right = draw(st.permutations(tuple(range(1, n+1))))
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


@property_settings
@given(ranks=rank_permutations())
def test_standardized_rank_transport_gap_is_twice_one_minus_spearman(ranks):
    """Ranks1..N have mean(N+1)/2 and population variance(N²−1)/12; sorted cost is zero."""
    left, right = ranks
    n = len(left)
    scale = math.sqrt((n*n-1)/12)
    x = (left-(n+1)/2)/scale
    y = (right-(n+1)/2)/scale
    empirical_w2 = np.mean((np.sort(x)-np.sort(y))**2)
    assert empirical_w2 == 0
    gap = np.mean((x-y)**2)-empirical_w2
    independent_rho = float(spearmanr(left, right).statistic)
    assert gap == pytest.approx(2*(1-independent_rho), abs=2e-12)
    assert rank_correlation(left, right) == pytest.approx(independent_rho, abs=2e-12)
