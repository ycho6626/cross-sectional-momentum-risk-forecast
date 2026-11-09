"""Independent copula geometry, fixed transforms and training-only centroid fixtures
(Methodology)."""


from copy import deepcopy
from datetime import date

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from polars.testing import assert_frame_equal

from momenta.copula import (
    CM_COLUMNS,
    Q_COLUMNS,
    cm_moments,
    copula_cloud,
    embedding,
    fit_centroids,
    fit_dated_centroids,
    quadratic_moments,
    separation_diagnostic,
    sliced_quantiles,
    stock_beta_sigma,
    sw2_distance,
    unit_directions,
)
from momenta.settings import load_settings


def test_cloud_uses_common_complete_rows_average_ties_and_n_plus_one():
    values = np.array([[1., 10., 2., 5.], [1., 30., 1., 8.], [3., 20., 3., 3.],
                       [np.nan, 0., 0., 0.]])
    cloud, complete = copula_cloud(values)
    assert_array_equal(complete, [True, True, True, False])
    assert_allclose(cloud, np.array([[1.5, 1., 2., 2.], [1.5, 3., 1., 3.],
                                    [3., 2., 3., 1.]])/4)
    transformed = values.copy()
    transformed[:, 0] = np.exp(transformed[:, 0])
    transformed[:, 1] = transformed[:, 1]**3
    assert_array_equal(copula_cloud(transformed)[0], cloud)


def test_beta_has_an_intercept_and_sigma_uses_population_window():
    market = np.array([-2., -1., 1., 2.])*.01
    stocks = np.column_stack([.01+2*market, np.full(4, .1), -market])
    stocks[1, 2] = np.nan
    beta, sigma = stock_beta_sigma(stocks, market)
    assert_allclose(beta[:2], [2., 0.], atol=1e-14)
    assert_allclose(sigma[:2], [2*np.sqrt(.00025), 0.])
    assert np.isnan(beta[2]) and np.isnan(sigma[2])


def test_projection_grid_and_embedding_distance_have_hand_calculated_scaling():
    cloud = np.array([[0., 0.], [2., 4.]])
    directions = np.eye(2)
    original = sliced_quantiles(cloud, directions, 3, method='linear')
    shifted = sliced_quantiles(cloud+[3., 4.], directions, 3, method='linear')
    assert_array_equal(original, [[0., 1., 2.], [0., 2., 4.]])
    assert_allclose(embedding(original), np.array([0., 1., 2., 0., 2., 4.])/np.sqrt(6))
    assert sw2_distance(original, shifted) == pytest.approx(np.sqrt((3**2+4**2)/2))
    assert sw2_distance(original, original) == 0.
    assert_array_equal(original, sliced_quantiles(cloud[::-1], directions, 3, method='linear'))
    fixed = unit_directions(8, 4, 20261007)
    assert_allclose(np.linalg.norm(fixed, axis=1), np.ones(8))
    assert_array_equal(fixed, unit_directions(8, 4, 20261007))


def test_cm_is_spearman_and_q_contains_only_the_registered_21_new_terms():
    # Coordinates are increasing, decreasing, increasing, decreasing.
    cloud, _ = copula_cloud(np.column_stack([np.arange(4), -np.arange(4),
                                            np.arange(4)**3, -np.arange(4)**3]))
    assert_allclose(cm_moments(cloud), [-1., 1., -1., -1., 1., -1.])
    assert len(CM_COLUMNS) == 6 and len(Q_COLUMNS) == 21
    quadratic = quadratic_moments(np.arange(1., 7.))
    assert_array_equal(quadratic, [1, 4, 9, 16, 25, 36,
                                   2, 3, 4, 5, 6, 6, 8, 10, 12, 12, 15, 18, 20, 24, 30])


def test_centroid_alignment_uses_previous_labels_and_hand_mean_centers():
    settings = load_settings()
    centers = np.array([[-3., -3.], [-3., 3.], [3., -3.], [3., 3.]])
    train = np.concatenate([center+np.array([[-.1, 0.], [.1, 0.]]) for center in centers])
    first = fit_centroids(train, settings)
    assert_allclose(first.centers, centers, atol=1e-14)
    assert_array_equal(first.counts, [2, 2, 2, 2])
    expected = centers[[2, 0, 3, 1]]
    aligned = fit_centroids(train, settings, previous_centroids=expected)
    assert_allclose(aligned.centers, expected, atol=1e-14)
    assert aligned.transform(np.array([[-3., -3.]]))[0, 1] == pytest.approx(0., abs=1e-14)
    assert first.inertia == pytest.approx(8*.1**2)


def test_dated_centroids_fit_unique_eligible_months_and_ignore_later_data():
    settings = load_settings()
    days = [date(2001, month, 28) for month in range(1, 11)]
    values = [[-3.1, -3.], [-2.9, -3.], [-3.1, 3.], [-2.9, 3.],
              [2.9, -3.], [3.1, -3.], [2.9, 3.], [3.1, 3.], [50., 50.], [-50., -50.]]
    frame = pl.DataFrame({'date': days, 'embedding': values})
    cutoff = date(2001, 10, 1)
    # September precedes the fit but its target is unfinished, so the caller
    # excludes it. Repeated daily copies of January must not upweight it.
    training = [*days[:8], days[0]]
    before, fit = fit_dated_centroids(frame, training, cutoff, settings)
    damaged = frame.with_columns(pl.when(pl.col('date') >= days[8])
                                 .then(pl.lit([1e6, -1e6])).otherwise(pl.col('embedding'))
                                 .alias('embedding'))
    after, changed_fit = fit_dated_centroids(damaged, training, cutoff, settings)
    assert fit == changed_fit
    assert fit['training_months'] == 8
    assert fit['training_last'] == str(days[7])
    assert_frame_equal(before.filter(pl.col('date') <= days[7]),
                       after.filter(pl.col('date') <= days[7]), check_exact=True)
    assert not before.equals(after)
    with pytest.raises(ValueError, match='precede cutoff'):
        fit_dated_centroids(frame, [*days[:8], days[-1]], cutoff, settings)


def test_separation_is_matched_size_and_descriptive_even_when_threshold_fails():
    settings = deepcopy(load_settings())
    settings['ot'].update(separation_splits=2, grid_points=3)
    # Every half is the same point cloud, so all distances vanish. This cannot
    # establish temporal separation and must report failure without raising.
    clouds = [np.tile([.5, .5, .5, .5], (size, 1)) for size in (6, 8, 10)]
    audit = separation_diagnostic(clouds, np.eye(4), settings)
    assert audit['half_size'] == 3
    assert audit['counts'] == {'within': 6, 'adjacent': 4, 'all_pairs': 6}
    assert audit['medians'] == {'within': 0., 'adjacent': 0., 'all_pairs': 0.}
    assert not audit['threshold_passed'] and not audit['is_gate']
