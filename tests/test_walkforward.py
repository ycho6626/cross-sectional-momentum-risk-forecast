"""Independent penalty, standardization and information-timing fixtures (Methodology)."""


import json
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal
from sklearn.linear_model import Ridge

from momenta.settings import load_settings
from momenta.walkforward import (
    Standardizer,
    assert_design_origins,
    feature_block_indices,
    fit_regression,
    month_end_origins,
    refit_for_origin,
    ridge_path,
    select_block_penalties,
    select_penalty,
    training_split,
    walk_forward,
)


def weekdays(first: date, last: date) -> list[date]:
    """Independent synthetic trading calendar; no market holidays are claimed."""
    return [first + timedelta(days=i) for i in range((last-first).days+1)
            if (first + timedelta(days=i)).weekday() < 5]


def test_ridge_uses_mean_loss_penalty_and_unpenalized_intercept():
    # x=(-1,1), y=(1,5): intercept=3; min (2-b)^2 + lambda*b^2.
    x, y = np.array([[-1.], [1.]]), np.array([1., 5.])
    coefficients, intercepts = ridge_path(x, y, [0., 1., 3.])
    assert_allclose(coefficients, [[2., 1., .5]])
    assert_allclose(intercepts, [3., 3., 3.])
    # Duplicating observations must not change a mean-loss-scale fit.
    repeated = ridge_path(np.repeat(x, 7, axis=0), np.repeat(y, 7), [0., 1., 3.])
    assert_allclose(repeated[0], coefficients)
    assert_allclose(repeated[1], intercepts)


def test_ridge_matches_independent_sklearn_alpha_n_lambda():
    x = np.array([[-2., 1.], [-1., 1.], [1., 2.], [3., 5.], [4., 1.]])
    y = np.array([.2, -.3, 1., 2., 1.5])
    z = Standardizer.fit(x, load_settings()["walkforward"]["feature_clip_sd"]).transform(x)
    coefficients, intercepts = ridge_path(z, y, [0., .1, 1.])
    for i, penalty in enumerate([0., .1, 1.]):
        reference = Ridge(alpha=len(y)*penalty).fit(z, y)
        assert_allclose(coefficients[:, i], reference.coef_, atol=1e-12)
        assert_allclose(intercepts[i], reference.intercept_, atol=1e-12)


def test_standardizer_uses_only_training_values():
    fit = Standardizer.fit(np.array([[1., 7.], [3., 7.]]), 3.0)
    assert_allclose(fit.mean, [2., 7.])
    assert_allclose(fit.scale, [1., 1.])
    assert_allclose(fit.transform(np.array([[100., 7.]])), [[3., 0.]])


def test_logistic_observation_duplication_keeps_mean_loss_objective():
    settings = load_settings()
    x = np.array([[-2.], [-1.], [0.], [1.], [2.], [3.]])
    y = np.array([0, 1, 0, 1, 0, 1])
    a = fit_regression(x, y, .03, logistic=True, settings=settings)
    b = fit_regression(np.repeat(x, 4, axis=0), np.repeat(y, 4), .03,
                       logistic=True, settings=settings)
    assert_allclose(a.predict(x), b.predict(x), atol=2e-8)
    # First-order condition for mean log-loss + lambda*b^2.
    z = a.standardizer.transform(x)
    gradient = z.T @ (a.predict(x)-y)/len(y) + 2*.03*a.coefficient
    assert_allclose(gradient, [0.], atol=2e-7)


def test_penalty_selection_refits_scaler_on_inner_training():
    settings = deepcopy(load_settings())
    settings['walkforward']['ridge_lambdas'] = [0., 1.]
    x, y = np.array([[-1.], [1.]]), np.array([-1., 1.])
    # Validation x=2, y=1 picks lambda=1: slope=.5, error0 (versus error1).
    penalty, losses = select_penalty(x, y, np.array([[2.]]), np.array([1.]),
                                     logistic=False, settings=settings)
    assert penalty == 1.
    assert_allclose(losses, [1., 0.])


def test_unequal_block_ridge_has_hand_calculated_shrinkage():
    # Orthogonal ±1 columns: min (2-b1)^2+(4-b2)^2+b1²+3b2².
    # The unequal-penalty answer is b=(1,1), intercept=3.
    settings = load_settings()
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    y = 3 + 2*x[:, 0] + 4*x[:, 1]
    fitted = fit_regression(x, y, [1., 3.], logistic=False, settings=settings)
    assert_allclose(fitted.coefficient, [1., 1.], atol=1e-14)
    assert fitted.intercept == pytest.approx(3.)
    repeated = fit_regression(np.repeat(x, 5, axis=0), np.repeat(y, 5), [1., 3.],
                              logistic=False, settings=settings)
    assert_allclose(repeated.predict(x), fitted.predict(x), atol=1e-14)


def test_clipped_training_matrix_preserves_a_free_intercept():
    # Raw training mean=1, population SD=sqrt(19). Clipping the high row to
    # 3 leaves standardized mean (3-sqrt(19))/20, which is not zero.
    settings = load_settings()
    x = np.array([[0.]]*19 + [[20.]])
    clipped = np.array([-1/np.sqrt(19)]*19 + [3.])
    y = 7 + 2*clipped
    fit = fit_regression(x, y, 0., logistic=False, settings=settings)
    assert_allclose(fit.standardizer.mean, [1.])
    assert_allclose(fit.standardizer.scale, [np.sqrt(19)])
    assert_allclose(fit.standardizer.transform(x)[:, 0], clipped)
    assert_allclose(fit.coefficient, [2.], atol=1e-13)
    assert fit.intercept == pytest.approx(7.)
    assert_allclose(fit.predict(np.array([[1000.], [-1000.]])), [13., 1.], atol=1e-13)


def test_sequential_choices_keep_earlier_penalties_even_when_joint_grid_is_better():
    settings = deepcopy(load_settings())
    settings['walkforward']['ridge_lambdas'] = [0., 1.]
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    y = 2*x[:, 0] + 2*x[:, 1]
    # At validation (x1,x2,y)=(1,2,5), HAR alone picks b1=2 (loss9)
    # over b1=1 (loss16). Keeping lambda_HAR=0, CORE b2=2 and b2=1
    # tie at loss1, so its first grid value wins. Joint selection instead
    # changes HAR to b1=1 and CORE to b2=2, giving loss0; that is forbidden.
    selected, audit = select_block_penalties(
        x, y, np.array([[1., 2.]]), np.array([5.]),
        blocks={'HAR': [0], 'CORE': [1]}, target='y1', settings=settings,
    )
    assert selected == {'HAR': 0., 'CORE': 0.}
    assert_allclose([row['loss'] for row in audit['HAR']], [9., 16.])
    assert_allclose([row['loss'] for row in audit['CORE']], [1., 1., 9.])
    assert [row['penalty'] for row in audit['CORE']] == [0., 1., 'exclude']
    joint = fit_regression(x, y, [1., 0.], logistic=False, settings=settings)
    assert_allclose(joint.predict(np.array([[1., 2.]])), [5.])


def test_exclusion_drops_columns_and_exactly_reproduces_prefix():
    settings = deepcopy(load_settings())
    settings['walkforward']['ridge_lambdas'] = [0., 1.]
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    y = 2*x[:, 0] + 2*x[:, 1]
    selected, audit = select_block_penalties(
        x, y, np.array([[1., 2.]]), np.array([2.]),
        blocks={'HAR': [0], 'CORE': [1]}, target='y1', settings=settings,
    )
    assert selected == {'HAR': 0., 'CORE': 'exclude'}
    assert_allclose([row['loss'] for row in audit['CORE']], [16., 4., 0.])
    larger = fit_regression(x, y, [0., np.inf], logistic=False, settings=settings)
    prefix = fit_regression(x[:, :1], y, 0., logistic=False, settings=settings)
    damaged = x.copy()
    damaged[:, 1] = [1e15, -1e15, 2e15, -2e15]
    assert larger.column_indices.tolist() == [0]
    assert larger.standardizer.mean.shape == (1,)
    np.testing.assert_array_equal(larger.predict(damaged), prefix.predict(x[:, :1]))
    np.testing.assert_array_equal(larger.coefficient, prefix.coefficient)
    assert larger.residual_variance == prefix.residual_variance


def test_later_block_is_selected_after_an_earlier_exclusion():
    settings = deepcopy(load_settings())
    settings['walkforward']['ridge_lambdas'] = [0., 1.]
    x = np.array([[a, b, c] for a in (-1., 1.) for b in (-1., 1.) for c in (-1., 1.)])
    y = 2*x.sum(axis=1)
    # Validation's irrelevant second column is clipped from ±4 to ±3.
    # HAR predicts2 (loss1); CORE candidates have losses25,4,1. With CORE
    # absent, M picks slope1 (loss0), keeping HAR's original slope2.
    selected, audit = select_block_penalties(
        x, y, np.array([[1., 4., 1.], [1., -4., -1.]]), np.array([3., 1.]),
        blocks={'HAR': [0], 'CORE': [1], 'M': [2]}, target='y1', settings=settings,
    )
    assert selected == {'HAR': 0., 'CORE': 'exclude', 'M': 1.}
    assert_allclose([row['loss'] for row in audit['CORE']], [25., 4., 1.])
    assert_allclose([row['loss'] for row in audit['M']], [1., 0., 1.])


@pytest.mark.parametrize('target', ['y2', 'y3'])
def test_all_blocks_may_be_excluded_for_mean_and_tail_benchmarks(target):
    settings = load_settings()
    x = np.array([[-1.], [-1.], [1.], [1.]])
    y = np.array([-1., -1., 1., 1.]) if target == 'y2' else np.array([0., 0., 1., 1.])
    validation_y = np.zeros(2) if target == 'y2' else np.array([1., 0.])
    selected, _ = select_block_penalties(
        x, y, np.array([[-1.], [1.]]), validation_y,
        blocks={'HAR': [0]}, target=target, settings=settings,
    )
    assert selected == {'HAR': 'exclude'}
    fit = fit_regression(x, y, [np.inf], logistic=(target == 'y3'), settings=settings)
    assert fit.coefficient.shape == (0,)
    np.testing.assert_array_equal(fit.predict(np.array([[-1e10], [1e10]])),
                                  np.full(2, y.mean()))


def test_logistic_unequal_penalties_satisfy_original_objective_gradient():
    settings = load_settings()
    x = np.array([[-2., 0.], [-1., 1.], [0., -1.], [1., 3.], [2., 1.], [3., -2.]])
    y, penalties = np.array([0, 1, 0, 1, 0, 1]), np.array([.001, .03])
    fit = fit_regression(x, y, penalties, logistic=True, settings=settings)
    probabilities = fit.predict(x)
    gradient = (fit.standardizer.transform(x).T @ (probabilities-y)/len(y)
                + 2*penalties*fit.coefficient)
    assert_allclose(gradient, np.zeros(2), atol=2e-7)
    assert abs(np.mean(probabilities-y)) < 2e-7
    prefix = fit_regression(x[:, :1], y, penalties[0], logistic=True, settings=settings)
    dropped = fit_regression(x, y, [penalties[0], np.inf], logistic=True, settings=settings)
    np.testing.assert_array_equal(dropped.predict(x), prefix.predict(x[:, :1]))


def test_excluded_logistic_is_bit_exact_training_base_rate():
    # A sigmoid(logit(.1)) round trip can introduce a one-ulp difference.
    # The empty prefix must equal the benchmark itself at every origin.
    x, y = np.arange(10.)[:, None], np.array([0.]*9 + [1.])
    fit = fit_regression(x, y, np.inf, logistic=True, settings=load_settings())
    np.testing.assert_array_equal(fit.predict(x), np.full(10, .1))


def test_registered_block_order_is_enforced_with_fred_appended():
    from momenta.features import FEATURE_BLOCKS

    columns = list(FEATURE_BLOCKS['HAR']) + list(FEATURE_BLOCKS['FRED'])
    resolved = feature_block_indices(columns, block_order=['HAR', 'CORE', 'M', 'L2', 'CM', 'OT'])
    assert list(resolved) == ['HAR', 'FRED']
    with pytest.raises(ValueError, match='partition'):
        feature_block_indices(['x'])
    with pytest.raises(ValueError, match='partition'):
        feature_block_indices(['x', 'z'], {'CORE': ['z'], 'HAR': ['x']})


def test_december_uses_prior_fit_and_january_begins_on_first_session():
    days = weekdays(date(2004, 1, 1), date(2006, 2, 28))
    # January 2 is retained in this synthetic calendar, not an exchange fixture.
    assert refit_for_origin(date(2005, 12, 30), days) == date(2005, 1, 3)
    assert refit_for_origin(date(2006, 1, 31), days) == date(2006, 1, 2)


def test_full_and_inner_target_windows_end_before_their_boundaries():
    settings = load_settings()
    days = weekdays(date(2001, 1, 1), date(2006, 12, 29))
    origins, ends = days[:-22], days[22:]
    refit = date(2005, 1, 3)
    split = training_split(origins, ends, days, refit,
                           np.ones(len(origins), dtype=bool), settings)
    assert all(ends[i] < refit for i in split.training)
    assert ends[split.training[-1]] == date(2004, 12, 31)
    validation_first = origins[split.validation[0]]
    assert validation_first == date(2003, 1, 3)
    distance = days.index(validation_first) - days.index(ends[split.inner[-1]])
    assert distance == 22
    assert split.inner_cutoff == ends[split.inner[-1]] + timedelta(days=1)
    assert not set(split.inner).intersection(split.validation)


def test_holdout_guard_rejects_any_later_origin():
    settings = load_settings()
    assert_design_origins([date(2015, 11, 30)], settings)
    with pytest.raises(ValueError, match='Holdout origins are locked'):
        assert_design_origins([date(2006, 1, 31), date(2015, 12, 1)], settings)


def fixture_panel():
    """Deterministic synthetic relation; expected timing comes from the calendar."""
    days = weekdays(date(2001, 1, 1), date(2007, 2, 28))
    n = len(days)-22
    x = np.sin(np.arange(n)/23.)
    features = pl.DataFrame({'date': days[:n], 'series': ['RAW']*n, 'x': x})
    targets = pl.DataFrame({'origin': days[:n], 'series': ['RAW']*n,
                            'construction': ['tercile_capped_vw']*n,
                            'target_start': days[2:n+2], 'target_end': days[22:],
                            'y1': 2+3*x, 'y2': x/100})
    return features, targets, days


def test_walk_forward_december_origin_and_future_damage_are_exact():
    settings = load_settings()
    features, targets, days = fixture_panel()
    cutoff = date(2006, 2, 28)
    kwargs = dict(series='RAW', target='y1', feature_set='fixture', columns=['x'],
                  blocks={'HAR': ['x']}, through=cutoff)
    original = walk_forward(features, targets, days, settings, **kwargs)
    changed_features = features.with_columns(
        pl.when(pl.col('date') > cutoff).then(10000.).otherwise(pl.col('x')).alias('x'))
    changed_targets = targets.with_columns([
        pl.when(pl.col('origin') > cutoff).then(10000.).otherwise(pl.col(c)).alias(c)
        for c in ('y1', 'y2')])
    changed = walk_forward(changed_features, changed_targets, days, settings, **kwargs)
    assert_frame_equal(original.forecasts, changed.forecasts, check_exact=True)
    assert_frame_equal(original.fits, changed.fits, check_exact=True)
    assert original.forecasts['origin'].to_list() == [date(2005, 12, 30),
                                                       date(2006, 1, 31), cutoff]
    assert original.forecasts['refit'].to_list() == [date(2005, 1, 3),
                                                      date(2006, 1, 2), date(2006, 1, 2)]
    assert_allclose(original.forecasts['prediction'], original.forecasts['actual'], atol=1e-12)


def test_walk_forward_excluded_extension_equals_prefix_and_audits_every_choice():
    settings = deepcopy(load_settings())
    settings['walkforward']['ridge_lambdas'] = [0.]
    features, targets, days = fixture_panel()
    z = np.where(np.arange(features.height) % 2, 1., -1.)
    features = features.with_columns(pl.Series('z', z))
    # z predicts outcomes in inner training, then its relation disappears in
    # validation. An expanded full fit would retain a nonzero z coefficient,
    # so only actual exclusion can exactly reproduce the prefix after refit.
    targets = targets.with_columns((
        pl.col('y1') + pl.when(pl.col('origin') < date(2003, 1, 1))
        .then(pl.Series(z)*2).otherwise(0.)
    ).alias('y1'))
    common = dict(series='RAW', target='y1', through=date(2005, 12, 30))
    prefix = walk_forward(features, targets, days, settings, feature_set='prefix',
                          columns=['x'], blocks={'HAR': ['x']}, **common)
    larger = walk_forward(features, targets, days, settings, feature_set='larger',
                          columns=['x', 'z'], blocks={'HAR': ['x'], 'CORE': ['z']}, **common)
    for column in ('prediction', 'residual_variance', 'training_count'):
        np.testing.assert_array_equal(larger.forecasts[column], prefix.forecasts[column])
    audit = json.loads(larger.fits['parameters_json'][0])
    assert audit['columns'] == ['x']
    assert audit['candidate_columns'] == ['x', 'z']
    assert audit['block_order'] == ['HAR', 'CORE']
    assert audit['block_lambdas'] == {'HAR': 0., 'CORE': 'exclude'}
    assert audit['clip_sd'] == 3.
    assert len(audit['validation_losses']['HAR']) == 1
    assert len(audit['validation_losses']['CORE']) == 2
    assert json.loads(larger.forecasts['block_lambdas_json'][0]) == audit['block_lambdas']
    assert json.loads(larger.forecasts['validation_losses_json'][0]) == audit['validation_losses']


def test_walk_forward_refuses_holdout_before_transform_is_called():
    settings = load_settings()
    features, targets, days = fixture_panel()
    calls = []

    def learned(frame, cutoff):
        calls.append(cutoff)
        return frame, {}

    with pytest.raises(ValueError, match='Holdout origins are locked'):
        walk_forward(features, targets, days, settings, series='RAW', target='y1',
                     feature_set='fixture', columns=['x'], blocks={'HAR': ['x']},
                     through=date(2015, 12, 31),
                     transform=learned)
    assert calls == []


def test_learned_transform_is_refit_at_inner_and_full_cutoffs():
    settings = load_settings()
    features, targets, days = fixture_panel()
    calls = []

    def learned(frame, cutoff):
        calls.append(cutoff)
        return frame, {'cutoff': cutoff.isoformat()}

    result = walk_forward(features, targets, days, settings, series='RAW', target='y1',
                          feature_set='fixture', columns=['x'], blocks={'HAR': ['x']},
                          through=date(2005, 12, 30),
                          transform=learned)
    assert len(calls) == 2
    assert calls[0] == date(2005, 1, 3)
    assert calls[1] == result.fits['inner_fit_cutoff'][0]
    assert calls[1] < result.fits['validation_origin_min'][0]


def test_tail_thresholds_are_separate_inner_and_full_training_quantiles():
    settings = load_settings()
    features, targets, days = fixture_panel()
    result = walk_forward(features, targets, days, settings, series='RAW', target='y3',
                          feature_set='fixture', columns=['x'], blocks={'HAR': ['x']},
                          through=date(2005, 12, 30))
    # Independent date filtering defines the expected target samples.
    full = targets.filter(pl.col('target_end') < date(2005, 1, 3))
    last_inner = result.fits['inner_target_end_max'][0]
    inner = targets.filter(pl.col('target_end') <= last_inner)
    assert result.fits['tail_threshold'][0] == pytest.approx(np.quantile(full['y2'], .1))
    assert result.fits['inner_tail_threshold'][0] == pytest.approx(np.quantile(inner['y2'], .1))
    assert result.fits['tail_threshold'][0] != result.fits['inner_tail_threshold'][0]
    expected_label = float(result.forecasts['y2'][0] < np.quantile(full['y2'], .1))
    assert result.forecasts['actual'][0] == expected_label


def test_month_end_calendar_keeps_december_friday():
    days = weekdays(date(2005, 12, 1), date(2006, 1, 31))
    assert month_end_origins(days) == [date(2005, 12, 30), date(2006, 1, 31)]


def test_partial_calendar_cannot_create_a_false_month_end():
    settings = load_settings()
    features, targets, days = fixture_panel()
    cutoff = date(2005, 12, 15)
    with pytest.raises(ValueError, match='calendar must extend'):
        walk_forward(features, targets, [d for d in days if d <= cutoff], settings,
                     series='RAW', target='y1', feature_set='fixture', columns=['x'],
                     blocks={'HAR': ['x']},
                     through=cutoff)


def test_training_never_uses_target_windows_ending_at_or_after_fit():
    settings = load_settings()
    features, targets, days = fixture_panel()
    kwargs = dict(series='RAW', target='y1', feature_set='fixture', columns=['x'],
                  blocks={'HAR': ['x']},
                  through=date(2005, 12, 30))
    original = walk_forward(features, targets, days, settings, **kwargs)
    damaged = targets.with_columns(
        pl.when(pl.col('target_end') >= date(2005, 1, 3)).then(99999.)
        .otherwise(pl.col('y1')).alias('y1'))
    changed = walk_forward(features, damaged, days, settings, **kwargs)
    assert_frame_equal(original.fits, changed.fits, check_exact=True)
    assert_allclose(original.forecasts['prediction'], changed.forecasts['prediction'], atol=0)
    assert original.forecasts['actual'][0] != changed.forecasts['actual'][0]
