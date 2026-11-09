"""Independent HMM recursion arithmetic, training-only PCA and fit timing (Methodology)."""


from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from momenta import regimes


@pytest.fixture
def settings():
    return {"walkforward": {"feature_clip_sd": 3., "blas_threads": 1,
                             "feature_start": "2001-01-01"},
            "hmm": {"states": 2, "secondary_states": 3, "restarts": 10, "pcs": 3,
                    "seed": 20261007, "n_iter": 500, "tol": 1e-4, "min_covar": 1e-3,
                    "occupancy_min": .05, "startprob_prior": 1., "transmat_prior": 1.,
                    "means_prior": 0., "means_weight": 0., "covars_prior": .01,
                    "covars_weight": 1.}}


@pytest.fixture
def known_hmm():
    return {"start": [.6, .4], "transition": [[.9, .1], [.2, .8]],
            "means": [[-1., 0., 0.], [1., 0., 0.]], "variances": [[1.]*3]*2,
            "stationary": [2/3, 1/3], "log_likelihood": -10., "restarts": []}


def test_log_forward_and_emission_control_match_hand_bayes_updates():
    densities = np.array([[.8, .2], [.1, .9], [.5, .5]])
    transition = np.array([[.9, .1], [.2, .8]])
    forward = regimes.log_forward(np.log(densities), transition, [.6, .4])
    assert_allclose(forward, [[6/7, 1/7], [4/13, 9/13], [27/65, 38/65]], rtol=1e-14)
    stationary = regimes.stationary_distribution(transition)
    assert_allclose(stationary, [2/3, 1/3], rtol=1e-14)
    emission = regimes.emission_probabilities(np.log(densities), stationary)
    assert_allclose(emission, [[8/9, 1/9], [2/11, 9/11], [2/3, 1/3]], rtol=1e-14)
    assert_allclose(regimes.log_forward(np.log(densities)-10000, transition, [.6, .4]),
                    forward, atol=1e-12)
    damaged = np.log(densities)
    damaged[2] = [-100, 100]
    assert_allclose(regimes.log_forward(damaged, transition, [.6, .4])[:2], forward[:2],
                    rtol=0, atol=0)
    with pytest.raises(ValueError, match="not unique"):
        regimes.stationary_distribution(np.eye(2))


def test_gaussian_log_density_matches_product_of_univariate_densities():
    x = np.array([[0., 2., -1.], [1., -2., 3.]])
    mu = np.array([[0., 0., 0.], [1., 2., 3.]])
    variance = np.array([[1., 4., 9.], [4., 1., 16.]])
    expected = []
    for row in x:
        expected.append([np.log(np.prod([
            np.exp(-((point-mean)/sd)**2/2)/(sd*np.sqrt(2*np.pi))
            for point, mean, sd in zip(row, means, np.sqrt(var), strict=True)
        ])) for means, var in zip(mu, variance, strict=True)])
    assert_allclose(regimes.gaussian_log_density(x, mu, variance), expected)


def test_pca_uses_population_training_moments_and_aligns_only_signs(settings):
    x = np.array([[-2., -1., 4.], [-1., 1., 4.], [1., -1., 4.], [2., 1., 4.]])
    fitted = regimes.fit_preprocessor(x, settings)
    assert_allclose(fitted["mean"], [0., 0., 4.])
    assert_allclose(fitted["scale"], [np.sqrt(2.5), 1., 1.])
    projected = regimes.project(x, fitted)
    expected_variance = [1+1/np.sqrt(10), 1-1/np.sqrt(10), 0]
    assert_allclose(np.cov(projected, rowvar=False, ddof=0), np.diag(expected_variance),
                    atol=1e-14)
    reversed_axes = -np.asarray(fitted["components"])
    aligned = regimes.fit_preprocessor(x, settings, reversed_axes)
    assert_allclose(aligned["components"], reversed_axes, atol=1e-14)
    extreme = np.array([[100., -100., 4.]])
    assert_allclose(regimes.project(extreme, fitted),
                    np.array([[3., -3., 0.]]) @ np.asarray(fitted["components"]).T)


def test_restarts_select_training_likelihood_but_reject_iteration_limit(settings, monkeypatch):
    from hmmlearn import hmm

    seed0 = settings["hmm"]["seed"]

    class FixtureHMM:
        def __init__(self, *, random_state, **kwargs):
            self.restart = random_state-seed0
            self.monitor_ = SimpleNamespace(history=[-100., -100.+1e-5],
                iter=500 if self.restart == 9 else 10)
            self.startprob_ = np.array([.4, .6])
            self.transmat_ = np.array([[.8, .2], [.1, .9]])
            self.means_ = np.array([[1., 0., 0.], [-1., 0., 0.]])
            self.covars_ = np.array([np.eye(3), np.eye(3)])

        def fit(self, x):
            return self

        def score(self, x):
            return -100+self.restart

        def predict_proba(self, x):
            raise AssertionError("Smoothed probabilities must never be requested")

    monkeypatch.setattr(hmm, "GaussianHMM", FixtureHMM)
    fitted = regimes.fit_hmm(np.zeros((30, 3)), settings)
    assert fitted["selected_restart"] == 8
    assert fitted["restarts"][-1]["status"] == "nonconverged"
    assert len(fitted["restarts"]) == 10
    assert [row["seed"] for row in fitted["restarts"]] == list(range(seed0, seed0+10))
    assert fitted["means"][0][0] == -1
    assert_allclose(fitted["start"], [.6, .4])
    assert_allclose(fitted["transition"], [[.9, .1], [.2, .8]])


@pytest.fixture
def daily_panel():
    start = date(2000, 12, 1)
    calendar = [start+timedelta(days=i) for i in range(1600)
                if (start+timedelta(days=i)).weekday() < 5]
    ends = [day for i, day in enumerate(calendar[:-1])
            if calendar[i+1].month != day.month]
    days = [day for day in calendar if date(2001, 1, 1) <= day <= date(2005, 2, 28)]
    locations = {day: i for i, day in enumerate(calendar)}
    x = np.arange(len(days), dtype=float)
    frame = pl.DataFrame({"date": days, "series": ["RAW"]*len(days),
        "target_end": [calendar[locations[day]+22] for day in days],
        "a": np.sin(x/30), "b": np.cos(x/17), "c": np.sin(x/51), "y1": -3+x/1000})
    initial = pl.DataFrame({"date": [date(2000, 12, 29)], "series": ["RAW"],
                            "a": [.3], "b": [-.2], "c": [.1]})
    return frame, ends, initial


def test_regime_transform_fits_completed_months_and_filters_warmup_only(
    settings, known_hmm, daily_panel, monkeypatch, tmp_path,
):
    frame, ends, initial = daily_panel
    calls = []

    def fit(values, settings, *, states):
        calls.append(values.copy())
        return known_hmm

    monkeypatch.setattr(regimes, "fit_hmm", fit)
    kwargs = {"columns": ["a", "b", "c"], "initial_monthly": initial,
              "cache_dir": tmp_path, "provenance": {"config_hash": "fixture"}}
    result, full = regimes.regime_transform(frame, date(2005, 1, 3), ends, settings, **kwargs)
    assert full["training_months"] == 47
    assert full["training_first_origin"] == "2001-01-31"
    assert full["training_last_origin"] == "2004-11-30"
    train = frame.filter(pl.col("date").is_in(ends), pl.col("target_end") < date(2005, 1, 3))
    assert_allclose(full["preprocessor"]["mean"], train.select("a", "b", "c").to_numpy().mean(0))
    assert result["hmmf_0"].null_count() == 0
    assert "hmmf_1" not in result.columns and "hmme_1" not in result.columns
    january = result.filter(pl.col("date") < date(2001, 1, 31))
    assert january["hmmf_0"].n_unique() == january["hmme_0"].n_unique() == 1
    _, inner = regimes.regime_transform(frame, date(2002, 12, 3), ends, settings, **kwargs)
    assert inner["training_months"] == 22
    assert len(calls) == 2
    repeated, repeated_metadata = regimes.regime_transform(
        frame, date(2005, 1, 3), ends, settings, **kwargs)
    assert len(calls) == 2 and repeated_metadata["fit_hash"] == full["fit_hash"]
    assert_frame_equal(result, repeated, check_exact=True)
    damaged = frame.with_columns([
        pl.when(pl.col("date") > date(2005, 1, 31)).then(10000.)
        .otherwise(pl.col(name)).alias(name) for name in ("a", "b", "c")])
    changed, metadata = regimes.regime_transform(
        damaged, date(2005, 1, 3), ends, settings, **kwargs)
    assert metadata["fit_hash"] == full["fit_hash"] and len(calls) == 2
    assert_frame_equal(result.filter(pl.col("date") <= date(2005, 1, 31)),
                       changed.filter(pl.col("date") <= date(2005, 1, 31)), check_exact=True)
    assert not result.equals(changed)
    with pytest.raises(ValueError, match="future fit"):
        regimes.regime_transform(frame, date(2005, 1, 3), ends, settings, previous=full, **kwargs)


def test_probability_diagnostics_use_training_occupancy_and_geometric_duration(known_hmm):
    result = regimes.diagnostics([[.9, .1]]*19+[[.1, .9]], known_hmm, .06)
    assert result["occupancy"] == [.95, .05]
    assert result["low_occupancy"] == [False, True]
    assert_allclose(result["mean_duration_months"], [10., 5.])


def test_three_state_features_omit_only_last_state_and_preserve_input_order(
    settings, daily_panel, monkeypatch,
):
    frame, ends, initial = daily_panel
    model = {"start": [1/3]*3, "transition": [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]],
             "means": [[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
             "variances": [[1.]*3]*3, "stationary": [1/3]*3,
             "log_likelihood": -20., "restarts": []}

    def fit(values, settings, *, states):
        assert states == 3
        return model

    monkeypatch.setattr(regimes, "fit_hmm", fit)
    result, record = regimes.regime_transform(
        frame.reverse(), date(2005, 1, 3), ends, settings, states=3,
        columns=["a", "b", "c"], initial_monthly=initial,
    )
    assert result["date"].equals(frame.reverse()["date"])
    assert {name for name in result.columns if name.startswith("hmm")} == {
        "hmmf_0", "hmmf_1", "hmme_0", "hmme_1"}
    monthly = pl.DataFrame(record["monthly_probabilities"])
    assert_allclose(monthly.select("hmmf_0", "hmmf_1", "hmmf_2").to_numpy().sum(1), 1)
    assert_allclose(monthly.select("hmme_0", "hmme_1", "hmme_2").to_numpy().sum(1), 1)
