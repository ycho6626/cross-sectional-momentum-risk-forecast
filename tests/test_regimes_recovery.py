"""The single registered HMM recovery fixture, rerun unchanged for validation (Methodology)."""


import numpy as np
import pytest

from momenta.regimes import synthetic_recovery
from momenta.settings import load_settings

pytestmark = pytest.mark.slow


def test_fixed_markov_fixture_recovers_states_and_filtering_beats_emissions():
    settings = load_settings()
    config = settings["hmm"]
    assert (config["seed"], config["synthetic_n"], config["synthetic_dims"],
            config["synthetic_persistence"], config["synthetic_separation"]) == (
                20261007, 300, 3, .95, 1.5)
    metrics, paths = synthetic_recovery(settings)
    truth = paths["truth"].to_numpy()
    filtered = paths["filtered"].to_numpy()
    emission = paths["emission"].to_numpy()
    # Independently score the returned probabilities against planted states.
    filtered_accuracy = np.mean((filtered > .5) == truth)
    emission_accuracy = np.mean((emission > .5) == truth)
    assert filtered_accuracy >= .85
    assert filtered_accuracy > emission_accuracy
    assert np.mean((filtered-truth)**2) < np.mean((emission-truth)**2)
    assert metrics["filtered_accuracy"] == filtered_accuracy
    assert metrics["emission_accuracy"] == emission_accuracy
    assert len(metrics["fit"]["restarts"]) == 10
