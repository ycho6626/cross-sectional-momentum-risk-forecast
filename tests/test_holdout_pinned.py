"""The real feature→forecast→strategy mock uses the production holdout path (Methodology)."""

import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from momenta.frozen_holdout import FROZEN_PATH, run_holdout
from momenta.settings import load_settings

pytestmark = pytest.mark.slow


def test_pinned_mock_holdout_rebuilds_every_frozen_table_and_reproduces_exactly():
    """Run twice without replacing builders, learned fits, forecasts or ledger accounts.

    The mock evaluates only December 2012 through November 2015, inside the design
    period. The design data root's inputs are linked into isolated attempt stages,
    never overwritten, and the second run must reproduce the first byte for byte.
    """
    settings = load_settings()
    root = Path(settings["data"]["data_root"])
    reference_path = root / "derived/final/study/forecasts.parquet"
    if not (root / "derived/final/summary.json").is_file() or not reference_path.is_file():
        pytest.skip("A completed design-period L3 run on the pinned data is required")
    frozen = tomllib.loads(FROZEN_PATH.read_text())
    first = run_holdout(settings, mock=True)
    second = run_holdout(settings, mock=True)
    assert first["execution"]["identity"]["bounds"]["first_origin"] == "2012-12-31"
    assert first["execution"]["identity"]["bounds"]["last_origin"] == "2015-11-30"
    assert second["execution"]["reproduced"]
    assert first["execution"]["stage"] != second["execution"]["stage"]
    assert first["execution"]["artifact_hashes"] == second["execution"]["artifact_hashes"]
    assert set(frozen["registry"]["tables"]) <= set(second["produced_tables"])

    # A later scoring boundary must preserve the original alignment/refit history.
    reference = pl.read_parquet(reference_path)
    reproduced = pl.read_parquet(
        Path(second["execution"]["stage"]) / "derived/final/forecasts.parquet"
    )
    keys = ["series", "target", "feature_set", "origin"]
    paired = reproduced.join(
        reference.select(
            *keys,
            pl.col("prediction").alias("frozen_prediction"),
            pl.col("block_lambdas_json").alias("frozen_penalties"),
            pl.col("refit").alias("frozen_refit"),
        ),
        on=keys,
        how="left",
        validate="1:1",
    )
    assert paired["frozen_prediction"].null_count() == 0
    np.testing.assert_allclose(
        paired["prediction"], paired["frozen_prediction"], atol=1e-12, rtol=0
    )
    assert paired["block_lambdas_json"].equals(paired["frozen_penalties"])
    assert paired["refit"].equals(paired["frozen_refit"])
