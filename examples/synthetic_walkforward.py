"""Run a deterministic, data-free risk-forecasting example using the public core.

The invented returns and weekday calendar are educational inputs. The example
does not reconstruct historical portfolios or evaluate an investment strategy.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.features import log_realized_variance
from momenta.settings import load_settings
from momenta.study_statistics import qlike
from momenta.targets import build_targets
from momenta.walkforward import ForecastRun, walk_forward

FIRST_DAY = date(2001, 1, 1)
LAST_DAY = date(2008, 12, 31)
LAST_ORIGIN = date(2008, 11, 28)


def demo_settings() -> dict[str, Any]:
    """Use released numerical settings; change only the construction's label."""
    settings = load_settings(Path(__file__).resolve().parents[1] / "config/settings.toml")
    settings["portfolios"]["primary"] = "synthetic"
    return settings


def synthetic_returns(settings: dict[str, Any]) -> pl.DataFrame:
    """Invent correlated returns with persistent volatility and a fixed seed.

    The illustrative log-scale state is s[t] = .97*s[t-1] + .12*epsilon[t].
    Independent standard-normal shocks drive that state and both return series.
    This generator is an example assumption, not a model fitted to market data.
    """
    days = [FIRST_DAY + timedelta(days=i) for i in range((LAST_DAY - FIRST_DAY).days + 1)]
    days = [day for day in days if day.weekday() < 5]
    shocks = np.random.default_rng(settings["evaluation"]["seed"]).normal(size=(len(days), 3))
    state = np.empty(len(days))
    previous = 0.0
    for i, shock in enumerate(shocks[:, 0]):
        state[i] = previous = 0.97 * previous + 0.12 * shock
    market = 0.009 * np.exp(state) * shocks[:, 1]
    own = 0.5 * market + 0.009 * np.exp(0.5 * state) * shocks[:, 2]
    return pl.DataFrame({
        "date": days, "series": "SYNTHETIC", "construction": "synthetic",
        "daily_return": own, "market_return": market,
    })


def feature_panel(returns: pl.DataFrame, settings: dict[str, Any]) -> pl.DataFrame:
    """Compute trailing log(horizon * mean(return squared)) with exported kernels."""
    horizon = settings["targets"]["horizon"]
    own = returns["daily_return"].to_numpy()
    features = {
        f"own_rv_{window}": log_realized_variance(own, window, horizon)
        for window in settings["features"]["own_rv_days"]
    }
    window = settings["features"]["market_rv_days"]
    features[f"mkt_rv_{window}"] = log_realized_variance(
        returns["market_return"].to_numpy(), window, horizon,
    )
    return returns.select("date", "series").with_columns(
        [pl.Series(name, values) for name, values in features.items()],
    )


def run_models(
    returns: pl.DataFrame, settings: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, ForecastRun]]:
    """Build completed targets, then delegate both forecasts to the existing engine."""
    days = returns["date"].to_list()
    features = feature_panel(returns, settings)
    targets = build_targets(returns, days, settings, last_origin=LAST_ORIGIN)
    own = [f"own_rv_{window}" for window in settings["features"]["own_rv_days"]]
    market = [f"mkt_rv_{settings['features']['market_rv_days']}"]
    specifications = {
        "own-risk": {"HAR": own},
        "own-risk + synthetic-market-risk": {"HAR": own, "M": market},
    }
    runs = {
        name: walk_forward(
            features, targets, days, settings, series="SYNTHETIC", target="y1",
            feature_set=name, columns=[column for block in blocks.values() for column in block],
            blocks=blocks, through=LAST_ORIGIN,
        )
        for name, blocks in specifications.items()
    }
    return features, targets, runs


def main() -> None:
    """Print descriptive losses; create no files and perform no historical analysis."""
    settings = demo_settings()
    returns = synthetic_returns(settings)
    _, _, runs = run_models(returns, settings)
    first_run = next(iter(runs.values()))
    forecasts = first_run.forecasts
    print("Synthetic risk forecasting example")
    print(f"Seed: {settings['evaluation']['seed']}; synthetic weekday returns: {returns.height}")
    print(f"Monthly origins: {forecasts['origin'].min()} to {forecasts['origin'].max()} "
          f"({forecasts.height} per model)")
    print("Annual refits: " + ", ".join(str(day) for day in first_run.fits["refit"]))
    print("Target: log(sum of squared returns at t+2 through t+22)")
    print("R2 compares each model with its training-mean forecast; lower MSE/QLIKE is better.")
    for name, run in runs.items():
        frame = run.forecasts
        actual = frame["actual"].to_numpy()
        prediction = frame["prediction"].to_numpy()
        mse = float(np.mean((actual - prediction) ** 2))
        baseline_mse = float(np.mean((actual - frame["base_prediction"].to_numpy()) ** 2))
        loss = qlike(actual, prediction, frame["residual_variance"].to_numpy()).mean()
        print(f"{name}: MSE={mse:.6f}; R2={1 - mse / baseline_mse:.6f}; QLIKE={loss:.6f}")
    augmented = runs["own-risk + synthetic-market-risk"]
    retained = sum(json.loads(value)["M"] != "exclude"
                   for value in augmented.fits["block_lambdas_json"])
    print(f"Synthetic market feature retained in {retained}/{augmented.fits.height} annual fits.")
    print("Illustration only: these synthetic losses are not historical or investment evidence.")


if __name__ == "__main__":
    main()
