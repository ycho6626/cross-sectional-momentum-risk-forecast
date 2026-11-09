"""Published BDP skew-t filter, with explicit training cutoffs (Methodology).

Source: Bianchi, De Polis and Petrella (2026), supplementary equations B1–B9,
https://doi.org/10.1093/rapstu/raag010 . We preserve the printed inverse
information normalization, including its scale entry (the density-derived
Fisher entry differs by nu). Positive rho denotes positive skew. The authors'
executable MATLAB code uses a different normalization and sign convention.
"""


from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class BDPFit:
    """A bounded likelihood fit, its initialization, and optimizer diagnostics."""

    loadings: tuple[float, float, float]
    nu: float
    initial: tuple[float, float, float]
    converged: bool
    objective: float
    observations: int
    attempts: tuple[dict[str, Any], ...]


def log_density(value: float, state: NDArray, nu: float) -> float:
    """BDP B9: log Cnu − log sigma − (nu+1)/2 log(1+z²/(nu q²))."""
    m, gamma, delta = state
    if nu <= 3 or not np.isfinite(state).all() or not math.isfinite(value):
        raise ValueError("BDP requires finite inputs and nu > 3")
    sigma, rho = math.exp(gamma), math.tanh(delta)
    z = (value - m) / sigma
    q = 1 + np.sign(z) * rho
    if sigma <= 0 or q <= 0 or abs(rho) >= 1:
        raise ValueError("BDP state reached its open parameter boundary")
    constant = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
    constant -= .5 * math.log(nu * math.pi)
    return constant - gamma - .5 * (nu + 1) * math.log1p(z * z / (nu * q * q))


def likelihood_score(value: float, state: NDArray, nu: float) -> NDArray:
    """BDP B1–B6: likelihood gradient in (m, log sigma, atanh rho)."""
    log_density(value, state, nu)
    m, gamma, delta = state
    sigma, rho = math.exp(gamma), math.tanh(delta)
    z = (value - m) / sigma
    sign = float(np.sign(z))
    q = 1 + sign * rho
    w = (nu + 1) / (nu * q * q + z * z)
    return np.array([w * z / sigma, w * z * z - 1,
                     (1 - rho * rho) * sign * w * z * z / q])


def published_information(state: NDArray, nu: float) -> NDArray:
    """Diagonal J' D J from printed B6–B7, preserving their scale normalization."""
    sigma, rho = math.exp(state[1]), math.tanh(state[2])
    return np.array([(nu + 1) / ((nu + 3) * (1 - rho * rho) * sigma * sigma),
                     2 / (nu + 3), 3 * (nu + 1) * (1 - rho * rho) / (nu + 3)])


def filter_step(value: float, state: NDArray, loadings: NDArray, nu: float) -> NDArray:
    """Published B8: f[t+1] = f[t] + diagonal(A) × (J'DJ)^−1 gradient."""
    result = np.asarray(state) + np.asarray(loadings) * (
        likelihood_score(value, state, nu) / published_information(state, nu)
    )
    if not np.isfinite(result).all():
        raise ValueError("BDP filter produced a nonfinite state")
    return result


def conditional_moments(state: NDArray, nu: float) -> tuple[float, float]:
    """BDP C: mean m+g*sigma*rho and variance sigma²[nu/(nu−2)+(3nu/(nu−2)−g²)rho²]."""
    sigma, rho = math.exp(state[1]), math.tanh(state[2])
    constant = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                        - .5 * math.log(nu * math.pi))
    g = 4 * nu * constant / (nu - 1)
    return (float(state[0] + g * sigma * rho),
            sigma * sigma * (nu / (nu - 2) * (1 + 3 * rho * rho) - g * g * rho * rho))


def skew_t_draw(state: NDArray, nu: float, rng: np.random.Generator) -> float:
    """BDP C10 density: choose its half-t side with probability (1+rho)/2."""
    rho = math.tanh(state[2])
    sign = 1 if rng.random() < (1 + rho) / 2 else -1
    return float(state[0] + math.exp(state[1]) * sign * (1 + sign * rho)
                 * abs(rng.standard_t(nu)))


def _filter(
    values: NDArray, loadings: NDArray, nu: float, initial: NDArray,
) -> tuple[float, NDArray]:
    """Scalar implementation of B8–B9; states[0] is prior, states[t+1] is post-close."""
    states = np.empty((len(values) + 1, 3))
    states[0] = initial
    m, gamma, delta = map(float, initial)
    a_m, a_gamma, a_delta = map(float, loadings)
    constant = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
    constant -= .5 * math.log(nu * math.pi)
    chi = (nu + 3) / (nu + 1)
    likelihood = 0.0
    for t, value in enumerate(values):
        sigma, rho = math.exp(gamma), math.tanh(delta)
        if sigma <= 0 or abs(rho) >= 1:
            raise ValueError("BDP filter reached an open state boundary")
        error = float(value) - m
        sign = (error > 0) - (error < 0)
        z = error / sigma
        q = 1 + sign * rho
        w = (nu + 1) / (nu * q * q + z * z)
        likelihood += constant - gamma - .5 * (nu + 1) * math.log1p(z * z / (nu*q*q))
        m += a_m * chi * (1 - rho*rho) * w * error
        gamma += a_gamma * .5 * (nu + 3) * (w*z*z - 1)
        delta += a_delta * chi * sign * w*z*z / (3*q)
        states[t + 1] = m, gamma, delta
    if not math.isfinite(likelihood) or not np.isfinite(states).all():
        raise ValueError("Nonfinite BDP likelihood or state path")
    return -likelihood / len(values), states


def fit_bdp(
    returns: NDArray, settings: dict[str, Any], *, initial: NDArray | None = None,
) -> BDPFit:
    """Fit percentage-point returns by bounded likelihood, with two preset starts (Methodology)."""
    from scipy.optimize import minimize

    cfg = settings["bdp"]
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < cfg["init_window"] or not np.isfinite(values).all():
        raise ValueError("BDP fit requires a complete training history and initialization window")
    if initial is None:
        warmup = values[:cfg["init_window"]]
        if np.std(warmup) <= 0:
            raise ValueError("BDP initialization has zero dispersion")
        initial = np.array([np.median(warmup), np.log(np.std(warmup)), 0.0])
    initial = np.asarray(initial, dtype=float)
    bounds = [tuple(np.log(cfg["loading_bounds"]))] * 3
    bounds.append(tuple(np.log(np.asarray(cfg["nu_bounds"]) - 3)))

    def objective(parameters: NDArray) -> float:
        try:
            loss, _ = _filter(values, np.exp(parameters[:3]), 3 + math.exp(parameters[3]),
                              initial)
            return loss
        except (ValueError, OverflowError, ZeroDivisionError):
            return 1e100

    attempts, results = [], []
    for start in cfg["optimizer_nu_starts"]:
        x0 = np.log([*cfg["optimizer_loading_start"], start - 3])
        fit = minimize(objective, x0, method=cfg["optimizer"], bounds=bounds, options={
            key: cfg[key] for key in ("maxiter", "maxfun", "ftol", "gtol")
        })
        attempts.append({"nu_start": start, "success": bool(fit.success),
                         "objective": float(fit.fun), "iterations": int(fit.nit),
                         "evaluations": int(fit.nfev), "message": str(fit.message)})
        results.append(fit)
    valid = [result for result in results if result.success and result.fun < 1e99]
    selected = min(valid or results, key=lambda result: result.fun)
    return BDPFit(tuple(np.exp(selected.x[:3])), 3 + math.exp(selected.x[3]), tuple(initial),
                  bool(valid), float(selected.fun), len(values), tuple(attempts))


def filter_bdp(returns: NDArray, fit: BDPFit) -> NDArray:
    """Return post-close states (m, log sigma, atanh rho) for each observed return."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("BDP filtering requires finite sequential returns")
    return _filter(values, np.array(fit.loadings), fit.nu, np.array(fit.initial))[1][1:]


def simulate_reference(settings: dict[str, Any]) -> tuple[NDArray, NDArray]:
    """Independent synthetic truth arithmetic from B8 and the two-piece density (Methodology).

    Deliberately does not call filter_step, _filter, or skew_t_draw: recovery
    must not inherit a mistake in the production filter under test.
    """
    cfg = settings["bdp"]
    rng = np.random.default_rng(cfg["seed"])
    nu = cfg["synthetic_nu"]
    a = cfg["synthetic_loadings"]
    m, gamma, delta = cfg["synthetic_initial"]
    values, states = [], []
    for _ in range(cfg["synthetic_n"]):
        scale, shape = np.exp(gamma), np.tanh(delta)
        side = 2 * (rng.uniform() < (1 + shape) / 2) - 1
        half_t = abs(rng.normal() / np.sqrt(rng.chisquare(nu) / nu))
        y = m + side * (1 + side * shape) * scale * half_t
        values.append(y)
        eps = y - m
        standardized2 = (eps / scale) ** 2
        q = 1 + np.sign(eps) * shape
        weight = (nu + 1) / (nu * q**2 + standardized2)
        gradient = np.array([weight * eps / scale**2, weight * standardized2 - 1,
                             (1-shape**2) * np.sign(eps)*weight*standardized2/q])
        information = np.array([(nu+1)/((nu+3)*(1-shape**2)*scale**2),
                                2/(nu+3), 3*(nu+1)*(1-shape**2)/(nu+3)])
        m, gamma, delta = np.array([m, gamma, delta]) + np.array(a)*gradient/information
        states.append([m, gamma, delta])
    return np.array(values), np.array(states)


def recovery_attempt(settings: dict[str, Any]) -> tuple[dict[str, Any], BDPFit]:
    """One preconfigured synthetic recovery; caller must persist settings and log first
    (Methodology)."""
    cfg = settings["bdp"]
    values, truth = simulate_reference(settings)
    fit = fit_bdp(values, settings, initial=np.array(cfg["synthetic_initial"]))
    estimated = filter_bdp(values, fit)
    scale_corr = float(np.corrcoef(np.exp(truth[:, 1]), np.exp(estimated[:, 1]))[0, 1])
    shape_corr = float(np.corrcoef(np.tanh(truth[:, 2]), np.tanh(estimated[:, 2]))[0, 1])
    relative_nu_error = abs(fit.nu / cfg["synthetic_nu"] - 1)
    passed = (fit.converged and relative_nu_error <= cfg["nu_recovery_tolerance"]
              and min(scale_corr, shape_corr) >= cfg["path_recovery_correlation"])
    return {"passed": bool(passed), "nu_relative_error": relative_nu_error,
            "scale_correlation": scale_corr, "shape_correlation": shape_corr,
            "fit": asdict(fit), "settings": cfg}, fit


def apply_bdp_features(
    frame: pl.DataFrame, books: pl.DataFrame, french: pl.DataFrame,
    cutoff: date, settings: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit French history strictly before cutoff, replace CI without changing keys/order
    (Methodology)."""
    train = french.filter(pl.col("date") < cutoff).sort("date")
    values = 100 * (train["d10"] - train["d1"]).to_numpy()
    fit = fit_bdp(values, settings)
    if not fit.converged:
        raise ValueError(f"BDP French fit failed to converge before {cutoff}")
    computed = []
    for series in frame["series"].unique().sort():
        needed = frame.filter(pl.col("series") == series)["date"].max()
        history = books.filter(pl.col("series") == series, pl.col("date") <= needed)
        history = history.sort("date")
        states = filter_bdp(100 * history["daily_return"].to_numpy(), fit)
        computed.append(history.select("date", "series").with_columns(
            pl.Series("bdp_ci", np.exp(states[:, 1]) * np.tanh(states[:, 2]) / 100),
        ))
    result = frame.drop("bdp_ci").join(pl.concat(computed), on=["date", "series"],
                                      how="left", maintain_order="left")
    result = result.with_columns(pl.lit("BDP-transfer").alias("bdp_method"))
    return result.select(frame.columns), {"fit_time": str(cutoff), "fit": asdict(fit),
                                          "french_last_observation": str(train["date"].max())}


def forecast_moments(state: NDArray, fit: BDPFit, settings: dict[str, Any]) -> dict[str, Any]:
    """Seeded dynamic 21-day simple-return moments, product(1+P)−1 (Methodology).

    States and simulated daily draws use percentage points; convert each draw
    to a decimal once. Keep model tail draws, count nonpositive wealth paths,
    and fail on nonfinite results. No frozen-state or 21×moment approximation.
    """
    cfg = settings["bdp"]
    n = cfg["horizon_simulations"]
    rng = np.random.default_rng(cfg["seed"])
    states = np.broadcast_to(state, (n, 3)).copy()
    wealth, nonpositive = np.ones(n), np.zeros(n, dtype=bool)
    nu = fit.nu
    chi = (nu+3)/(nu+1)
    for _ in range(settings["targets"]["horizon"]):
        m, scale, rho = states[:, 0], np.exp(states[:, 1]), np.tanh(states[:, 2])
        side = np.where(rng.random(n) < (1+rho)/2, 1., -1.)
        draws = m + scale*side*(1+side*rho)*abs(rng.standard_t(nu, n))
        wealth *= 1 + draws/100
        nonpositive |= wealth <= 0
        error = draws-m
        sign, z = np.sign(error), error/scale
        q = 1+sign*rho
        weight = (nu+1)/(nu*q*q+z*z)
        states[:, 0] += fit.loadings[0]*chi*(1-rho*rho)*weight*error
        states[:, 1] += fit.loadings[1]*(nu+3)/2*(weight*z*z-1)
        states[:, 2] += fit.loadings[2]*chi*sign*weight*z*z/(3*q)
        if not np.isfinite(states).all() or not np.isfinite(wealth).all():
            raise ValueError("BDP horizon simulation produced nonfinite paths")
    return {"mean": float(np.mean(wealth-1)), "variance": float(np.var(wealth-1, ddof=1)),
            "nonpositive_wealth_paths": int(nonpositive.sum()), "simulations": n,
            "seed": cfg["seed"], "horizon": settings["targets"]["horizon"]}


class BDPRecoveryFailure(RuntimeError):
    """A registered full-model acceptance test failed, permitting the labeled fallback."""


def ensure_bdp_decision(settings: dict[str, Any]) -> dict[str, Any]:
    """Persist configuration, log one recovery and first real fit, then pin the decision
    (Methodology).

    Only numerical/convergence/recovery failures permit fallback. Programming,
    missing-source and persistence errors propagate. Existing attempts are
    reused, never silently retuned or repeated after code/config/data changes.
    This is a data-heavy operation; the CLI coordinates its exclusive job slot.
    """
    from momenta.experiments import trial
    from momenta.manifest import Manifest, provenance, sha256_file
    from momenta.settings import REPO

    root = Path(settings["data"]["data_root"])
    output = root/'derived/baseline'
    artifacts = output/"bdp_attempt"
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = output/"bdp_decision.json"
    french_path = root/"curated/french/momentum_deciles_daily.parquet"
    identity = {
        "bdp_settings_hash": hashlib.sha256(json.dumps(settings["bdp"], sort_keys=True)
                                             .encode()).hexdigest(),
        "source_hash": sha256_file(Path(__file__)),
        "french_hash": sha256_file(french_path),
        "data_manifest_hash": sha256_file(Manifest(root).path),
    }
    if destination.exists():
        decision = json.loads(destination.read_text())
        if any(decision.get(key) != value for key, value in identity.items()):
            raise ValueError("Pinned BDP attempt inputs changed; do not silently repeat recovery")
        for path, digest in decision.get("artifacts", {}).items():
            if not Path(path).is_file() or sha256_file(Path(path)) != digest:
                raise ValueError("Pinned BDP attempt artifact changed")
        return decision
    configuration = artifacts/"configuration.json"
    if configuration.exists():
        raise ValueError("An interrupted BDP attempt exists; inspect its trial before resuming")
    metadata = {**provenance(REPO, Path(settings["config_path"]), Manifest(root)), **identity}
    configuration.write_text(json.dumps({"settings": settings["bdp"], "provenance": metadata},
                                         sort_keys=True, indent=2)+"\n")
    numerical_failures = (ValueError, OverflowError, FloatingPointError,
                          ZeroDivisionError, np.linalg.LinAlgError)
    decision: dict[str, Any] = {**metadata, "full_model": False, "method": "approximation",
                                "synthetic_passed": False, "first_french_passed": False}
    # The random state and the complete synthetic design were saved above,
    # before this trial starts and before either optimization is evaluated.
    try:
        with trial(settings, experiment="bdp_synthetic_recovery", kind="diagnostic",
                   series="synthetic", sample=f"T={settings['bdp']['synthetic_n']}",
                   specification={"model": "published_B8_inverse_information",
                                  "estimator": "bounded_MLE", **settings["bdp"]}) as attempt:
            try:
                values, truth = simulate_reference(settings)
            except numerical_failures as error:
                raise BDPRecoveryFailure(f"Synthetic dynamics failed: {error}") from error
            pl.DataFrame({"return": values, "m": truth[:, 0], "gamma": truth[:, 1],
                           "delta": truth[:, 2]}).write_parquet(artifacts/"synthetic_truth.parquet")
            try:
                result, fit = recovery_attempt(settings)
            except numerical_failures as error:
                raise BDPRecoveryFailure(f"Synthetic likelihood fit failed: {error}") from error
            attempt.metrics.update(result)
            (artifacts/"recovery.json").write_text(json.dumps(result, indent=2)+"\n")
            decision["recovery"] = result
            try:
                estimated = filter_bdp(values, fit)
            except numerical_failures as error:
                raise BDPRecoveryFailure(f"Recovered filter failed: {error}") from error
            pl.DataFrame({"m": estimated[:, 0], "gamma": estimated[:, 1],
                           "delta": estimated[:, 2]}).write_parquet(
                artifacts/"synthetic_estimated.parquet",
            )
            if not result["passed"]:
                raise BDPRecoveryFailure("Registered synthetic recovery criterion failed")
            decision["synthetic_passed"] = True
    except BDPRecoveryFailure as error:
        decision.update(reason=str(error), failure_type=type(error).__name__)
    if decision["synthetic_passed"]:
        french = pl.read_parquet(french_path).sort("date")
        year = date.fromisoformat(settings["walkforward"]["design_first_origin"]).year
        refit = french.filter((pl.col("date").dt.year() == year)
                              & (pl.col("date").dt.month() == 1))["date"].min()
        if refit is None:
            raise ValueError("French calendar lacks the first January refit")
        history = french.filter(pl.col("date") < refit)
        try:
            with trial(settings, experiment="bdp_first_french_fit", kind="diagnostic",
                       series="French D10-D1", sample=f"date < {refit}",
                       specification={"model": "published_B8_inverse_information",
                                      "estimator": "bounded_MLE", **settings["bdp"]}) as attempt:
                values = 100*(history["d10"]-history["d1"]).to_numpy()
                try:
                    fit = fit_bdp(values, settings)
                except numerical_failures as error:
                    raise BDPRecoveryFailure(f"First French likelihood failed: {error}") from error
                result = {"passed": fit.converged, "fit": asdict(fit), "fit_time": str(refit),
                          "last_observation": str(history["date"].max())}
                attempt.metrics.update(result)
                (artifacts/"first_french_fit.json").write_text(json.dumps(result, indent=2)+"\n")
                decision["first_french_fit"] = result
                if not fit.converged:
                    raise BDPRecoveryFailure("First French likelihood fit did not converge")
                decision.update(first_french_passed=True, full_model=True, method="BDP-transfer")
        except BDPRecoveryFailure as error:
            decision.update(reason=str(error), failure_type=type(error).__name__)
    decision["artifacts"] = {str(path): sha256_file(path) for path in sorted(artifacts.iterdir())}
    destination.write_text(json.dumps(decision, indent=2, sort_keys=True)+"\n")
    table = REPO/"results/tables/bdp_recovery.csv"
    records = [{"criterion": "synthetic_recovery", "passed": decision["synthetic_passed"]},
               {"criterion": "first_french_fit", "passed": decision["first_french_passed"]}]
    for key in ("nu_relative_error", "scale_correlation", "shape_correlation"):
        value = decision.get("recovery", {}).get(key)
        records.append({"criterion": key, "value": value})
    table.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records, infer_schema_length=None).write_csv(table)
    table.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return decision
