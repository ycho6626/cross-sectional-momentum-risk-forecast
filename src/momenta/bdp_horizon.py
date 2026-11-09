"""BDP holding-window moments after the execution lag (Methodology)."""


from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from momenta.bdp import BDPFit


def holding_moments(state: NDArray, fit: BDPFit, settings: dict[str, Any]) -> dict[str, Any]:
    """Simulate BDP dynamics, then compound earned returns t+lag+1..t+lag+horizon.

    Published BDP supplementary B8–B9 updates apply on every simulated day,
    including the execution lag. Wealth starts only after that lag (Methodology). Draws use
        percentage points and convert to decimals once. Return
    sample variance across seeded paths, retaining all tail draws and counting
    paths whose earned wealth becomes nonpositive at any holding date.
    """
    cfg = settings["bdp"]
    n = cfg["horizon_simulations"]
    lag, horizon = settings["targets"]["lag"], settings["targets"]["horizon"]
    if n < 2 or lag < 0 or horizon < 1:
        raise ValueError("BDP moments need at least two paths and a valid holding window")
    state = np.asarray(state, dtype=float)
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError("BDP moments require a finite three-dimensional state")
    rng = np.random.default_rng(cfg["seed"])
    states = np.broadcast_to(state, (n, 3)).copy()
    wealth, nonpositive = np.ones(n), np.zeros(n, dtype=bool)
    nu = fit.nu
    chi = (nu+3)/(nu+1)
    for day in range(lag+horizon):
        m, scale, rho = states[:, 0], np.exp(states[:, 1]), np.tanh(states[:, 2])
        side = np.where(rng.random(n) < (1+rho)/2, 1., -1.)
        draws = m + scale*side*(1+side*rho)*abs(rng.standard_t(nu, n))
        if day >= lag:
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
            raise ValueError("BDP holding simulation produced nonfinite paths")
    mean, variance = float(np.mean(wealth-1)), float(np.var(wealth-1, ddof=1))
    if not np.isfinite([mean, variance]).all():
        raise ValueError("BDP holding moments are nonfinite")
    return {"mean": mean, "variance": variance,
            "nonpositive_wealth_paths": int(nonpositive.sum()),
            "nonpositive_earned_wealth_paths": int(nonpositive.sum()),
            "simulations": n, "seed": cfg["seed"], "horizon": horizon,
            "lag": lag, "horizon_start": lag+1, "horizon_end": lag+horizon}
