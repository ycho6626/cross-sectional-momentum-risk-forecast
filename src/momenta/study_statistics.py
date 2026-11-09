"""Small statistical functions for the design study (Methodology).

Primary sources:
Clark–West: https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf
Diebold–Mariano: https://users.ssc.wisc.edu/~behansen/718/DieboldMariano1995.pdf
Stationary bootstrap: https://www.math.ucsd.edu/~politis/SBblock-revER.pdf
Lo (2002): https://doi.org/10.2469/faj.v58.n4.2453
Bailey–Lopez de Prado: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
"""


from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata


def _vector(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("Statistics require a nonempty finite vector")
    return result


def _paired(*values: ArrayLike) -> list[NDArray[np.float64]]:
    arrays = [_vector(value) for value in values]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("Paired vectors must have the same length")
    return arrays


def holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm p_(i)^adj=min(1,max_{j≤i}((m−j+1)p_(j))) (Methodology).

    Preserve input order. An unavailable test leaves the preregistered family
    incomplete, so every adjusted p-value in that family remains unavailable.
    """
    if any(
        value is not None and (not np.isfinite(value) or not 0 <= value <= 1) for value in pvalues
    ):
        raise ValueError("Holm p-values must be finite probabilities or None")
    if any(value is None for value in pvalues):
        return [None] * len(pvalues)
    result = [None] * len(pvalues)
    largest = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        largest = max(largest, (len(pvalues) - rank) * pvalues[index])
        result[index] = min(1.0, largest)
    return result


def nw_test(
    loss_difference: ArrayLike,
    lags: int,
    alternative: Literal["greater", "two-sided"] = "greater",
) -> dict[str, Any]:
    """Test mean(d)=0 using Bartlett HAC: γ0+2Σ(1−k/(L+1))γk (Methodology).

    Autocovariances use denominator n, without a finite-sample correction.
    Positive differences favour the candidate. Constant losses are descriptive.
    """
    values = _vector(loss_difference)
    if lags < 0 or alternative not in {"greater", "two-sided"}:
        raise ValueError("Invalid Newey–West lag or alternative")
    n = len(values)
    used_lags = min(lags, n - 1)
    centered = values - values.mean()
    variance = float(centered @ centered / n)
    for lag in range(1, used_lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        variance += 2 * (1 - lag / (used_lags + 1)) * covariance
    result: dict[str, Any] = {
        "n": n,
        "mean_difference": float(values.mean()),
        "lags": used_lags,
        "standard_error": None,
        "statistic": None,
        "pvalue": None,
        "status": "degenerate",
    }
    if n < 2 or np.ptp(values) == 0 or variance <= 0:
        return result
    standard_error = float(np.sqrt(variance / n))
    statistic = float(values.mean() / standard_error)
    pvalue = float(ndtr(-statistic) if alternative == "greater" else 2 * ndtr(-abs(statistic)))
    result.update(standard_error=standard_error, statistic=statistic, pvalue=pvalue, status="ok")
    return result


def clark_west(
    actual: ArrayLike,
    baseline: ArrayLike,
    prediction: ArrayLike,
    lags: int,
) -> dict[str, Any]:
    """CW adjusted difference e0²−e1²+(f0−f1)²; one-sided HAC (Methodology)."""
    y, base, model = _paired(actual, baseline, prediction)
    return nw_test((y - base) ** 2 - (y - model) ** 2 + (base - model) ** 2, lags)


def diebold_mariano(
    actual: ArrayLike,
    baseline: ArrayLike,
    prediction: ArrayLike,
    lags: int,
) -> dict[str, Any]:
    """DM unadjusted squared-loss difference; two-sided descriptive HAC (Methodology)."""
    y, base, model = _paired(actual, baseline, prediction)
    return nw_test((y - base) ** 2 - (y - model) ** 2, lags, "two-sided")


def stationary_indices(
    dates: Sequence[date],
    reps: int,
    block_months: float,
    seed: int,
) -> NDArray[np.int64]:
    """Circular geometric blocks, restart probability 1/b, shared by pairs (Methodology).

    One observation per consecutive calendar month is required. A missing month
    must not silently become an adjacent month in the resampling calendar.
    """
    months = np.array([value.year * 12 + value.month for value in dates])
    if not len(months) or (len(months) > 1 and not np.all(np.diff(months) == 1)):
        raise ValueError("Bootstrap requires ordered, unique, consecutive calendar months")
    if reps < 1 or not np.isfinite(block_months) or block_months < 1:
        raise ValueError("Positive replications and mean block length at least one are required")
    random = np.random.default_rng(seed)
    n = len(months)
    starts = random.integers(0, n, size=(reps, n))
    restarts = random.random((reps, n - 1)) < 1 / block_months
    indices = np.empty((reps, n), dtype=np.int64)
    indices[:, 0] = starts[:, 0]
    for index in range(1, n):
        indices[:, index] = np.where(
            restarts[:, index - 1], starts[:, index], (indices[:, index - 1] + 1) % n
        )
    return indices


def interval(draws: ArrayLike, confidence: float = 0.95) -> dict[str, Any]:
    """Percentile bootstrap interval; disclose undefined resamples (Methodology)."""
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap draws or confidence level")
    finite = values[np.isfinite(values)]
    lower, upper = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    return {
        "lower": float(np.quantile(finite, lower)) if len(finite) else None,
        "upper": float(np.quantile(finite, upper)) if len(finite) else None,
        "valid_draws": len(finite),
        "undefined_draws": len(values) - len(finite),
    }


def qlike(
    actual_log_variance: ArrayLike, prediction: ArrayLike, residual_variance: ArrayLike
) -> NDArray[np.float64]:
    """QLIKE = v/vhat−log(v/vhat)−1, vhat=exp(prediction+s²/2) (Methodology)."""
    actual, prediction, residual = _paired(actual_log_variance, prediction, residual_variance)
    if np.any(residual < 0):
        raise ValueError("Training residual variance must be nonnegative")
    log_ratio = actual - prediction - residual / 2
    with np.errstate(over="raise"):
        return np.expm1(log_ratio) - log_ratio


def brier(actual: ArrayLike, probability: ArrayLike) -> float:
    """Brier score = mean((binary outcome−probability)²), Methodology."""
    y, p = _paired(actual, probability)
    if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Brier needs binary outcomes and probabilities in [0,1]")
    return float(np.mean((y - p) ** 2))


def _auc_rows(actual: NDArray, probability: NDArray) -> NDArray[np.float64]:
    ranks = rankdata(probability, method="average", axis=-1)
    positives = actual.sum(axis=-1)
    negatives = actual.shape[-1] - positives
    numerator = (ranks * actual).sum(axis=-1) - positives * (positives + 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(numerator / (positives * negatives))


def auc(actual: ArrayLike, probability: ArrayLike) -> float | None:
    """AUC = P(score+ > score−)+½P(tie); undefined for one class (Methodology)."""
    y, p = _paired(actual, probability)
    brier(y, p)
    value = float(_auc_rows(y, p))
    return value if np.isfinite(value) else None


def forecast_metrics(
    frame: pl.DataFrame,
    *,
    target: str,
    reps: int,
    block_months: float,
    seed: int,
    nw_lags: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired monthly forecast losses, HAC tests and stationary intervals (Methodology)."""
    frame = frame.sort("origin")
    actual, prediction, baseline = _paired(
        frame["actual"].to_numpy(),
        frame["prediction"].to_numpy(),
        frame["base_prediction"].to_numpy(),
    )
    indices = stationary_indices(frame["origin"].to_list(), reps, block_months, seed)
    result: dict[str, Any] = {
        "n": len(actual),
        "first_origin": str(frame["origin"].min()),
        "last_origin": str(frame["origin"].max()),
    }
    if target == "y3":
        loss = (actual - prediction) ** 2
        result.update(
            brier=brier(actual, prediction),
            auc=auc(actual, prediction),
            baseline_brier=brier(actual, baseline),
            events=int(actual.sum()),
            brier_interval=interval(loss[indices].mean(axis=1), confidence),
            auc_interval=interval(_auc_rows(actual[indices], prediction[indices]), confidence),
            inference="descriptive",
        )
        return result
    if target not in {"y1", "y2"}:
        raise ValueError("Unknown forecast target")
    model_loss, base_loss = (actual - prediction) ** 2, (actual - baseline) ** 2
    difference = base_loss - model_loss
    denominator = float(base_loss.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        bootstrap_r2 = 1 - model_loss[indices].sum(axis=1) / base_loss[indices].sum(axis=1)
    result.update(
        mse=float(model_loss.mean()),
        baseline_mse=float(base_loss.mean()),
        mse_improvement=float(difference.mean()),
        r2_oos=1 - float(model_loss.sum()) / denominator if denominator else None,
        mse_interval=interval(model_loss[indices].mean(axis=1), confidence),
        mse_improvement_interval=interval(difference[indices].mean(axis=1), confidence),
        r2_oos_interval=interval(bootstrap_r2, confidence),
        cw=clark_west(actual, baseline, prediction, nw_lags),
        dm=diebold_mariano(actual, baseline, prediction, nw_lags),
    )
    if target == "y1":
        result["qlike"] = float(
            qlike(actual, prediction, frame["residual_variance"].to_numpy()).mean()
        )
    return result


def h2_contrast(
    actual_a: ArrayLike,
    full_a: ArrayLike,
    reduced_a: ArrayLike,
    actual_b: ArrayLike,
    full_b: ArrayLike,
    reduced_b: ArrayLike,
    indices: NDArray[np.int64],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Δa−Δb, Δ=1−SSE(full)/SSE(reduced), with shared bootstrap draws (Methodology)."""
    ya, fa, ra, yb, fb, rb = _paired(actual_a, full_a, reduced_a, actual_b, full_b, reduced_b)
    loss_fa, loss_ra = (ya - fa) ** 2, (ya - ra) ** 2
    loss_fb, loss_rb = (yb - fb) ** 2, (yb - rb) ** 2
    if loss_ra.sum() <= 0 or loss_rb.sum() <= 0:
        raise ValueError("H2 reduced-model SSE must be positive")
    delta_a = 1 - float(loss_fa.sum() / loss_ra.sum())
    delta_b = 1 - float(loss_fb.sum() / loss_rb.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = -loss_fa[indices].sum(axis=1) / loss_ra[indices].sum(axis=1) + loss_fb[indices].sum(
            axis=1
        ) / loss_rb[indices].sum(axis=1)
    return {
        "delta_a": delta_a,
        "delta_b": delta_b,
        "contrast": delta_a - delta_b,
        "interval": interval(draws, confidence),
        "n": len(ya),
    }


def lo_sharpe(returns: ArrayLike, periods: int = 12) -> float | None:
    """Lo (2002), eqs.19–20: qμ/sqrt(qγ0+2Σ(q−k)γk), 1/T moments."""
    values = _vector(returns)
    if periods < 1:
        raise ValueError("Aggregation periods must be positive")
    if len(values) < periods or len(values) < 2:
        return None
    centered = values - values.mean()
    variance = periods * float(centered @ centered / len(values))
    for lag in range(1, periods):
        variance += 2 * (periods - lag) * float(centered[lag:] @ centered[:-lag] / len(values))
    return float(periods * values.mean() / np.sqrt(variance)) if variance > 0 else None


def lo_sharpe_draws(returns: NDArray[np.float64], periods: int = 12) -> NDArray[np.float64]:
    """Apply Lo's same q-period aggregation to each bootstrap row (Methodology)."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or periods < 1:
        raise ValueError("Invalid Sharpe bootstrap matrix")
    if values.shape[1] < max(periods, 2):
        return np.full(values.shape[0], np.nan)
    centered = values - values.mean(axis=1, keepdims=True)
    variance = periods * np.mean(centered**2, axis=1)
    for lag in range(1, periods):
        variance += (
            2
            * (periods - lag)
            * np.sum(centered[:, lag:] * centered[:, :-lag], axis=1)
            / values.shape[1]
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variance > 0, periods * values.mean(axis=1) / np.sqrt(variance), np.nan)


def deflated_sharpe(
    returns: ArrayLike,
    trial_sharpes: ArrayLike,
    distinct_trials: int,
) -> dict[str, Any]:
    """DSR = Φ((SR−SR0)√(T−1)/√(1−skew·SR+(kurt−1)SR²/4)).

    SR0 uses across-trial SD and the expected maximum over N specifications
    (Bailey–Lopez de Prado 2014). All Sharpes are monthly, not annualized.
    Failed specifications stay in N; missing estimates make DSR descriptive.
    """
    values = _vector(returns)
    sharpes = np.asarray(trial_sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    if distinct_trials < 1 or len(sharpes) > distinct_trials:
        raise ValueError("Invalid distinct-strategy count")
    result: dict[str, Any] = {
        "dsr": None,
        "distinct_trading_strategies": distinct_trials,
        "strategies_with_sharpe": len(sharpes),
        "benchmark_monthly_sharpe": None,
        "descriptive": len(sharpes) < distinct_trials,
    }
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return result
    if distinct_trials == 1:
        threshold = 0.0
    elif len(sharpes) < 2:
        return result
    else:
        threshold = float(
            np.std(sharpes, ddof=1)
            * (
                (1 - np.euler_gamma) * ndtri(1 - 1 / distinct_trials)
                + np.euler_gamma * ndtri(1 - 1 / (distinct_trials * np.e))
            )
        )
    sharpe = float(np.mean(values) / np.std(values, ddof=1))
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    skew = float(np.mean(centered**3) / variance**1.5)
    kurtosis = float(np.mean(centered**4) / variance**2)
    denominator = 1 - skew * sharpe + (kurtosis - 1) * sharpe**2 / 4
    if denominator > 0:
        result["dsr"] = float(ndtr((sharpe - threshold) * np.sqrt((len(values) - 1) / denominator)))
    result.update(benchmark_monthly_sharpe=threshold, monthly_sharpe=sharpe)
    return result
