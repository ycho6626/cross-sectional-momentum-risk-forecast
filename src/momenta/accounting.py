"""Pure daily accounting for academic legs and strategy books (Methodology)."""


from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AcademicStep:
    """A leg's daily return and surviving normalized weights (Methodology)."""

    leg_return: float
    weights: NDArray[np.float64]


@dataclass(frozen=True)
class BookStep:
    """Daily book return, ending NAV/positions, and signed terminal cash (Methodology)."""

    book_return: float
    nav: float
    positions: NDArray[np.float64]
    terminal_cash: float


@dataclass(frozen=True)
class RebalanceStep:
    """New positions, signed trades, and traded notional divided by NAV (Methodology)."""

    positions: NDArray[np.float64]
    deltas: NDArray[np.float64]
    turnover: float


def _day_arrays(
    holdings: NDArray[np.float64],
    returns: NDArray[np.float64],
    terminal_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Enforce finite held returns r ≥ −1 and aligned vectors (Methodology)."""
    held = np.asarray(holdings, dtype=np.float64)
    observed = np.asarray(returns, dtype=np.float64)
    terminal = np.asarray(terminal_mask)
    if held.ndim != 1 or observed.shape != held.shape or terminal.shape != held.shape:
        raise ValueError("Holdings, returns, and terminal mask must be aligned vectors")
    if terminal.dtype != np.bool_:
        raise ValueError("Terminal mask must contain booleans")
    if not np.all(np.isfinite(held)):
        raise ValueError("Holdings must be finite")
    active = held != 0
    if not np.all(np.isfinite(observed[active])):
        raise ValueError("Every held position requires a finite daily or terminal return")
    if np.any(observed[active] < -1):
        raise ValueError("A stock return cannot be below -100%")
    # Mask before arithmetic: zero times an inactive NaN would still be NaN.
    return held, np.where(active, observed, 0.0), terminal


def academic_leg_day(
    weights: NDArray[np.float64],
    returns: NDArray[np.float64],
    terminal_mask: NDArray[np.bool_],
) -> AcademicStep:
    """Earn R = Σω_i r_i, drift by (1+r), then renormalize survivors (Methodology).

    Input weights are nonnegative and sum to one, or are all zero for a leg
    held entirely in cash. A terminal return is earned before removing that
    security. When no positive-value survivors remain, the leg stays in cash
    with zero subsequent returns until its next formation. Inputs are unchanged.
    """
    held, observed, terminal = _day_arrays(weights, returns, terminal_mask)
    total = float(np.sum(held))
    if np.any(held < 0) or (total != 0 and not isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("Academic leg weights must be nonnegative and sum to one or zero")
    leg_return = float(np.dot(held, observed))
    drifted = held * (1 + observed)
    drifted[terminal] = 0.0
    remaining = float(np.sum(drifted))
    if remaining > 0:
        drifted /= remaining
    return AcademicStep(leg_return, drifted)


def book_day(
    positions: NDArray[np.float64],
    nav: float,
    returns: NDArray[np.float64],
    terminal_mask: NDArray[np.bool_],
) -> BookStep:
    """Earn P = Σx_i r_i / NAV, drift x, and settle terminals to cash (Methodology).

    NAV_new = NAV + Σx_i r_i. Each position becomes x_i(1+r_i) before terminal
    settlement, then terminal positions become zero. Their signed proceeds
    are terminal_cash: positive for long proceeds and negative for short
    settlement. Total cash is NAV_new − Σx_new and earns zero excess return.
    A dated runner calls this before any same-close rebalance. Inputs are unchanged.
    """
    if not isfinite(nav) or nav <= 0:
        raise ValueError("Book NAV must be positive and finite")
    held, observed, terminal = _day_arrays(positions, returns, terminal_mask)
    profit = float(np.dot(held, observed))
    new_nav = nav + profit
    if not isfinite(new_nav) or new_nav <= 0:
        raise ValueError("Book NAV must remain positive and finite after daily returns")
    drifted = held * (1 + observed)
    terminal_cash = float(np.sum(drifted[terminal]))
    drifted[terminal] = 0.0
    return BookStep(profit / nav, new_nav, drifted, terminal_cash)


def rebalance_book(
    pretrade_positions: NDArray[np.float64],
    nav: float,
    target_signed_weights: NDArray[np.float64],
) -> RebalanceStep:
    """Set x_new = NAV·a and turnover = Σ|x_new−x_pretrade|/NAV (Methodology).

    Targets already include the chosen scale and any allocation left in cash
    because it cannot be executed. Terminal positions must already be zero in
    pretrade_positions. Net formation/scaling trades once per security; do not
    sum separate turnover estimates. No costs are applied and inputs are unchanged.
    """
    if not isfinite(nav) or nav <= 0:
        raise ValueError("Rebalance NAV must be positive and finite")
    previous = np.asarray(pretrade_positions, dtype=np.float64)
    target = np.asarray(target_signed_weights, dtype=np.float64)
    if previous.ndim != 1 or target.shape != previous.shape:
        raise ValueError("Pretrade positions and target weights must be aligned vectors")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(target)):
        raise ValueError("Pretrade positions and target weights must be finite")
    positions = nav * target
    deltas = positions - previous
    turnover = float(np.sum(np.abs(deltas)) / nav)
    if not np.all(np.isfinite(positions)) or not isfinite(turnover):
        raise ValueError("Rebalanced positions and turnover must be finite")
    return RebalanceStep(positions, deltas, turnover)
