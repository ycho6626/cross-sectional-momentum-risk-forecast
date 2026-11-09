"""Hand fixtures and conservation properties for portfolio accounting."""


import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from momenta.accounting import academic_leg_day, book_day, rebalance_book


def test_two_day_example_distinguishes_monthly_legs_daily_spread_and_strategy():
    winner = np.array([1.0])
    loser = np.array([1.0])
    winner_returns, loser_returns, spreads = [], [], []
    for winner_return, loser_return in [(0.1, 0.0), (0.0, 0.1)]:
        win = academic_leg_day(winner, np.array([winner_return]), np.array([False]))
        lose = academic_leg_day(loser, np.array([loser_return]), np.array([False]))
        winner, loser = win.weights, lose.weights
        winner_returns.append(win.leg_return)
        loser_returns.append(lose.leg_return)
        spreads.append(win.leg_return - lose.leg_return)
    assert spreads == pytest.approx([0.1, -0.1])
    assert np.prod(1 + np.array(winner_returns)) - np.prod(1 + np.array(loser_returns)) == 0
    assert np.prod(1 + np.array(spreads)) - 1 == pytest.approx(-0.01)

    first = book_day(np.array([1.0, -1.0]), 1.0, np.array([0.1, 0.0]), np.zeros(2, bool))
    second = book_day(first.positions, first.nav, np.array([0.0, 0.1]), np.zeros(2, bool))
    assert first.book_return == pytest.approx(0.1)
    assert second.book_return == pytest.approx(-1 / 11)
    assert second.nav == pytest.approx(1.0)
    assert (1 + first.book_return) * (1 + second.book_return) - 1 == pytest.approx(0.0)


def test_previous_book_earns_execution_day_before_new_formation_is_traded():
    initial = np.array([1.0, -1.0, 0.0, 0.0])
    # New names move +50% and -50% today, but only the prior book is held.
    earned = book_day(initial, 1.0, np.array([0.1, 0.0, 0.5, -0.5]), np.zeros(4, bool))
    assert earned.book_return == pytest.approx(0.1)
    trade = rebalance_book(earned.positions, earned.nav, np.array([0.0, 0.0, 1.0, -1.0]))
    assert trade.positions == pytest.approx([0.0, 0.0, 1.1, -1.1])
    assert trade.deltas == pytest.approx([-1.1, 1.0, 1.1, -1.1])
    assert trade.turnover == pytest.approx(43 / 11)
    next_day = book_day(
        trade.positions, earned.nav, np.array([np.nan, np.nan, 0.0, 0.1]), np.zeros(4, bool)
    )
    assert next_day.book_return == pytest.approx(-0.1)
    assert next_day.nav == pytest.approx(0.99)


def test_first_execution_day_is_cash_and_unavailable_target_allocation_stays_cash():
    initial = book_day(np.zeros(3), 1.0, np.full(3, np.nan), np.array([True, False, False]))
    assert initial.book_return == 0
    assert initial.nav == 1
    # Runner zeroes the unavailable first target without redistributing its half leg.
    trade = rebalance_book(initial.positions, initial.nav, np.array([0.0, 0.5, -1.0]))
    assert trade.positions == pytest.approx([0.0, 0.5, -1.0])
    assert trade.turnover == 1.5
    next_day = book_day(trade.positions, 1.0, np.array([np.nan, 0.1, 0.0]), np.zeros(3, bool))
    assert next_day.book_return == pytest.approx(0.05)


def test_terminal_long_earns_then_settles_with_academic_survivor_renormalization():
    book = book_day(
        np.array([0.5, 0.5, -1.0]), 1.0,
        np.array([-0.4, 0.2, 0.0]), np.array([True, False, False]),
    )
    leg = academic_leg_day(
        np.array([0.5, 0.5]), np.array([-0.4, 0.2]), np.array([True, False])
    )
    assert book.book_return == pytest.approx(-0.1)
    assert book.nav == pytest.approx(0.9)
    assert book.positions == pytest.approx([0.0, 0.6, -1.0])
    assert book.terminal_cash == pytest.approx(0.3)
    assert book.nav - sum(book.positions) == pytest.approx(1.3)
    assert leg.leg_return == pytest.approx(-0.1)
    assert leg.weights == pytest.approx([0.0, 1.0])

    next_book = book_day(
        book.positions, book.nav, np.array([np.nan, 0.1, 0.0]), np.zeros(3, bool)
    )
    next_leg = academic_leg_day(leg.weights, np.array([np.nan, 0.1]), np.zeros(2, bool))
    assert next_book.book_return == pytest.approx(1 / 15)
    assert next_book.nav == pytest.approx(0.96)
    assert next_book.terminal_cash == 0
    assert next_book.nav - sum(next_book.positions) == pytest.approx(1.3)
    assert next_leg.leg_return == pytest.approx(0.1)
    assert (1 + leg.leg_return) * (1 + next_leg.leg_return) - 1 == pytest.approx(-0.01)

    # Reinvesting terminal cash at a later scheduled close is a trade; settlement itself is not.
    trade = rebalance_book(book.positions, book.nav, np.array([0.0, 1.0, -1.0]))
    assert trade.deltas == pytest.approx([0.0, 0.3, 0.1])
    assert trade.turnover == pytest.approx(4 / 9)


def test_terminal_short_settlement_has_signed_cash_and_positive_profit():
    result = book_day(
        np.array([1.0, -1.0]), 1.0, np.array([0.0, -0.3]), np.array([False, True])
    )
    assert result.book_return == pytest.approx(0.3)
    assert result.nav == pytest.approx(1.3)
    assert result.positions == pytest.approx([1.0, 0.0])
    assert result.terminal_cash == pytest.approx(-0.7)
    assert result.nav - sum(result.positions) == pytest.approx(0.3)


@pytest.mark.parametrize("terminal_return", [-1.0, -0.3, 0.2])
def test_academic_leg_without_survivors_holds_cash_until_next_formation(terminal_return):
    terminal = academic_leg_day(
        np.array([1.0]), np.array([terminal_return]), np.array([True])
    )
    assert terminal.leg_return == terminal_return
    assert terminal.weights.tolist() == [0]
    following = academic_leg_day(terminal.weights, np.array([np.nan]), np.array([False]))
    assert following.leg_return == 0
    assert following.weights.tolist() == [0]


def test_ledger_nets_offsetting_formation_and_scaling_trades_once():
    trade = rebalance_book(np.array([0.6, 0.4, -1.0]), 1.0, np.array([0.6, 0.0, -0.6]))
    assert trade.positions == pytest.approx([0.6, 0.0, -0.6])
    assert trade.deltas == pytest.approx([0.0, -0.4, 0.4])
    assert trade.turnover == pytest.approx(0.8)


def test_drifted_positions_determine_rebalance_turnover_even_when_nav_is_flat():
    day = book_day(np.array([1.0, -1.0]), 1.0, np.array([0.1, 0.1]), np.zeros(2, bool))
    assert day.nav == 1
    assert day.positions == pytest.approx([1.1, -1.1])
    trade = rebalance_book(day.positions, day.nav, np.array([1.0, -1.0]))
    assert trade.deltas == pytest.approx([-0.1, 0.1])
    assert trade.turnover == pytest.approx(0.2)


@pytest.mark.parametrize("bad_return", [np.nan, np.inf, -np.inf])
def test_nonfinite_return_on_a_held_security_fails_even_when_terminal(bad_return):
    with pytest.raises(ValueError, match="held position requires a finite"):
        academic_leg_day(np.array([1.0]), np.array([bad_return]), np.array([True]))
    with pytest.raises(ValueError, match="held position requires a finite"):
        book_day(np.array([-1.0]), 1.0, np.array([bad_return]), np.array([True]))


def test_inactive_nonfinite_returns_are_ignored_and_inputs_are_unchanged():
    weights = np.array([1.0, 0.0, 0.0])
    returns = np.array([0.1, np.nan, np.inf])
    terminal = np.array([False, False, True])
    original_returns = returns.copy()
    leg = academic_leg_day(weights, returns, terminal)
    book = book_day(weights, 2.0, returns, terminal)
    trade = rebalance_book(weights, 2.0, np.array([0.0, 0.5, -0.5]))
    assert leg.leg_return == pytest.approx(0.1)
    assert book.book_return == pytest.approx(0.05)
    assert trade.turnover == 1.5
    assert weights.tolist() == [1.0, 0.0, 0.0]
    assert np.array_equal(returns, original_returns, equal_nan=True)
    assert terminal.tolist() == [False, False, True]


def test_invalid_stock_return_and_insolvent_nav_fail_loudly():
    with pytest.raises(ValueError, match="below -100%"):
        academic_leg_day(np.array([1.0]), np.array([-1.01]), np.array([True]))
    with pytest.raises(ValueError, match="remain positive"):
        book_day(np.array([-1.0]), 1.0, np.array([1.0]), np.array([False]))


@given(
    positions=st.lists(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False), min_size=4, max_size=4
    ),
    returns=st.lists(
        st.floats(-0.25, 0.25, allow_nan=False, allow_infinity=False), min_size=4, max_size=4
    ),
    terminal=st.lists(st.booleans(), min_size=4, max_size=4),
)
def test_book_cash_conservation_for_bounded_positions_and_returns(positions, returns, terminal):
    """For |x_i|≤2, |r_i|≤1/4 and NAV=10, cash changes only by terminal settlement."""
    held, observed, ended = np.array(positions), np.array(returns), np.array(terminal)
    initial_cash = 10.0 - sum(positions)
    result = book_day(held, 10.0, observed, ended)
    final_cash = result.nav - float(np.sum(result.positions))
    assert final_cash == pytest.approx(initial_cash + result.terminal_cash, abs=1e-12)
    assert result.positions[ended] == pytest.approx(np.zeros(np.count_nonzero(ended)))


@given(
    positions=st.lists(
        st.floats(-2, 2, allow_nan=False, allow_infinity=False), min_size=3, max_size=3
    ),
    target=st.lists(
        st.floats(-1, 1, allow_nan=False, allow_infinity=False), min_size=3, max_size=3
    ),
    scale=st.floats(0.1, 100, allow_nan=False, allow_infinity=False),
)
def test_ledger_turnover_is_invariant_to_account_currency_scale(positions, target, scale):
    """For positive account scale, trade notional and NAV scale together (Methodology)."""
    held, weights = np.array(positions), np.array(target)
    base = rebalance_book(held, 10.0, weights)
    scaled = rebalance_book(held * scale, 10.0 * scale, weights)
    assert scaled.turnover == pytest.approx(base.turnover)
    assert scaled.deltas == pytest.approx(base.deltas * scale)
