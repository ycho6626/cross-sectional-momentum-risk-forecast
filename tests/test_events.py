"""Hand-calculated terminal-value and SIC fixtures (Methodology)."""


from datetime import date

import pytest

from momenta.events import Action, classify_delisting, rollback_sic

DAY = date(2020, 1, 31)


SETTINGS = {"performance_return": -0.30, "unexplained_return": 0.0, "election_choice": "cash"}


def test_cash_and_stock_are_additive_and_override_generic_delisting():
    result = classify_delisting(
        20,
        [
            Action(DAY, "delisted"),
            Action(DAY, "acquisitioncash", 8),
            Action(DAY, "acquisitionstock", 0.5, "BUYER"),
        ],
        {"BUYER": 30},
        **SETTINGS,
    )
    # $8 + 0.5 × $30 = $23; $23 / $20 − 1 = 15%.
    assert result.event_class == "acquisition_with_terms"
    assert result.terminal_value == 23
    assert result.terminal_return == pytest.approx(0.15)
    assert result.exception is None


def test_cash_election_is_not_added_to_stock_alternative():
    result = classify_delisting(
        20,
        [
            Action(DAY, "acquisitionelectcash", 24),
            Action(DAY, "acquisitionelectstock", 1, "BUYER"),
            Action(DAY, "acquisitionby", contraticker="BUYER"),
        ],
        {"BUYER": 30},
        **SETTINGS,
    )
    assert result.event_class == "election"
    assert result.election_choice == "cash"
    assert result.terminal_value == 24
    assert result.terminal_return == pytest.approx(0.20)


def test_stock_election_when_cash_is_absent():
    result = classify_delisting(
        20, [Action(DAY, "acquisitionelectstock", 0.5, "BUYER")], {"BUYER": 30}, **SETTINGS
    )
    assert result.election_choice == "stock"
    assert result.terminal_value == 15
    assert result.terminal_return == -0.25


def test_terms_take_precedence_over_election_and_bankruptcy():
    result = classify_delisting(
        20,
        [
            Action(DAY, "acquisitioncash", 22),
            Action(DAY, "acquisitionelectcash", 25),
            Action(DAY, "bankruptcyliquidation"),
        ],
        {},
        **SETTINGS,
    )
    assert result.event_class == "acquisition_with_terms"
    assert result.terminal_value == 22
    assert result.election_choice is None


def test_missing_acquirer_price_zeroes_entire_return_and_records_exception():
    result = classify_delisting(
        20,
        [
            Action(DAY, "acquisitioncash", 8),
            Action(DAY, "acquisitionstock", 0.5, "BUYER"),
            Action(DAY, "delisted"),
        ],
        {},
        **SETTINGS,
    )
    assert result.event_class == "acquisition_with_terms"
    assert result.terminal_value == 20
    assert result.terminal_return == 0
    assert result.exception == "missing_acquirer_price"


def test_invalid_cash_election_does_not_silently_choose_stock():
    result = classify_delisting(
        20,
        [
            Action(DAY, "acquisitionelectcash"),
            Action(DAY, "acquisitionelectstock", 1, "BUYER"),
        ],
        {"BUYER": 30},
        **SETTINGS,
    )
    assert result.election_choice == "cash"
    assert result.terminal_return == 0
    assert result.exception == "missing_or_invalid_consideration"


@pytest.mark.parametrize("action", ["acquisitionby", "mergerto"])
def test_acquisition_without_terms_precedes_voluntary_and_generic_delisting(action):
    result = classify_delisting(
        20,
        [Action(DAY, action), Action(DAY, "voluntarydelisting"), Action(DAY, "delisted")],
        {},
        **SETTINGS,
    )
    assert result.event_class == "acquisition_without_terms"
    assert result.terminal_return == 0
    assert result.exception == "acquisition_without_terms"


def test_voluntary_delisting_precedes_performance():
    result = classify_delisting(
        20, [Action(DAY, "voluntarydelisting"), Action(DAY, "delisted")], {}, **SETTINGS
    )
    assert result.event_class == "voluntary_delisting"
    assert result.terminal_value == 20
    assert result.exception is None


@pytest.mark.parametrize("action", ["bankruptcyliquidation", "regulatorydelisting", "delisted"])
def test_performance_delisting_return_and_proceeds(action):
    result = classify_delisting(20, [Action(DAY, action)], {}, **SETTINGS)
    assert result.event_class == "performance_delisting"
    assert result.terminal_value == 14
    assert result.terminal_return == -0.30


def test_unrelated_action_does_not_explain_the_end():
    result = classify_delisting(20, [Action(DAY, "dividend", 0.5)], {}, **SETTINGS)
    assert result.event_class == "unexplained_end"
    assert result.terminal_value == 20
    assert result.exception == "no_matching_terminal_action"


def test_configured_sensitivity_is_applied_to_performance_returns():
    result = classify_delisting(
        20,
        [Action(DAY, "delisted")],
        {},
        **{**SETTINGS, "performance_return": -1.0},
    )
    assert result.terminal_value == 0
    assert result.terminal_return == -1


def test_sic_rolls_back_all_changes_latest_first_with_exact_date_boundary():
    first, second = date(2010, 5, 1), date(2015, 5, 1)
    actions = [
        Action(first, "sicchangefrom", 6770),
        Action(first, "sicchangeto", 2834),
        Action(second, "sicchangefrom", 2834),
        Action(second, "sicchangeto", 3571),
    ]
    before = rollback_sic(3571, actions, date(2009, 12, 31))
    assert before.siccode == 6770
    assert before.changes_rolled_back == 2
    assert before.issues == ()
    on_first = rollback_sic(3571, actions[::-1], first)
    assert on_first.siccode == 2834
    assert on_first.changes_rolled_back == 1
    assert rollback_sic(3571, actions, second).siccode == 3571


def test_missing_sic_to_row_is_audited_but_old_code_is_known():
    result = rollback_sic(3571, [Action(DAY, "sicchangefrom", 6770)], date(2019, 12, 31))
    assert result.siccode == 6770
    assert [issue.issue for issue in result.issues] == ["missing_sicchangeto"]


def test_missing_or_ambiguous_from_does_not_leak_current_sic_backwards():
    result = rollback_sic(3571, [Action(DAY, "sicchangeto", 3571)], date(2019, 12, 31))
    assert result.siccode is None
    assert result.issues[0].issue == "missing_sicchangefrom"
    result = rollback_sic(
        3571,
        [
            Action(DAY, "sicchangefrom", 6770),
            Action(DAY, "sicchangefrom", 2834),
            Action(DAY, "sicchangeto", 3571),
        ],
        date(2019, 12, 31),
    )
    assert result.siccode is None
    assert result.issues[0].issue == "ambiguous_sicchangefrom"


def test_sic_chain_mismatch_is_reported_and_older_from_still_used():
    result = rollback_sic(
        9999,
        [Action(DAY, "sicchangefrom", 6770), Action(DAY, "sicchangeto", 3571)],
        date(2019, 12, 31),
    )
    assert result.siccode == 6770
    assert result.issues[0].issue == "sic_chain_mismatch"


def test_older_sic_can_be_recovered_after_newer_missing_pair():
    result = rollback_sic(
        3571,
        [
            Action(DAY, "sicchangeto", 3571),
            Action(date(2010, 1, 1), "sicchangefrom", 6770),
            Action(date(2010, 1, 1), "sicchangeto", 2834),
        ],
        date(2009, 12, 31),
    )
    assert result.siccode == 6770
    assert result.issues[0].issue == "missing_sicchangefrom"


def test_future_classification_change_with_complete_pair_preserves_past_sic():
    cutoff = date(2019, 12, 31)
    original = rollback_sic(6770, [], cutoff)
    updated = rollback_sic(
        3571,
        [Action(DAY, "sicchangefrom", 6770), Action(DAY, "sicchangeto", 3571)],
        cutoff,
    )
    assert updated.siccode == original.siccode == 6770
