"""Point-in-time classifications and terminal values from Sharadar actions."""


from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite


@dataclass(frozen=True)
class Action:
    """One security's action; field meanings follow Sharadar's actions dictionary."""

    date: date
    action: str
    value: float | None = None
    contraticker: str | None = None


@dataclass(frozen=True)
class DelistingResult:
    """Terminal per-share proceeds and the diagnostic required by Methodology."""

    event_class: str
    terminal_return: float
    terminal_value: float
    exception: str | None = None
    election_choice: str | None = None


def _imputed(
    event_class: str,
    last_close: float,
    terminal_return: float,
    exception: str | None = None,
    election_choice: str | None = None,
) -> DelistingResult:
    """Return V = last_close × (1 + imputed return), per Methodology."""
    return DelistingResult(
        event_class, terminal_return, last_close * (1 + terminal_return), exception, election_choice
    )


def _consideration(
    actions: Sequence[Action],
    acquirer_prices: Mapping[str, float | None],
) -> tuple[float | None, str | None]:
    """Compute Σ cash + Σ ratio × acquirer close on credit date (Methodology)."""
    value = 0.0
    for action in actions:
        if action.value is None or not isfinite(action.value) or action.value < 0:
            return None, "missing_or_invalid_consideration"
        if action.action.endswith("cash"):
            value += action.value
        else:
            price = acquirer_prices.get(action.contraticker or "")
            if price is None or not isfinite(price) or price <= 0:
                return None, "missing_acquirer_price"
            value += action.value * price
    return value, None


def classify_delisting(
    last_closeunadj: float,
    actions: Sequence[Action],
    acquirer_prices: Mapping[str, float | None],
    *,
    performance_return: float,
    unexplained_return: float,
    election_choice: str,
) -> DelistingResult:
    """Apply Methodology precedence, then r_terminal = V / last_closeunadj − 1.

    The caller matches actions within the configured trading-day window of the
    last SEP date and supplies acquirer unadjusted closes on the first trading
    day afterward. Keys are the actions' exact contra ticker strings, resolved
    through the SEP security master by the caller. Election alternatives are
    never added. Missing consideration or acquirer prices yields the specified
    0% exception return; it does not fall through to generic delisting.
    """
    if not isfinite(last_closeunadj) or last_closeunadj <= 0:
        raise ValueError("The last unadjusted close must be positive and finite")
    if any(not isfinite(r) or r < -1 for r in (performance_return, unexplained_return)):
        raise ValueError("Imputed terminal returns must be finite and at least -1")
    if election_choice != "cash":
        raise ValueError('The event convention requires choosing cash when available')

    grouped: dict[str, list[Action]] = defaultdict(list)
    for action in actions:
        grouped[action.action].append(action)

    terms = grouped["acquisitioncash"] + grouped["acquisitionstock"]
    choice = None
    if terms:
        event_class = "acquisition_with_terms"
    elif grouped["acquisitionelectcash"] or grouped["acquisitionelectstock"]:
        event_class = "election"
        choice = "cash" if grouped["acquisitionelectcash"] else "stock"
        terms = grouped[f"acquisitionelect{choice}"]
    elif grouped["acquisitionby"] or grouped["mergerto"]:
        return _imputed(
            "acquisition_without_terms", last_closeunadj, 0.0, "acquisition_without_terms"
        )
    elif grouped["voluntarydelisting"]:
        return _imputed("voluntary_delisting", last_closeunadj, 0.0)
    elif any(grouped[a] for a in ("bankruptcyliquidation", "regulatorydelisting", "delisted")):
        return _imputed("performance_delisting", last_closeunadj, performance_return)
    else:
        return _imputed(
            "unexplained_end", last_closeunadj, unexplained_return, "no_matching_terminal_action"
        )

    terminal_value, exception = _consideration(terms, acquirer_prices)
    if terminal_value is None:
        return _imputed(event_class, last_closeunadj, 0.0, exception, choice)
    return DelistingResult(
        event_class, terminal_value / last_closeunadj - 1, terminal_value, None, choice
    )


@dataclass(frozen=True)
class SicIssue:
    """A missing, ambiguous, or inconsistent SIC action pair for the audit."""

    date: date | None
    issue: str


@dataclass(frozen=True)
class SicResult:
    """SIC at the requested date, counted rollbacks, and pair audit issues."""

    siccode: int | None
    changes_rolled_back: int
    issues: tuple[SicIssue, ...]


def _sic_value(value: float | int | None) -> int | None:
    """Accept integer SIC values in (0, 10000), the code domain of Methodology."""
    if value is None or not isfinite(value) or value <= 0 or value >= 10000:
        return None
    return int(value) if int(value) == value else None


def rollback_sic(
    current_sic: float | int | None,
    actions: Sequence[Action],
    as_of: date,
) -> SicResult:
    """Reverse every change after as_of: sic ← sicchangefrom, per Methodology.

    Work from latest to earliest. An unambiguous from value remains usable when
    the to row is missing or contradicts the newer code, with an audit issue.
    Missing or ambiguous from values make SIC unknown until an older usable
    change establishes the earlier code. Actions on as_of are already effective.
    """
    sic = _sic_value(current_sic)
    issues = [] if sic is not None else [SicIssue(None, "missing_current_sic")]
    by_date: dict[date, list[Action]] = defaultdict(list)
    for action in actions:
        if action.action in ("sicchangefrom", "sicchangeto") and action.date > as_of:
            by_date[action.date].append(action)

    for change_date in sorted(by_date, reverse=True):
        rows = by_date[change_date]
        from_rows = [a for a in rows if a.action == "sicchangefrom"]
        to_rows = [a for a in rows if a.action == "sicchangeto"]
        before = _sic_value(from_rows[0].value) if len(from_rows) == 1 else None
        after = _sic_value(to_rows[0].value) if len(to_rows) == 1 else None
        for side, side_rows, code in (("from", from_rows, before), ("to", to_rows, after)):
            if not side_rows:
                issues.append(SicIssue(change_date, f"missing_sicchange{side}"))
            elif len(side_rows) > 1:
                issues.append(SicIssue(change_date, f"ambiguous_sicchange{side}"))
            elif code is None:
                issues.append(SicIssue(change_date, f"invalid_sicchange{side}"))
        if sic is not None and after is not None and sic != after:
            issues.append(SicIssue(change_date, "sic_chain_mismatch"))
        sic = before

    return SicResult(sic, len(by_date), tuple(issues))
