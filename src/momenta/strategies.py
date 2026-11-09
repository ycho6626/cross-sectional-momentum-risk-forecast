"""Lagged overlays, exact ledger costs and training-only DM transfer (Methodology)."""


from __future__ import annotations

import calendar as month_calendar
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class StrategyResult:
    """Dated account values, rebalance totals and exact per-name trades (Methodology)."""

    daily: pl.DataFrame
    rebalances: pl.DataFrame
    ledger: pl.DataFrame


@dataclass(frozen=True)
class DMMeanFit:
    """French next-month mean fit and its complete outcome cutoff (Methodology)."""

    intercept: float
    slope: float
    fit_date: date
    train_first: date
    train_last: date
    target_end: date
    n: int


@dataclass(frozen=True)
class DMCalibration:
    """A clipped gross-book scale, achieved volatility and dated audit (Methodology)."""

    scale: float
    annual_volatility: float
    attained: bool
    audit: dict[str, Any]


def volatility_multiples(
    variance: NDArray[np.float64], annual_target: float, horizon: int,
    trading_days: int, max_multiple: float,
) -> NDArray[np.float64]:
    """Return min(max_multiple, annual_target·sqrt(horizon/days/variance)) (Methodology).

    A zero variance reaches the exposure cap. Negative/nonfinite variances
    are errors; missing forecasts must be handled as unavailable upstream.
    """
    values = np.asarray(variance, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Variance forecasts must be a finite nonnegative vector")
    if annual_target <= 0 or horizon < 1 or trading_days < 1 or max_multiple <= 0:
        raise ValueError("Invalid volatility-target parameters")
    result = np.full(len(values), float(max_multiple))
    positive = values > 0
    result[positive] = np.minimum(
        max_multiple, annual_target * np.sqrt(horizon / trading_days / values[positive]),
    )
    return result


def _one_book(frame: pl.DataFrame) -> dict[str, Any]:
    """Prevent combining separately denominated base books (Methodology)."""
    keys = [key for key in ("series", "construction") if key in frame.columns]
    if keys and frame.select(keys).unique().height != 1:
        raise ValueError("Pass exactly one series and construction to the strategy engine")
    return frame.select(keys).row(0, named=True) if keys else {}


def simulate_strategy(
    base_daily: pl.DataFrame,
    base_ledger: pl.DataFrame,
    overlays: pl.DataFrame,
    *,
    cost_bp: float,
    initial_nav: float,
    last_date: date,
    record_ledger: bool = True,
) -> StrategyResult:
    """Replay x_strategy=k·x_base, net trades once, and charge fees from cash (Methodology).

    k is constant between executions. Daily profit is k times base dollar
    profit, including terminal credits. At execution k_new=w·NAV_pre/base_NAV.
    Per-name trade is k_new·base_post−k_old·base_pre. Fees use pre-cost NAV,
    and net return is ending NAV/previous NAV−1. The initial book is cash.
    Targets must include every base rebalance until last_date and use only
    models fitted by their origins. Run cost_bp=0 separately for the gross book.
    """
    if not np.isfinite(cost_bp) or cost_bp < 0 or not np.isfinite(initial_nav) or initial_nav <= 0:
        raise ValueError("Costs must be nonnegative and initial NAV positive")
    labels = _one_book(base_daily)
    ledger_labels = _one_book(base_ledger)
    if labels and ledger_labels and labels != ledger_labels:
        raise ValueError("Base returns and ledger must identify the same book")
    overlays = overlays.filter(pl.col("execution_date") <= last_date).sort("execution_date")
    if overlays.is_empty():
        raise ValueError("At least one executable overlay is required")
    if overlays["execution_date"].n_unique() != overlays.height:
        raise ValueError("Duplicate overlay execution date")
    for row in overlays.iter_rows(named=True):
        if not np.isfinite(row["multiple"]) or row["execution_date"] <= row["origin"]:
            raise ValueError("Overlays require finite multiples and execution after origin")
        if row.get("fit_date") is not None and row["fit_date"] > row["origin"]:
            raise ValueError("A model cannot be fitted after the decision it serves")
    first_date = overlays["execution_date"].min()
    base = base_daily.filter(pl.col("date").is_between(first_date, last_date)).sort("date")
    if base.is_empty() or base["date"].n_unique() != base.height:
        raise ValueError("Base daily dates must be present and unique")
    ledger = base_ledger.filter(pl.col("date").is_between(first_date, last_date))
    expected = set(ledger["date"].to_list())
    decisions = {row["execution_date"]: row for row in overlays.iter_rows(named=True)}
    if expected != set(decisions) or not expected.issubset(set(base["date"].to_list())):
        raise ValueError("Every base rebalance requires exactly one dated overlay")
    trade_groups = {group["date"][0]: group for group in ledger.partition_by("date")}
    daily_rows, rebalance_rows, ledger_rows = [], [], []
    nav, ratio = float(initial_nav), 0.0
    for row in base.iter_rows(named=True):
        day, base_nav, base_return = row["date"], row["nav"], row["daily_return"]
        if not np.isfinite(base_nav) or base_nav <= 0 or not np.isfinite(base_return):
            raise ValueError("Base NAV and returns must be finite; NAV must be positive")
        if base_return <= -1:
            raise ValueError("A positive-NAV base book cannot return -100%")
        previous_nav = nav
        profit = ratio * base_nav * base_return / (1 + base_return)
        nav += profit
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("Scaled strategy NAV is nonpositive after market returns")
        terminal_cash = ratio * row["terminal_cash"]
        cost = turnover = 0.0
        if day in decisions:
            decision, group = decisions[day], trade_groups[day]
            if "formation" in group.columns and group["formation"].unique().to_list() != [
                decision["origin"]
            ]:
                raise ValueError("Overlay origin does not match the frozen base formation")
            before = ratio * group["pretrade"].to_numpy()
            new_ratio = decision["multiple"] * nav / base_nav
            after = new_ratio * group["posttrade"].to_numpy()
            trades = after - before
            fees = cost_bp / 10_000 * np.abs(trades)
            turnover = float(np.abs(trades).sum() / nav)
            cost = float(fees.sum())
            for index, permaticker in enumerate(group["permaticker"].to_list()):
                if record_ledger and (before[index] != 0 or after[index] != 0):
                    ledger_rows.append({
                        **labels, "date": day, "origin": decision["origin"],
                        "permaticker": permaticker, "nav": nav,
                        "pretrade": float(before[index]), "posttrade": float(after[index]),
                        "trade": float(trades[index]), "cost": float(fees[index]),
                        "turnover": float(abs(trades[index]) / nav),
                    })
            rebalance_rows.append({
                **labels, "date": day, "origin": decision["origin"],
                "fit_date": decision.get("fit_date"), "multiple": decision["multiple"],
                "nav_before_cost": nav, "turnover": turnover, "cost": cost,
                "cost_fraction": cost / nav,
            })
            nav -= cost
            ratio = new_ratio
            if not np.isfinite(nav) or nav <= 0:
                raise ValueError("Scaled strategy NAV is nonpositive after trading costs")
        daily_rows.append({
            **labels, "date": day, "daily_return": nav / previous_nav - 1,
            "return_before_cost": profit / previous_nav, "nav": nav,
            "cash": nav - ratio * (base_nav - row["cash"]),
            "gross_exposure": abs(ratio) * base_nav * row["gross_exposure"] / nav,
            "terminal_cash": terminal_cash, "cost": cost, "turnover": turnover,
            "base_position_ratio": ratio,
        })
    ledger_schema = {
        **{key: pl.String for key in labels}, "date": pl.Date, "origin": pl.Date,
        "permaticker": pl.Int64, "nav": pl.Float64, "pretrade": pl.Float64,
        "posttrade": pl.Float64, "trade": pl.Float64, "cost": pl.Float64,
        "turnover": pl.Float64,
    }
    return StrategyResult(
        pl.DataFrame(daily_rows), pl.DataFrame(rebalance_rows).with_columns(
            pl.col("fit_date").cast(pl.Date),
        ), pl.DataFrame(ledger_rows, schema=ledger_schema),
    )


def _next_month(day: date) -> date:
    """Return the next calendar month's last date for the next-month mean outcome."""
    year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
    return date(year, month, month_calendar.monthrange(year, month)[1])


def build_dm_pairs(
    ff3_daily: pl.DataFrame, french_deciles_monthly: pl.DataFrame, settings: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Pair French bear×variance with next-month published D10−D1 (Methodology).

    Bear uses compounded total market returns over 504 source sessions; variance
    uses 21/126 times squared excess-market returns. The original daily source
    calendar, including historical Saturdays, is retained. Incomplete warmup
    windows are omitted and counted, never padded.
    """
    factors = ff3_daily.sort("date")
    if factors["date"].n_unique() != factors.height:
        raise ValueError("French daily dates must be unique")
    days = factors["date"].to_list()
    market, risk_free = factors["mkt_rf"].to_numpy(), factors["rf"].to_numpy()
    outcomes = {
        (row["date"].year, row["date"].month): row
        for row in french_deciles_monthly.iter_rows(named=True)
    }
    bear_days = int(settings["features"]["bear_days"])
    variance_days = int(settings["features"]["market_rv_days"])
    horizon = int(settings["targets"]["horizon"])
    nominal_start = date.fromisoformat(settings["strategy"]["dm_first_month"])
    last_origin = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    rows, warmup_omissions, missing_windows = [], 0, 0
    for index, day in enumerate(days):
        if day > last_origin:
            break
        if day < nominal_start or (index + 1 < len(days) and days[index + 1].month == day.month):
            continue
        target_end = _next_month(day)
        outcome = outcomes.get((target_end.year, target_end.month))
        if index + 1 < max(bear_days, variance_days):
            warmup_omissions += 1
            continue
        bear_window = market[index + 1 - bear_days:index + 1] + risk_free[
            index + 1 - bear_days:index + 1
        ]
        variance_window = market[index + 1 - variance_days:index + 1]
        if not np.isfinite(bear_window).all() or not np.isfinite(variance_window).all():
            missing_windows += 1
            continue
        if outcome is None or outcome["d10"] is None or outcome["d1"] is None:
            continue
        bear = float(np.prod(1 + bear_window) < 1)
        variance = horizon / variance_days * float(variance_window @ variance_window)
        rows.append({"origin": day, "target_end": target_end, "bear_x_var": bear * variance,
                     "next_month_return": outcome["d10"] - outcome["d1"]})
    pairs = pl.DataFrame(rows, schema={"origin": pl.Date, "target_end": pl.Date,
                                      "bear_x_var": pl.Float64, "next_month_return": pl.Float64})
    return pairs, {"nominal_first_month": nominal_start.isoformat(),
                   "first_supported_origin": str(pairs["origin"].min()) if pairs.height else None,
                   "warmup_omissions": warmup_omissions, "missing_windows": missing_windows,
                   "source_sessions": len(days)}


def fit_dm_mean(pairs: pl.DataFrame, refit: date) -> DMMeanFit:
    """OLS next-month French WML=a+b·bear_x_var using outcome ends < refit (Methodology)."""
    training = pairs.filter(pl.col("target_end") < refit, pl.col("origin") < refit).sort("origin")
    if training.height < 2:
        raise ValueError("DM mean regression needs at least two completed training months")
    x, y = training["bear_x_var"].to_numpy(), training["next_month_return"].to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("DM mean training values must be finite")
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    return DMMeanFit(float(coefficients[0]), float(coefficients[1]), refit,
                     training["origin"].min(), training["origin"].max(),
                     training["target_end"].max(), training.height)


def dm_multiples(
    bear_x_var: NDArray[np.float64], variance: NDArray[np.float64],
    fit: DMMeanFit, scale: float, max_multiple: float,
) -> NDArray[np.float64]:
    """Return clip(c·(a+b·bear_x_var)/variance, −max_multiple, max_multiple) (Methodology)."""
    state, risk = np.asarray(bear_x_var, dtype=float), np.asarray(variance, dtype=float)
    if state.ndim != 1 or risk.shape != state.shape or not np.isfinite(state).all():
        raise ValueError("DM states and variance forecasts must be aligned finite vectors")
    if not np.isfinite(risk).all() or np.any(risk <= 0) or not np.isfinite(scale) or scale < 0:
        raise ValueError("DM variance must be positive and scale nonnegative")
    return np.clip(scale * (fit.intercept + fit.slope * state) / risk,
                   -max_multiple, max_multiple)


def gross_scaled_monthly(
    execution_nav: NDArray[np.float64], month_end_nav: NDArray[np.float64],
    multiples: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact calendar-month gross returns from base NAV at execution and month-end (Methodology).

    Before each execution the previous month's book earns its remaining daily
    P&L; afterwards the new multiple applies. The first execution starts from
    cash. This is algebraically identical to the daily ledger-ratio replay.
    """
    executions, ends, weights = (np.asarray(x, dtype=float)
                                 for x in (execution_nav, month_end_nav, multiples))
    if executions.ndim != 1 or ends.shape != executions.shape or weights.shape != ends.shape:
        raise ValueError("Monthly base NAV and multiples must be aligned vectors")
    if not all(np.isfinite(x).all() for x in (executions, ends, weights)) or (
        np.any(executions <= 0) or np.any(ends <= 0)
    ):
        raise ValueError("Monthly base NAV must be positive and all inputs finite")
    after = 1 + weights * (ends / executions - 1)
    before = np.ones(len(weights))
    if len(weights) > 1:
        old_at_execution = 1 + weights[:-1] * (executions[1:] / executions[:-1] - 1)
        if np.any(old_at_execution <= 0) or np.any(after[:-1] <= 0):
            raise ValueError("Scaled gross NAV becomes nonpositive before execution")
        before[1:] = old_at_execution / after[:-1]
    if np.any(after <= 0):
        raise ValueError("Scaled gross NAV becomes nonpositive by month-end")
    return before * after - 1


def calibrate_scale(
    scores: NDArray[np.float64], execution_nav: NDArray[np.float64],
    month_end_nav: NDArray[np.float64], sample: NDArray[np.bool_],
    settings: dict[str, Any],
    *,
    interval_minimum: NDArray[np.float64] | None = None,
    interval_maximum: NDArray[np.float64] | None = None,
) -> DMCalibration:
    """Find the first bracketed c attaining target gross volatility under clipping (Methodology).

    Search an increasing configured geometric grid plus every clipping
    breakpoint, then bisect its first crossing. Volatility is not assumed
    globally monotone. This is the first bracketed crossing, not a claim about
    unsampled local peaks. If unattained, retain the highest-volatility finite
    evaluated scale and report the failure explicitly.
    """
    score = np.asarray(scores, dtype=float)
    selected = np.asarray(sample)
    if score.ndim != 1 or selected.shape != score.shape or selected.dtype != np.bool_:
        raise ValueError("Calibration scores and boolean sample mask must align")
    if not np.isfinite(score).all() or selected.sum() < 2:
        raise ValueError("Calibration needs finite scores and at least two training months")
    if (interval_minimum is None) != (interval_maximum is None):
        raise ValueError("Both interval NAV extrema must be supplied together")
    if interval_minimum is not None:
        low, high = np.asarray(interval_minimum), np.asarray(interval_maximum)
        if low.shape != score.shape or high.shape != score.shape or (
            not np.isfinite(low).all() or not np.isfinite(high).all() or
            np.any(low <= 0) or np.any(high < low)
        ):
            raise ValueError("Interval NAV extrema must be positive and aligned")
    config = settings["strategy"]
    target, cap = float(config["vol_target_annual"]), float(config["max_multiple"])
    maximum = float(config["calibration_max_scale"])
    minimum = float(config["calibration_min_scale"])
    tolerance = float(config["calibration_tolerance"])
    points = int(config["calibration_grid_points"])
    iterations = int(config["calibration_max_iterations"])
    annualization = int(settings["replication"]["months_per_year"])
    if not (target > 0 and cap > 0 and maximum >= minimum > 0 and tolerance > 0):
        raise ValueError("Invalid scale calibration bounds")
    if points < 2 or iterations < 1 or annualization < 1:
        raise ValueError("Invalid scale calibration grid or annualization")
    nonzero = np.abs(score[score != 0])
    if not len(nonzero):
        return DMCalibration(0.0, 0.0, False, {"reason": "zero_mean_scores", "evaluations": 1})
    knots = cap / nonzero
    upper = min(maximum, float(knots.max()))
    grid = np.unique(np.concatenate([
        [0.0, upper], np.geomspace(min(minimum, upper), upper, points), knots[knots <= upper],
    ]))
    evaluations = 0

    def annual_volatility(scale: float) -> float:
        nonlocal evaluations
        evaluations += 1
        weights = np.clip(scale * score, -cap, cap)
        if interval_minimum is not None and (
            np.any(1 + weights * (low - 1) <= 0) or np.any(1 + weights * (high - 1) <= 0)
        ):
            raise ValueError("Scaled gross NAV becomes nonpositive inside a holding month")
        returns = gross_scaled_monthly(execution_nav, month_end_nav, weights)
        return float(np.std(returns[selected], ddof=1) * np.sqrt(annualization))

    left, best_scale, best_volatility = 0.0, 0.0, 0.0
    invalid_paths = 0
    for right in grid[1:]:
        try:
            volatility = annual_volatility(float(right))
        except ValueError:
            invalid_paths += 1
            # Do not report an insolvent path as an attainable volatility.
            break
        if volatility > best_volatility:
            best_scale, best_volatility = float(right), volatility
        if volatility >= target:
            initial_left, initial_right = left, float(right)
            for _ in range(iterations):
                middle = (left + right) / 2
                value = annual_volatility(float(middle))
                if abs(value - target) <= tolerance:
                    return DMCalibration(float(middle), value, True, {
                        "reason": "first_bracketed_crossing", "evaluations": evaluations,
                        "bracket_low": initial_left, "bracket_high": initial_right,
                        "search_upper": upper, "invalid_paths": invalid_paths,
                    })
                if value >= target:
                    right = middle
                else:
                    left = float(middle)
            value = annual_volatility(float(right))
            return DMCalibration(float(right), value, abs(value - target) <= tolerance, {
                "reason": "first_bracketed_crossing", "evaluations": evaluations,
                "bracket_low": initial_left, "bracket_high": initial_right,
                "search_upper": upper, "invalid_paths": invalid_paths,
            })
        left = float(right)
    return DMCalibration(best_scale, best_volatility, False, {
        "reason": "nonpositive_nav_before_target" if invalid_paths else "target_not_attained",
        "evaluations": evaluations, "search_upper": upper, "invalid_paths": invalid_paths,
        "fully_saturated": bool(float(knots.max()) <= maximum),
    })


def calibrate_dm(
    base_daily: pl.DataFrame,
    training_variances: pl.DataFrame,
    bear_features: pl.DataFrame,
    mean_fit: DMMeanFit,
    settings: dict[str, Any],
) -> DMCalibration:
    """Calibrate c on complete pre-refit Sharadar months and selected L1 fitted risk (Methodology).

    Variance rows are the selected L1 fit's in-sample monthly-origin variance
    predictions, including its training residual correction. Only rows whose
    target_end < refit are eligible. Gross accounting runs from cash through
    the last complete holding month before refit. Missing training decisions
    hold cash and their holding months are excluded from the volatility sample.
    No DM parameter enters penalty selection; call after selecting the L1 fit.
    """
    _one_book(base_daily)
    _one_book(training_variances)
    refit = mean_fit.fit_date
    if mean_fit.target_end >= refit or mean_fit.train_last >= refit:
        raise ValueError("DM mean parameters must use outcomes ending before their refit")
    training = training_variances.filter(pl.col("origin") < refit, pl.col("target_end") < refit)
    if "refit" in training.columns:
        training = training.filter(pl.col("refit") == refit)
    if training["origin"].n_unique() != training.height:
        raise ValueError("DM calibration requires unique monthly training origins")
    training = training.join(bear_features.select("origin", "bear_x_var"),
                             on="origin", how="left", validate="1:1").sort("origin")
    if training.height < 2:
        raise ValueError("DM calibration needs at least two completed training months")
    risk, state = training["variance_prediction"].to_numpy(), training["bear_x_var"].to_numpy()
    if not np.isfinite(risk).all() or np.any(risk <= 0) or not np.isfinite(state).all():
        raise ValueError("Training-only DM variance and state values must exist and be finite")
    scores = (mean_fit.intercept + mean_fit.slope * state) / risk
    result = calibrate_training_scores(
        base_daily, training.with_columns(pl.Series("score", scores)), refit, settings,
    )
    return DMCalibration(
        result.scale, result.annual_volatility, result.attained,
        {**result.audit, "mean_target_end": str(mean_fit.target_end)},
    )


def calibrate_training_scores(
    base_daily: pl.DataFrame, training_scores: pl.DataFrame, refit: date,
    settings: dict[str, Any],
) -> DMCalibration:
    """Calibrate clipped mean/variance scores on completed training holding months (Methodology).

    Both DM and BDP supply origin, target_end and score. Outcomes and holding
    months must end before refit. Initial holdings are cash; missing decisions
    also target cash, and their holding months are excluded from the sample.
    """
    _one_book(base_daily)
    training = training_scores.filter(
        pl.col("origin") < refit, pl.col("target_end") < refit,
    ).sort("origin")
    if training["origin"].n_unique() != training.height:
        raise ValueError("Scale calibration requires unique monthly training origins")
    if not np.isfinite(training["score"].to_numpy()).all():
        raise ValueError("Training-only mean/variance scores must be finite")
    decisions: dict[date, tuple[date, float]] = {}
    for origin, score in training.select("origin", "score").iter_rows():
        end = _next_month(origin)
        if end < refit:
            if end in decisions:
                raise ValueError("DM calibration origins must be monthly")
            decisions[end] = origin, float(score)
    if len(decisions) < 2:
        raise ValueError("DM calibration needs at least two complete holding months before refit")
    first_end, final_end = min(decisions), max(decisions)
    base = base_daily.filter(pl.col("date") <= final_end).sort("date")
    if base["date"].n_unique() != base.height:
        raise ValueError("Base book dates must be unique for calibration")
    months = base.with_columns(pl.col("date").dt.month_end().alias("month_end")).partition_by(
        "month_end", maintain_order=True,
    )
    lag = int(settings["portfolios"]["rebalance_lag_days"])
    if lag < 1:
        raise ValueError("Execution lag must be positive")
    scores, executions, ends, sample = [], [], [], []
    interval_minimum, interval_maximum = [], []
    for month_index, month in enumerate(months):
        end = month["month_end"][0]
        if end < first_end or end > final_end:
            continue
        if month.height < lag:
            raise ValueError("Incomplete base holding month")
        if end in decisions:
            origin = decisions[end][0]
            origin_month = base.filter(
                pl.col("date").dt.year() == origin.year,
                pl.col("date").dt.month() == origin.month,
            )
            if origin_month.is_empty() or origin_month["date"][-1] != origin:
                raise ValueError("Training origin must be an actual base-book month-end")
        scores.append(decisions[end][1] if end in decisions else 0.0)
        executions.append(month["nav"][lag - 1])
        ends.append(month["nav"][-1])
        sample.append(end in decisions)
        interval_nav = month["nav"].to_numpy()[lag - 1:]
        if month_index + 1 < len(months):
            interval_nav = np.r_[interval_nav, months[month_index + 1]["nav"].to_numpy()[:lag]]
        interval_minimum.append(float(interval_nav.min() / executions[-1]))
        interval_maximum.append(float(interval_nav.max() / executions[-1]))
    if sum(sample) != len(decisions):
        raise ValueError("Base book does not cover every completed DM calibration month")
    calibration = calibrate_scale(np.array(scores), np.array(executions), np.array(ends),
                                  np.array(sample, dtype=bool), settings,
                                  interval_minimum=np.array(interval_minimum),
                                  interval_maximum=np.array(interval_maximum))
    used_training = training.filter(pl.col("origin").is_in([row[0] for row in decisions.values()]))
    audit = {
        **calibration.audit, "fit_date": str(refit),
        "training_origin_first": str(used_training["origin"].min()),
        "training_origin_last": str(used_training["origin"].max()),
        "training_target_end": str(used_training["target_end"].max()),
        "holding_month_first": str(first_end), "holding_month_last": str(final_end),
        "training_months": int(sum(sample)), "cash_decision_months": int(len(sample) - sum(sample)),
        "cost_bp": 0.0, "return_convention": "exact_calendar_month_gross_book",
    }
    return DMCalibration(
        calibration.scale, calibration.annual_volatility, calibration.attained, audit,
    )


def break_even_cost(
    base_daily: pl.DataFrame,
    base_ledger: pl.DataFrame,
    variant_overlays: pl.DataFrame,
    bsc_overlays: pl.DataFrame,
    settings: dict[str, Any],
    *,
    last_date: date,
    initial_nav: float,
) -> dict[str, Any]:
    """Replay costs to locate annual arithmetic net-mean equality with BSC (Methodology).

    All strategy/model parameters stay frozen. Search only the configured bp
    grid and bisect its sign-changing brackets. Report every located crossing,
    grid nonmonotonicity, or no crossing within the searched range. This is a
    descriptive cost comparison, never a strategy-selection step.
    """
    left_dates = variant_overlays.filter(pl.col("execution_date") <= last_date)[
        "execution_date"
    ].sort().to_list()
    right_dates = bsc_overlays.filter(pl.col("execution_date") <= last_date)[
        "execution_date"
    ].sort().to_list()
    if left_dates != right_dates:
        raise ValueError("Break-even comparison requires the same decision and holding sample")
    config = settings["strategy"]
    grid = sorted(set(float(value) for value in config["break_even_cost_grid_bp"]))
    if len(grid) < 2 or grid[0] != 0 or not np.isfinite(grid).all() or grid[-1] <= 0:
        raise ValueError("Break-even grid must start at zero and contain positive finite costs")
    tolerance = float(config["calibration_tolerance"])
    iterations = int(config["calibration_max_iterations"])
    annualization = int(settings["replication"]["months_per_year"])
    values: dict[float, float] = {}

    def difference(cost: float) -> float:
        if cost not in values:
            results = []
            for overlays in (variant_overlays, bsc_overlays):
                daily = simulate_strategy(
                    base_daily, base_ledger, overlays, cost_bp=cost,
                    initial_nav=initial_nav, last_date=last_date, record_ledger=False,
                ).daily
                results.append(daily.with_columns(pl.col("date").dt.month_end()).group_by(
                    "date", maintain_order=True,
                ).agg(((1 + pl.col("daily_return")).product() - 1).alias("return")))
            matched = results[0].join(results[1], on="date", suffix="_bsc", validate="1:1")
            if matched.height != results[0].height or matched.height != results[1].height:
                raise ValueError("Strategy monthly return samples differ")
            values[cost] = float(annualization * (matched["return"] - matched["return_bsc"]).mean())
        return values[cost]

    valid_grid, grid_differences, failed_cost = [], [], None
    for cost in grid:
        try:
            value = difference(cost)
        except ValueError as error:
            if "strategy NAV is nonpositive" not in str(error):
                raise
            failed_cost = cost
            break
        valid_grid.append(cost)
        grid_differences.append(value)
    changes = np.diff(grid_differences)
    nonmonotone = bool(np.any(changes < -tolerance) and np.any(changes > tolerance))
    roots = [cost for cost, value in zip(valid_grid, grid_differences, strict=True)
             if abs(value) <= tolerance]
    if len(valid_grid) == len(grid) and all(abs(value) <= tolerance for value in grid_differences):
        return {"break_even_bp": None, "status": "equal_on_cost_grid", "crossings_bp": [],
                "nonmonotone": False, "grid_bp": valid_grid,
                "annual_mean_differences": grid_differences, "evaluations": len(values)}
    for left, right in zip(valid_grid, valid_grid[1:], strict=False):
        initial_left, initial_right = difference(left), difference(right)
        if initial_left * initial_right >= 0 or (
            min(abs(initial_left), abs(initial_right)) <= tolerance
        ):
            continue
        left_value = initial_left
        for _ in range(iterations):
            middle = (left + right) / 2
            value = difference(middle)
            if abs(value) <= tolerance:
                break
            if left_value * value <= 0:
                right = middle
            else:
                left, left_value = middle, value
        roots.append(middle)
    roots.sort()
    if len(roots) > 1:
        status = "multiple_crossings"
    elif roots:
        status = "nonmonotone_crossing" if nonmonotone else "crossing"
    else:
        status = "no_crossing_nonmonotone" if nonmonotone else "no_crossing_in_cost_grid"
    if failed_cost is not None:
        status += "_incomplete_nonpositive_nav"
    return {"break_even_bp": roots[0] if roots else None, "status": status,
            "crossings_bp": roots, "nonmonotone": nonmonotone, "grid_bp": valid_grid,
            "annual_mean_differences": grid_differences, "failed_cost_bp": failed_cost,
            "evaluations": len(values)}
