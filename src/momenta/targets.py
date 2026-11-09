"""Forward targets on the unscaled strategy book (Methodology)."""


from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


def forward_targets(
    returns: NDArray[np.float64], horizon: int, lag: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """At i use i+lag+1…i+lag+horizon: log ΣP² and Σlog(1+P) (Methodology).

    Arrays align with origins. Incomplete windows and zero realized variance
    have NaN log-variance; incomplete windows also have NaN log-return. No
    variance floor, demeaning, annualization, or fitted tail threshold is used.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Book returns must be a finite vector greater than -100%")
    if horizon < 1 or lag < 0:
        raise ValueError("Target horizon must be positive and lag nonnegative")
    y1, y2 = np.full(len(values), np.nan), np.full(len(values), np.nan)
    count = len(values) - lag - horizon
    if count > 0:
        windows = np.lib.stride_tricks.sliding_window_view(values[lag + 1:], horizon)
        variance = np.square(windows).sum(axis=1)
        positive = variance > 0
        y1[:count][positive] = np.log(variance[positive])
        y2[:count] = np.log1p(windows).sum(axis=1)
    return y1, y2


def tail_labels(
    log_returns: NDArray[np.float64], threshold: float,
) -> NDArray[np.int64]:
    """Label 1{y2 < q}; q is supplied by the current training-only fit (Methodology)."""
    values = np.asarray(log_returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Tail labels require finite log returns and a finite fitted threshold")
    return (values < threshold).astype(np.int64)


def _calendar_dates(calendar: list[date] | pl.DataFrame) -> list[date]:
    """Require a strictly increasing, unique supplied trading calendar (Methodology)."""
    dates = calendar["date"].to_list() if isinstance(calendar, pl.DataFrame) else list(calendar)
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Trading calendar must be nonempty, sorted and unique")
    return dates


def build_targets(
    book_daily: pl.DataFrame,
    calendar: list[date] | pl.DataFrame,
    settings: dict[str, Any],
    *,
    first_origin: date | None = None,
    last_origin: date | None = None,
) -> pl.DataFrame:
    """Build complete dated y1/y2 windows; never fit y3 globally (Methodology).

    Book rows must cover every calendar session between their endpoints. Only
    origins with a full target window are emitted. A zero-variance window has
    null y1 and an explicit flag, allowing complete-feature/target filtering.
    """
    dates = _calendar_dates(calendar)
    config = settings["targets"]
    horizon, lag = int(config["horizon"]), int(config["lag"])
    walkforward = settings.get("walkforward", {})
    if first_origin is None and "feature_start" in walkforward:
        first_origin = date.fromisoformat(walkforward["feature_start"])
    if "design_last_origin" in walkforward:
        design_end = date.fromisoformat(walkforward["design_last_origin"])
        if last_origin is not None and last_origin > design_end:
            raise ValueError('Holdout-origin targets are restricted to the design period')
        last_origin = design_end if last_origin is None else last_origin
    if last_origin is not None:
        final_index = np.searchsorted(dates, last_origin, side="right") - 1 + lag + horizon
        data_end = dates[min(len(dates) - 1, max(0, final_index))]
        book_daily = book_daily.filter(pl.col("date") <= data_end)
    records: list[dict[str, Any]] = []
    for group in book_daily.sort("date").partition_by(["series", "construction"]):
        series, construction = group.select("series", "construction").row(0)
        observed = group["date"].to_list()
        expected = [d for d in dates if observed[0] <= d <= observed[-1]]
        if observed != expected:
            raise ValueError(f"Missing or duplicate book date for {series}/{construction}")
        y1, y2 = forward_targets(group["daily_return"].to_numpy(), horizon, lag)
        for index in range(max(0, len(observed) - lag - horizon)):
            origin = observed[index]
            if (first_origin is not None and origin < first_origin) or (
                last_origin is not None and origin > last_origin
            ):
                continue
            records.append({
                "series": series, "construction": construction, "origin": origin,
                "target_start": observed[index + lag + 1],
                "target_end": observed[index + lag + horizon],
                "y1": float(y1[index]) if np.isfinite(y1[index]) else None,
                "y2": float(y2[index]), "zero_realized_variance": bool(np.isnan(y1[index])),
            })
    schema = {
        "series": pl.String, "construction": pl.String, "origin": pl.Date,
        "target_start": pl.Date, "target_end": pl.Date,
        "y1": pl.Float64, "y2": pl.Float64, "zero_realized_variance": pl.Boolean,
    }
    return pl.DataFrame(records, schema=schema).sort("series", "construction", "origin")


def overlap_counts(
    targets: pl.DataFrame, calendar: list[date] | pl.DataFrame,
) -> pl.DataFrame:
    """Count inclusive target intersections and unused sessions at month-ends (Methodology).

    Counts use the supplied trading calendar, not weekdays or calendar-day
    differences. The caller can restrict targets to its scored sample first.
    """
    dates = _calendar_dates(calendar)
    indices = {day: index for index, day in enumerate(dates)}
    month_ends = {day for i, day in enumerate(dates)
                  if i + 1 == len(dates) or dates[i + 1].month != day.month}
    records = []
    selected = targets.filter(pl.col("origin").is_in(sorted(month_ends)))
    for group in selected.sort("origin").partition_by(["series", "construction"]):
        rows = group.to_dicts()
        for left, right in zip(rows, rows[1:], strict=False):
            start, end = indices[left["target_start"]], indices[left["target_end"]]
            next_start, next_end = indices[right["target_start"]], indices[right["target_end"]]
            records.append({
                "series": left["series"], "construction": left["construction"],
                "origin": left["origin"], "next_origin": right["origin"],
                "origin_distance_days": indices[right["origin"]] - indices[left["origin"]],
                "overlap_days": max(0, min(end, next_end) - max(start, next_start) + 1),
                "gap_days": max(0, next_start - end - 1),
            })
    schema = {
        "series": pl.String, "construction": pl.String,
        "origin": pl.Date, "next_origin": pl.Date,
        "origin_distance_days": pl.Int64, "overlap_days": pl.Int64, "gap_days": pl.Int64,
    }
    return pl.DataFrame(records, schema=schema).sort("series", "construction", "origin")
