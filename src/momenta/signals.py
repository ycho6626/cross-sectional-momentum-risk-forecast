"""Point-in-time monthly RAW and FF3 residual momentum (Methodology)."""


from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


def raw_momentum(returns: NDArray[np.float64]) -> float:
    """Return product(1 + R_k) − 1 for the supplied formation months (Methodology)."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("RAW requires a complete finite formation window")
    if np.any(values < -1):
        raise ValueError("Monthly total returns cannot be below -100%")
    result = float(np.prod(1 + values) - 1)
    if not np.isfinite(result):
        raise ValueError("RAW compounding produced a non-finite signal")
    return result


def residual_momentum(
    returns: NDArray[np.float64],
    factors: NDArray[np.float64],
    recent: NDArray[np.bool_],
    min_observations: int,
) -> float:
    """Return mean(e_recent)/sd(e_recent, ddof=1), from FF3 OLS (Methodology).

    The caller supplies the m−35..m window and marks m−11..m−1 as recent.
    Factors are [Mkt−RF, SMB, HML, RF]; OLS regresses R−RF on an intercept
    and the first three columns. Missing pairs are omitted from the fit,
    but every marked recent month must be observed. There is no clipping or
    zero-return exclusion. Sample SD matches JKP's res_mom implementation.
    """
    values = np.asarray(returns, dtype=np.float64)
    ff = np.asarray(factors, dtype=np.float64)
    mask = np.asarray(recent, dtype=np.bool_)
    if values.ndim != 1 or ff.shape != (values.size, 4) or mask.shape != values.shape:
        raise ValueError("RES requires aligned returns, four factors, and a recent-month mask")
    valid = np.isfinite(values) & np.isfinite(ff).all(axis=1)
    if valid.sum() < min_observations:
        raise ValueError(
            f"RES has {valid.sum()} complete observations; requires {min_observations}",
        )
    if mask.sum() < 2 or not valid[mask].all():
        raise ValueError("RES requires complete recent residual months")
    if np.any(values[valid] < -1):
        raise ValueError("Monthly total returns cannot be below -100%")
    y = values[valid] - ff[valid, 3]
    design = np.column_stack((np.ones(valid.sum()), ff[valid, :3]))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    residuals = (y - fitted)[mask[valid]]
    dispersion = float(np.std(residuals, ddof=1))
    # A perfect fit leaves floating-point noise, not an economically defined
    # residual signal. The bound scales with machine precision and fit size.
    roundoff = np.finfo(np.float64).eps * max(design.shape) * max(
        float(np.max(np.abs(y))), float(np.max(np.abs(fitted))),
    )
    if not np.isfinite(dispersion) or dispersion <= roundoff:
        raise ValueError("RES residual standard deviation is zero at numerical precision")
    result = float(np.mean(residuals) / dispersion)
    if not np.isfinite(result):
        raise ValueError("RES produced a non-finite signal")
    return result


def _check_keys(frame: pl.DataFrame, keys: list[str], name: str) -> None:
    """Require one non-null observation per calendar-month key (Methodology)."""
    if any(frame[column].null_count() for column in keys):
        raise ValueError(f"{name} has null keys")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{name} has duplicate calendar-month keys")


def build_signals(
    monthly_returns: pl.DataFrame,
    factors: pl.DataFrame,
    universe: pl.DataFrame,
    settings: dict[str, Any],
) -> pl.DataFrame:
    """Append RAW/RES to desired U_m rows using only months ≤m (Methodology).

    Inputs are small monthly frames: stock returns have permaticker, date and
    monthly_return; FF3 has date, mkt_rf, smb, hml and rf. Returns are decimals.
    Calendar-month alignment allows factors/monthly returns dated on calendar
    month-end to serve the last-trading-day formation. All universe fields,
    including its original date and RES eligibility, are preserved. Output is
    sorted by date/permaticker, with RES null only for ineligible members.
    """
    config = settings["signals"]
    formation = int(config["formation_months"])
    window = int(config["res_window_months"])
    minimum = int(config["res_min_months"])
    if not 1 <= formation < window or not 1 <= minimum <= window:
        raise ValueError("Invalid momentum window settings")
    month = (pl.col("date").dt.year() * 12 + pl.col("date").dt.month()).alias("_month")
    monthly = monthly_returns.select("permaticker", "date", "monthly_return").with_columns(month)
    ff = factors.select("date", "mkt_rf", "smb", "hml", "rf").with_columns(month)
    desired = universe.with_columns(month).sort("date", "permaticker")
    _check_keys(monthly, ["permaticker", "_month"], "Monthly returns")
    _check_keys(ff, ["_month"], "FF3 factors")
    _check_keys(desired, ["permaticker", "_month"], "Universe")
    if desired["res_eligible"].null_count():
        raise ValueError("Universe has missing RES eligibility")
    if desired.is_empty():
        return desired.drop("_month").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("raw"),
            pl.lit(None, dtype=pl.Float64).alias("res"),
        )

    panel = monthly.join(ff.drop("date"), on="_month", how="left").sort("permaticker", "_month")
    histories = {
        key[0]: (
            group["_month"].to_numpy(),
            group["monthly_return"].cast(pl.Float64).to_numpy(),
            group.select("mkt_rf", "smb", "hml", "rf").cast(pl.Float64).to_numpy(),
        )
        for key, group in panel.partition_by("permaticker", as_dict=True).items()
    }
    raw_values: list[float] = []
    residual_values: list[float | None] = []
    for permaticker, date, eligible, origin in desired.select(
        "permaticker", "date", "res_eligible", "_month",
    ).iter_rows():
        if permaticker not in histories:
            raise ValueError(f"Missing return history for {permaticker} at {date}")
        months, returns, ff_values = histories[permaticker]
        left, right = np.searchsorted(months, [origin - formation, origin])
        if right - left != formation:
            raise ValueError(f"Incomplete RAW formation months for {permaticker} at {date}")
        try:
            raw_values.append(raw_momentum(returns[left:right]))
            if not eligible:
                residual_values.append(None)
                continue
            fit_left = int(np.searchsorted(months, origin - window + 1))
            fit_right = int(np.searchsorted(months, origin, side="right"))
            fit_months = months[fit_left:fit_right]
            recent = (fit_months >= origin - formation) & (fit_months < origin)
            residual_values.append(residual_momentum(
                returns[fit_left:fit_right], ff_values[fit_left:fit_right], recent, minimum,
            ))
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(f"Invalid signal for {permaticker} at {date}: {error}") from error
    return desired.drop("_month").with_columns(
        pl.Series("raw", raw_values, dtype=pl.Float64),
        pl.Series("res", residual_values, dtype=pl.Float64),
    )
