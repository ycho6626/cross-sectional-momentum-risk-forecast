"""Established risk predictors and historical portfolio variance (Methodology)."""


from __future__ import annotations

import bisect
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO

OWN_FEATURES = ("own_rv_5", "own_rv_21", "own_rv_126")


MARKET_FEATURES = ("mkt_rv_126", "bear_24", "bear_x_var", "mkt_ret_36")


CORE_FEATURES = ("book_rv_126", "wml_beta_126", "bdp_ci", "disp_21", "mom_gap", "comom", "illiq")


FEATURE_COLUMNS = (*OWN_FEATURES, *CORE_FEATURES, *MARKET_FEATURES)


MACRO_FEATURES = ("log_vix", "baa10y", "t10y3m")


FEATURE_BLOCKS = {"HAR": OWN_FEATURES, "CORE": CORE_FEATURES,
                  "M": MARKET_FEATURES, "FRED": MACRO_FEATURES}


FEATURE_SETS = {
    "L1-own": OWN_FEATURES,
    "L1_without_M": tuple(c for c in FEATURE_COLUMNS if c not in MARKET_FEATURES),
    "L1": FEATURE_COLUMNS,
    "L1m": (*FEATURE_COLUMNS, *MACRO_FEATURES),
}


def rolling_mean(values: NDArray, window: int) -> NDArray:
    """Trailing arithmetic mean requiring every observation, with no future padding
    (Methodology)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or window < 1:
        raise ValueError("Rolling mean requires a vector and positive window")
    output = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        output[window-1:] = windows.mean(axis=1)
    return output


def log_realized_variance(values: NDArray, window: int, horizon: int) -> NDArray:
    """log((horizon/window) sum last-window P²), Methodology."""
    variance = horizon * rolling_mean(np.asarray(values)**2, window)
    return np.log(variance, out=np.full(len(variance), np.nan), where=variance > 0)


def rolling_beta(returns: NDArray, market: NDArray, window: int) -> NDArray:
    """OLS slope with intercept: cov(P,Mkt−RF)/var(Mkt−RF), Methodology."""
    y, x = np.asarray(returns), np.asarray(market)
    cov = rolling_mean(x*y, window) - rolling_mean(x, window)*rolling_mean(y, window)
    variance = rolling_mean(x*x, window) - rolling_mean(x, window)**2
    return np.divide(cov, variance, out=np.full(len(x), np.nan), where=variance > 0)


def quantile_skewness(values: NDArray, quantiles: list[float]) -> float:
    """(q_high+q_low−2q_median)/(q_high−q_low), BDP approximation in Methodology."""
    if not np.isfinite(values).all():
        return float("nan")
    low, median, high = np.quantile(values, quantiles)
    return float((high + low - 2*median)/(high-low)) if high > low else float("nan")


def approximate_bdp(returns: NDArray, settings: dict[str, Any]) -> NDArray:
    """Daily RMS over 126 × quantile skewness over 252 (labeled approximation; Methodology)."""
    cfg = settings["features"]
    scale = np.sqrt(rolling_mean(np.asarray(returns)**2, cfg["approximation_scale_days"]))
    shape = np.full(len(returns), np.nan)
    window = cfg["approximation_shape_days"]
    for index in range(window-1, len(returns)):
        shape[index] = quantile_skewness(returns[index-window+1:index+1],
                                        cfg["approximation_quantiles"])
    return scale * shape


def average_residual_correlation(returns: NDArray, factors: NDArray) -> tuple[float, int]:
    """Balanced weekly excess-return FF3 residual correlation, Methodology.

    Columns are stocks; factors are Mkt−RF, SMB, HML, RF. All stocks use the
    same weeks and OLS intercept. Standardize using 1/T moments, then apply
    (n Var(mean(z_i))−1)/(n−1). Undefined zero-dispersion residuals are excluded
    and counted in the returned usable panel size. This is the pairwise,
    loser-only, no-industry-adjustment variant of Lou–Polk (2022), footnote 16.
    """
    panel, ff = np.asarray(returns, dtype=float), np.asarray(factors, dtype=float)
    if panel.ndim != 2 or ff.shape != (panel.shape[0], 4):
        raise ValueError("Weekly panel and four factors must align")
    if not np.isfinite(ff).all():
        return float("nan"), 0
    panel = panel[:, np.isfinite(panel).all(axis=0)]
    if panel.shape[1] < 2:
        return float("nan"), panel.shape[1]
    design = np.column_stack([np.ones(len(ff)), ff[:, :3]])
    excess = panel - ff[:, 3, None]
    residuals = excess - design @ np.linalg.lstsq(design, excess, rcond=None)[0]
    sd = residuals.std(axis=0)
    threshold = np.finfo(float).eps * max(design.shape) * np.max(np.abs(excess), axis=0)
    residuals = residuals[:, sd > threshold]
    sd = sd[sd > threshold]
    n = len(sd)
    if n < 2:
        return float("nan"), n
    standardized = (residuals - residuals.mean(axis=0)) / sd
    return float((n*np.var(standardized.mean(axis=1))-1)/(n-1)), n


def historical_book_variance(
    returns: NDArray, weights: NDArray, horizon: int,
) -> tuple[float, float]:
    """horizon × mean((omega' r)²), missing r=0; absolute-weight coverage (Methodology)."""
    returns, weights = np.asarray(returns), np.asarray(weights)
    if returns.ndim != 2 or returns.shape[1] != len(weights):
        raise ValueError("Historical returns and formation weights must align")
    total = np.abs(weights).sum()
    if total <= 0:
        return float("nan"), float("nan")
    observed = np.isfinite(returns)
    path = np.where(observed, returns, 0) @ weights
    variance = horizon * np.mean(path*path)
    coverage = np.mean(observed @ np.abs(weights) / total)
    return (float(np.log(variance)) if variance > 0 else float("nan"), float(coverage))


def completed_week_indices(calendar: list[date], origin: date) -> list[list[int]]:
    """Whole calendar weeks whose last scheduled trading date is ≤origin (Methodology)."""
    weeks: dict[date, list[int]] = {}
    for index, day in enumerate(calendar):
        monday = day - timedelta(days=day.weekday())
        weeks.setdefault(monday, []).append(index)
    return [indices for indices in weeks.values() if calendar[indices[-1]] <= origin
            and (calendar[indices[-1]] < calendar[-1]
                 or calendar[indices[-1]].weekday() == 4)]


def weekly_compound(panel: NDArray, groups: list[list[int]]) -> NDArray:
    """Compound daily stock returns or factor returns, retaining any missing value (Methodology)."""
    return np.array([np.prod(1 + panel[indices], axis=0) - 1 for indices in groups])


def own_features(books: pl.DataFrame, factors: pl.DataFrame, settings: dict) -> pl.DataFrame:
    """Return per-book trailing own features, retaining warmup nulls (Methodology)."""
    cfg, horizon = settings["features"], settings["targets"]["horizon"]
    frames = []
    for frame in books.sort("date").partition_by("series", maintain_order=True):
        frame = frame.join(factors.select("date", "mkt_rf"), on="date", how="left")
        y = frame["daily_return"].to_numpy()
        columns = [pl.Series(f"own_rv_{n}", log_realized_variance(y, n, horizon))
                   for n in cfg["own_rv_days"]]
        columns += [pl.Series("wml_beta_126", rolling_beta(y, frame["mkt_rf"].to_numpy(),
                                                        cfg["beta_days"])),
                    pl.Series("bdp_ci", approximate_bdp(y, settings))]
        frames.append(frame.select("date", "series").with_columns(columns))
    return pl.concat(frames).sort("date", "series")


def market_features(factors: pl.DataFrame, settings: dict) -> pl.DataFrame:
    """French-only trailing market risk and trends; RAW market = Mkt−RF+RF (Methodology)."""
    cfg = settings["features"]
    frame = factors.sort("date")
    market = frame["mkt_rf"].to_numpy()
    total = market + frame["rf"].to_numpy()
    variance = settings["targets"]["horizon"] * rolling_mean(market**2, cfg["market_rv_days"])
    bear_path = rolling_mean(np.log1p(total), cfg["bear_days"]) * cfg["bear_days"]
    bear = np.where(np.isfinite(bear_path), (bear_path < 0).astype(float), np.nan)
    trend = np.expm1(rolling_mean(np.log1p(total), cfg["market_return_days"])
                     * cfg["market_return_days"])
    return frame.select("date").with_columns(
        pl.Series("mkt_rv_126", np.log(variance)), pl.Series("bear_24", bear),
        pl.Series("bear_x_var", bear * variance), pl.Series("mkt_ret_36", trend),
    )


def macro_features(dates: pl.DataFrame, observations: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Backward observation-date joins for the three authorized FRED inputs (Methodology)."""
    frame = dates.select("date").unique().sort("date")
    for series, column, logarithm in (("VIXCLS", "log_vix", True),
                                     ("BAA10Y", "baa10y", False),
                                     ("T10Y3M", "t10y3m", False)):
        observed = observations[series].filter(pl.col("value").is_not_null()).sort("date")
        frame = frame.join_asof(observed.select("date", pl.col("value").alias(column)),
                               on="date", strategy="backward")
        if logarithm:
            frame = frame.with_columns(pl.col(column).log())
    return frame


def _stock_panel(
    curated: Path, days: list[date], ids: NDArray, delistings: pl.DataFrame,
) -> tuple[NDArray, NDArray, list[Path]]:
    """Read each source year separately, retaining selected IDs and bounded dates (Methodology)."""
    day_array = np.array(days, dtype="datetime64[D]")
    returns = np.full((len(days), len(ids)), np.nan)
    dollar_volume = np.full_like(returns, np.nan)
    sources = []
    for year in sorted({d.year for d in days}):
        files = sorted((curated / f"prices/year={year}").glob("*.parquet"))
        if not files:
            continue
        sources.extend(files)
        frame = pl.scan_parquet(files).filter(
            pl.col("permaticker").is_in(ids), pl.col("date").is_in(days),
        ).select("permaticker", "date", "daily_return",
                 (pl.col("close")*pl.col("volume")).alias("dollar_volume")).collect()
        rows = np.searchsorted(day_array, frame["date"].to_numpy())
        cols = np.searchsorted(ids, frame["permaticker"].to_numpy())
        returns[rows, cols] = frame["daily_return"].to_numpy()
        dollar_volume[rows, cols] = frame["dollar_volume"].to_numpy()
    terminal = delistings.filter(pl.col("permaticker").is_in(ids),
                                pl.col("date").is_in(days))
    if terminal.height:
        rows = np.searchsorted(day_array, terminal["date"].to_numpy())
        cols = np.searchsorted(ids, terminal["permaticker"].to_numpy())
        if np.isfinite(returns[rows, cols]).any():
            raise ValueError("Ordinary and terminal returns overlap")
        returns[rows, cols] = terminal["terminal_return"].to_numpy()
    return returns, dollar_volume, sources


def cross_features(
    days: list[date], ids: NDArray, returns: NDArray, dollar_volume: NDArray,
    universe: pl.DataFrame, signals: pl.DataFrame, weights: pl.DataFrame,
    factors: pl.DataFrame, calendar: list[date], settings: dict,
) -> pl.DataFrame:
    """Compute shared cross-section features and formation-weight book risk (Methodology).

    Pure bounded-frame entry point for real-vintage perturbation tests. The
    caller supplies enough history and trims the requested output dates later.
    """
    cfg, horizon = settings["features"], settings["targets"]["horizon"]
    formations = sorted(universe["date"].unique().to_list())
    u_map = {key[0]: group["permaticker"].to_numpy()
             for key, group in universe.partition_by("date", as_dict=True).items()}
    s_map = {key[0]: group for key, group in signals.partition_by("date", as_dict=True).items()}
    w_map = {key: group for key, group in weights.partition_by(
        ["formation", "series", "construction"], as_dict=True,
    ).items()}
    day_array = np.array(days, dtype="datetime64[D]")
    ff = pl.DataFrame({"date": days}).join(factors, on="date", how="left").select(
        "mkt_rf", "smb", "hml", "rf",
    ).to_numpy()
    # Global week endpoints preserve holiday-shortened weeks and exclude a
    # month-end's unfinished calendar week even in a truncated build.
    all_weeks = completed_week_indices(calendar, days[-1])
    groups = [[int(np.searchsorted(day_array, np.datetime64(calendar[j]))) for j in group]
              for group in all_weeks if calendar[group[0]] >= days[0]
              and calendar[group[-1]] <= days[-1]]
    week_ends = [days[group[-1]] for group in groups]
    weekly_returns = weekly_compound(returns, groups)
    weekly_ff = weekly_compound(ff, groups)
    month_features: dict[date, dict[str, Any]] = {}
    for formation in formations:
        if formation < days[0] or formation > days[-1] or formation not in s_map:
            continue
        raw = s_map[formation]["raw"].to_numpy()
        low, high = np.quantile(raw, cfg["momentum_gap_quantiles"])
        loser = w_map.get((formation, "RAW", "decile_vw"))
        loser_ids = loser.filter(pl.col("weight") < 0)["permaticker"].to_numpy()
        columns = np.searchsorted(ids, loser_ids)
        end = bisect.bisect_right(week_ends, formation)
        comom, balanced = float("nan"), 0
        if end >= cfg["comom_weeks"]:
            begin = end - cfg["comom_weeks"]
            comom, balanced = average_residual_correlation(
                weekly_returns[begin:end, columns], weekly_ff[begin:end],
            )
        month_features[formation] = {
            "mom_gap": float(high-low), "comom": comom, "comom_count": balanced,
            "comom_total": len(loser_ids), "comom_asof": formation,
        }
    dispersion, illiquidity, counts, sizes, dv_counts = [], [], [], [], []
    matched = []
    for index, day in enumerate(days):
        pos = bisect.bisect_right(formations, day) - 1
        formation = formations[pos] if pos >= 0 else None
        matched.append(formation)
        if formation is None:
            dispersion.append(np.nan)
            illiquidity.append(np.nan)
            counts.append(0)
            sizes.append(0)
            dv_counts.append(0)
            continue
        cols = np.searchsorted(ids, u_map[formation])
        # Year panels contain the union of all applicable formation members.
        r, dv = returns[index, cols], dollar_volume[index, cols]
        valid = np.isfinite(r)
        liquid = valid & np.isfinite(dv) & (dv > 0)
        dispersion.append(float(np.var(r[valid])) if valid.sum() >= 2 else np.nan)
        illiquidity.append(float(np.mean(abs(r[liquid])/dv[liquid]))
                           if liquid.any() else np.nan)
        counts.append(int(valid.sum()))
        sizes.append(len(cols))
        dv_counts.append(int(liquid.sum()))
    disp = rolling_mean(np.array(dispersion), cfg["dispersion_days"])
    illiq = rolling_mean(np.array(illiquidity), cfg["illiquidity_days"])
    records = []
    for index, day in enumerate(days):
        formation = matched[index]
        if formation not in month_features:
            continue
        shared = {"date": day, **month_features[formation],
                  "disp_21": float(np.log(disp[index])) if disp[index] > 0 else None,
                  "illiq": float(np.log(illiq[index]*cfg["illiquidity_dollar_scale"]))
                  if illiq[index] > 0 else None,
                  "universe_return_count": counts[index], "universe_size": sizes[index],
                  "illiquidity_count": dv_counts[index]}
        for series in ("RAW", "RAW_c", "RES"):
            selected = w_map.get((formation, series, settings["portfolios"]["primary"]))
            if selected is None:
                continue
            variance, coverage = float("nan"), float("nan")
            window = cfg["book_rv_days"]
            if index + 1 >= window:
                cols = np.searchsorted(ids, selected["permaticker"].to_numpy())
                variance, coverage = historical_book_variance(
                    returns[index-window+1:index+1, cols], selected["weight"].to_numpy(), horizon,
                )
            records.append({**shared, "series": series, "book_rv_126": variance,
                            "book_rv_coverage": coverage})
    return pl.DataFrame(records).sort("date", "series")


def build_features(settings: dict[str, Any], through: date | None = None) -> Path:
    """Build static design-period L1 features; BDP full-model transforms remain cutoff-specific."""
    root = Path(settings["data"]["data_root"])
    curated, portfolio_dir, output = root/"curated", root/'derived/portfolios', root/'derived/baseline'  # noqa: E501
    output.mkdir(parents=True, exist_ok=True)
    cfg = settings["features"]
    design_end = date.fromisoformat(settings["walkforward"]["design_last_origin"])
    bound = min(through, design_end) if through else design_end
    feature_start = date.fromisoformat(settings["walkforward"]["feature_start"])
    inputs = [portfolio_dir/f"{name}.parquet" for name in ("signals", "formation_weights", "book_daily")]  # noqa: E501
    inputs += [portfolio_dir/"_metadata.json", curated/"_metadata.json", curated/"universe.parquet",
               curated/"cap_repairs.parquet", curated/"delistings.parquet",
               curated/"french/ff3_daily.parquet"]
    signals = pl.read_parquet(portfolio_dir/"signals.parquet").filter(pl.col("date") <= bound)
    universe = pl.read_parquet(curated/"universe.parquet").select("date", "permaticker")
    universe = universe.filter(pl.col("date") <= bound)
    weights = pl.read_parquet(portfolio_dir/"formation_weights.parquet").filter(pl.col("formation") <= bound)  # noqa: E501
    books = pl.read_parquet(portfolio_dir/"book_daily.parquet").filter(
        pl.col("construction") == settings["portfolios"]["primary"], pl.col("date") <= bound,
    ).select("date", "series", "daily_return")
    factors = pl.read_parquet(curated/"french/ff3_daily.parquet").sort("date")
    calendar = factors["date"].to_list()
    delistings = pl.read_parquet(curated/"delistings.parquet").select(
        "permaticker", "date", "terminal_return",
    )
    own = own_features(books, factors, settings)
    market = market_features(factors.filter(pl.col("date") <= bound), settings)
    frames = []
    for year in range(feature_start.year, bound.year+1):
        print(f"Build L1 features {year}", flush=True)
        first = max(date(year, 1, 1), feature_start)
        end = min(date(year, 12, 31), bound)
        # At most 371 preceding sessions: enough for 52 whole weeks, including
        # holiday weeks and the preceding formation at the year boundary.
        warmup = max(cfg["book_rv_days"], cfg["comom_weeks"]*7+7,
                     cfg["dispersion_days"], cfg["illiquidity_days"])
        start_index = max(0, bisect.bisect_left(calendar, first)-warmup)
        days = calendar[start_index:bisect.bisect_right(calendar, end)]
        prior = universe.filter(pl.col("date") <= days[0])["date"].max()
        subset = universe.filter(pl.col("date").is_between(prior, end))
        ids = np.sort(subset["permaticker"].unique().to_numpy())
        panel, dv, sources = _stock_panel(curated, days, ids, delistings)
        inputs.extend(sources)
        cross = cross_features(days, ids, panel, dv, subset,
                               signals.filter(pl.col("date").is_between(prior, end)),
                               weights.filter(pl.col("formation").is_between(prior, end)),
                               factors, calendar, settings)
        frames.append(cross.filter(pl.col("date").is_between(first, end)))
    if not frames:
        raise ValueError("No dates in the feature build range")
    features = pl.concat(frames).join(own, on=["date", "series"], how="inner")
    features = features.join(market, on="date", how="left").sort("date", "series")
    observations = {}
    for series in ("VIXCLS", "BAA10Y", "T10Y3M"):
        path = curated/f"fred/{series}.parquet"
        inputs.append(path)
        observations[series] = pl.read_parquet(path)
    features = features.join(macro_features(features, observations), on="date", how="left")
    decision_path = output/"bdp_decision.json"
    method = "approximation_pending_recovery"
    if decision_path.exists():
        inputs.append(decision_path)
        decision = json.loads(decision_path.read_text())
        method = "BDP-transfer_requires_cutoff_fit" if decision["full_model"] else "approximation"
    features = features.with_columns(pl.lit(method).alias("bdp_method"))
    features = features.with_columns(pl.col(pl.Float64).fill_nan(None))
    path = output/"features.parquet"
    features.write_parquet(path, compression=settings["storage"]["compression"])
    metadata = {
        **provenance(REPO, Path(settings["config_path"]), Manifest(root)),
        "through": str(bound), "bdp_method": method,
        "input_hashes": {str(source): sha256_file(source) for source in sorted(set(inputs))},
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_hash": sha256_file(path),
        "rows": features.height,
        "availability": {
            name: {series: str(features.filter(pl.col("series") == series).drop_nulls(columns)
                               ["date"].min()) for series in ("RAW", "RAW_c", "RES")}
            for name, columns in FEATURE_SETS.items()
        },
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)+"\n",
    )
    return path
