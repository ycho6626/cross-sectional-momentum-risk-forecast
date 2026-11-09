"""Formation and dated academic/strategy accounting (Methodology)."""


from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from momenta.manifest import Manifest, provenance, sha256_file
from momenta.settings import REPO

CONSTRUCTIONS = (
    ("RAW", "tercile_capped_vw"),
    ("RAW", "decile_vw"),
    ("RES", "tercile_capped_vw"),
    ("RES", "decile_vw"),
    ("RAW_c", "tercile_capped_vw"),
)


def tail_weights(
    signals: NDArray[np.float64],
    caps: NDArray[np.float64],
    bins: int,
    cap_ceiling: float | None,
) -> NDArray[np.float64]:
    """Sort by ceil(bins × upper ECDF); normalize each extreme leg (Methodology).

    Equal signals share a bin. Weights are signed; cap_ceiling is ME_p80 from
    the full U_m for capped weights, and None for plain value weights.
    """
    signals, caps = np.asarray(signals, dtype=float), np.asarray(caps, dtype=float)
    if signals.ndim != 1 or signals.shape != caps.shape or len(signals) < bins:
        raise ValueError("Signal and cap vectors must align and support every sort bin")
    if not np.isfinite(signals).all() or not np.isfinite(caps).all() or np.any(caps <= 0):
        raise ValueError("Formation requires finite signals and positive usable caps")
    if bins < 2 or (cap_ceiling is not None and (not np.isfinite(cap_ceiling) or cap_ceiling <= 0)):
        raise ValueError("Invalid bin count or cap ceiling")
    ranks = np.searchsorted(np.sort(signals), signals, side="right")
    # Integer arithmetic avoids floating-point rounding at exact tercile boundaries.
    groups = (bins * ranks + len(signals) - 1) // len(signals)
    base = caps if cap_ceiling is None else np.minimum(caps, cap_ceiling)
    weights = np.zeros(len(signals))
    for group, sign in ((1, -1), (bins, 1)):
        selected = groups == group
        if not selected.any():
            raise ValueError("Signal ties leave an empty extreme portfolio leg")
        weights[selected] = sign * base[selected] / base[selected].sum()
    return weights


def form_portfolios(signals: pl.DataFrame, settings: dict[str, Any]) -> pl.DataFrame:
    """Use usable caps and full-universe p80 for all five registered constructions."""
    config = settings["portfolios"]
    frames = []
    for group in signals.sort("date", "permaticker").partition_by("date", maintain_order=True):
        ceiling = float(np.quantile(group["marketcap"].to_numpy(), config["cap_percentile"]))
        for series, construction in CONSTRUCTIONS:
            selected = group if series == "RAW" else group.filter("res_eligible")
            if selected.is_empty():
                continue
            column = "res" if series == "RES" else "raw"
            primary = construction == config["primary"]
            weights = tail_weights(
                selected[column].to_numpy(), selected["marketcap"].to_numpy(),
                config["primary_bins"] if primary else config["secondary_bins"],
                ceiling if primary else None,
            )
            frames.append(selected.select(
                "permaticker", pl.col("date").alias("formation"), "marketcap", "cap_source",
                pl.col(column).alias("signal"),
            ).with_columns(
                pl.lit(series).alias("series"), pl.lit(construction).alias("construction"),
                pl.lit(ceiling).alias("cap_p80"), pl.Series("weight", weights),
            ).filter(pl.col("weight") != 0))
    if not frames:
        raise ValueError("No formations available")
    return pl.concat(frames).sort("formation", "series", "construction", "permaticker")


def _year_panel(
    curated: Path,
    days: list[date],
    ids: NDArray[np.int64],
    delistings: pl.DataFrame,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Read one price year and add dated terminal credits; never fill held gaps."""
    year = days[0].year
    panel = pl.scan_parquet(curated / f"prices/year={year}/*.parquet").filter(
        pl.col("permaticker").is_in(ids), pl.col("date").is_in(days),
    ).select("permaticker", "date", "daily_return").collect()
    day_array = np.array(days, dtype="datetime64[D]")
    returns = np.full((len(days), len(ids)), np.nan)
    terminals = np.zeros(returns.shape, dtype=bool)
    rows = np.searchsorted(day_array, panel["date"].to_numpy())
    columns = np.searchsorted(ids, panel["permaticker"].to_numpy())
    returns[rows, columns] = panel["daily_return"].to_numpy()
    endings = delistings.filter(
        pl.col("permaticker").is_in(ids), pl.col("credit_date").is_in(days),
        pl.col("terminal_return").is_not_null(),
    )
    if not endings.is_empty():
        rows = np.searchsorted(day_array, endings["credit_date"].to_numpy())
        columns = np.searchsorted(ids, endings["permaticker"].to_numpy())
        if np.isfinite(returns[rows, columns]).any():
            raise ValueError("Ordinary and terminal returns overlap on a credit date")
        returns[rows, columns] = endings["terminal_return"].to_numpy()
        terminals[rows, columns] = True
    return returns, terminals


def run_portfolios(
    weights: pl.DataFrame,
    calendar: list[date],
    curated: Path,
    output: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Credit daily returns before the close-of-day trade (Methodology).

    Academic legs start immediately after formation. The strategy executes the
    frozen target after rebalance_lag_days sessions; already-terminated targets
    remain cash. Every nonzero holding requires a finite daily/terminal return.
    """
    from momenta.accounting import academic_leg_day, book_day, rebalance_book

    output.mkdir(parents=True, exist_ok=True)
    ids = np.sort(weights["permaticker"].unique().to_numpy()).astype(np.int64)
    delistings = pl.read_parquet(curated / "delistings.parquet").rename({"date": "credit_date"})
    lag = int(settings["portfolios"]["rebalance_lag_days"])
    if lag < 1:
        raise ValueError("Strategy implementation lag must be at least one trading day")
    calendar = sorted(calendar)
    day_array = np.array(calendar, dtype="datetime64[D]")
    academic_events: dict[date, list[tuple[tuple[str, str], date, NDArray]]] = {}
    execution_events: dict[date, list[tuple[tuple[str, str], date, NDArray]]] = {}
    for group in weights.partition_by(["formation", "series", "construction"]):
        formation, series, construction = group.select(
            "formation", "series", "construction"
        ).row(0)
        next_index = int(np.searchsorted(day_array, np.datetime64(formation), side="right"))
        target = np.zeros(len(ids))
        target[np.searchsorted(ids, group["permaticker"].to_numpy())] = group["weight"].to_numpy()
        event = ((series, construction), formation, target)
        if next_index < len(calendar):
            academic_events.setdefault(calendar[next_index], []).append(event)
        if next_index + lag - 1 < len(calendar):
            execution_events.setdefault(calendar[next_index + lag - 1], []).append(event)
    states: dict[tuple[str, str], dict[str, Any]] = {}
    daily_academic, daily_book, rebalances = [], [], []
    held_days = academic_held_days = terminal_credits = unavailable_total = 0
    terminated = np.zeros(len(ids), dtype=bool)
    # Only events credited by the first simulated date can affect execution.
    prior = delistings.filter(pl.col("credit_date") < calendar[0])
    prior_ids = prior["permaticker"].to_numpy()
    terminated[np.isin(ids, prior_ids)] = True
    for year in sorted({d.year for d in calendar}):
        days = [d for d in calendar if d.year == year]
        print(f"Build portfolio returns {year}", flush=True)
        returns, terminals = _year_panel(curated, days, ids, delistings)
        ledger: list[dict[str, Any]] = []
        for offset, day in enumerate(days):
            ret, terminal = returns[offset], terminals[offset]
            for key, formation, target in academic_events.get(day, []):
                if key not in states:
                    states[key] = {
                        "positions": np.zeros(len(ids)),
                        "nav": float(settings["portfolios"]["initial_nav"]),
                    }
                states[key].update(
                    winner=np.maximum(target, 0), loser=np.maximum(-target, 0),
                    formation=formation,
                )
            terminated |= terminal
            for (series, construction), state in sorted(states.items()):
                active = (state["positions"] != 0) | (state["winner"] != 0) | (state["loser"] != 0)
                missing = active & ~np.isfinite(ret)
                if missing.any():
                    raise ValueError(
                        f"Missing held return on {day} for {series}/{construction}: "
                        f"permatickers {ids[missing][:10].tolist()}"
                    )
                held_days += int(np.count_nonzero(state["positions"]))
                academic_held_days += int(np.count_nonzero(state["winner"]) +
                                          np.count_nonzero(state["loser"]))
                terminal_credits += int(np.count_nonzero((state["positions"] != 0) & terminal))
                winner = academic_leg_day(state["winner"], ret, terminal)
                loser = academic_leg_day(state["loser"], ret, terminal)
                book = book_day(state["positions"], state["nav"], ret, terminal)
                state.update(winner=winner.weights, loser=loser.weights,
                             positions=book.positions, nav=book.nav)
                daily_academic.append({
                    "date": day, "series": series, "construction": construction,
                    "winner_return": winner.leg_return, "loser_return": loser.leg_return,
                    "wml": winner.leg_return - loser.leg_return,
                })
                # The day's NAV and return precede trading; exposures below are at close.
                state["daily_return"] = book.book_return
                state["terminal_cash"] = book.terminal_cash
            for key, formation, target in execution_events.get(day, []):
                state = states[key]
                unavailable = int(np.count_nonzero((target != 0) & terminated))
                unavailable_total += unavailable
                executable = np.where(terminated, 0, target)
                pretrade = state["positions"].copy()
                trade = rebalance_book(pretrade, state["nav"], executable)
                state["positions"] = trade.positions
                series, construction = key
                rebalances.append({
                    "date": day, "formation": formation, "series": series,
                    "construction": construction, "nav": state["nav"],
                    "turnover": trade.turnover, "unavailable_targets": unavailable,
                    "executed_long_weight": float(executable[executable > 0].sum()),
                    "executed_short_weight": float(-executable[executable < 0].sum()),
                })
                for index in np.flatnonzero((pretrade != 0) | (trade.positions != 0)):
                    ledger.append({
                        "date": day, "formation": formation, "series": series,
                        "construction": construction, "permaticker": int(ids[index]),
                        "nav": state["nav"], "pretrade": float(pretrade[index]),
                        "posttrade": float(trade.positions[index]),
                        "trade": float(trade.deltas[index]),
                        "turnover": float(abs(trade.deltas[index]) / state["nav"]),
                    })
            for (series, construction), state in sorted(states.items()):
                daily_book.append({
                    "date": day, "series": series, "construction": construction,
                    "daily_return": state["daily_return"], "nav": state["nav"],
                    "cash": float(state["nav"] - state["positions"].sum()),
                    "gross_exposure": float(np.abs(state["positions"]).sum() / state["nav"]),
                    "terminal_cash": state["terminal_cash"],
                })
        if ledger:
            destination = output / f"ledger/year={year}"
            destination.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(ledger).write_parquet(
                destination / "part.parquet", compression=settings["storage"]["compression"],
            )
    if not daily_book:
        raise ValueError("No holding days follow the supplied formations")
    academic = pl.DataFrame(daily_academic).sort("series", "construction", "date")
    book = pl.DataFrame(daily_book).sort("series", "construction", "date")
    academic_monthly = academic.with_columns(pl.col("date").dt.month_end()).group_by(
        "series", "construction", "date", maintain_order=True,
    ).agg(
        ((pl.col("winner_return") + 1).product() - 1).alias("winner_return"),
        ((pl.col("loser_return") + 1).product() - 1).alias("loser_return"),
    ).with_columns((pl.col("winner_return") - pl.col("loser_return")).alias("wml"))
    book_monthly = book.with_columns(pl.col("date").dt.month_end()).group_by(
        "series", "construction", "date", maintain_order=True,
    ).agg(((pl.col("daily_return") + 1).product() - 1).alias("monthly_return"),
          pl.col("nav").last())
    final = []
    for (series, construction), state in sorted(states.items()):
        for index in np.flatnonzero(state["positions"]):
            final.append({"date": calendar[-1], "series": series, "construction": construction,
                          "permaticker": int(ids[index]), "position": state["positions"][index],
                          "nav": state["nav"]})
    for name, frame in (
        ("academic_daily", academic), ("academic_monthly", academic_monthly),
        ("book_daily", book), ("book_monthly", book_monthly),
        ("rebalance_summary", pl.DataFrame(rebalances)), ("final_positions", pl.DataFrame(final)),
    ):
        frame.write_parquet(
            output / f"{name}.parquet", compression=settings["storage"]["compression"],
        )
    return {"held_position_days": held_days, "academic_held_position_days": academic_held_days,
            "missing_held_returns": 0, "terminal_credits": terminal_credits,
            "unavailable_execution_targets": unavailable_total,
            "formation_count": weights["formation"].n_unique()}


def build(settings: dict[str, Any], through: date | None = None) -> bool:
    """Build the portfolio layer from a checked, pinned data vintage (Methodology)."""
    import shutil

    from momenta.portfolio_report import evaluate_portfolios
    from momenta.signals import build_signals

    root = Path(settings["data"]["data_root"])
    curated, output = root / "curated", root / 'derived/portfolios'
    checks = json.loads((root / "derived/m1_checks.json").read_text())
    if not checks["passed"]:
        raise ValueError('Data checks must pass before building portfolios')
    manifest = Manifest(root)
    for key, path in (
        ("config_hash", Path(settings["config_path"])), ("data_manifest_hash", manifest.path),
        ("curated_metadata_hash", curated / "_metadata.json"),
        ("cap_repairs_hash", curated / "cap_repairs.parquet"),
        ("observed_cap_exceptions_hash", curated / "observed_cap_exceptions.parquet"),
        ("price_corrections_hash", REPO / "config/price_corrections.csv"),
    ):
        if checks["provenance"].get(key) != sha256_file(path):
            raise ValueError(f"Stale data checks ({key}); rerun data check")
    curation = json.loads((curated / "_metadata.json").read_text())
    common_end = date.fromisoformat(curation["common_end"])
    bound = min(through, common_end) if through else common_end
    universe = pl.read_parquet(curated / "universe.parquet").filter(pl.col("date") <= bound)
    if universe.is_empty():
        raise ValueError("No formations before the build cutoff")
    monthly = pl.scan_parquet(curated / "monthly_returns.parquet").filter(
        pl.col("permaticker").is_in(universe["permaticker"].unique().implode()),
    ).select("permaticker", "date", "monthly_return").collect()
    factors = pl.read_parquet(curated / "french/ff3_monthly.parquet")
    calendar = pl.read_parquet(curated / "trading_calendar.parquet").filter(
        pl.col("date") > universe["date"].min(), pl.col("date") <= bound,
    )["date"].to_list()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    print(f"Build signals for {universe['date'].n_unique()} formations", flush=True)
    signals = build_signals(monthly, factors, universe, settings)
    signals.write_parquet(output / "signals.parquet")
    weights = form_portfolios(signals, settings)
    weights.write_parquet(output / "formation_weights.parquet")
    audit = run_portfolios(weights, calendar, curated, output, settings)
    audit["signal_rows"] = signals.height
    metadata = {
        **provenance(REPO, Path(settings["config_path"]), manifest),
        "mode": "thin" if through else "full", "common_end": str(bound),
        "curated_metadata_hash": sha256_file(curated / "_metadata.json"),
        "universe_hash": sha256_file(curated / "universe.parquet"),
        "cap_repairs_hash": sha256_file(curated / "cap_repairs.parquet"),
        "price_corrections_hash": sha256_file(REPO / "config/price_corrections.csv"),
    }
    (output / "_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    for path in output.rglob("*.parquet"):
        path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = evaluate_portfolios(settings, output, metadata, audit)
    print(
        f"Portfolio {metadata['mode']} build complete; acceptance passed: {summary['passed']}", flush=True,  # noqa: E501
    )
    return bool(summary["passed"])
