"""Training-only copula and regime transforms on the registered refit lineages (Methodology).

Each annual refit and its inner validation cutoff form separate lineages. A
centroid or HMM fit at a cutoff sees only months whose targets ended before it,
and is aligned to the previous fit of the same lineage, never to a later one.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from momenta.baseline_study import _structural_transform
from momenta.copula import OT_COLUMNS, fit_dated_centroids
from momenta.regimes import REGIME_INPUTS, regime_transform
from momenta.walkforward import month_end_origins, training_split


def carry(daily: pl.DataFrame, monthly: pl.DataFrame) -> pl.DataFrame:
    """Carry a completed month's observables forward, never backwards."""
    names = [name for name in monthly.columns if name != "date"]
    if monthly["date"].n_unique() != monthly.height:
        raise ValueError("Monthly transforms require unique formation dates")
    return (
        daily.drop(names, strict=False)
        .with_row_index("_order")
        .sort("date")
        .join_asof(monthly.sort("date"), on="date", strategy="backward")
        .sort("_order")
        .drop("_order")
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class LearnedTransforms:
    """Share expensive dated fits; full and inner alignments have separate pasts.

    Alignment starts at the registered first design refit even when a mock or
    holdout scores later origins. Earlier fits are never aligned to a later fit.
    The caches are local to this execution; BDP paths come from the cutoff-specific cache.
    """

    def __init__(
        self,
        settings: dict[str, Any],
        bdp_settings: dict[str, Any],
        days: list[date],
        monthly: pl.DataFrame,
        warmup: pl.DataFrame,
        output: Path,
        metadata: dict[str, Any],
    ) -> None:
        self.settings, self.days, self.monthly = settings, days, monthly
        self.ends = month_end_origins(days)
        self.warmup, self.output, self.metadata = warmup, output, metadata
        self.bdp = _structural_transform(bdp_settings, True)
        self.schedules: dict[str, dict[str, list[date]]] = {}
        self.base: dict[tuple[str, date], tuple[pl.DataFrame, dict]] = {}
        self.geometry: dict[tuple[str, date], tuple[pl.DataFrame, dict]] = {}
        self.regimes: dict[tuple[str, int, str, date], tuple[pl.DataFrame, dict]] = {}
        self.records: list[dict] = []
        self.centroid_records: list[dict] = []

    def _base(self, frame: pl.DataFrame, cutoff: date) -> tuple[pl.DataFrame, dict]:
        key = frame["series"][0], cutoff
        if key not in self.base:
            transformed, record = self.bdp(frame, cutoff)
            self.base[key] = (
                transformed,
                {
                    "fit_time": record["fit_time"],
                    "identity_hash": record["identity_hash"],
                    "artifact_hash": record["artifact_hash"],
                },
            )
        return self.base[key]

    def _schedule(self, frame: pl.DataFrame) -> dict[str, list[date]]:
        series = frame["series"][0]
        if series not in self.schedules:
            first = date.fromisoformat(
                self.settings.get(
                    "_registered_design_first_origin",
                    self.settings["walkforward"]["design_first_origin"],
                )
            )
            last = min(
                frame["date"].max(),
                date.fromisoformat(self.settings["walkforward"]["design_last_origin"]),
            )
            full = [
                day
                for i, day in enumerate(self.days)
                if day.month == 1
                and (i == 0 or self.days[i - 1].year != day.year)
                and first.year <= day.year <= last.year
            ]
            inner = []
            for cutoff in full:
                values, _ = self._base(frame, cutoff)
                valid = np.isfinite(values.select(REGIME_INPUTS).to_numpy()).all(
                    axis=1
                ) & np.isfinite(values["y1"].to_numpy())
                split = training_split(
                    values["date"].to_list(),
                    values["target_end"].to_list(),
                    self.days,
                    cutoff,
                    valid,
                    self.settings,
                )
                inner.append(split.inner_cutoff)
            self.schedules[series] = {"full": full, "inner": sorted(set(inner))}
        return self.schedules[series]

    def _lineage(self, frame: pl.DataFrame, cutoff: date) -> tuple[str, list[date]]:
        schedules = self._schedule(frame)
        matches = [name for name, dates in schedules.items() if cutoff in dates]
        if len(matches) != 1:
            raise ValueError(f"Cutoff {cutoff} is absent or ambiguous in refit lineages")
        lineage = matches[0]
        return lineage, [day for day in schedules[lineage] if day <= cutoff]

    def _geometry(self, frame: pl.DataFrame, cutoff: date) -> tuple[pl.DataFrame, dict]:
        lineage, history = self._lineage(frame, cutoff)
        previous = None
        for day in history:
            key = lineage, day
            base, _ = self._base(frame, day)
            finite = pl.all_horizontal([pl.col(name).is_finite() for name in REGIME_INPUTS])
            dates = base.filter(
                pl.col("date").is_in(self.ends), pl.col("target_end") < day,
                finite, pl.col("y1").is_finite(),
            )["date"].sort().to_list()
            if key in self.geometry and self.geometry[key][1]["training_dates"] != [
                str(value) for value in dates
            ]:
                raise ValueError("Shared centroid cache cannot change training-month support")
            if key not in self.geometry:
                transformed, record = fit_dated_centroids(
                    self.monthly,
                    dates,
                    day,
                    self.settings,
                    None if previous is None else np.asarray(previous["centroids"]),
                )
                record.update(lineage=lineage)
                record["fit_hash"] = _hash(record)
                directory = self.output / "centroids"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{lineage}_{day}.json"
                path.write_text(
                    json.dumps({**record, "provenance": self.metadata}, sort_keys=True, indent=2)
                    + "\n"
                )
                self.geometry[key] = transformed, record
                self.centroid_records.append(record)
            previous = self.geometry[key][1]
        transformed, record = self.geometry[lineage, cutoff]
        return carry(frame, transformed.select("date", *OT_COLUMNS)), {
            "fit_hash": record["fit_hash"],
            "lineage": lineage,
            "training_months": record["training_months"],
            "training_first": record["training_first"],
            "training_last": record["training_last"],
        }

    def _regime(self, frame: pl.DataFrame, cutoff: date, states: int):
        lineage, history = self._lineage(frame, cutoff)
        series = frame["series"][0]
        previous = None
        for day in history:
            key = series, states, lineage, day
            if key not in self.regimes:
                base, _ = self._base(frame, day)
                initial = self.warmup.filter(pl.col("series") == series)
                initial, _ = self.bdp(initial, day)
                transformed, record = regime_transform(
                    base,
                    day,
                    self.ends,
                    self.settings,
                    states=states,
                    previous=previous,
                    initial_monthly=initial,
                    cache_dir=self.output / "regimes",
                    provenance=self.metadata,
                )
                record = {**record, "lineage": lineage}
                self.regimes[key] = transformed, record
                self.records.append(record)
            previous = self.regimes[key][1]
        transformed, record = self.regimes[series, states, lineage, cutoff]
        names = [f"{kind}_{i}" for kind in ("hmmf", "hmme") for i in range(states - 1)]
        result = frame.drop(names, strict=False).join(
            transformed.select("date", *names), on="date", maintain_order="left"
        )
        return result, {
            "fit_hash": record["fit_hash"],
            "lineage": lineage,
            "training_months": record["training_months"],
            "training_target_end_max": record["training_target_end_max"],
        }

    def transform(self, *, ot: bool = False, states: int | None = None):
        """Compose BDP, optional OT and optional monthly filtered/emission states."""

        def apply(frame: pl.DataFrame, cutoff: date):
            transformed, bdp = self._base(frame, cutoff)
            # The cache includes the full feature frame, so extensions remain available.
            metadata = {"bdp": bdp}
            if ot:
                transformed, metadata["ot"] = self._geometry(transformed, cutoff)
            if states is not None:
                transformed, metadata["hmm"] = self._regime(transformed, cutoff, states)
            return transformed, metadata

        return apply
