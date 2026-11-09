"""Pinned French, JKP and FRED reference data (Methodology)."""


import calendar
import csv
import io
import json
import math
import os
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv

from momenta.manifest import Manifest, download
from momenta.settings import REPO

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


FRENCH_FILES = {
    "ff3_daily": "F-F_Research_Data_Factors_daily",
    "ff3_monthly": "F-F_Research_Data_Factors",
    "momentum_daily": "F-F_Momentum_Factor_daily",
    "momentum_monthly": "F-F_Momentum_Factor",
    "momentum_deciles_daily": "10_Portfolios_Prior_12_2_Daily",
    "momentum_deciles_monthly": "10_Portfolios_Prior_12_2",
    "momentum_breakpoints": "Prior_2-12_Breakpoints",
}


JKP_BASE = "https://jkpfactors-data.s3.amazonaws.com/public/"


JKP_FACTORS = ("ret_12_1", "resff3_12_1")


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def _csv_from_zip(path: Path) -> str:
    """Read the single CSV member without extracting provider paths to disk."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in reference archive {path.name}")
        return archive.read(members[0]).decode("utf-8-sig")


def _french_date(token: str) -> date:
    """Map YYYYMM monthly observations to calendar month-end (Methodology)."""
    year, month = int(token[:4]), int(token[4:6])
    day = int(token[6:8]) if len(token) == 8 else calendar.monthrange(year, month)[1]
    return date(year, month, day)


def _french_value(token: str) -> float | None:
    """Return x/100; −99.99 and −999 denote missing, per Methodology."""
    value = float(token)
    if value in {-99.99, -999.0}:
        return None
    if not math.isfinite(value):
        raise ValueError("Non-finite value in French reference file")
    return value / 100.0


def parse_french(text: str, name: str) -> tuple[pl.DataFrame, str]:
    """Parse the first dated block, retaining percent/missing semantics (Methodology).

    For portfolios this block is the published value-weighted return series;
    annual, equal-weighted and diagnostic sections remain in the pinned ZIP.
    Breakpoint stock counts retain their original unit, while returns use x/100.
    """
    if name not in FRENCH_FILES:
        raise ValueError(f"Unknown French input: {name}")
    vintage_match = re.search(r"[^\r\n]*created[^\r\n]*\d{6} CRSP[^\r\n]*", text)
    if vintage_match is None:
        raise ValueError(f"Missing CRSP vintage in French {name}")
    vintage = vintage_match.group().strip()
    if name.startswith("ff3"):
        columns = ["mkt_rf", "smb", "hml", "rf"]
    elif name.startswith("momentum_deciles"):
        columns = [f"d{number}" for number in range(1, 11)]
    elif name == "momentum_breakpoints":
        columns = ["n_stocks", *[f"p{number:02d}" for number in range(5, 101, 5)]]
    else:
        columns = ["mom"]
    digits = 8 if name.endswith("daily") else 6
    rows: list[dict[str, Any]] = []
    started = False
    for fields in csv.reader(io.StringIO(text)):
        token = fields[0].strip() if fields else ""
        if not re.fullmatch(rf"\d{{{digits}}}", token):
            if started:
                break
            continue
        started = True
        if len(fields) != len(columns) + 1:
            raise ValueError(f"Wrong number of columns in French {name}, date {token}")
        row: dict[str, Any] = {"date": _french_date(token)}
        for column, value in zip(columns, fields[1:], strict=True):
            row[column] = int(value) if column == "n_stocks" else _french_value(value)
        rows.append(row)
    if not rows:
        raise ValueError(f"No observations in French {name}")
    schema = {"date": pl.Date, **dict.fromkeys(columns, pl.Float64)}
    if "n_stocks" in schema:
        schema["n_stocks"] = pl.Int64
    frame = pl.DataFrame(rows, schema=schema).sort("date")
    if frame["date"].n_unique() != frame.height:
        raise ValueError(f"Duplicate dates in French {name}")
    return frame, vintage


def parse_jkp(text: str, factor: str, frequency: str) -> pl.DataFrame:
    """Read JKP US capped-VW factors; returns are already decimals (Methodology)."""
    frame = pl.read_csv(io.StringIO(text), try_parse_dates=True, null_values=["NA", "NaN"])
    required = {"location", "name", "freq", "weighting", "direction", "date", "ret"}
    if not required.issubset(frame.columns) or frame.is_empty():
        raise ValueError(f"Invalid JKP {factor} {frequency} schema")
    definition = ["location", "name", "freq", "weighting", "direction"]
    if any(frame[column].null_count() for column in definition):
        raise ValueError(f"Missing factor definition in JKP {factor} {frequency}")
    if frame.filter(
        (pl.col("location") != "usa")
        | (pl.col("name") != factor)
        | (pl.col("freq") != frequency)
        | (pl.col("weighting") != "vw_cap")
        | (pl.col("direction") != 1)
    ).height:
        raise ValueError(f"Unexpected factor definition in JKP {factor} {frequency}")
    frame = frame.with_columns(pl.col("date").cast(pl.Date), pl.col("ret").cast(pl.Float64))
    if frame["date"].n_unique() != frame.height or frame["date"].null_count():
        raise ValueError(f"Missing or duplicate dates in JKP {factor} {frequency}")
    if frame.filter(pl.col("ret").is_not_null() & ~pl.col("ret").is_finite()).height:
        raise ValueError(f"Non-finite returns in JKP {factor} {frequency}")
    return frame.sort("date")


def parse_fred(payload: dict[str, Any]) -> pl.DataFrame:
    """Preserve FRED native units and map '.' to null; do not forward-fill (Methodology)."""
    observations = payload.get("observations", [])
    if not observations or int(payload.get("count", len(observations))) != len(observations):
        raise ValueError("FRED response is empty or incomplete")
    rows = []
    for observation in observations:
        raw = observation["value"]
        value = None if raw == "." else float(raw)
        if value is not None and not math.isfinite(value):
            raise ValueError("Non-finite value in FRED reference file")
        rows.append({
            "date": date.fromisoformat(observation["date"]),
            "value": value,
            "realtime_start": date.fromisoformat(observation["realtime_start"]),
            "realtime_end": date.fromisoformat(observation["realtime_end"]),
        })
    frame = pl.DataFrame(rows, schema={
        "date": pl.Date, "value": pl.Float64, "realtime_start": pl.Date, "realtime_end": pl.Date,
    }).sort("date")
    if frame["date"].n_unique() != frame.height:
        raise ValueError("Duplicate dates in FRED reference file")
    return frame


def _annotate(
    manifest: Manifest, name: str, path: Path, frame: pl.DataFrame, vintage: str,
    publication_timing: str, **metadata: Any,
) -> None:
    entry = manifest.get(name)
    if entry is None:
        raise ValueError(f"Unrecorded reference file: {name}")
    manifest.record(
        name, path, entry["source_url"], vintage,
        date_min=frame["date"].min().isoformat(), date_max=frame["date"].max().isoformat(),
        observations=frame.height, publication_timing=publication_timing, **metadata,
    )


def pull_references(settings: dict[str, Any], manifest: Manifest) -> None:
    """Fetch reference inputs sequentially and pin raw bytes and source vintages."""
    root = Path(settings["data"]["data_root"])
    for name, stem in FRENCH_FILES.items():
        key = f"french/{name}"
        path = download(
            manifest, key, f"{FRENCH_BASE}{stem}_CSV.zip", root / "raw/french" / f"{stem}.zip",
        )
        frame, vintage = parse_french(_csv_from_zip(path), name)
        _annotate(manifest, key, path, frame, vintage,
                  "Published after observation month; publication lag not modeled (final vintage).",
                  units="decimal returns; breakpoint counts in stocks")
    for factor in JKP_FACTORS:
        for frequency in ("monthly", "daily"):
            name = f"{factor}_{frequency}"
            key = f"jkp/{name}"
            filename = f"%5Busa%5D_%5B{factor}%5D_%5B{frequency}%5D_%5Bvw_cap%5D.zip"
            path = download(manifest, key, JKP_BASE + filename, root / "raw/jkp" / f"{name}.zip")
            frame = parse_jkp(_csv_from_zip(path), factor, frequency)
            entry = manifest.get(key)
            vintage = entry["vintage"].split("; observations through")[0]
            vintage += f"; observations through {frame['date'].max().isoformat()}"
            _annotate(manifest, key, path, frame, vintage,
                      "Periodic historical release; publication lag not modeled (final vintage).",
                      units="decimal returns", weighting="vw_cap", location="usa",
                      license="CC BY-NC 4.0")
    load_dotenv(REPO / ".env")
    api_key = os.environ.get("FRED_API_KEY")
    for series in settings["data"]["fred_series"]:
        key = f"fred/{series}"
        if not api_key and manifest.get(key) is None:
            raise RuntimeError("FRED_API_KEY is missing from .env")
        path = download(
            manifest, key, FRED_URL, root / "raw/fred" / f"{series}.json",
            params={"series_id": series, "api_key": api_key, "file_type": "json",
                    "limit": 100000, "sort_order": "asc"},
        )
        payload = json.loads(path.read_text())
        frame = parse_fred(payload)
        vintage = f"FRED realtime {payload['realtime_start']} to {payload['realtime_end']}"
        timing = (
            "Retrospective NBER recession dating; figures only, never a predictor."
            if series == "USREC" else
            "Daily FRED release; release lag not modeled, aligned by observation date."
        )
        _annotate(manifest, key, path, frame, vintage, timing,
                  units="indicator (0/1)" if series == "USREC" else "native index/percentage units",
                  predictor_allowed=series != "USREC")


def curate_references(settings: dict[str, Any], manifest: Manifest) -> dict[str, Path]:
    """Normalize pinned references and take the trading calendar from FF3 daily."""
    root = Path(settings["data"]["data_root"])
    outputs: dict[str, Path] = {}

    def write(name: str, frame: pl.DataFrame) -> None:
        path = root / "curated" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path, compression=settings["storage"]["compression"])
        outputs[name] = path

    for name in FRENCH_FILES:
        key = f"french/{name}"
        if not manifest.verify(key):
            raise ValueError(f"Run data pull first: missing {key}")
        frame, _ = parse_french(_csv_from_zip(Path(manifest.get(key)["path"])), name)
        write(key, frame)
        if name == "ff3_daily":
            write("trading_calendar", frame.select("date"))
    for factor in JKP_FACTORS:
        for frequency in ("monthly", "daily"):
            key = f"jkp/{factor}_{frequency}"
            if not manifest.verify(key):
                raise ValueError(f"Run data pull first: missing {key}")
            write(key, parse_jkp(_csv_from_zip(Path(manifest.get(key)["path"])), factor, frequency))
    for series in settings["data"]["fred_series"]:
        key = f"fred/{series}"
        if not manifest.verify(key):
            raise ValueError(f"Run data pull first: missing {key}")
        write(key, parse_fred(json.loads(Path(manifest.get(key)["path"]).read_text())))
    return outputs
