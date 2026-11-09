"""Pinned full-history source pulls and streaming raw CSV staging (Methodology)."""


import json
import shutil
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from momenta.manifest import Manifest, download
from momenta.settings import REPO, connect, sql_path


def extract_csv(archive: Path, destination: Path) -> Path:
    """Extract exactly one CSV without trusting archive member paths."""
    target = destination / (archive.stem + ".csv")
    if target.exists():
        return target
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = [n for n in zipped.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {archive.name}; found {len(members)}")
        partial = target.with_suffix(".csv.part")
        with zipped.open(members[0]) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        partial.replace(target)
    return target


def stage_years(settings: dict[str, Any], manifest: Manifest, table: str) -> dict:
    """Stream a full CSV to year partitions, then process each year independently."""
    root = Path(settings["data"]["data_root"])
    entry = manifest.get(f"sharadar/{table}")
    if entry is None:
        raise ValueError(f"Missing pinned Sharadar {table}")
    stage = root / "raw/staged" / table
    receipt = stage / "_receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["source_sha256"] == entry["sha256"]:
            return saved
        shutil.rmtree(stage)
    archive = root / "raw/sharadar" / f"{table}.zip"
    csv = extract_csv(archive, root / "raw/extracted")
    stage.mkdir(parents=True, exist_ok=True)
    with connect(settings) as con:
        # COPY's bounded CSV scan only demultiplexes rows; all calculations are yearly.
        con.execute(f"""COPY (
            SELECT *, year(date::DATE) AS source_year
            FROM read_csv({sql_path(csv)}, all_varchar=true)
        ) TO {sql_path(stage)} (FORMAT PARQUET, PARTITION_BY(source_year),
          COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)""")
        count, first, last = con.execute(f"""SELECT count(*), min(date::DATE), max(date::DATE)
            FROM read_parquet({sql_path(stage / "source_year=*/*.parquet")})""").fetchone()
    saved = {
        "source_sha256": entry["sha256"],
        "rows": count,
        "first_date": str(first),
        "last_date": str(last),
    }
    receipt.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def pull(settings: dict[str, Any]) -> None:
    """Fetch or verify pinned source inputs, stocks first, and assert full SEP history."""
    from momenta.reference_data import pull_references

    root = Path(settings["data"]["data_root"])
    manifest = Manifest(root)
    secrets = dotenv_values(REPO / ".env")
    key = secrets.get("SHARADAR_API_KEY")
    if not key:
        raise ValueError("Missing SHARADAR_API_KEY in .env")
    for table in settings["data"]["sharadar_tables"]:
        print(f"Pull Sharadar {table}", flush=True)
        path = root / "raw/sharadar" / f"{table}.zip"
        download(
            manifest,
            f"sharadar/{table}",
            settings["data"]["sharadar_base_url"] + "/" + table,
            path,
            params={"api_key": key, "years": "full"},
        )
        if table == "stocks":
            print("Stage stocks by year and verify full-history entitlement", flush=True)
            receipt = stage_years(settings, manifest, table)
            required = date.fromisoformat(settings["data"]["sep_required_start"])
            if date.fromisoformat(receipt["first_date"]) > required:
                raise ValueError(
                    f"SEP starts {receipt['first_date']}; full history through "
                    f"{required} is required. Check the Sharadar entitlement."
                )
            print(f"SEP: {receipt['first_date']} through {receipt['last_date']}", flush=True)
    pull_references(settings, manifest)
