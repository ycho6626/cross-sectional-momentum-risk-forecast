"""Project settings and bounded DuckDB connections (Methodology)."""


import tomllib
from pathlib import Path
from typing import Any

import duckdb

REPO = Path(__file__).resolve().parents[2]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Read the project settings; default paths are anchored to the repository."""
    path = path or REPO / "config/settings.toml"
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    settings["config_path"] = str(path.resolve())
    return settings


def connect(settings: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    """Limit DuckDB to the approved memory, threads and SSD spill location."""
    temp = Path(settings["data"]["data_root"]) / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(
        config={
            "memory_limit": settings["storage"]["memory_limit"],
            "threads": settings["storage"]["threads"],
            "temp_directory": str(temp),
            "preserve_insertion_order": False,
        }
    )


def sql_path(path: Path | str) -> str:
    """Quote a local path as a SQL string literal."""
    return "'" + str(path).replace("'", "''") + "'"
