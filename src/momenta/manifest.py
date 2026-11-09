"""Pinned raw downloads and result provenance (Methodology)."""


from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

_SECRET_KEY = re.compile(
    r"api[_-]?key|token|secret|password|credential|authorization|signature|^key$|^auth$",
    re.IGNORECASE,
)


_CORE_FIELDS = {"path", "source_url", "sha256", "bytes", "vintage", "downloaded_at"}


class IntegrityError(RuntimeError):
    """A pinned file is absent or differs from its recorded bytes."""


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory, including the original raw ZIP bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_url(url: str) -> str:
    """Retain the source endpoint and public parameters, omitting credentials."""
    parts = urlsplit(url)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query) if not _SECRET_KEY.search(key)]
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Exclude credential fields and sanitize URL metadata before persistence."""
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if _SECRET_KEY.search(key):
            continue
        if isinstance(value, dict):
            value = _safe_metadata(value)
        elif isinstance(value, str) and value.startswith(("https://", "http://")):
            value = sanitize_url(value)
        result[key] = value
    return result


class Manifest:
    """One atomic JSON manifest for all raw sources under a data root."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "raw" / "manifest.json"
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            content = json.loads(self.path.read_text())
            if content.get("schema_version") != 1:
                raise IntegrityError("Unsupported raw-manifest schema")
            self._entries = content["entries"]

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        """Return source IDs and their metadata without exposing mutable state."""
        return json.loads(json.dumps(self._entries))

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a source entry, or None if it has never been pinned."""
        entry = self._entries.get(name)
        return json.loads(json.dumps(entry)) if entry is not None else None

    def verify(self, name: str) -> bool:
        """Verify raw bytes; missing or altered pinned files fail loudly."""
        entry = self.get(name)
        if entry is None:
            return False
        path = Path(entry["path"])
        if not path.is_file():
            raise IntegrityError(
                f"Pinned file is missing for {name}; restore it or refresh explicitly"
            )
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise IntegrityError(f"SHA-256 mismatch for {name}; restore it or refresh explicitly")
        if not str(entry.get("vintage", "")).strip():
            raise IntegrityError(f"Pinned file has no vintage for {name}")
        return True

    def record(
        self,
        name: str,
        path: Path,
        source_url: str,
        vintage: str,
        downloaded_at: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        """Pin bytes, or enrich the same bytes while preserving download timing."""
        return self._record(
            name, path, source_url, vintage, sha256_file(path), downloaded_at, metadata
        )

    def _record(
        self,
        name: str,
        path: Path,
        source_url: str,
        vintage: str,
        digest: str,
        downloaded_at: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        path = path.resolve()
        if not path.is_relative_to(self.root / "raw"):
            raise ValueError("Raw manifest files must be inside data_root/raw")
        if not str(vintage).strip():
            raise ValueError("A non-empty source vintage is required")
        if _CORE_FIELDS.intersection(metadata):
            raise ValueError("Extra metadata must not override manifest integrity fields")
        previous = self._entries.get(name, {})
        same_bytes = previous.get("sha256") == digest
        entry = dict(previous) if same_bytes else {}
        entry.update(_safe_metadata(metadata))
        entry.update(
            path=str(path),
            source_url=sanitize_url(source_url),
            sha256=digest,
            bytes=path.stat().st_size,
            vintage=str(vintage),
            downloaded_at=(
                downloaded_at
                or (previous.get("downloaded_at") if same_bytes else None)
                or datetime.now(UTC).isoformat()
            ),
        )
        updated = {**self._entries, name: entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.part")
        with temporary.open("w") as output:
            json.dump({"schema_version": 1, "entries": updated}, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path)
        self._entries = updated
        return json.loads(json.dumps(entry))


def download(
    manifest: Manifest,
    name: str,
    url: str,
    path: Path,
    *,
    vintage: str | None = None,
    refresh: bool = False,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    max_attempts: int = 3,
    **metadata: Any,
) -> Path:
    """Stream one raw snapshot, reusing a verified pin unless refresh is explicit.

    Partial downloads never replace a prior snapshot. HTTP exceptions are reduced
    to safe messages because the original exception may contain an API key.
    ZIP files remain ZIP files; extraction belongs to the source-specific parser.
    """
    if not refresh and manifest.verify(name):
        entry = manifest.get(name)
        assert entry is not None
        return Path(entry["path"])
    path = path.resolve()
    if not path.is_relative_to(manifest.root / "raw"):
        raise ValueError("Downloads must be inside data_root/raw")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if vintage is not None and not vintage.strip():
        raise ValueError("A non-empty source vintage is required")
    if _CORE_FIELDS.intersection(metadata):
        raise ValueError("Extra metadata must not override manifest integrity fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    source_url = sanitize_url(str(httpx.Request("GET", url, params=params).url))
    own_client = client is None
    if client is None:
        client = httpx.Client(timeout=httpx.Timeout(120, read=300), follow_redirects=True)
    try:
        for attempt in range(max_attempts):
            try:
                digest = hashlib.sha256()
                total = 0
                last_progress = 0
                with client.stream("GET", url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    source_vintage = vintage or response.headers.get("last-modified")
                    with temporary.open("wb") as output:
                        for block in response.iter_bytes(chunk_size=1024 * 1024):
                            output.write(block)
                            digest.update(block)
                            total += len(block)
                            if total - last_progress >= 50_000_000:
                                print(f"{name}: {total / 1e6:.0f} MB downloaded", flush=True)
                                last_progress = total
                        output.flush()
                        os.fsync(output.fileno())
                if total == 0:
                    raise RuntimeError(f"Download failed for {name}: empty response")
                downloaded_at = datetime.now(UTC).isoformat()
                temporary.replace(path)
                manifest._record(
                    name,
                    path,
                    source_url,
                    source_vintage or f"downloaded {downloaded_at}",
                    digest.hexdigest(),
                    downloaded_at,
                    metadata,
                )
                return path
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                failure = f"HTTP {status}"
                retryable = status == 429 or status >= 500
            except httpx.HTTPError as error:
                failure = type(error).__name__
                retryable = True
            finally:
                temporary.unlink(missing_ok=True)
            if not retryable or attempt + 1 == max_attempts:
                raise RuntimeError(f"Download failed for {name}: {failure}") from None
            print(f"{name}: {failure}; retrying ({attempt + 2}/{max_attempts})", flush=True)
            time.sleep(2**attempt)
    finally:
        if own_client:
            client.close()
    raise AssertionError("Unreachable download state")


def provenance(repo_root: Path, config_path: Path, manifest: Manifest) -> dict[str, Any]:
    """Record the config, code state and pinned data for each result file."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {
        "config_hash": sha256_file(config_path),
        "git_hash": commit,
        "git_dirty": dirty,
        "data_manifest_hash": sha256_file(manifest.path),
        "data_vintage": {name: entry["vintage"] for name, entry in manifest.entries.items()},
    }
