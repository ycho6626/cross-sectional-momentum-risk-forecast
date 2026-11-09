"""Integrity and credential protection of pinned source snapshots."""


from pathlib import Path

import httpx
import pytest

from momenta.manifest import IntegrityError, Manifest, download


def test_changed_pinned_bytes_fail_instead_of_silently_becoming_a_new_vintage(tmp_path: Path):
    raw = tmp_path / "raw" / "example.zip"
    raw.parent.mkdir()
    raw.write_bytes(b"abc")
    manifest = Manifest(tmp_path)
    entry = manifest.record("example", raw, "https://example.org/data", "2026-10")
    # Published SHA-256 test vector for the ASCII string "abc".
    assert entry["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert manifest.verify("example")
    raw.write_bytes(b"abd")
    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        Manifest(tmp_path).verify("example")
    raw.unlink()
    with pytest.raises(IntegrityError, match="missing"):
        manifest.verify("example")


def test_pinned_download_reuses_original_bytes_and_time_without_network(tmp_path: Path):
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b"abc", headers={"Last-Modified": "2026-10-01"})

    manifest = Manifest(tmp_path)
    raw = tmp_path / "raw" / "example.zip"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        download(manifest, "example", "https://example.org/data", raw, client=client)
        original = manifest.get("example")
        download(manifest, "example", "https://example.org/data", raw, client=client)
    assert len(calls) == 1
    assert manifest.get("example") == original
    assert raw.read_bytes() == b"abc"


def test_interrupted_refresh_preserves_the_existing_snapshot(tmp_path: Path):
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"new snapshot begins"
            raise httpx.ReadError("request included api_key=do-not-echo")

    raw = tmp_path / "raw" / "example.zip"
    raw.parent.mkdir()
    raw.write_bytes(b"abc")
    manifest = Manifest(tmp_path)
    original = manifest.record("example", raw, "https://example.org/data", "2026-09")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=BrokenStream()))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="ReadError") as caught:
            download(
                manifest,
                "example",
                "https://example.org/data?api_key=do-not-echo",
                raw,
                client=client,
                refresh=True,
                max_attempts=1,
            )
    assert "do-not-echo" not in str(caught.value)
    assert raw.read_bytes() == b"abc"
    assert manifest.get("example") == original
    assert not raw.with_suffix(".zip.part").exists()
    assert manifest.verify("example")


def test_saved_source_urls_keep_public_query_parameters_and_omit_credentials(tmp_path: Path):
    manifest = Manifest(tmp_path)
    raw = tmp_path / "raw" / "example.zip"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"abc"))
    with httpx.Client(transport=transport) as client:
        download(
            manifest,
            "example",
            "https://example.org/data",
            raw,
            params={"api_key": "do-not-save", "years": "full"},
            client=client,
        )
    original = manifest.get("example")
    assert original is not None
    assert original["source_url"] == "https://example.org/data?years=full"
    manifest.record(
        "example",
        raw,
        "https://user:do-not-save@example.org/data?token=do-not-save&years=full",
        "release 2026-10",
        documentation_url="https://example.org/readme?signature=do-not-save",
        api_key="do-not-save",
    )
    enriched = manifest.get("example")
    assert enriched is not None
    assert enriched["downloaded_at"] == original["downloaded_at"]
    assert enriched["vintage"] == "release 2026-10"
    assert "do-not-save" not in manifest.path.read_text()


def test_manifest_retains_canonical_source_when_download_redirects(tmp_path: Path):
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.org":
            return httpx.Response(
                302, headers={"Location": "https://storage.example/file?signature=do-not-save"}
            )
        return httpx.Response(200, content=b"abc")

    manifest = Manifest(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        download(
            manifest,
            "example",
            "https://example.org/data?years=full",
            tmp_path / "raw" / "example.zip",
            client=client,
        )
    entry = manifest.get("example")
    assert entry is not None
    assert entry["source_url"] == "https://example.org/data?years=full"
    assert "storage.example" not in manifest.path.read_text()
