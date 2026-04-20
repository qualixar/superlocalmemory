# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-06 §9.3 / H8, H9

"""Tests for ``scripts/release_manifest.py``.

Covers H8 (every release asset has SHA-256 in manifest), H9 (manifest
shape validated), and deterministic serialization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import release_manifest as rm  # noqa — path injected by conftest


def _seed_asset(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def _expected_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Asset entry
# ---------------------------------------------------------------------------


def test_build_asset_entry_happy(tmp_path):
    path = _seed_asset(tmp_path, "slm-hook-macos-arm64.tar.gz",
                       b"dummy-bytes")
    spec = rm.AssetSpec(
        path=path, platform="macos", arch="arm64",
        signing=rm.SIGNING_APPLE_NOTARIZED,
    )
    entry = rm.build_asset_entry(
        spec, version="3.4.22",
        base_url=rm.DEFAULT_BASE_URL,
    )
    assert entry["name"] == path.name
    assert entry["sha256"] == _expected_sha(path)
    assert entry["size_bytes"] == len(b"dummy-bytes")
    assert entry["signing"] == "apple-notarized"
    assert entry["platform"] == "macos"
    assert entry["arch"] == "arm64"
    assert entry["url"].startswith(
        "https://github.com/qualixar/superlocalmemory/releases/download"
    )
    assert "v3.4.22" in entry["url"]
    assert entry["url"].endswith(path.name)


def test_build_asset_entry_missing_file_raises(tmp_path):
    spec = rm.AssetSpec(
        path=tmp_path / "nope.tar.gz",
        platform="linux", arch="x86_64",
        signing=rm.SIGNING_UNSIGNED,
    )
    with pytest.raises(FileNotFoundError):
        rm.build_asset_entry(spec, version="3.4.22",
                             base_url=rm.DEFAULT_BASE_URL)


def test_build_asset_entry_rejects_bad_platform(tmp_path):
    path = _seed_asset(tmp_path, "weird.tar.gz", b"x")
    spec = rm.AssetSpec(
        path=path, platform="solaris", arch="arm64",
        signing=rm.SIGNING_UNSIGNED,
    )
    with pytest.raises(ValueError, match="unsupported platform"):
        rm.build_asset_entry(spec, version="3.4.22",
                             base_url=rm.DEFAULT_BASE_URL)


def test_build_asset_entry_rejects_bad_signing(tmp_path):
    path = _seed_asset(tmp_path, "slm-hook-linux-x86_64.tar.gz", b"x")
    spec = rm.AssetSpec(
        path=path, platform="linux", arch="x86_64",
        signing="not-a-signing",
    )
    with pytest.raises(ValueError, match="invalid signing"):
        rm.build_asset_entry(spec, version="3.4.22",
                             base_url=rm.DEFAULT_BASE_URL)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _three_assets(tmp_path: Path) -> list[rm.AssetSpec]:
    return [
        rm.AssetSpec(
            path=_seed_asset(tmp_path, "slm-hook-macos-arm64.tar.gz",
                             b"mac-arm"),
            platform="macos", arch="arm64",
            signing=rm.SIGNING_APPLE_NOTARIZED,
        ),
        rm.AssetSpec(
            path=_seed_asset(tmp_path, "slm-hook-linux-x86_64.tar.gz",
                             b"linux-x86"),
            platform="linux", arch="x86_64",
            signing=rm.SIGNING_UNSIGNED,
        ),
        rm.AssetSpec(
            path=_seed_asset(tmp_path,
                             "slm-hook-windows-x86_64-setup.exe",
                             b"win-setup"),
            platform="windows", arch="x86_64",
            signing=rm.SIGNING_AUTHENTICODE,
        ),
    ]


def test_manifest_has_sha256_per_asset(tmp_path):
    manifest = rm.build_manifest(
        "3.4.22", _three_assets(tmp_path),
        released_at="2026-04-17T00:00:00+00:00",
    )
    for entry in manifest.assets:
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64


def test_manifest_shape(tmp_path):
    manifest = rm.build_manifest(
        "3.4.22", _three_assets(tmp_path),
        released_at="2026-04-17T00:00:00+00:00",
    )
    doc = json.loads(rm.serialize(manifest))
    errors = rm.validate_shape(doc)
    assert errors == [], errors
    # Explicit keys.
    assert doc["version"] == "3.4.22"
    assert doc["released_at"].startswith("2026-04-17")
    assert isinstance(doc["assets"], list) and len(doc["assets"]) == 3
    assert len(doc["manifest_sha256_self"]) == 64


def test_manifest_self_sha_is_deterministic(tmp_path):
    m1 = rm.build_manifest(
        "3.4.22", _three_assets(tmp_path),
        released_at="2026-04-17T00:00:00+00:00",
    )
    m2 = rm.build_manifest(
        "3.4.22", _three_assets(tmp_path),
        released_at="2026-04-17T00:00:00+00:00",
    )
    assert m1.manifest_sha256_self == m2.manifest_sha256_self


def test_manifest_requires_version():
    with pytest.raises(ValueError, match="version"):
        rm.build_manifest("", [])


def test_manifest_requires_at_least_one_asset():
    with pytest.raises(ValueError, match="at least one asset"):
        rm.build_manifest("3.4.22", [])


def test_write_manifest_emits_json_and_sig(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    manifest = rm.build_manifest(
        "3.4.22",
        [rm.AssetSpec(
            path=_seed_asset(assets_dir, "slm-hook-linux-arm64.tar.gz",
                             b"linux-arm"),
            platform="linux", arch="arm64",
            signing=rm.SIGNING_UNSIGNED,
        )],
        released_at="2026-04-17T00:00:00+00:00",
    )
    out_dir = tmp_path / "release"
    manifest_path, sig_path = rm.write_manifest(manifest, out_dir)
    assert manifest_path.is_file()
    assert sig_path.is_file()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["version"] == "3.4.22"
    sig_body = sig_path.read_text(encoding="utf-8")
    assert "placeholder" in sig_body
    assert manifest.manifest_sha256_self in sig_body


# ---------------------------------------------------------------------------
# validate_shape — negative cases
# ---------------------------------------------------------------------------


def test_validate_shape_flags_missing_top_level_keys():
    errors = rm.validate_shape({})
    assert any("version" in e for e in errors)
    assert any("assets" in e for e in errors)


def test_validate_shape_flags_non_list_assets():
    errors = rm.validate_shape({
        "version": "3.4.22", "released_at": "now",
        "assets": "nope", "manifest_sha256_self": "x",
        "manifest_signature": "y",
    })
    assert any("assets must be a list" in e for e in errors)


def test_validate_shape_flags_empty_assets():
    errors = rm.validate_shape({
        "version": "3.4.22", "released_at": "now",
        "assets": [], "manifest_sha256_self": "x",
        "manifest_signature": "y",
    })
    assert any("empty" in e for e in errors)


def test_validate_shape_flags_bad_sha256_length():
    doc = {
        "version": "3.4.22", "released_at": "now",
        "assets": [{
            "name": "a.tgz", "url": "https://x/a.tgz",
            "size_bytes": 1, "sha256": "short", "signing": "unsigned",
        }],
        "manifest_sha256_self": "x", "manifest_signature": "y",
    }
    errors = rm.validate_shape(doc)
    assert any("sha256" in e for e in errors)


def test_validate_shape_flags_bad_signing():
    doc = {
        "version": "3.4.22", "released_at": "now",
        "assets": [{
            "name": "a.tgz", "url": "https://x/a.tgz",
            "size_bytes": 1, "sha256": "a" * 64, "signing": "WRONG",
        }],
        "manifest_sha256_self": "x", "manifest_signature": "y",
    }
    errors = rm.validate_shape(doc)
    assert any("invalid signing" in e for e in errors)


def test_validate_shape_flags_missing_field():
    doc = {
        "version": "3.4.22", "released_at": "now",
        "assets": [{"name": "a.tgz"}],
        "manifest_sha256_self": "x", "manifest_signature": "y",
    }
    errors = rm.validate_shape(doc)
    assert any("missing field" in e for e in errors)


def test_validate_shape_flags_non_dict_asset():
    doc = {
        "version": "3.4.22", "released_at": "now",
        "assets": ["not-a-dict"],
        "manifest_sha256_self": "x", "manifest_signature": "y",
    }
    errors = rm.validate_shape(doc)
    assert any("must be an object" in e for e in errors)
