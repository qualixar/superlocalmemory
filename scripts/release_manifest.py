# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-06 §6.1

"""Generate the release ``manifest.json`` for ``slm-hook`` binaries.

LLD reference: ``.backup/active-brain/lld/LLD-06-windows-binary-and-legacy-migration.md``
Section 6.1 (Release manifest).

Shape (per LLD-06 §6.1):

    {
      "version": "3.4.22",
      "released_at": "2026-04-17T00:00:00Z",
      "assets": [
        {
          "name": "slm-hook-macos-arm64.tar.gz",
          "url":  "https://.../manifest.json",
          "size_bytes": 5312000,
          "sha256": "a7c3...",
          "signing": "apple-notarized"
        }, ...
      ],
      "manifest_sha256_self": "...",
      "manifest_signature": "minisign: placeholder"
    }

Minisign signing is OUT of scope at Stage 6 — this module emits a
placeholder and a ``manifest.minisig`` stub. Real signing is a CI job
that layers on top (needs the private key in GitHub Secrets).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Signing envelope strings — one per expected platform. Must match the
# values the dispatcher + postinstall check. Kept as constants so tests
# can lock them.
SIGNING_APPLE_NOTARIZED = "apple-notarized"
SIGNING_AUTHENTICODE = "authenticode"
SIGNING_UNSIGNED = "unsigned"

# Default base URL used when the caller doesn't supply one. Kept in a
# constant so tests can assert the release shape without a network call.
DEFAULT_BASE_URL = (
    "https://github.com/qualixar/superlocalmemory/releases/download"
)

# Allowed platform/arch combos we ship binaries for. Any asset whose
# filename does not encode one of these is rejected.
_VALID_PLATFORMS: frozenset[tuple[str, str]] = frozenset({
    ("macos", "arm64"),
    ("macos", "x86_64"),
    ("linux", "x86_64"),
    ("linux", "arm64"),
    ("windows", "x86_64"),
})


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Input spec for a single release artefact on disk."""

    path: Path
    platform: str
    arch: str
    signing: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """In-memory representation of the manifest to be serialized."""

    version: str
    released_at: str
    assets: list[dict] = field(default_factory=list)
    manifest_sha256_self: str = ""
    manifest_signature: str = "minisign: placeholder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_platform(platform: str, arch: str) -> None:
    if (platform, arch) not in _VALID_PLATFORMS:
        raise ValueError(
            f"unsupported platform/arch: {platform}/{arch}"
        )


def _validate_signing(signing: str) -> None:
    allowed = {SIGNING_APPLE_NOTARIZED, SIGNING_AUTHENTICODE,
               SIGNING_UNSIGNED}
    if signing not in allowed:
        raise ValueError(
            f"invalid signing value: {signing!r} (allowed: {allowed})"
        )


def _asset_url(base_url: str, version: str, filename: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/v{version}/{filename}"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_asset_entry(
    spec: AssetSpec,
    *,
    version: str,
    base_url: str,
) -> dict:
    """Compute a single manifest asset entry."""
    if not spec.path.is_file():
        raise FileNotFoundError(f"asset missing: {spec.path}")
    _validate_platform(spec.platform, spec.arch)
    _validate_signing(spec.signing)

    return {
        "name": spec.path.name,
        "url": _asset_url(base_url, version, spec.path.name),
        "size_bytes": spec.path.stat().st_size,
        "sha256": _sha256_file(spec.path),
        "signing": spec.signing,
        "platform": spec.platform,
        "arch": spec.arch,
    }


def build_manifest(
    version: str,
    assets: list[AssetSpec],
    *,
    base_url: str = DEFAULT_BASE_URL,
    released_at: str | None = None,
) -> Manifest:
    """Assemble the full Manifest object (including self SHA)."""
    if not version:
        raise ValueError("version is required")
    if not assets:
        raise ValueError("at least one asset is required")

    entries: list[dict] = [
        build_asset_entry(spec, version=version, base_url=base_url)
        for spec in assets
    ]
    payload = {
        "version": version,
        "released_at": released_at or _utcnow_iso(),
        "assets": entries,
    }
    # Compute the manifest-self SHA over the canonical JSON WITHOUT the
    # self-SHA field (otherwise it would be recursive).
    serialized = json.dumps(payload, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    self_sha = hashlib.sha256(serialized).hexdigest()

    return Manifest(
        version=version,
        released_at=payload["released_at"],
        assets=entries,
        manifest_sha256_self=self_sha,
        manifest_signature="minisign: placeholder",
    )


def serialize(manifest: Manifest) -> str:
    """Deterministic JSON serialization for on-disk storage."""
    doc = {
        "version": manifest.version,
        "released_at": manifest.released_at,
        "assets": manifest.assets,
        "manifest_sha256_self": manifest.manifest_sha256_self,
        "manifest_signature": manifest.manifest_signature,
    }
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def write_manifest(
    manifest: Manifest,
    dest_dir: Path,
) -> tuple[Path, Path]:
    """Write ``manifest.json`` + a stub ``manifest.minisig`` to ``dest_dir``.

    Returns the two paths (for caller convenience).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"
    sig_path = dest_dir / "manifest.minisig"

    manifest_path.write_text(serialize(manifest), encoding="utf-8")
    # Placeholder — real minisig produced by a separate CI job.
    sig_path.write_text(
        "untrusted comment: placeholder - real signature added by CI\n"
        f"manifest_sha256_self: {manifest.manifest_sha256_self}\n",
        encoding="utf-8",
    )
    return manifest_path, sig_path


# ---------------------------------------------------------------------------
# Validation helpers (used by tests + postinstall)
# ---------------------------------------------------------------------------

def validate_shape(doc: dict) -> list[str]:
    """Return a list of validation errors on a loaded manifest dict."""
    errors: list[str] = []
    for key in ("version", "released_at", "assets",
                "manifest_sha256_self", "manifest_signature"):
        if key not in doc:
            errors.append(f"missing top-level key: {key}")

    assets = doc.get("assets")
    if not isinstance(assets, list):
        errors.append("assets must be a list")
        return errors
    if not assets:
        errors.append("assets list is empty")

    for i, entry in enumerate(assets):
        if not isinstance(entry, dict):
            errors.append(f"asset[{i}] must be an object")
            continue
        for field_name in ("name", "url", "size_bytes",
                           "sha256", "signing"):
            if field_name not in entry:
                errors.append(
                    f"asset[{i}] missing field: {field_name}"
                )
        sha = entry.get("sha256", "")
        if not isinstance(sha, str) or len(sha) != 64:
            errors.append(f"asset[{i}] sha256 must be 64-char hex")
        signing = entry.get("signing")
        if signing is not None:
            try:
                _validate_signing(signing)
            except ValueError as exc:
                errors.append(f"asset[{i}] {exc}")
    return errors


__all__ = (
    "AssetSpec",
    "Manifest",
    "SIGNING_APPLE_NOTARIZED",
    "SIGNING_AUTHENTICODE",
    "SIGNING_UNSIGNED",
    "DEFAULT_BASE_URL",
    "build_asset_entry",
    "build_manifest",
    "serialize",
    "write_manifest",
    "validate_shape",
)
