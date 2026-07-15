#!/usr/bin/env python3
"""Prepare and validate the frozen payload carried by the macOS DMG.

This helper deliberately does not build Python packages, contact a registry, or
read signing credentials.  The release input is exactly one local wheel whose
embedded metadata must match the version in the checked-out ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any


PRODUCT = "SuperLocalMemory"
PACKAGE = "superlocalmemory"
LICENSE_ID = "AGPL-3.0-or-later"
MANIFEST_NAME = "ARTIFACT-MANIFEST.json"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _project_version(project_root: Path) -> str:
    config_path = project_root / "pyproject.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    try:
        name = str(config["project"]["name"])
        version = str(config["project"]["version"])
        license_id = str(config["project"]["license"]["text"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid project metadata in {config_path}") from exc
    if _canonical_name(name) != PACKAGE:
        raise ValueError(f"unexpected project package: {name}")
    if license_id != LICENSE_ID:
        raise ValueError(
            f"project license must be {LICENSE_ID}, found {license_id}"
        )
    return version


def _wheel_identity(wheel: Path) -> tuple[str, str]:
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"frozen wheel does not exist: {wheel}")
    try:
        with zipfile.ZipFile(wheel) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "wheel must contain exactly one .dist-info/METADATA file"
                )
            metadata = BytesParser().parsebytes(archive.read(candidates[0]))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid wheel archive: {wheel}") from exc
    name = metadata.get("Name", "")
    version = metadata.get("Version", "")
    if _canonical_name(name) != PACKAGE:
        raise ValueError(f"wheel package must be {PACKAGE}, found {name or '<empty>'}")
    if not version:
        raise ValueError("wheel metadata has no Version")
    return name, version


def _write_command(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _inventory(volume: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in volume.rglob("*") if candidate.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(volume).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "executable": bool(path.stat().st_mode & stat.S_IXUSR),
            }
        )
    return entries


def prepare_stage(
    *,
    wheel: Path,
    project_root: Path,
    stage_dir: Path,
) -> dict[str, Any]:
    """Create the deterministic DMG source directory from one frozen wheel."""
    wheel = wheel.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    if stage_dir.exists():
        raise ValueError(f"stage directory already exists: {stage_dir}")

    project_version = _project_version(project_root)
    _, wheel_version = _wheel_identity(wheel)
    if wheel_version != project_version:
        raise ValueError(
            f"wheel version {wheel_version} does not match project version "
            f"{project_version}"
        )

    volume = stage_dir / PRODUCT
    artifacts = volume / "artifacts"
    scripts = volume / "scripts"
    artifacts.mkdir(parents=True)
    scripts.mkdir()

    copied_wheel = artifacts / wheel.name
    shutil.copyfile(wheel, copied_wheel)
    shutil.copyfile(project_root / "scripts" / "install.sh", scripts / "install.sh")
    (scripts / "install.sh").chmod(
        (scripts / "install.sh").stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )
    for notice in ("LICENSE", "NOTICE", "ATTRIBUTION.md"):
        shutil.copyfile(project_root / notice, volume / notice)

    install_command = f'''#!/usr/bin/env bash
set -euo pipefail
readonly HERE="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd -P)"
exec "$HERE/scripts/install.sh" install --package "$HERE/artifacts/{wheel.name}" "$@"
'''
    uninstall_command = '''#!/usr/bin/env bash
set -euo pipefail
readonly HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$HERE/scripts/install.sh" uninstall "$@"
'''
    _write_command(volume / "INSTALL.command", install_command)
    _write_command(volume / "UNINSTALL.command", uninstall_command)

    readme = f"""SuperLocalMemory {project_version} for macOS
=========================================

This disk image contains one frozen Python wheel. Its checksum and complete
payload inventory are recorded in {MANIFEST_NAME}.

Install
-------
Double-click INSTALL.command, or run it in Terminal. The installer requires an
existing uv or pipx installation and creates an isolated user-scoped tool
environment. It does not bootstrap a package manager or edit IDE settings.
After installation, run `slm setup` to choose integrations explicitly.

Uninstall
---------
Double-click UNINSTALL.command, or run it in Terminal. Uninstall removes the
isolated application package but preserves SuperLocalMemory runtime data.

Security and release state
--------------------------
The adjacent external .manifest.json and .sha256 files describe the final DMG
checksum, signing state, and notarization state. A release artifact is valid
only when strict validation confirms both signing and notarization.

License: GNU Affero General Public License v3.0 or later ({LICENSE_ID})
Repository: https://github.com/qualixar/superlocalmemory
"""
    (volume / "README-INSTALLATION.txt").write_text(
        readme, encoding="utf-8", newline="\n"
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "product": PRODUCT,
        "package": PACKAGE,
        "version": project_version,
        "license": LICENSE_ID,
        "source": "frozen-local-wheel",
        "wheel": {
            "name": wheel.name,
            "size_bytes": copied_wheel.stat().st_size,
            "sha256": _sha256(copied_wheel),
        },
        "installer": {
            "manager": "uv-or-pipx",
            "network_bootstrap": False,
            "automatic_ide_mutation": False,
            "preserves_runtime_data_on_uninstall": True,
        },
        "files": _inventory(volume),
    }
    (volume / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_stage(volume)
    return manifest


def validate_stage(volume: Path) -> dict[str, Any]:
    """Verify every staged file against the internal artifact manifest."""
    manifest_path = volume / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing {MANIFEST_NAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported artifact manifest schema")
    if manifest.get("product") != PRODUCT or manifest.get("package") != PACKAGE:
        raise ValueError("artifact manifest identifies the wrong product")
    if manifest.get("license") != LICENSE_ID:
        raise ValueError("artifact manifest has the wrong license")

    expected_files = {entry["path"]: entry for entry in manifest.get("files", [])}
    actual_files = {
        path.relative_to(volume).as_posix(): path
        for path in volume.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if set(expected_files) != set(actual_files):
        raise ValueError("staged file inventory does not match artifact manifest")
    for relative, path in actual_files.items():
        expected = expected_files[relative]
        if path.stat().st_size != expected.get("size_bytes"):
            raise ValueError(f"checksum or size mismatch for {relative}")
        if _sha256(path) != expected.get("sha256"):
            raise ValueError(f"checksum mismatch for {relative}")
        is_executable = bool(path.stat().st_mode & stat.S_IXUSR)
        if is_executable != expected.get("executable"):
            raise ValueError(f"executable mode mismatch for {relative}")

    wheel = manifest.get("wheel", {})
    wheel_path = volume / "artifacts" / str(wheel.get("name", ""))
    if not wheel_path.is_file() or _sha256(wheel_path) != wheel.get("sha256"):
        raise ValueError("frozen wheel checksum does not match artifact manifest")
    _, wheel_version = _wheel_identity(wheel_path)
    if wheel_version != manifest.get("version"):
        raise ValueError("frozen wheel version does not match artifact manifest")
    return manifest


def _sidecar_paths(dmg: Path) -> tuple[Path, Path]:
    return (
        dmg.with_name(f"{dmg.name}.manifest.json"),
        dmg.with_name(f"{dmg.name}.sha256"),
    )


def write_release_sidecars(
    *,
    dmg: Path,
    version: str,
    signed: bool,
    notarized: bool,
) -> tuple[Path, Path]:
    """Write checksum/provenance sidecars after all DMG mutations finish."""
    if not dmg.is_file():
        raise ValueError(f"DMG does not exist: {dmg}")
    if notarized and not signed:
        raise ValueError("a notarized DMG must also be signed")
    digest = _sha256(dmg)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "product": PRODUCT,
        "version": version,
        "dmg": {
            "name": dmg.name,
            "size_bytes": dmg.stat().st_size,
            "sha256": digest,
        },
        "signing": "developer-id" if signed else "unsigned",
        "notarization": "apple-notarized" if notarized else "not-submitted",
        "release_ready": bool(signed and notarized),
    }
    manifest_path, checksum_path = _sidecar_paths(dmg)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_path.write_text(
        f"{digest}  {dmg.name}\n", encoding="utf-8", newline="\n"
    )
    return manifest_path, checksum_path


def validate_release_sidecars(
    dmg: Path, *, require_release_ready: bool
) -> dict[str, Any]:
    """Validate final DMG bytes and truthfully enforce release state."""
    manifest_path, checksum_path = _sidecar_paths(dmg)
    if not dmg.is_file() or not manifest_path.is_file() or not checksum_path.is_file():
        raise ValueError("DMG and both release sidecars are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported DMG release manifest schema")
    if manifest.get("product") != PRODUCT:
        raise ValueError("DMG release manifest identifies the wrong product")
    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("DMG release manifest has no version")
    expected_name = f"SuperLocalMemory-v{version}-macos-universal.dmg"
    if dmg.name != expected_name:
        raise ValueError(
            f"DMG filename does not match manifest version: expected {expected_name}"
        )
    digest = _sha256(dmg)
    dmg_record = manifest.get("dmg", {})
    if dmg_record.get("name") != dmg.name:
        raise ValueError("DMG filename does not match release manifest")
    if dmg_record.get("size_bytes") != dmg.stat().st_size:
        raise ValueError("DMG size does not match release manifest")
    if dmg_record.get("sha256") != digest:
        raise ValueError("DMG checksum does not match release manifest")
    if checksum_path.read_text(encoding="utf-8") != f"{digest}  {dmg.name}\n":
        raise ValueError("DMG checksum sidecar does not match")
    signed = manifest.get("signing") == "developer-id"
    notarized = manifest.get("notarization") == "apple-notarized"
    if manifest.get("release_ready") != bool(signed and notarized):
        raise ValueError("release_ready is inconsistent with signing state")
    if require_release_ready and not signed:
        raise ValueError("DMG is not signed with Developer ID")
    if require_release_ready and not notarized:
        raise ValueError("DMG is not notarized by Apple")
    return manifest


def validate_release_pair(
    volume: Path,
    dmg: Path,
    *,
    require_release_ready: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind mounted payload identity to the final container provenance."""
    artifact = validate_stage(volume)
    release = validate_release_sidecars(
        dmg, require_release_ready=require_release_ready
    )
    if artifact.get("version") != release.get("version"):
        raise ValueError("mounted wheel version does not match DMG release version")
    return artifact, release


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--wheel", type=Path, required=True)
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--stage-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate-stage")
    validate.add_argument("--volume", type=Path, required=True)

    sidecars = subparsers.add_parser("sidecars")
    sidecars.add_argument("--dmg", type=Path, required=True)
    sidecars.add_argument("--version", required=True)
    sidecars.add_argument("--signed", action="store_true")
    sidecars.add_argument("--notarized", action="store_true")

    release = subparsers.add_parser("validate-sidecars")
    release.add_argument("--dmg", type=Path, required=True)
    release.add_argument("--require-release-ready", action="store_true")

    pair = subparsers.add_parser("validate-pair")
    pair.add_argument("--volume", type=Path, required=True)
    pair.add_argument("--dmg", type=Path, required=True)
    pair.add_argument("--require-release-ready", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        manifest = prepare_stage(
            wheel=args.wheel,
            project_root=args.project_root,
            stage_dir=args.stage_dir,
        )
    elif args.command == "validate-stage":
        manifest = validate_stage(args.volume)
    elif args.command == "sidecars":
        paths = write_release_sidecars(
            dmg=args.dmg,
            version=args.version,
            signed=args.signed,
            notarized=args.notarized,
        )
        print("\n".join(str(path) for path in paths))
        return 0
    elif args.command == "validate-sidecars":
        manifest = validate_release_sidecars(
            args.dmg,
            require_release_ready=args.require_release_ready,
        )
    else:
        artifact, release = validate_release_pair(
            args.volume,
            args.dmg,
            require_release_ready=args.require_release_ready,
        )
        manifest = {"artifact": artifact, "release": release}
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
