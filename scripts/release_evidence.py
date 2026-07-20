#!/usr/bin/env python3
"""Create and verify byte-level evidence for coordinated release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import tomllib
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


EVIDENCE_SCHEMA = "superlocalmemory.release-evidence/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _artifact_kinds(paths: Iterable[Path]) -> set[str]:
    kinds: set[str] = set()
    for path in paths:
        if path.suffix == ".whl":
            kinds.add("wheel")
        elif path.name.endswith(".tar.gz"):
            kinds.add("sdist")
        elif path.suffix == ".tgz":
            kinds.add("npm tarball")
    return kinds


def _component(*, ecosystem: str, name: str, version: str) -> dict:
    normalized = name.lower().replace("_", "-") if ecosystem == "pypi" else name
    encoded_name = urllib.parse.quote(normalized, safe="/")
    encoded_version = urllib.parse.quote(str(version), safe="")
    purl = f"pkg:{ecosystem}/{encoded_name}@{encoded_version}"
    return {
        "type": "library",
        "bom-ref": purl,
        "name": name,
        "version": str(version),
        "purl": purl,
    }


def build_cyclonedx_sbom(
    *,
    version: str,
    commit: str,
    uv_lock: Path,
    package_lock: Path,
    generated_at: str | None = None,
) -> dict:
    """Build a deterministic CycloneDX inventory from both release locks."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a 40-character lowercase Git SHA")
    with uv_lock.open("rb") as stream:
        python_lock = tomllib.load(stream)
    npm_lock = json.loads(package_lock.read_text(encoding="utf-8"))

    components: dict[str, dict] = {}
    for package in python_lock.get("package", []):
        name = str(package.get("name", ""))
        dependency_version = str(package.get("version", ""))
        if not name or not dependency_version or name == "superlocalmemory":
            continue
        component = _component(
            ecosystem="pypi", name=name, version=dependency_version
        )
        components[component["bom-ref"]] = component

    for location, package in npm_lock.get("packages", {}).items():
        if not location or "node_modules/" not in location:
            continue
        name = location.rsplit("node_modules/", 1)[-1]
        dependency_version = str(package.get("version", ""))
        if not name or not dependency_version:
            continue
        component = _component(
            ecosystem="npm", name=name, version=dependency_version
        )
        components[component["bom-ref"]] = component

    serial = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"https://github.com/qualixar/superlocalmemory/{commit}/{version}",
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": generated_at or _utc_now(),
            "component": {
                "type": "application",
                "bom-ref": f"pkg:pypi/superlocalmemory@{version}",
                "name": "superlocalmemory",
                "version": version,
                "purl": f"pkg:pypi/superlocalmemory@{version}",
                "properties": [
                    {"name": "qualixar:git-commit", "value": commit},
                ],
            },
        },
        "components": [components[key] for key in sorted(components)],
    }


def verify_npm_tarball(tarball: Path, expected_version: str) -> list[str]:
    """Verify the exact npm candidate archive without running install hooks."""
    errors: list[str] = []
    required = {
        "package/package.json",
        "package/bin/slm-npm",
        "package/src/superlocalmemory/__init__.py",
        "package/LICENSE",
        "package/NOTICE",
    }
    try:
        with tarfile.open(tarball, "r:gz") as archive:
            members = archive.getmembers()
            names = {member.name.removeprefix("./") for member in members}
            for member in members:
                name = member.name.removeprefix("./")
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    errors.append(f"unsafe npm archive path: {name}")
                if member.issym() or member.islnk() or member.isdev():
                    errors.append(f"unsafe npm archive member type: {name}")
                if name.endswith((".pyc", ".pyo")) or "__pycache__" in pure.parts:
                    errors.append(f"compiled cache in npm archive: {name}")

            missing = sorted(required - names)
            if missing:
                errors.append(f"required npm files missing: {missing}")
            package_member = archive.getmember("package/package.json")
            package_stream = archive.extractfile(package_member)
            if package_stream is None:
                errors.append("package/package.json is not a regular file")
                return errors
            package = json.load(package_stream)
    except (OSError, tarfile.TarError, KeyError, json.JSONDecodeError) as exc:
        return [f"invalid npm tarball: {exc}"]

    if package.get("name") != "superlocalmemory":
        errors.append(f"npm package name mismatch: {package.get('name')!r}")
    if package.get("version") != expected_version:
        errors.append(
            f"npm version mismatch: expected {expected_version!r}, "
            f"got {package.get('version')!r}"
        )
    if package.get("license") != "AGPL-3.0-or-later":
        errors.append(f"npm license mismatch: {package.get('license')!r}")
    bins = package.get("bin", {})
    slm_bin = bins.get("slm", "") if isinstance(bins, dict) else ""
    if str(slm_bin).removeprefix("./") != "bin/slm-npm":
        errors.append("npm slm bin entry is missing or incorrect")
    return errors


def build_evidence(
    *,
    version: str,
    commit: str,
    artifacts: Iterable[Path],
    artifact_root: Path,
    generated_at: str | None = None,
    workflow_run: str | None = None,
) -> dict:
    """Build a deterministic manifest binding a version to exact local bytes."""
    if not version or version.startswith("v"):
        raise ValueError("version must be non-empty and omit the v prefix")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a 40-character lowercase Git SHA")

    root = artifact_root.resolve()
    candidates = [path.resolve() for path in artifacts]
    if len(candidates) != len(set(candidates)):
        raise ValueError("artifact paths must be unique")
    missing_kinds = {"wheel", "sdist", "npm tarball"} - _artifact_kinds(candidates)
    if missing_kinds:
        raise ValueError(f"missing required artifact kind(s): {', '.join(sorted(missing_kinds))}")

    entries: list[dict] = []
    for path in sorted(candidates):
        if not path.is_file():
            raise FileNotFoundError(f"release artifact missing: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact is outside artifact root: {path}") from exc
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "version": version,
        "commit": commit,
        "generated_at": generated_at or _utc_now(),
        "artifacts": entries,
    }
    if workflow_run:
        evidence["workflow_run"] = workflow_run
    return evidence


def verify_evidence(evidence: dict, artifact_root: Path) -> list[str]:
    """Return every manifest/byte mismatch; an empty list means verified."""
    errors: list[str] = []
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        errors.append("unsupported or missing evidence schema")
    if not re.fullmatch(r"[0-9a-f]{40}", str(evidence.get("commit", ""))):
        errors.append("invalid commit SHA")

    root = artifact_root.resolve()
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return [*errors, "artifacts must be a non-empty list"]

    seen: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            errors.append("artifact entry must be an object")
            continue
        relative = str(entry.get("path", ""))
        if relative in seen:
            errors.append(f"duplicate artifact path: {relative}")
            continue
        seen.add(relative)
        path = (root / relative).resolve()
        if path != root and not path.is_relative_to(root):
            errors.append(f"artifact path escapes root: {relative}")
            continue
        if not path.is_file():
            errors.append(f"artifact missing: {relative}")
            continue
        if path.stat().st_size != entry.get("size_bytes"):
            errors.append(f"size mismatch: {relative}")
        if _sha256(path) != entry.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
    return errors


def write_evidence(evidence: dict, artifact_root: Path) -> tuple[Path, Path]:
    """Write the JSON manifest and a sha256sum-compatible checksum file."""
    root = artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "release-evidence.json"
    checksums = root / "SHA256SUMS"
    manifest.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"{entry['sha256']}  {entry['path']}"
        for entry in sorted(evidence["artifacts"], key=lambda item: item["path"])
    ]
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest, checksums


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--version", required=True)
    create.add_argument("--commit", required=True)
    create.add_argument("--artifact-root", required=True, type=Path)
    create.add_argument("--workflow-run")
    create.add_argument("artifacts", nargs="+", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--artifact-root", required=True, type=Path)

    sbom = subparsers.add_parser("sbom")
    sbom.add_argument("--version", required=True)
    sbom.add_argument("--commit", required=True)
    sbom.add_argument("--uv-lock", required=True, type=Path)
    sbom.add_argument("--package-lock", required=True, type=Path)
    sbom.add_argument("--output", required=True, type=Path)

    npm = subparsers.add_parser("verify-npm")
    npm.add_argument("--tarball", required=True, type=Path)
    npm.add_argument("--version", required=True)

    args = parser.parse_args(argv)
    if args.command == "create":
        evidence = build_evidence(
            version=args.version,
            commit=args.commit,
            artifacts=args.artifacts,
            artifact_root=args.artifact_root,
            workflow_run=args.workflow_run,
        )
        manifest, checksums = write_evidence(evidence, args.artifact_root)
        print(json.dumps({"manifest": str(manifest), "checksums": str(checksums)}))
        return 0

    if args.command == "sbom":
        payload = build_cyclonedx_sbom(
            version=args.version,
            commit=args.commit,
            uv_lock=args.uv_lock,
            package_lock=args.package_lock,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.output)
        return 0

    if args.command == "verify-npm":
        errors = verify_npm_tarball(args.tarball, args.version)
    else:
        evidence = json.loads(args.manifest.read_text(encoding="utf-8"))
        errors = verify_evidence(evidence, args.artifact_root)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
