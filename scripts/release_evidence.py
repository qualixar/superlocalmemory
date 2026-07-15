#!/usr/bin/env python3
"""Create and verify byte-level evidence for coordinated release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
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

    evidence = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = verify_evidence(evidence, args.artifact_root)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
