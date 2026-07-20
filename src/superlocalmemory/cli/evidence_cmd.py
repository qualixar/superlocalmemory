# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""CLI for versioned evidence export, verification, import, and rebuild."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path


def _database():
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager

    config = SLMConfig.load()
    return DatabaseManager(config.db_path)


def _emit(value: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, sort_keys=True))
        return
    for key, item in value.items():
        print(f"{key}: {item}")


def cmd_evidence(args: Namespace) -> None:
    """Dispatch ``slm evidence`` without implicit destructive behavior."""
    from superlocalmemory.core.evidence_bundle import (
        export_evidence_bundle,
        import_evidence_bundle,
        rebuild_derived_state,
        verify_evidence_bundle,
    )

    action = getattr(args, "evidence_command", None)
    as_json = bool(getattr(args, "json", False))
    profile = str(getattr(args, "profile", "default") or "default")

    if action == "verify":
        report = verify_evidence_bundle(args.bundle)
        _emit({
            "valid": report.valid,
            "bundle_id": report.bundle_id,
            "counts": report.counts,
            "errors": list(report.errors),
            "warnings": list(report.warnings),
        }, as_json=as_json)
        if not report.valid:
            raise SystemExit(1)
        return

    if action == "export":
        db = _database()
        if not Path(db.db_path).is_file():
            raise SystemExit(f"memory database not found: {db.db_path}")
        manifest = export_evidence_bundle(db, profile, args.destination)
        _emit({
            "bundle_id": manifest["bundle_id"],
            "profile_id": profile,
            "destination": str(Path(args.destination).resolve()),
            "unresolved_source_links": manifest["unresolved_source_links"],
        }, as_json=as_json)
        return

    if action == "import":
        if not bool(getattr(args, "execute", False)):
            print(
                "Dry run only. Verify the bundle, then re-run with --execute; "
                "replacement also requires --rollback-dir."
            )
            return
        report = import_evidence_bundle(
            _database(),
            args.bundle,
            target_profile_id=profile,
            replace=bool(getattr(args, "replace", False)),
            rollback_dir=getattr(args, "rollback_dir", None),
        )
        _emit({
            "valid": report.valid,
            "bundle_id": report.bundle_id,
            "profile_id": profile,
            "counts": report.counts,
        }, as_json=as_json)
        return

    if action == "rebuild":
        if not bool(getattr(args, "execute", False)):
            print(
                "Dry run only. Re-run with --execute to rebuild derived lexical state."
            )
            return
        result = rebuild_derived_state(_database(), profile)
        _emit({"profile_id": profile, **result}, as_json=as_json)
        return

    raise SystemExit("choose an evidence subcommand: export, verify, import, or rebuild")


__all__ = ["cmd_evidence"]
