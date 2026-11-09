# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Shared I/O helpers and compliance constants for ``slm gdpr`` subcommands.

Separated from gdpr_cmd.py to honour the 800-line file cap (coding-style.md).
All symbols here are internal (prefixed ``_``) or ALL_CAPS constants.
"""

from __future__ import annotations

import json as _json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _die(message: str, code: int = 1) -> None:
    """Print an actionable error to stderr and exit non-zero."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def _print_json(data: Any) -> None:
    """Emit pretty-printed JSON to stdout (evidence-pipeline format)."""
    print(_json.dumps(data, indent=2, default=str))


def _json_envelope(
    command: str, *, data: dict | None = None, error: dict | None = None,
    success: bool | None = None,
) -> dict:
    """Build the standard agent-native JSON envelope used across all SLM CLIs.

    ``success`` defaults to "no error". A command whose work partly happened
    (an incomplete erasure) passes ``success=False`` together with both an
    ``error`` and the ``data`` of what did happen, so the envelope says exactly
    what was and was not done and agrees with the exit code.
    """
    from superlocalmemory.cli.json_output import _get_version

    env: dict = {
        "success": (error is None) if success is None else bool(success),
        "command": command,
        "version": _get_version(),
    }
    if error is not None:
        env["error"] = error
    if error is None or data is not None:
        env["data"] = data if data is not None else {}
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Erasure result text
# ─────────────────────────────────────────────────────────────────────────────

def _print_erase_text(profile: str, statement: dict, ok: bool) -> None:
    """Say what was removed and what remains, in the words a person can act on."""
    if ok:
        status = "COMPLETE"
    elif statement["complete"]:
        status = ("COMPLETE, BUT NOT PROVABLE - the data is gone and the "
                  "tamper-evident receipt did not persist")
    else:
        status = "INCOMPLETE"
    print(f"Art.17 erasure for profile '{profile}': {status}")
    if statement["removed"]:
        print("  Removed:")
        for key, value in statement["removed"].items():
            print(f"    {key}: {value}")
    if statement["remains"]:
        print("  Still to do:")
        for item in statement["remains"]:
            print(f"    - {item['what']}")
            print(f"      {item['retry']}")
    print("  Verify receipts: slm gdpr verify --profile", profile)


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _data_root() -> Path:
    """Resolve the canonical SLM data root (honours SLM_DATA_DIR env var)."""
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def _db_path() -> Path:
    return _data_root() / "memory.db"


def _audit_chain_path() -> Path:
    return _data_root() / "audit_chain.db"


# ─────────────────────────────────────────────────────────────────────────────
# Export delivery
# ─────────────────────────────────────────────────────────────────────────────

def stream_export_to_file(gdpr: Any, profile: str, out_path: Path, *,
                          include_embeddings: bool) -> dict:
    """Stream an Art.15/20 export into ``out_path``; the file appears only when whole.

    The export is written to ``<out>.part`` (owner-only) and renamed on success,
    so a failure never leaves a truncated file that looks like a finished export.
    """
    from superlocalmemory.infra.private_files import open_private_new

    part = out_path.with_name(out_path.name + ".part")
    fd = open_private_new(part)
    try:
        with os.fdopen(fd, "w+b") as fp:
            summary = gdpr.write_profile_export(profile, fp, include_embeddings=include_embeddings)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(part, out_path)  # a new owner-only file replaces any broader one, or a link
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return summary


def stream_export_to_stdout(gdpr: Any, profile: str, *, include_embeddings: bool,
                            envelope: bool) -> dict:
    """Stream an export to stdout, optionally inside the standard JSON envelope.

    Nothing reaches stdout until the export has been read in full, so a failure
    prints no half of a document.
    """
    from superlocalmemory.compliance.export_stream import copy_file

    with tempfile.TemporaryFile("w+b") as spool:
        summary = gdpr.write_profile_export(profile, spool, include_embeddings=include_embeddings)
        out = sys.stdout
        if envelope:
            head = _json_envelope("gdpr-export", data={})
            head.pop("data")
            out.write(_json.dumps(head, indent=2)[:-2] + ',\n  "data": ')
        out.flush()
        copy_file(spool, out)
        if envelope:
            out.write("}\n")
        out.flush()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# GDPR compliance constants (DPO-facing, surfaced in ``status`` output)
# ─────────────────────────────────────────────────────────────────────────────

# Known gaps in this release — reported honestly so a DPO can plan remediation.
# Gaps C1 and C2 (cross-device erase verification and audit-log replay) remain open.
KNOWN_GAPS = [
    {
        "ref": "C1",
        "summary": "backup copies are not rewritten by an erasure",
        "detail": (
            "Backup snapshots (backups/, pre-migration-snapshots/, pre-restore/) "
            "that hold an erased profile, or cannot be read to show they do not, "
            "are recorded as erasure obligations and the erasure reports itself "
            "incomplete while any exists. A backup is never rewritten (its "
            "checksums would stop matching); the erasure is replayed on every "
            "restore of it, and the obligation ends only when the backup is "
            "deleted or re-read clean. Age does not end it."
        ),
    },
    {
        "ref": "C2",
        "summary": "a shared code graph is only partly erased",
        "detail": (
            "code_graph.db has no owner column. When several profiles exist, an "
            "erasure removes the links between the erased profile's memories and "
            "code, and keeps the repository structure, which describes the "
            "source rather than the person. The result says so."
        ),
    },
]

# Art. coverage flags that compliance/gdpr.py actively implements.
ART_COVERAGE = {
    "art15_right_to_access": True,
    "art17_right_to_erasure": True,
    "art20_right_to_portability": True,
    "backup_scope": "outstanding_obligation_recorded_on_receipt",
    "audit_chain": "hash-chained tamper-evident",
    "erasure_fails_closed": True,
}
