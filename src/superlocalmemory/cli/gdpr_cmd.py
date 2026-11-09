# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm gdpr`` — GDPR subject-rights CLI for enterprise DPOs.

Subcommands
-----------
  status   Compliance posture: receipts, audit counts, known gaps.
             Safe read-only. Never prints raw memory content.
  export   Art.15/20 subject access / data portability.
             Requires --profile. Writes JSON to --output FILE or stdout.
  erase    Art.17 right to erasure — IRREVERSIBLE.
             Requires --profile PROFILE AND --yes (both mandatory).
             Default output with neither flag is a dry-run preview.
  verify   Check HMAC integrity of an erasure receipt.
             Exit 0 = VERIFIED, 1 = TAMPERED, 2 = NOT_FOUND.

Every subcommand accepts --json for evidence-pipeline integration.
Exit codes: 0 success, 1 erasure/verify failure, 2 refusal/not-found.

Safety design (DPO-grade):
  - erase is the only irreversible operation.  It refuses to run unless BOTH
    --profile PROFILE AND --yes are supplied.  A dry-run is always printed first
    when --dry-run is given (or when --yes is absent).
  - --json never emits raw memory content in status or verify output.
  - verify exits non-zero on tampered receipts so a pipeline can branch on it.
  - status honestly reports known compliance gaps (backups, code_graph) rather
    than claiming completeness it cannot support.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

# I/O helpers and GDPR compliance constants extracted to gdpr_io.py to keep
# this file under the 800-line hard cap (coding-style.md).
from superlocalmemory.cli.gdpr_io import (
    ART_COVERAGE as _ART_COVERAGE,
)
from superlocalmemory.cli.gdpr_io import (
    KNOWN_GAPS as _KNOWN_GAPS,
)
from superlocalmemory.cli.gdpr_io import (
    _audit_chain_path,
    _data_root,
    _die,
    _json_envelope,
    _print_erase_text,
    _print_json,
    stream_export_to_file,
    stream_export_to_stdout,
)

# ─────────────────────────────────────────────────────────────────────────────
# status


def _cmd_gdpr_status(args: Namespace) -> None:
    """``slm gdpr status [--profile P] [--json]``."""
    use_json = getattr(args, "json", False)
    profile = getattr(args, "profile", None)

    root = _data_root()
    db_path = root / "memory.db"
    db_exists = db_path.exists()

    receipts: list[dict] = []
    audit_count = 0
    audit_recent: list[dict] = []
    profiles_found: list[str] = []

    if db_exists:
        try:
            from superlocalmemory.storage.memory_write import memory_read
            with memory_read(db_path) as conn:
                conn.row_factory = _json_row_factory(conn)
                # Receipts — never include audit_hash (HMAC key material)
                scope_clause = "WHERE profile_id = ?" if profile else ""
                scope_params = (profile,) if profile else ()
                rows = conn.execute(
                    f"SELECT erasure_id, profile_id, subject_type, subject_id, "
                    f"requested_by, fact_count, state, all_erased, requested_at, "
                    f"completed_at FROM erasure_receipts "
                    f"{scope_clause} ORDER BY completed_at DESC LIMIT 50",
                    scope_params,
                ).fetchall()
                for r in rows:
                    receipts.append(_safe_receipt_row(r))
        except Exception:
            pass  # Table absent on a fresh install — not an error

        try:
            from superlocalmemory.storage.memory_write import memory_read
            with memory_read(db_path) as conn:
                # Profile list
                profile_rows = conn.execute(
                    "SELECT profile_id FROM profiles"
                ).fetchall()
                profiles_found = [r[0] for r in profile_rows if r]
        except Exception:
            pass

    if db_exists:
        try:
            from superlocalmemory.compliance.audit import AuditChain
            chain = AuditChain(str(_audit_chain_path()))
            scope_filter = {"profile_id": profile} if profile else {}
            audit_recent = chain.query(limit=10, **scope_filter)
            audit_count = chain.count(**scope_filter)
        except Exception:
            pass

    data = {
        "data_root": str(root),
        "db_exists": db_exists,
        "active_profile": profile or "default",
        "profiles": profiles_found,
        "receipts": {
            "count": len(receipts),
            "entries": receipts,
        },
        "audit": {
            "count": audit_count,
            "recent_10": _sanitise_audit(audit_recent),
        },
        "coverage": _ART_COVERAGE,
        "known_gaps": _KNOWN_GAPS,
        "next_actions": [
            "slm gdpr export --profile PROFILE --output export.json",
            "slm gdpr erase --profile PROFILE --dry-run",
            "slm gdpr verify --receipt-id RECEIPT_ID",
        ],
    }

    if use_json:
        _print_json(_json_envelope("gdpr-status", data=data))
    else:
        _print_human_status(data)


def _json_row_factory(conn):
    """Return a row_factory that makes rows indexable by position."""
    import sqlite3
    return sqlite3.Row


def _safe_receipt_row(row) -> dict:
    """Convert an erasure_receipts row to a safe-to-emit dict (no HMAC)."""
    try:
        keys = (
            "erasure_id", "profile_id", "subject_type", "subject_id",
            "requested_by", "fact_count", "state", "all_erased",
            "requested_at", "completed_at",
        )
        if hasattr(row, "keys"):
            return {k: row[k] for k in keys if k in row.keys()}
        return dict(zip(keys, row))
    except Exception:
        return {}


def _sanitise_audit(events: list) -> list[dict]:
    """Strip any raw content from audit events; keep only metadata fields."""
    safe: list[dict] = []
    _ALLOWED = {
        "event_id", "operation", "agent_id", "profile_id",
        "timestamp", "created_at",
    }
    for ev in events:
        if isinstance(ev, dict):
            safe.append({k: v for k, v in ev.items() if k in _ALLOWED})
        else:
            safe.append({})
    return safe


def _print_human_status(data: dict) -> None:
    root = data["data_root"]
    db = "EXISTS" if data["db_exists"] else "NOT FOUND"
    print("\nGDPR Compliance Status")
    print(f"  Data root : {root}")
    print(f"  Database  : {db}")
    print(f"  Profiles  : {', '.join(data['profiles']) or '(none)'}")
    print(f"  Receipts  : {data['receipts']['count']}")
    print(f"  Audit events: {data['audit']['count']}")
    print()
    print("Coverage:")
    for k, v in data["coverage"].items():
        print(f"  {k}: {v}")
    print()
    print("Known gaps (not yet covered):")
    for gap in data["known_gaps"]:
        print(f"  [{gap['ref']}] {gap['summary']}")
    print()
    print("Next steps:")
    for action in data["next_actions"]:
        print(f"  slm {action}" if not action.startswith("slm") else f"  {action}")


# ─────────────────────────────────────────────────────────────────────────────
# export
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_gdpr_export(args: Namespace) -> None:
    """``slm gdpr export --profile P [--output FILE] [--include-embeddings] [--json]``."""
    use_json = getattr(args, "json", False)
    profile = getattr(args, "profile", None)
    output_file = getattr(args, "output", None)

    if not profile:
        if use_json:
            _print_json(_json_envelope(
                "gdpr-export",
                error={
                    "code": "MISSING_PROFILE",
                    "message": "Art.15 export requires --profile PROFILE_ID.",
                    "hint": "Run `slm gdpr status` to list known profiles.",
                },
            ))
        else:
            _die("Art.15 export requires --profile PROFILE_ID.\n"
                 "  Run `slm gdpr status` to list known profiles.")
        sys.exit(1)

    root = _data_root()
    db_path = root / "memory.db"

    if not db_path.exists():
        err = {
            "code": "NO_DATABASE",
            "message": "No SuperLocalMemory database found at the data root.",
            "hint": (
                f"Data root: {root}. "
                "Start the daemon (`slm serve`) to create a database, "
                "or set SLM_DATA_DIR to the correct path."
            ),
        }
        if use_json:
            _print_json(_json_envelope("gdpr-export", error=err))
        else:
            _die(f"{err['message']}\n  {err['hint']}")
        sys.exit(2)

    include_embeddings = bool(getattr(args, "include_embeddings", False))
    try:
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        from superlocalmemory.storage.database import DatabaseManager

        # Do NOT call db.initialize() — GDPRCompliance drives all DDL via
        # _profile_scoped_tables() / sqlite_master and works with the existing
        # schema. Pattern mirrors commands.py:3233.
        db = DatabaseManager(db_path)
        gdpr = GDPRCompliance(db, data_root=root)
        if output_file:
            out_path = Path(output_file)
            summary = stream_export_to_file(
                gdpr, profile, out_path, include_embeddings=include_embeddings)
        else:
            summary = stream_export_to_stdout(
                gdpr, profile, include_embeddings=include_embeddings, envelope=use_json)
    except OSError as exc:
        if use_json:
            _print_json(_json_envelope(
                "gdpr-export",
                error={"code": "WRITE_ERROR", "message": str(exc)},
            ))
        else:
            _die(f"Failed to write export to {output_file or 'stdout'}: {exc}")
        sys.exit(1)
    except Exception as exc:
        if use_json:
            _print_json(_json_envelope(
                "gdpr-export",
                error={"code": "EXPORT_ERROR", "message": str(exc)},
            ))
        else:
            _die(f"Export failed: {exc}\n  Run `slm gdpr status` to check DB state.")
        sys.exit(1)

    if not output_file:
        return  # the export itself was the output

    total = summary["total_items"]
    confirmation = {
        "profile": profile,
        "total_items": total,
        "output_file": str(out_path.resolve()),
        "bytes": summary["bytes"],
        "embeddings": "included_base64" if include_embeddings else "omitted",
        "note": "Output file contains personal data. Handle as confidential.",
    }
    if use_json:
        _print_json(_json_envelope("gdpr-export", data=confirmation))
    else:
        print(f"Art.15 export complete — {total} items written to {out_path.resolve()}")
        print(f"  {summary['bytes']:,} bytes. Handle as confidential personal data.")
        if not include_embeddings:
            print("  Embeddings and other binary columns are omitted (see export_manifest);"
                  " add --include-embeddings to include them as base64.")


# ─────────────────────────────────────────────────────────────────────────────
# erase
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_gdpr_erase(args: Namespace) -> None:
    """``slm gdpr erase --profile P [--yes] [--dry-run] [--json]``."""
    use_json = getattr(args, "json", False)
    profile = getattr(args, "profile", None)
    confirmed = getattr(args, "yes", False)
    dry_run = getattr(args, "dry_run", False)

    # ── SAFETY GATE 1: profile must be named explicitly ──────────────────────
    if not profile:
        msg = (
            "Art.17 erasure requires --profile PROFILE_ID.\n"
            "  This operation is IRREVERSIBLE. Name the profile explicitly.\n"
            "  Run `slm gdpr status` to list known profiles.\n"
            "  Run `slm gdpr erase --profile PROFILE --dry-run` to preview."
        )
        if use_json:
            _print_json(_json_envelope(
                "gdpr-erase",
                error={
                    "code": "MISSING_PROFILE",
                    "message": (
                        "Art.17 erasure requires --profile PROFILE_ID. "
                        "This operation is IRREVERSIBLE. "
                        "Run with --dry-run first."
                    ),
                    "hint": "slm gdpr status  →  list known profiles",
                },
            ))
        else:
            print(msg, file=sys.stderr)
        sys.exit(2)

    # ── SAFETY GATE 2: "default" profile cannot be erased ───────────────────
    if profile == "default":
        msg = (
            "Cannot erase the 'default' profile via GDPR Art.17. "
            "The default profile is a system profile.  "
            "Use a named profile: slm profile list"
        )
        if use_json:
            _print_json(_json_envelope(
                "gdpr-erase",
                error={"code": "DEFAULT_PROFILE", "message": msg},
            ))
        else:
            _die(msg, code=2)
        sys.exit(2)

    root = _data_root()
    db_path = root / "memory.db"

    if not db_path.exists():
        err = {
            "code": "NO_DATABASE",
            "message": "No SuperLocalMemory database found at the data root.",
            "hint": (
                f"Data root: {root}. "
                "Start the daemon (`slm serve`) to create a database."
            ),
        }
        if use_json:
            _print_json(_json_envelope("gdpr-erase", error=err))
        else:
            _die(f"{err['message']}\n  {err['hint']}")
        sys.exit(2)

    # ── DRY-RUN: show exactly what would be erased ───────────────────────────
    if dry_run or not confirmed:
        _show_erase_preview(profile, root, db_path, use_json)
        if not confirmed:
            # Explicit --dry-run with no --yes → preview succeeded (exit 0).
            # Implicit dry-run (neither --dry-run nor --yes) → refused (exit 2).
            # A pipeline that doesn't supply --yes must see a non-zero exit.
            if dry_run:
                sys.exit(0)
            if use_json:
                _print_json(_json_envelope(
                    "gdpr-erase",
                    error={
                        "code": "CONFIRMATION_REQUIRED",
                        "message": (
                            "Erasure aborted — --yes not supplied. "
                            "This is a dry-run preview. "
                            "Re-run with --yes to confirm (IRREVERSIBLE)."
                        ),
                        "hint": f"slm gdpr erase --profile {profile} --yes",
                    },
                ))
            else:
                print(
                    "\nErasure aborted — no --yes flag supplied.\n"
                    "  Re-run with --yes to confirm (IRREVERSIBLE):\n"
                    f"    slm gdpr erase --profile {profile} --yes",
                    file=sys.stderr,
                )
            sys.exit(2)
        return  # dry-run with --dry-run flag: preview only, exit 0

    # ── LIVE ERASURE ─────────────────────────────────────────────────────────
    try:
        from superlocalmemory.compliance.gdpr import GDPRCompliance
        from superlocalmemory.storage.database import DatabaseManager

        db = DatabaseManager(db_path)
        gdpr = GDPRCompliance(db, data_root=root)
        counts = gdpr.forget_profile(profile)
    except Exception as exc:
        if use_json:
            _print_json(_json_envelope(
                "gdpr-erase",
                error={
                    "code": "ERASE_ERROR",
                    "message": str(exc),
                    "hint": "The erasure may be incomplete. Check the audit_chain.db receipt.",
                },
            ))
        else:
            print(f"error: Erasure failed: {exc}", file=sys.stderr)
            print("  The erasure may be incomplete.  Check the audit_chain.db receipt.", file=sys.stderr)
        sys.exit(1)

    from superlocalmemory.compliance import erasure_outcome

    # One verdict for every surface (compliance/erasure_outcome.py): the JSON
    # envelope, the printed text and the exit code below all read it, so they
    # cannot disagree about the same erasure.
    statement = erasure_outcome.describe(counts)
    ok = erasure_outcome.erasure_succeeded(counts)
    result_data = {
        "profile": profile,
        "erasure_complete": bool(statement["complete"]),
        "erasure_provable": bool(statement["provable"]),
        "removed": statement["removed"],
        "remains": statement["remains"],
        "counts": {k: v for k, v in counts.items() if k != "erasure_report"},
        "note": (
            "Erasure recorded in audit_chain.db. Run "
            "`slm gdpr verify --profile PROFILE` to confirm receipts."
        ),
    }

    if use_json:
        error = None if ok else {
            "code": "ERASURE_INCOMPLETE",
            "message": erasure_outcome.failure_message(counts),
            "hint": "; ".join(item["retry"] for item in statement["remains"]),
        }
        _print_json(_json_envelope("gdpr-erase", data=result_data, error=error, success=ok))
    else:
        _print_erase_text(profile, statement, ok)

    if not ok:
        sys.exit(1)


def _show_erase_preview(
    profile: str, root: Path, db_path: Path, use_json: bool
) -> None:
    """Show what would be erased without making any changes."""
    tables: list[str] = []
    row_counts: dict[str, int] = {}

    if db_path.exists():
        try:
            from superlocalmemory.storage.memory_write import memory_read
            with memory_read(db_path) as conn:
                # Discover profile-scoped tables (mirroring GDPRCompliance logic)
                _NON_MEMORY_SCOPED = {"profiles", "erasure_receipts"}
                tbl_rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                for (tname,) in tbl_rows:
                    if tname.startswith("sqlite_") or tname in _NON_MEMORY_SCOPED:
                        continue
                    try:
                        cols = {
                            r[1] for r in conn.execute(
                                f"PRAGMA table_info({tname})"
                            ).fetchall()
                        }
                        if "profile_id" not in cols:
                            continue
                        cnt = conn.execute(
                            f"SELECT COUNT(*) FROM {tname} WHERE profile_id = ?",
                            (profile,),
                        ).fetchone()[0]
                        tables.append(tname)
                        row_counts[tname] = cnt
                    except Exception:
                        continue
        except Exception:
            pass

    preview = {
        "dry_run": True,
        "profile": profile,
        "would_erase": row_counts,
        "tables_affected": len(tables),
        "total_rows": sum(row_counts.values()),
        "also_purged": [
            "learning.db, pending.db and admission_journal.db rows of the profile",
            "write receipts (write_commits) of the profile",
            "active_brain_cache.db, derive_cache.db and the MCP cache / compression originals",
            "vector index entries and the profile's entry in profiles.json",
            "pictures, documents and the code-graph links of its memories",
        ],
        "survives": [
            "erasure_receipts and the audit chain (proof that the erasure happened; hashes only)",
            "backups (backups/, pre-migration-snapshots/, pre-restore/): each is recorded "
            "as an erasure obligation and the result stays INCOMPLETE until it is deleted",
        ],
        "warning": (
            "This operation is IRREVERSIBLE. "
            "Re-run with --yes to confirm."
        ),
    }

    if use_json:
        _print_json(_json_envelope("gdpr-erase-dryrun", data=preview))
    else:
        print(f"\nGDPR Art.17 dry-run for profile '{profile}':")
        print(f"  Tables to erase : {preview['tables_affected']}")
        print(f"  Total rows      : {preview['total_rows']}")
        for t, c in row_counts.items():
            print(f"    {t}: {c} rows")
        print("  Also purged: " + ", ".join(preview["also_purged"]))
        print("  Survives   : " + ", ".join(preview["survives"]))
        print()
        print(f"  WARNING: {preview['warning']}")


# ─────────────────────────────────────────────────────────────────────────────
# verify
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_gdpr_verify(args: Namespace) -> None:
    """``slm gdpr verify --receipt-id ID [--profile P] [--json]``.

    Exit codes:
      0 — receipt found and HMAC is intact (VERIFIED)
      1 — receipt found but HMAC fails (TAMPERED — integrity violation)
      2 — receipt not found (NOT_FOUND)
    """
    use_json = getattr(args, "json", False)
    receipt_id = getattr(args, "receipt_id", None)
    profile = getattr(args, "profile", None)

    # Allow listing all receipts for a profile
    list_mode = not receipt_id and profile

    root = _data_root()
    db_path = root / "memory.db"

    if not db_path.exists():
        if use_json:
            _print_json(_json_envelope(
                "gdpr-verify",
                error={
                    "code": "NOT_FOUND",
                    "message": "No database found at the data root.",
                    "hint": f"Data root: {root}",
                },
            ))
        else:
            _die(f"No database at {db_path}. Run `slm gdpr status` first.", code=2)
        sys.exit(2)

    if list_mode:
        _list_receipts(profile, db_path, use_json)
        return

    if not receipt_id:
        if use_json:
            _print_json(_json_envelope(
                "gdpr-verify",
                error={
                    "code": "MISSING_RECEIPT_ID",
                    "message": "Provide --receipt-id ID to verify a specific receipt.",
                    "hint": "slm gdpr verify --profile PROFILE  →  list receipts",
                },
            ))
        else:
            _die(
                "Provide --receipt-id ID to verify, or --profile PROFILE to list.\n"
                "  Example: slm gdpr verify --receipt-id <id>",
                code=2,
            )
        sys.exit(2)

    # Check if receipt exists first (to distinguish NOT_FOUND from TAMPERED)
    receipt_meta: dict = {}
    exists = False
    try:
        from superlocalmemory.storage.memory_write import memory_read
        with memory_read(db_path) as conn:
            params: tuple = (receipt_id,)
            extra = ""
            if profile:
                extra = " AND profile_id = ?"
                params = (receipt_id, profile)
            row = conn.execute(
                "SELECT erasure_id, profile_id, subject_type, subject_id, "
                "fact_count, state, all_erased, requested_at, completed_at "
                f"FROM erasure_receipts WHERE erasure_id = ?{extra}",
                params,
            ).fetchone()
            if row:
                exists = True
                receipt_meta = {
                    "erasure_id": row[0],
                    "profile_id": row[1],
                    "subject_type": row[2],
                    "subject_id": row[3],
                    "fact_count": row[4],
                    "state": row[5],
                    "all_erased": bool(row[6]),
                    "requested_at": row[7],
                    "completed_at": row[8],
                }
    except Exception as exc:
        if use_json:
            _print_json(_json_envelope(
                "gdpr-verify",
                error={"code": "DB_ERROR", "message": str(exc)},
            ))
        else:
            _die(f"Could not read database: {exc}", code=1)
        sys.exit(1)

    if not exists:
        result = {
            "erasure_id": receipt_id,
            "status": "NOT_FOUND",
            "verified": False,
            "message": "Receipt ID not found in erasure_receipts.",
        }
        if use_json:
            _print_json(_json_envelope("gdpr-verify", data=result))
        else:
            print(f"NOT_FOUND: Receipt '{receipt_id}' does not exist.")
        sys.exit(2)

    # Run HMAC verification
    hmac_ok = False
    verify_error: str | None = None
    try:
        from superlocalmemory.core.transactions.erasure import verify_receipt
        from superlocalmemory.storage.memory_write import memory_read

        with memory_read(db_path) as conn:
            hmac_ok = verify_receipt(
                conn, receipt_id, profile_id=profile
            )
    except Exception as exc:
        verify_error = str(exc)

    if verify_error:
        result = {
            "erasure_id": receipt_id,
            "status": "VERIFY_ERROR",
            "verified": False,
            "error": verify_error,
            "receipt": receipt_meta,
        }
        if use_json:
            _print_json(_json_envelope("gdpr-verify", data=result))
        else:
            print(f"VERIFY_ERROR: {verify_error}", file=sys.stderr)
        sys.exit(1)

    status_str = "VERIFIED" if hmac_ok else "TAMPERED"
    result = {
        "erasure_id": receipt_id,
        "status": status_str,
        "verified": hmac_ok,
        "receipt": receipt_meta,
        "message": (
            "HMAC chain is intact — deletion can be proven."
            if hmac_ok else
            "HMAC mismatch — receipt may have been tampered with. "
            "Do NOT rely on this receipt as Art.17 evidence."
        ),
    }

    if use_json:
        _print_json(_json_envelope("gdpr-verify", data=result))
    else:
        print(f"{status_str}: receipt '{receipt_id}'")
        for k, v in receipt_meta.items():
            print(f"  {k}: {v}")
        print(f"  HMAC: {'OK' if hmac_ok else 'MISMATCH'}")

    sys.exit(0 if hmac_ok else 1)


def _list_receipts(profile: str, db_path: Path, use_json: bool) -> None:
    """List all erasure receipts for a profile."""
    receipts: list[dict] = []
    try:
        from superlocalmemory.storage.memory_write import memory_read
        with memory_read(db_path) as conn:
            rows = conn.execute(
                "SELECT erasure_id, profile_id, subject_type, subject_id, "
                "fact_count, state, all_erased, requested_at, completed_at "
                "FROM erasure_receipts WHERE profile_id = ? "
                "ORDER BY completed_at DESC",
                (profile,),
            ).fetchall()
            for r in rows:
                receipts.append({
                    "erasure_id": r[0],
                    "profile_id": r[1],
                    "subject_type": r[2],
                    "subject_id": r[3],
                    "fact_count": r[4],
                    "state": r[5],
                    "all_erased": bool(r[6]),
                    "requested_at": r[7],
                    "completed_at": r[8],
                })
    except Exception as exc:
        if use_json:
            _print_json(_json_envelope(
                "gdpr-verify",
                error={"code": "DB_ERROR", "message": str(exc)},
            ))
        else:
            _die(f"Could not list receipts: {exc}")
        sys.exit(1)

    result = {
        "profile": profile,
        "count": len(receipts),
        "receipts": receipts,
        "hint": "Use --receipt-id ID to verify a specific receipt's HMAC.",
    }

    if use_json:
        _print_json(_json_envelope("gdpr-verify", data=result))
    else:
        print(f"Erasure receipts for profile '{profile}': {len(receipts)}")
        for r in receipts:
            print(
                f"  [{r['erasure_id'][:16]}...] "
                f"state={r['state']} "
                f"facts={r['fact_count']} "
                f"complete={r['all_erased']}"
            )
        if receipts:
            print("\nVerify a receipt: slm gdpr verify --receipt-id <id>")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def cmd_gdpr(args: Namespace) -> None:
    """Dispatch ``slm gdpr`` subcommands."""
    sub = getattr(args, "gdpr_command", None)
    handlers = {
        "status": _cmd_gdpr_status,
        "export": _cmd_gdpr_export,
        "erase": _cmd_gdpr_erase,
        "verify": _cmd_gdpr_verify,
    }
    handler = handlers.get(sub)
    if handler:
        handler(args)
    else:
        print(
            "Usage: slm gdpr <status|export|erase|verify> [options]\n"
            "\n"
            "  slm gdpr status [--profile P] [--json]\n"
            "  slm gdpr export --profile P [--output FILE] [--include-embeddings] [--json]\n"
            "  slm gdpr erase --profile P --yes [--dry-run] [--json]\n"
            "  slm gdpr verify --receipt-id ID [--profile P] [--json]\n"
            "\n"
            "All subcommands accept --json for evidence-pipeline integration.\n"
            "Exit codes: 0=success, 1=failure/tampered, 2=refused/not-found.\n"
        )
        sys.exit(1)
