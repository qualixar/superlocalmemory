# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm doctor --truth``: the signatures and audit-chain checks (WS-5, D-16), read-only.

A person ran this on purpose, so it reads the signing key from the keychain itself, once,
with a timeout (the daemon holds its own copy; a hook never asks). It never creates a key.
Without a key it still reports counts and says why nothing was verified; a modified fact is
reported by id, never by text.

Both checks only report. A modified signature or a broken keyed link exits non-zero from
``--truth`` (the store's own evidence says something changed); unsigned facts, signatures
under another key and plain links after keying began are reported and are not failures.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from superlocalmemory.trust import signatures
from superlocalmemory.trust import signing_key as sk


def read_key(store: Any = None) -> sk.Loaded:
    """The one keychain read of this command. Never creates a key."""
    return sk.load(store, create=False)


def signatures_section(conn: sqlite3.Connection, loaded: sk.Loaded, *,
                       all_facts: bool = False) -> dict[str, Any]:
    status = loaded.status if loaded.signer is None else sk.READY
    report = signatures.verify(conn, loaded.signer, key_status=status,
                               sample=None if all_facts else signatures.SAMPLE)
    return report.as_dict()


def audit_section(audit_path: Path, loaded: sk.Loaded) -> dict[str, Any]:
    """The chain's verification, on a read-only connection. ``present`` is False without a file."""
    if not audit_path.is_file():
        return {"present": False}
    from superlocalmemory.compliance.audit import AuditChain

    try:
        report = AuditChain(audit_path, read_only=True).verify_report(loaded.signer)
    except Exception as exc:  # noqa: BLE001 - an unrun check is not a passed one
        return {"present": True, "ok": False, "reason": f"could not be read ({type(exc).__name__})"}
    return {"present": True, **report}


def lines(sig: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    """The text a person reads."""
    out = ["Signatures (D-16; a sample, not a full proof):"
           if not sig["exhaustive"] else "Signatures (D-16; every signed memory checked):"]
    out.append(f"  {sig['key_words']}")
    out.append(f"  {sig['signed']} of {sig['facts']} memories are signed, {sig['unsigned']} are not yet.")
    if sig["checked"]:
        out.append(f"  Checked {sig['checked']}: {sig['modified']} modified"
                   + (f" ({', '.join(sig['modified_ids'])})" if sig["modified_ids"] else "")
                   + f", {sig['signed_under_another_key']} signed under another key (cannot be "
                   f"checked), {sig['moved_by_a_fold']} waiting to be signed again after a profile move.")
    if not audit.get("present"):
        out.append("Audit chain: no file next to this store.")
        return out
    if not audit.get("ok"):
        out.append(f"Audit chain: FAILED: {audit.get('reason', 'unknown')}"
                   + (f" (entry {audit['failed_at']})" if audit.get("failed_at") else ""))
        return out
    out.append(f"Audit chain: {audit['entries']} entries, order and links verified; "
               f"{audit['keyed']} keyed links checked by HMAC, {audit['keyed_unverified']} keyed "
               "under a key this computer does not hold (order only), "
               f"{audit['legacy']} written before 4.1.25 record less.")
    if audit.get("plain_after_keyed"):
        out.append(f"  Note: {audit['plain_after_keyed']} plain links follow the first keyed one: "
                   "a command run while the daemon was down writes them, and so could someone "
                   "rewriting the file without the key. They are counted, not trusted as keyed.")
    return out


def failed(sig: dict[str, Any], audit: dict[str, Any]) -> bool:
    return bool(sig["modified"]) or (audit.get("present") is True and audit.get("ok") is False)


__all__ = ["audit_section", "failed", "lines", "read_key", "signatures_section"]
