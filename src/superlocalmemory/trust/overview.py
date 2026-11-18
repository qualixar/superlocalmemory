# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Trust overview for the dashboard (WS-5, §8.10.3 h) and the Truth checks card.

One read of what the trust work knows, in words a person reads: how many memories came from
each kind of source, how many wait for the person's OK, whether memories are signed and
whether any no longer match their signature, and whether the audit log is intact. Counts and
ids only: no memory text and never the value of a credential.

Read-only. The signature check uses the key the daemon already holds in memory (``signer``)
and never asks the keychain; without it the counts are still given and nothing is verified,
and the words say so. The audit chain is opened with ``read_only=True``. Each part fails open:
a store missing a table gives an empty part, a failed read says it could not be read (never "intact").
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from superlocalmemory.trust import origin_class as oc
from superlocalmemory.trust import report as trust_report
from superlocalmemory.trust import signatures
from superlocalmemory.trust import signing_key as sk

logger = logging.getLogger(__name__)

#: Signature check size for a dashboard read (the same sample the doctor uses).
SAMPLE = signatures.SAMPLE


def signature_state(conn: sqlite3.Connection, signer: sk.Signer | None, key_status: str,
                    *, sample: int | None = SAMPLE) -> dict[str, Any]:
    """Signed / unsigned / modified / moved, with a sentence. ``modified`` and ``moved`` are
    of the facts checked (a sample unless ``sample`` is ``None``)."""
    try:
        r = signatures.verify(conn, signer, key_status=key_status, sample=sample)
    except Exception as exc:  # noqa: BLE001 - an unrun check is not a passed one
        logger.debug("signature check failed", exc_info=True)
        return {"ok": None, "words": f"The signatures could not be checked ({type(exc).__name__}).",
                "key_words": sk.words(key_status)}
    out = r.as_dict()
    out["ok"] = r.ok if r.checked else None
    if r.modified:
        words = (f"{r.modified} of the {r.checked} memories checked no longer match their "
                 "signature. They were changed outside SuperLocalMemory.")
    elif not r.signed:
        words = "No memory is signed yet. " + sk.words(key_status)
    elif not r.checked:
        words = (f"{r.signed} of {r.facts} memories are signed. This process does not hold the key, "
                 "so nothing was checked now.")
    else:
        words = (f"{r.signed} of {r.facts} memories are signed. "
                 f"{r.checked} checked: none changed since signing.")
    out["words"] = words
    return out


def audit_state(audit_path: Path, signer: sk.Signer | None) -> dict[str, Any]:
    """The audit log's verification in words. ``present`` is false without a file."""
    if not Path(audit_path).is_file():
        return {"present": False, "ok": None, "words": "There is no audit log next to this store yet."}
    from superlocalmemory.compliance.audit import AuditChain

    try:
        r = AuditChain(Path(audit_path), read_only=True).verify_report(signer)
    except Exception as exc:  # noqa: BLE001 - never report "intact" for a check that did not run
        logger.debug("audit chain unreadable", exc_info=True)
        return {"present": True, "ok": False,
                "words": f"The audit log could not be read ({type(exc).__name__})."}
    if not r.get("ok"):
        where = f" at entry {r['failed_at']}" if r.get("failed_at") else ""
        return {**r, "present": True,
                "words": f"The audit log failed its check{where}: {r.get('reason') or 'unknown'}."}
    words = f"The audit log is intact: {r.get('entries', 0)} entries, order and links verified"
    if r.get("keyed"):
        words += f"; {r['keyed']} also checked with your key"
    return {**r, "present": True, "words": words + "."}


def origins(conn: sqlite3.Connection, profile_id: str) -> list[dict[str, Any]]:
    """Memories by where they came from, most trusted first, in words."""
    counts = trust_report.origins(conn, profile_id)
    return [{"class": name, "label": oc.label(name), "count": n} for name, n in counts.items()]


def build(conn: sqlite3.Connection, profile_id: str, *, slm_home: Path | None,
          audit_path: Path, signer: sk.Signer | None, key_status: str | None = None) -> dict[str, Any]:
    """The overview for one profile (JSON-ready). Signatures and the log are store-wide."""
    section = trust_report.build(conn, profile_id, slm_home=slm_home)
    status = key_status or (sk.READY if signer is not None else sk.status())
    return {
        "firewall": {"state": section.get("taint", "unknown"), "waiting": section.get("waiting", 0),
                     "approved": section.get("approved", 0),
                     "unclassified": section.get("unclassified", 0)},
        "secret_screen": {"state": section.get("secret_screen", "unknown"),
                          "by_type": (section.get("secrets_held_back") or {}).get("by_type", {}),
                          "memories": (section.get("secrets_held_back") or {}).get("memories", 0)},
        "origins": origins(conn, profile_id),
        "signatures": signature_state(conn, signer, status),
        "audit_chain": audit_state(audit_path, signer),
    }


__all__ = ["SAMPLE", "audit_state", "build", "origins", "signature_state"]
