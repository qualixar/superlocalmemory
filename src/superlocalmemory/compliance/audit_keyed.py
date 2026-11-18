# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Keyed links for the audit chain (WS-5, D-16): hash version 3.

Versions 1 and 2 are plain SHA-256: anyone who can rewrite the file can recompute every
hash. A version-3 link is an HMAC-SHA256 under the signing key that lives in the OS
keychain (``trust/signing_key``), so a copy of the file without the user's login session
cannot rewrite entries and keep the chain valid. The same threat statement as signed facts:
it does not stop a process running as the user.

*Which links are keyed.* A link is keyed when the process writing it holds the key (the
daemon, with ``trust.signing`` on). A CLI command run while the daemon is down writes a plain
link: the chain keeps working, and the report counts such links after the first keyed one
(``plain_after_keyed``) so a run of them is visible and never called "fully keyed".

*Anchors.* The first keyed link, and the first one after the key changed (a lost or replaced
key), is preceded by an anchor entry (operation ``key_anchor``) that is itself keyed with the
new key and records the new key id, the previous key id and why. Links made under an earlier
key cannot be checked by HMAC any more (the key is gone, nothing is re-keyed); their order
and prev-links are still checked and they are counted ``keyed_unverified``. Plain links
before the anchor stay verifiable by their plain hash.

Pure functions; the chain itself is ``compliance/audit.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

KEYED_VERSION = 3
ANCHOR_OPERATION = "key_anchor"
_V3_DOMAIN = "superlocalmemory-audit-v3"


def keyed_hash(signer: Any, prev_hash: str, operation: str, agent_id: str, profile_id: str,
               content_hash: str, timestamp: str, metadata: str) -> str:
    """The HMAC of one link: every stored field plus the key id, as canonical JSON."""
    record = {"domain": _V3_DOMAIN, "v": KEYED_VERSION, "prev": prev_hash, "ts": timestamp,
              "op": operation, "agent": agent_id, "profile": profile_id,
              "content": content_hash, "meta": metadata, "key": signer.key_id}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return signer.mac(canonical.encode("utf-8"))


def anchor_reason(last_key_id: str | None, signer: Any, chain_was_empty: bool) -> str | None:
    """Why an anchor must be written before the next keyed link, or ``None`` if none is needed."""
    if last_key_id == signer.key_id:
        return None
    if last_key_id:
        return "key_changed"
    return "chain_started" if chain_was_empty else "keying_started"


def anchor_metadata(reason: str, signer: Any, previous_key_id: str | None) -> dict[str, str]:
    return {"reason": reason, "key_id": signer.key_id, "previous_key_id": previous_key_id or ""}


@dataclass
class Tally:
    """What a walk counted about keyed links, and where it stands (one per walk)."""

    keyed: int = 0               # links whose HMAC was checked and held
    keyed_unverified: int = 0    # keyed under a key this process does not hold
    plain_after_keyed: int = 0   # plain links after the first keyed one
    anchors: int = 0
    seen_keyed: bool = False
    active_key: str | None = None  # the key id the latest anchor declared

    def as_report(self) -> dict[str, int]:
        return {"keyed": self.keyed, "keyed_unverified": self.keyed_unverified,
                "plain_after_keyed": self.plain_after_keyed, "anchors": self.anchors}


def _declared_key(row: dict) -> str:
    try:
        return str(json.loads(row.get("metadata") or "{}").get("key_id", ""))
    except (ValueError, AttributeError):
        return ""


def check_row(row: dict, signer: Any, tally: Tally) -> str | None:
    """Judge one keyed link: ``None`` if it passes, else why it does not.

    A keyed link must name a key id; an anchor must declare the key it is keyed with; any
    other keyed link must carry the key the latest anchor declared, so a plain or foreign link
    cannot be relabelled as keyed to escape its check. Its HMAC is checked when this process
    holds that key; otherwise it is counted ``keyed_unverified`` (order is still checked).
    """
    key_id = row.get("key_id") or ""
    if not key_id:
        return "a keyed entry names no key"
    if row.get("operation") == ANCHOR_OPERATION:
        if _declared_key(row) != key_id:
            return "a key anchor declares a different key than it is keyed with"
        tally.anchors += 1
        tally.active_key = key_id
    elif key_id != tally.active_key:
        return "a keyed entry carries a key that no anchor declares"
    tally.seen_keyed = True
    if signer is None or key_id != signer.key_id:
        tally.keyed_unverified += 1
        return None
    expected = keyed_hash(signer, row["prev_hash"], row["operation"], row["agent_id"] or "",
                          row["profile_id"] or "", row["content_hash"] or "", row["timestamp"],
                          row.get("metadata") or "")
    if expected != row["event_hash"]:
        return "the entry's fields do not match its keyed hash"
    tally.keyed += 1
    return None


__all__ = ["ANCHOR_OPERATION", "KEYED_VERSION", "Tally", "anchor_metadata", "anchor_reason",
           "check_row", "keyed_hash"]
