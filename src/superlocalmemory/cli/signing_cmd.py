# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm signing``: signed memories and the keyed audit chain (WS-5, D-16).

    slm signing status [--json] [--all]   # is signing available; how many memories are signed
    slm signing resign [--yes]            # sign again what a lost key or a removed signature left

The key lives in your operating system's keychain and nowhere else; there is no key file.
Signing detects a change made by something that has the file but not your login session (a
synced or restored copy, another user account, a disk image). It does not stop a process
running as you. A lost key loses no data: memories signed under it are listed as "signed
under another key", never as modified, and ``resign`` signs them again under the new key.

``status`` reads the keychain once and the store read-only. ``resign`` blesses what is in the
store as it stands. Any caller may run it (D-49: an agent passes ``--yes``; at a terminal it
asks first); each run is recorded as a decision (who, channel ``cli``, counts) and listed under
"Recent decisions by your agents". It has no undo: it changes no memory, and the marks it
replaces could no longer be checked. It refuses while the daemon runs (stop it first), like
``slm db fidelity``, and never creates a key. Signing never claimed to stop a process running
as you, so asking for a terminal here protected nothing.
"""

from __future__ import annotations

import sqlite3
import sys
from argparse import Namespace
from typing import Any


def register_signing_parsers(sub: Any) -> None:
    """Attach ``slm signing``. Called from cli/main.py."""
    p = sub.add_parser("signing", help="Signed memories and the keyed audit chain: status, resign")
    ssub = p.add_subparsers(dest="signing_command")
    st = ssub.add_parser("status", help="Is signing available; how many memories are signed")
    st.add_argument("--all", action="store_true", help="Verify every signed memory, not a sample")
    st.add_argument("--db", default=None, help="Store file (default: yours)")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    rs = ssub.add_parser("resign", help="Sign again the memories a lost key signed "
                                        "(recorded as a decision; stop the daemon first)")
    rs.add_argument("--yes", action="store_true",
                    help="Do not ask for the final yes (needed when no terminal is attached)")
    rs.add_argument("--profile", default="", help="Profile the decision is recorded under")
    rs.add_argument("--db", default=None, help="Store file (default: yours)")


def _status(args: Namespace) -> int:
    from superlocalmemory.cli import doctor_trust
    from superlocalmemory.cli.doctor_truth import _db_path
    from superlocalmemory.truth.invariants import open_read_only

    path = _db_path(args)
    use_json = bool(getattr(args, "json", False))
    if not path.is_file():
        print(f"No store at {path}", file=sys.stderr)
        return 2
    loaded = doctor_trust.read_key()
    conn = open_read_only(str(path))
    try:
        sig = doctor_trust.signatures_section(conn, loaded, all_facts=bool(getattr(args, "all", False)))
    finally:
        conn.close()
    audit = doctor_trust.audit_section(path.parent / "audit_chain.db", loaded)
    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("signing", data={"signatures": sig, "audit_chain": audit})
    else:
        for line in doctor_trust.lines(sig, audit):
            print(line)
    return 1 if doctor_trust.failed(sig, audit) else 0


def _daemon_running() -> bool:
    from superlocalmemory.cli.daemon import owned_daemon_process_alive

    return bool(owned_daemon_process_alive())


def _question(other: int, removed: int, record_bad: bool) -> str:
    parts = []
    if other:
        parts.append(f"{other} memories were signed under a key this computer no longer holds")
    if removed:
        parts.append(f"{removed} memories were signed and lost their signature (it was removed, "
                     "so they may have been changed: look at them with `slm why <id>` first)")
    if record_bad and not (other or removed):
        parts.append("the signing record does not check under the current key")
    return "; ".join(parts).capitalize() + ". Sign them again as they are now? [y/N] "


def _resign(args: Namespace) -> int:
    from superlocalmemory.cli import doctor_trust
    from superlocalmemory.cli.doctor_truth import _db_path
    from superlocalmemory.trust import signatures
    from superlocalmemory.trust import signing_key as sk

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive and not getattr(args, "yes", False):
        print("Nothing changed: without a terminal to ask at, pass --yes to sign again.",
              file=sys.stderr)
        return 2
    if _daemon_running():
        print("SLM is running. Stop it first (`slm serve stop`), then run `slm signing resign` "
              "again.", file=sys.stderr)
        return 2
    path = _db_path(args)
    if not path.is_file():
        print(f"No store at {path}", file=sys.stderr)
        return 2
    loaded = doctor_trust.read_key()
    if loaded.signer is None:
        print(sk.words(loaded.status), file=sys.stderr)
        return 2
    conn = sqlite3.connect(str(path), timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        other = conn.execute("SELECT COUNT(*) FROM fact_signatures WHERE key_id != ?",
                             (loaded.signer.key_id,)).fetchone()[0]
        report = signatures.verify(conn, loaded.signer, sample=0)
        removed = report.missing
        record_bad = report.record in (signatures.RECORD_MISSING, signatures.RECORD_FORGED,
                                       signatures.RECORD_OTHER_KEY)
        if not (other or removed or record_bad):
            print("Nothing to sign again: no memory is signed under another key, none lost its "
                  "signature, and the signing record checks.")
            return 0
        if not getattr(args, "yes", False):
            answer = input(_question(other, removed, record_bad))
            if answer.strip().lower() not in ("y", "yes"):
                print("Nothing changed.")
                return 1
        with conn:
            done = signatures.resign(conn, loaded.signer, only_other_key=True,
                                     include_missing=True)
            _record(conn, args, loaded.signer.key_id, done, other, removed)
    finally:
        conn.close()
    print(f"Signed {done} memories again under the current key.")
    return 0


def _record(conn: sqlite3.Connection, args: Namespace, key_id: str, done: int, other: int,
            removed: int) -> None:
    """The decision record (D-49): who signed again, through the command line, and how many."""
    from superlocalmemory.cli.health_memory import _profile
    from superlocalmemory.cli.trust_cmd import _who
    from superlocalmemory.core import decisions

    summary = (f"Signed {done} memories again under the current key ({other} under a lost key, "
               f"{removed} that had lost their signature)")
    decisions.log(conn, profile_id=_profile(args), kind=decisions.SIGNING, subject_id=key_id,
                  action="resign", decider=decisions.Decider(_who(), "cli"), summary=summary,
                  detail={"signed": done, "other_key": other, "removed": removed})


def run(args: Namespace) -> None:
    command = getattr(args, "signing_command", None) or "status"
    code = _resign(args) if command == "resign" else _status(args)
    if code:
        sys.exit(code)


__all__ = ["register_signing_parsers", "run"]
