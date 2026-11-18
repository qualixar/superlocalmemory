# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm doctor --truth``: the truth layer's invariants on a store, read-only (R-27, D-24).

Run it on a copy of the store (``--db``) at every milestone. It opens the file
with ``mode=ro`` and never starts a daemon or the engine, so it cannot change
what it checks. Exit 1 when any invariant it checks is violated.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


def _db_path(args: Namespace) -> Path:
    given = getattr(args, "db", None)
    if given:
        return Path(given).expanduser()
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root() / "memory.db"


def _refuse_no_store(path: Path, use_json: bool) -> None:
    message = f"No store at {path}"
    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("doctor", error={"code": "NO_STORE", "message": message})
    else:
        print(message, file=sys.stderr)
    sys.exit(2)


def _print_certificates(certificates: list, second: list) -> None:
    live = [c for c in certificates if not c["voided_at"]]
    print(f"Judge certificates: {len(live)} open, {len(certificates) - len(live)} voided")
    for c in certificates[:20]:
        state = (f"voided {c['voided_at'][:10]} ({c['void_reason']})" if c["voided_at"]
                 else f"open since {c['issued_at'][:10]}")
        print(f"  {c['tier']}/{c['model']} {c['verdict']} [{c['population']}, {c['regime']}]: "
              f"{state}; {c['n']} audited, {c['errors']} wrong, threshold {c['threshold']:.2f}")
    print("Second opinions (M-06: a stronger judge asked in shadow):"
          + ("" if second else " none yet"))
    for s in second:
        print(f"  {s['lower_tier']}/{s['lower_model']} -> {s['upper_tier']}/{s['upper_model']}: "
              f"sure of something else on {s['disagreed']} of {s['pairs']} pairs "
              f"(95% bounds {s['rate_lower']:.2f}-{s['rate_upper']:.2f})")


def _print_text(path: Path, results: list, violated: list, certificates: list, second: list,
                sig: dict, audit: dict) -> None:
    from superlocalmemory.cli import doctor_trust
    from superlocalmemory.truth import invariants as inv

    print(f"Truth invariants on {path} (read-only)")
    for r in results:
        mark = {inv.OK: "ok  ", inv.VIOLATED: "FAIL", inv.NOT_APPLICABLE: "n/a ",
                inv.NOT_CHECKED: "----"}[r.status]
        tail = f": {r.count} ({', '.join(r.sample)})" if r.status == inv.VIOLATED else ""
        print(f"  [{mark}] {r.invariant} {r.title}{tail}")
    print(f"{len(violated)} violated" if violated else "No violations among the checked invariants")
    _print_certificates(certificates, second)
    for line in doctor_trust.lines(sig, audit):
        print(line)


def cmd_doctor_truth(args: Namespace) -> None:
    from superlocalmemory.truth import invariants as inv

    path = _db_path(args)
    use_json = getattr(args, "json", False)
    if not path.is_file():
        _refuse_no_store(path, use_json)
    from superlocalmemory.cli import doctor_trust
    from superlocalmemory.core import judge_certificates as certs
    from superlocalmemory.core import shadow_escalation as shadow

    conn = inv.open_read_only(str(path))
    try:
        results = inv.check(conn)
        certificates = certs.records(conn, None)
        second = [s.as_dict() for s in shadow.disagreements(conn, None)]
        # WS-5 (D-16): one keychain read by this command, then signatures and the audit chain.
        loaded = doctor_trust.read_key()
        sig = doctor_trust.signatures_section(
            conn, loaded, all_facts=bool(getattr(args, "signatures_all", False)))
    finally:
        conn.close()
    audit = doctor_trust.audit_section(path.parent / "audit_chain.db", loaded)
    violated = [r for r in results if r.status == inv.VIOLATED]
    if use_json:
        from superlocalmemory.cli.json_output import json_print
        json_print("doctor", data={"store": str(path), "truth": [
            {"invariant": r.invariant, "title": r.title, "status": r.status, "count": r.count,
             "sample": list(r.sample)} for r in results], "certificates": certificates,
            "second_opinions": second, "signatures": sig, "audit_chain": audit})
    else:
        _print_text(path, results, violated, certificates, second, sig, audit)
    if violated or doctor_trust.failed(sig, audit):
        sys.exit(1)


__all__ = ["cmd_doctor_truth"]
