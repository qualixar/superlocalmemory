# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm consent``: what may leave this computer, and your say over it (W2, D-20).

    slm consent                              # list each purpose and its state
    slm consent give judge_ladder_hosted     # agree, after reading the notice (a person, at a terminal)
    slm consent withdraw judge_ladder_hosted # stop at once (works anywhere)

Today there is one purpose: letting the background judges send pairs of your
memories to the online Jev judge (the same provider as the online answer
check). Giving consent asks a person: it refuses without a terminal, so an
agent cannot agree for you. Withdrawing always works. Every change is written
to the audit chain first; if that fails, nothing changes.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from typing import Any

from superlocalmemory.cli.corrections_cmd import _out, _refused


def register_consent_parsers(sub: Any) -> None:
    """Attach ``slm consent``. Called from cli/main.py."""
    p = sub.add_parser("consent", help="What may leave this computer: list, give, withdraw")
    p.add_argument("action", nargs="?", default="list", choices=["list", "give", "withdraw"])
    p.add_argument("purpose", nargs="?", default="", help="e.g. judge_ladder_hosted")
    p.add_argument("--json", action="store_true", help="machine-readable output")


def _home():
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def _answer_check() -> tuple[str, str, bool]:
    """(judge, provider, consent) of the online answer check, from the saved settings."""
    from superlocalmemory.core.config import SLMConfig

    r = SLMConfig.load().retrieval
    return (str(r.sufficiency_judge), str(r.sufficiency_jev_provider or ""),
            r.sufficiency_jev_consent is True)


def cmd_consent(args: Namespace) -> int:
    from superlocalmemory.core import consent_store

    use_json = bool(getattr(args, "json", False))
    action, purpose = getattr(args, "action", "list"), (getattr(args, "purpose", "") or "").strip()
    command = "consent"
    if action == "list":
        rows = [consent_store.state(_home(), p) for p in consent_store.PURPOSES]
        if use_json:
            _out(command, True, data={"consents": rows})
        else:
            for r in rows:
                since = f" since {str(r['at'])[:10]}" if r["given"] and r["at"] else ""
                print(f"{r['purpose']}: {'given' + since if r['given'] else 'not given'}")
        return 0
    if purpose not in consent_store.PURPOSES:
        return _refused(command, f"unknown purpose {purpose!r}; one of: "
                                 + ", ".join(consent_store.PURPOSES), use_json, "UNKNOWN_PURPOSE")
    judge, provider, checked = _answer_check()
    if action == "give":
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return _refused(command, "giving consent asks a person at a terminal; an agent or a "
                                     "script cannot agree for you. Withdrawing works anywhere.",
                            use_json, "NEEDS_A_PERSON")
        if judge != "jev" or not checked:
            return _refused(command, "turn on the online answer check with Jev first (dashboard, "
                                     "Settings); the background judges use the same connection.",
                            use_json, "ANSWER_CHECK_NOT_JEV")
        print(consent_store.notice(purpose, provider))
        if input("Type yes to agree: ").strip().lower() != "yes":
            print("Nothing changed.")
            return 0
    actor = "cli"
    try:
        changed = consent_store.change(_home(), purpose, action == "give", via="cli", actor=actor,
                                       provider=provider or "unknown")
    except consent_store.ConsentNotRecorded:
        return _refused(command, "the consent could not be recorded, so nothing was changed",
                        use_json, "NOT_RECORDED")
    state = consent_store.state(_home(), purpose)
    if use_json:
        _out(command, True, data={"changed": changed, **state})
    else:
        print(("Given." if state["given"] else "Withdrawn.") if changed else "Nothing to change.")
    return 0


def run(args: Namespace) -> None:
    rc = cmd_consent(args)
    if rc:
        sys.exit(rc)


__all__ = ["cmd_consent", "register_consent_parsers", "run"]
