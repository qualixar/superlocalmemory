# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The kill switch as every surface shows and sets it (D-23, blueprint §8.10.3 (k)).

One place for the state and the plain words, so the command line, HTTP, MCP and the dashboard
say the same thing. The switch itself lives in ``core/feature_flags`` (``kill_switch`` in
``features.json``, or ``SLM_4230_OFF`` in the environment); this module only reads and sets it.

D-23: the switch returns the store to pre-4.2.30 read behaviour without a downgrade. It stops
every 4.2.30 pass and makes reads ignore slots, leases, disputes and taint. It does not undo
retirements: those come back through the separate ledger command
(``truth.retirement_rollback``, ``slm flags rollback-retirements``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from superlocalmemory.core.feature_flags import KILL_ENV, kill_switch_on, set_kill_switch

DOES_WORDS = ("Stops everything new in 4.2.30 (the background passes, the checks on what you "
              "save, and the extra notes on recalled memories) and makes SuperLocalMemory read "
              "your memories the way it did before 4.2.30. Nothing is deleted and you do not "
              "need to downgrade.")
DOES_NOT_WORDS = ("It does not bring back memories that 4.2.30 already retired. Those come back "
                  "with \"Bring back memories that 4.2.30 retired\" (slm flags "
                  "rollback-retirements).")
ENVIRONMENT_WORDS = (f"The environment variable {KILL_ENV} is set for SuperLocalMemory, so "
                     "everything new stays off until it is removed.")


def view(home: Path | None) -> dict[str, Any]:
    """``on``, where it comes from (``file``, ``environment``, ``both`` or ``off``) and words."""
    in_file = kill_switch_on(home, {})
    in_env = kill_switch_on(None)
    source = "both" if in_file and in_env else "file" if in_file else (
        "environment" if in_env else "off")
    return {"on": in_file or in_env, "source": source,
            "words": ("Everything new in 4.2.30 is off." if in_file or in_env
                      else "Everything new in 4.2.30 is working as set.")
            + (" " + ENVIRONMENT_WORDS if in_env else "")}


def current() -> dict[str, Any]:
    """:func:`view` of this process's data folder; never raises (a status call must not fail)."""
    try:
        from superlocalmemory.infra.data_root import canonical_data_root

        return view(canonical_data_root())
    except Exception:  # noqa: BLE001 - one line of a status answer is never worth failing it
        return {"on": False, "source": "unknown", "words": ""}


def set_state(home: Path, on: bool) -> dict[str, Any]:
    """Set or clear the switch in ``features.json`` and return the new :func:`view`."""
    set_kill_switch(home, on)
    return view(home)


def record(conn: Any, *, profile_id: str, on: bool, decider: Any) -> str:
    """Record the change as a decision (D-49): who, through which channel, when; undoable."""
    from superlocalmemory.core import decisions

    summary = ("Turned off everything new in 4.2.30" if on
               else "Turned the 4.2.30 features back on")
    return decisions.log(conn, profile_id=profile_id, kind=decisions.KILL_SWITCH,
                         subject_id="kill_switch", action="on" if on else "off", decider=decider,
                         summary=summary, detail={"on": on})


def undo(conn: Any, *, found: dict, decider: Any, home: Path | None = None) -> dict[str, Any]:
    """Set the switch back to how it was before ``found``, only if nobody changed it since.

    ``ValueError`` (in words) when it was changed again after this decision: undoing it then
    would silently overrule the later change.
    """
    from superlocalmemory.core import decisions
    from superlocalmemory.infra.data_root import canonical_data_root

    home = home or canonical_data_root()
    was_on = found["action"] == "on"
    if kill_switch_on(home, {}) != was_on:
        raise ValueError("The switch was changed again after this decision, so this undo would "
                         "overrule that later change. Nothing was changed; use the switch itself.")
    set_kill_switch(home, not was_on)
    decisions.mark_undone(conn, found["decision_id"], decider)
    return {"kill_switch": "off" if was_on else "on"}


__all__ = ["DOES_NOT_WORDS", "DOES_WORDS", "ENVIRONMENT_WORDS", "current", "record", "set_state",
           "undo", "view"]
