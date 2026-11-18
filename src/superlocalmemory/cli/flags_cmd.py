# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm flags``: show and set the 4.2.30 subsystem flags and the kill switch.

    slm flags                       every flag, its state, and the kill switch
    slm flags set <flag> on|shadow|off
    slm flags kill on|off           D-23: every 4.2.30 subsystem off at once
    slm flags rollback-retirements [--dry-run]
                                    D-23: bring back memories 4.2.30 retired (says how many first;
                                    the kill switch does not undo retirements)

Files only (``<SLM_HOME>/features.json``); no daemon is needed. Subsystems
read the flags at their next run (session start, the maintenance pass).
(``slm features`` is a different command: images, documents and mesh.)
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace

from superlocalmemory.core.feature_flags import (
    FLAGS,
    FeatureFlags,
    FlagState,
    kill_switch_on,
    set_kill_switch,
    write_flag,
)


def _home():
    from superlocalmemory.infra.data_root import state_path

    return state_path("features.json").parent


def _show(home, as_json: bool) -> None:
    flags = FeatureFlags(home)
    states = flags.as_dict()
    if as_json:
        print(json.dumps({"kill_switch": flags.killed, "flags": states}, indent=2))
        return
    print(f"kill switch: {'ON (every 4.2.30 subsystem off)' if flags.killed else 'off'}")
    width = max(len(f) for f in FLAGS)
    for flag_id, flag in FLAGS.items():
        print(f"  {flag_id:<{width}}  {states[flag_id]:<6}  {flag.summary}")


def _kill_through_daemon(on: bool) -> bool:
    """Set the switch through a running daemon, which records the decision (D-49) with channel
    ``cli``. False when no daemon answers: the caller then sets the file itself, unrecorded."""
    from superlocalmemory.cli.daemon import DaemonNotFound, daemon_request

    try:
        answer = daemon_request("POST", "/api/flags/kill", {"on": on})
    except DaemonNotFound:
        return False
    except Exception:  # noqa: BLE001 - the switch must still move; the record is what is lost
        return False
    # Only a daemon that set it and says so counts: no daemon answers None, and an older one has
    # no such route.
    state = answer.get("kill_switch") if isinstance(answer, dict) else None
    return bool(isinstance(state, dict) and answer.get("success") and state.get("on") is on)


def cmd_flags(args: Namespace) -> int:
    home = _home()
    action = getattr(args, "action", None)
    if action is None:
        _show(home, getattr(args, "json", False))
        return 0
    value = str(getattr(args, "value", "") or "").strip().lower()
    if action == "kill":
        if not _kill_through_daemon(value == "on"):
            set_kill_switch(home, value == "on")
            print("(SuperLocalMemory is not running, so this change is not in the list of "
                  "decisions)", file=sys.stderr)
        print(f"kill switch {'on' if kill_switch_on(home, {}) else 'off'}")
        if value == "on":
            from superlocalmemory.core.kill_switch import DOES_NOT_WORDS, DOES_WORDS

            print(DOES_WORDS)
            print(DOES_NOT_WORDS)
        return 0
    if action == "rollback-retirements":
        from superlocalmemory.cli.flags_rollback import rollback_retirements

        return rollback_retirements(args)
    flag_id = getattr(args, "flag", "") or ""
    if flag_id not in FLAGS:
        print(f"unknown flag {flag_id!r}; one of: {', '.join(FLAGS)}")
        return 2
    write_flag(home, flag_id, FlagState(value))
    print(f"{flag_id} = {FeatureFlags(home, environ={}).state(flag_id).value}")
    return 0


__all__ = ["cmd_flags"]
