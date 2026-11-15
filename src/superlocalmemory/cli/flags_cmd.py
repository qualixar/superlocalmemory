# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm flags``: show and set the 4.2.30 subsystem flags and the kill switch.

    slm flags                       every flag, its state, and the kill switch
    slm flags set <flag> on|shadow|off
    slm flags kill on|off           D-23: every 4.2.30 subsystem off at once

Files only (``<SLM_HOME>/features.json``); no daemon is needed. Subsystems
read the flags at their next run (session start, the maintenance pass).
(``slm features`` is a different command: images, documents and mesh.)
"""

from __future__ import annotations

import json
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


def cmd_flags(args: Namespace) -> int:
    home = _home()
    action = getattr(args, "action", None)
    if action is None:
        _show(home, getattr(args, "json", False))
        return 0
    value = str(getattr(args, "value", "") or "").strip().lower()
    if action == "kill":
        set_kill_switch(home, value == "on")
        print(f"kill switch {'on' if kill_switch_on(home, {}) else 'off'}")
        return 0
    flag_id = getattr(args, "flag", "") or ""
    if flag_id not in FLAGS:
        print(f"unknown flag {flag_id!r}; one of: {', '.join(FLAGS)}")
        return 2
    write_flag(home, flag_id, FlagState(value))
    print(f"{flag_id} = {FeatureFlags(home, environ={}).state(flag_id).value}")
    return 0


__all__ = ["cmd_flags"]
