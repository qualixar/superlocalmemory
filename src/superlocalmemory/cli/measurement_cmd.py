# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm measurement``: what SLM measures, and the switch to turn it off (D-28).

    slm measurement                 # status: the notice (first time only) and each setting
    slm measurement off             # stop withholding and swapping (receipts stay)
    slm measurement off --receipts  # stop the receipts as well
    slm measurement on              # back on

Files only (``<SLM_HOME>/features.json``, ``disclosures.json``); no daemon is
needed. Running ``status`` as a person (not ``--json``) counts as having seen
the notice. ``--json`` never does: an agent reading it is not a person reading
it, and it carries the notice text and ``pending`` so the agent can pass it on.
An agent turns this off only when the user asks.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any


def register_measurement_parsers(sub: Any) -> None:
    """Attach ``slm measurement``. Called from cli/main.py."""
    p = sub.add_parser("measurement",
                       help="What SLM measures, and the switch to turn it off: status | on | off")
    p.add_argument("action", nargs="?", default="status", choices=["status", "on", "off"])
    p.add_argument("--receipts", action="store_true",
                   help="with on/off: change the receipts too (ids and numbers kept on this computer)")
    p.add_argument("--json", action="store_true", help="machine-readable output")


def _home():
    from superlocalmemory.infra.data_root import state_path

    return state_path("features.json").parent


def _emit(data: dict, use_json: bool, command: str = "measurement") -> None:
    if use_json:
        from superlocalmemory.cli.json_output import json_print

        json_print(command, data=data)


def _print_status(report: dict) -> None:
    if report["kill_switch"]:
        print("The kill switch is on (slm flags kill off to release it): every 4.2.30 "
              "measure is off.")
    for f in report["flags"]:
        print(f"{f['label']}: {f['state']}. {f['what']}")
    print(f"Measuring: {'yes' if report['measuring'] else 'no'}")


def _status(home, use_json: bool) -> int:
    from superlocalmemory.outcomes import disclosure, switch

    report = switch.status(home)
    if use_json:
        _emit({**report, "notice": disclosure.text()}, True)
        return 0
    if report["pending"]:
        print(disclosure.text())
        print()
        try:
            disclosure.acknowledge(home, "cli")
        except OSError as exc:  # the notice was shown; saying so is best effort
            print(f"(Could not record that you saw this: {exc})")
    _print_status(report)
    return 0


def _change(home, on: bool, with_receipts: bool, use_json: bool) -> int:
    from superlocalmemory.outcomes import switch

    result = switch.set_measurement(home, on, channel="cli", include_receipts=with_receipts)
    report = switch.status(home)
    if use_json:
        _emit({**report, "changed": result["changed"]}, True)
        return 0
    word = "on" if on else "off"
    if result["changed"]:
        names = ", ".join(sorted(result["changed"]))
        print(f"Measuring is now {word} ({names}). It takes effect at the next session start.")
    else:
        print(f"Already {word}; nothing to change.")
    if on is False and not with_receipts:
        print("Receipts (ids and numbers, this computer only) are still written. "
              "Add --receipts to stop them too.")
    return 0


def cmd_measurement(args: Namespace) -> int:
    action = getattr(args, "action", None) or "status"
    use_json = bool(getattr(args, "json", False))
    home = _home()
    if action == "status":
        return _status(home, use_json)
    return _change(home, action == "on", bool(getattr(args, "receipts", False)), use_json)


def announce_if_pending() -> bool:
    """Install and upgrade paths (``slm setup``, ``slm upgrade-hosts``): print the
    notice to stderr when it is pending. No prompt; never raises."""
    try:
        from superlocalmemory.outcomes import disclosure

        return disclosure.announce(_home(), channel="install")
    except Exception:  # noqa: BLE001 - the notice never breaks setup or an upgrade
        return False


def run(args: Namespace) -> None:
    rc = cmd_measurement(args)
    if rc:
        raise SystemExit(rc)


__all__ = ["announce_if_pending", "cmd_measurement", "register_measurement_parsers", "run"]
