# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm media pack status|enable|disable``: the optional picture and document models (WS-8, D-19).

``captions`` describes each saved picture in a few words (on every platform); ``documents`` reads scanned
pages with a larger model. Each is off until turned on, and turning on downloads its model: the size is
shown first and nothing is downloaded without ``--yes`` (or a typed yes at a terminal). The work runs in
the daemon, like ``slm media enable``.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from superlocalmemory.cli.daemon import DaemonConflict, daemon_request
from superlocalmemory.cli.features_cmd import EXIT_DAEMON_DOWN, NOT_RUNNING, die
from superlocalmemory.cli.media_cmd import EXIT_LOW_RAM, _confirmed, _emit, _running

PACKS_PATH = "/api/v3/features/media/packs"
PACK_NAMES = ("captions", "documents", "regions")
EXIT_REFUSED = EXIT_LOW_RAM


def _call(args: Namespace, method: str, body: dict | None = None, suffix: str = "", **kwargs: Any) -> dict[str, Any]:
    data = daemon_request(method, PACKS_PATH + suffix, body, **kwargs)
    if data is None:
        die(args, NOT_RUNNING, EXIT_DAEMON_DOWN)
    return data


def _line(pack: dict[str, Any]) -> str:
    state = {"on": "on", "ready": "downloaded, off", "downloading": "downloading", "failed": "failed",
             "off": "off"}.get(str(pack.get("state")), str(pack.get("state")))
    extra = ""
    if pack.get("state") == "downloading":
        extra = f" ({round(float(pack.get('progress') or 0) * 100)}%)"
    elif pack.get("state") == "failed" and pack.get("error"):
        extra = f" ({pack['error']})"
    elif pack.get("state") == "off":
        extra = f" ({pack.get('download_words')})"
    return f"{pack.get('pack')}: {state}{extra}. {pack.get('what')}"


def _status(args: Namespace) -> None:
    packs = _call(args, "GET")["packs"]
    name = getattr(args, "pack", None)
    shown = [p for p in packs if not name or p["pack"] == name]
    _emit(args, {"packs": shown}, "\n".join(_line(p) for p in shown))


def _enable(args: Namespace) -> None:
    _running(args)
    pack = next(p for p in _call(args, "GET")["packs"] if p["pack"] == args.pack)
    if not getattr(args, "json", False):
        print(pack["download_words"] + f" ({pack['model']}, {pack['licence']} licence).")
        print("The model files stay on this computer. Nothing is downloaded until you say yes.")
    if pack["state"] in ("on", "ready"):
        print("The files are already here; no download.")
    elif not _confirmed(args):
        print("Nothing changed.")
        return
    try:
        reply = _call(args, "POST", {"pack": args.pack, "yes": True, "source": "cli"}, "/enable",
                      preserve_conflict=True)
    except DaemonConflict as exc:
        die(args, str(exc), EXIT_REFUSED)
    got = reply["pack"]
    _emit(args, reply, "Turning on. The download runs in the background; check: slm media pack status "
          + args.pack if got["state"] == "downloading" else _line(got))


def _disable(args: Namespace) -> None:
    reply = _call(args, "POST", {"pack": args.pack, "remove_files": bool(getattr(args, "remove_files", False))},
                  "/disable")
    _emit(args, reply, f"{args.pack} is off. Your saved memories are kept.")


def cmd_pack(args: Namespace) -> None:
    sub = getattr(args, "pack_command", None) or "status"
    args.media_command = f"pack {sub}"
    {"status": _status, "enable": _enable, "disable": _disable}[sub](args)


def register_pack_parser(msub: Any, json_flag: dict[str, Any]) -> None:
    p = msub.add_parser("pack", help="optional picture models: describe pictures, read scanned documents better")
    p.add_argument("--json", **json_flag)
    psub = p.add_subparsers(dest="pack_command", title="pack subcommands")
    s = psub.add_parser("status", help="what is on, and the download size of each part")
    s.add_argument("pack", nargs="?", choices=PACK_NAMES)
    s.add_argument("--json", **json_flag)
    e = psub.add_parser("enable", help="turn a part on (shows the download size first)")
    e.add_argument("pack", choices=PACK_NAMES)
    e.add_argument("--yes", "-y", action="store_true", help="do not ask")
    e.add_argument("--json", **json_flag)
    d = psub.add_parser("disable", help="turn a part off; memories are kept")
    d.add_argument("pack", choices=PACK_NAMES)
    d.add_argument("--remove-files", action="store_true", help="also delete the downloaded model")
    d.add_argument("--json", **json_flag)
