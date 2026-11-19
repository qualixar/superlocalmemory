# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm media pack status|enable|disable``: the optional picture and document models (WS-8, D-19).

``captions`` describes each saved picture in a few words (on every platform); ``documents`` reads scanned
pages with a larger model. Both are on by default: on a computer with 16 GB or more the daemon downloads each
model by itself; on a smaller one this command is the one click (the size is shown first and nothing is
downloaded by hand without ``--yes`` or a typed yes). The work runs in the daemon, like ``slm media enable``.
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
    """One part in plain words: ``captions: Downloading the picture description model, 40%. <what it does>``."""
    words = str(pack.get("words") or "").strip()
    if not words:  # an older daemon that does not word its own state
        words = {"on": "on", "ready": "downloaded, off", "downloading": "downloading", "failed": "failed",
                 "off": "off"}.get(str(pack.get("state")), str(pack.get("state")))
    if not words.endswith((".", "!", "?")):
        words += "."
    return f"{pack.get('pack')}: {words} {pack.get('what')}"


def _warning(packs: list[dict[str, Any]]) -> str:
    return next((str(p["warning"]) for p in packs if p.get("warning")), "")


def _status(args: Namespace) -> None:
    packs = _call(args, "GET")["packs"]
    name = getattr(args, "pack", None)
    shown = [p for p in packs if not name or p["pack"] == name]
    lines = [_line(p) for p in shown]
    if _warning(shown):
        lines.append(_warning(shown))
    _emit(args, {"packs": shown}, "\n".join(lines))


def _enable(args: Namespace) -> None:
    _running(args)
    pack = next(p for p in _call(args, "GET")["packs"] if p["pack"] == args.pack)
    if not getattr(args, "json", False):
        if pack.get("warning"):  # a small computer is warned, never refused
            print(pack["warning"])
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
