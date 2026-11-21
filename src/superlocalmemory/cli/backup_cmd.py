# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm backup` — cloud backup encryption from the command line.

  slm backup status                  encryption status and any notices
  slm backup recovery-key            show this computer's recovery key
  slm backup recovery-key --import   put a recovery key on this computer
  slm backup decrypt FILE [-o OUT]   turn a downloaded backup into a .db file
  slm backup restore-safety-copy     put back the safety copy a failed dashboard
                                     restore left (service stopped)
  slm backup clear-recovery-needed   let the service start after a manual repair

A recovery key is only ever read from stdin (or a hidden prompt), never from
the command line, so it stays out of shell history and the process list.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any


def add_backup_parser(sub: Any) -> None:
    backup_p = sub.add_parser(
        "backup", help="Cloud backup encryption: status, recovery key, decrypt a download",
    )
    backup_sub = backup_p.add_subparsers(dest="backup_command", title="backup subcommands")

    status_p = backup_sub.add_parser("status", help="Show cloud backup encryption status")
    status_p.add_argument("--json", action="store_true", help="Output JSON")

    key_p = backup_sub.add_parser(
        "recovery-key", help="Show the recovery key, or --import one on a new computer",
    )
    key_p.add_argument("--import", dest="import_key", action="store_true",
                       help="Read a recovery key from stdin and use it on this computer")
    key_p.add_argument("--json", action="store_true", help="Output JSON")

    dec_p = backup_sub.add_parser(
        "decrypt", help="Turn a downloaded cloud backup into a plain .db file",
    )
    dec_p.add_argument("file", help="The downloaded backup (.slmenc, or an old .db)")
    dec_p.add_argument("-o", "--out", default=None,
                       help="Where to write the .db (default: beside the download)")
    dec_p.add_argument("--recovery-key-stdin", dest="recovery_key_stdin", action="store_true",
                       help="Read the recovery key from stdin instead of using this computer's key")
    dec_p.add_argument("--force", action="store_true", help="Overwrite the output file")
    dec_p.add_argument("--json", action="store_true", help="Output JSON")

    safe_p = backup_sub.add_parser(
        "restore-safety-copy",
        help="Put back the safety copy a failed dashboard restore kept (service stopped)",
    )
    safe_p.add_argument("--id", default=None,
                        help="Recovery point to restore (default: the last restore's safety copy)")
    safe_p.add_argument("--json", action="store_true", help="Output JSON")

    clear_p = backup_sub.add_parser(
        "clear-recovery-needed",
        help="Let the service start again after you repaired the stores yourself",
    )
    clear_p.add_argument("--json", action="store_true", help="Output JSON")


def cmd_backup(args: argparse.Namespace) -> None:
    from superlocalmemory.infra.backup_crypto import BackupCryptoError
    from superlocalmemory.server.backup_restore_txn import RolledBack, StoreBusy
    from superlocalmemory.server.backup_rollback import NotEnoughSpace

    handlers = {"status": _status, "recovery-key": _recovery_key, "decrypt": _decrypt,
                "restore-safety-copy": _restore_safety_copy,
                "clear-recovery-needed": _clear_recovery_needed}
    handler = handlers.get(getattr(args, "backup_command", None) or "status")
    try:
        handler(args)
    except (BackupCryptoError, FileExistsError, FileNotFoundError, ValueError) as exc:
        _fail(args, str(exc))
    except NotEnoughSpace as exc:
        _fail(args, str(exc))
    except StoreBusy:
        _fail(args, "SuperLocalMemory is running. Stop it with `slm serve stop`, then retry.")
    except RolledBack:
        _fail(args, "The safety copy could not be restored; nothing was changed.")


def _data_root() -> Path:
    from superlocalmemory.core.config import SLMConfig

    return Path(SLMConfig.load().db_path).parent


def _clear_recovery_needed(args: argparse.Namespace) -> None:
    from superlocalmemory.server.backup_restore_txn import clear_recovery_needed

    cleared = clear_recovery_needed(_data_root())
    text = ("Cleared: the service may start again. Only do this after repairing the stores; "
            "`slm backup restore-safety-copy` is the safe way back." if cleared
            else "Nothing to clear: no restore is waiting for recovery.")
    _emit(args, {"cleared": cleared}, text)


def _restore_safety_copy(args: argparse.Namespace) -> None:
    from superlocalmemory.server.backup_restore_txn import OUTCOME_NAME, recover_safety_copy
    from superlocalmemory.storage._durable_json import read_json

    root = _data_root()
    point_id = args.id or (read_json(root / OUTCOME_NAME) or {}).get("safety_id")
    if not point_id:
        raise ValueError("No safety copy is recorded. Pass --id with a recovery point.")
    result = recover_safety_copy(root, str(point_id))
    _emit(args, result, (
        f"Safety copy {result['restored']} restored. The state just before this command "
        f"was kept as recovery point {result['safety_id']}. Start the service with "
        "`slm serve start`."))


def _status(args: argparse.Namespace) -> None:
    from superlocalmemory.infra.backup_keys import encryption_status

    status = {**encryption_status(), "recovery": _recovery_status()}
    if getattr(args, "json", False):
        print(json.dumps(status, indent=2))
        return
    if status["enabled"]:
        print(f"Cloud backups are encrypted on this computer before upload (key {status['key_id']}).")
    else:
        print("No backup encryption key yet. One is created when you connect a cloud "
              "destination or make the first cloud backup.")
    for notice in status["notices"]:
        print(f"\n{notice}")
    if status["recovery"]:
        print(f"\n{status['recovery']['message']}")


def _recovery_status() -> dict | None:
    """A restore waiting for recovery, with the pre-restore copy it kept (or None)."""
    from superlocalmemory.server import recovery_marker

    try:
        root = _data_root()
        marker = recovery_marker.read(root)
        message = recovery_marker.refusal(root)
    except Exception:  # noqa: BLE001 - status never fails on this optional line
        return None
    if marker is None or not message:
        return None
    copies = recovery_marker.kept_copies(marker)
    return {"phase": marker.get("phase"), "safety_id": marker.get("safety_id"),
            "kept_copy": copies[0] if copies else None, "message": message}


def _recovery_key(args: argparse.Namespace) -> None:
    from superlocalmemory.infra import backup_keys as bk

    if args.import_key:
        result = bk.install_recovery_key(_read_secret("Recovery key: "))
        verb = "installed" if result["installed"] else "already in use"
        _emit(args, result, f"Backup key {result['key_id']} {verb} on this computer.")
        return
    shown = bk.reveal_recovery_key()
    lines = [
        f"Recovery key (key {shown['key_id']}):",
        "",
        f"  {shown['recovery_key']}",
        "",
        shown["advice"],
    ]
    legacy = bk.encryption_status()
    if legacy["legacy_plaintext_uploads"]:
        lines += ["", bk.LEGACY_NOTICE]
    _emit(args, shown, "\n".join(lines))


def _decrypt(args: argparse.Namespace) -> None:
    from superlocalmemory.infra.cloud_backup_crypto import restore_backup_file

    recovery = _read_secret("Recovery key: ") if args.recovery_key_stdin else None
    result = restore_backup_file(
        Path(args.file).expanduser(),
        Path(args.out).expanduser() if args.out else None,
        recovery_key=recovery,
        overwrite=args.force,
    )
    text = (
        f"{result['message']}\nWritten to: {result['output']}\n"
        "To use it, stop SuperLocalMemory (`slm serve stop`) and put this file in "
        "place of the database it was copied from (for example memory.db)."
    )
    _emit(args, result, text)


def _read_secret(prompt: str) -> str:
    if sys.stdin is not None and sys.stdin.isatty():
        return getpass.getpass(prompt)
    return sys.stdin.readline().strip() if sys.stdin is not None else ""


def _emit(args: argparse.Namespace, data: dict[str, Any], text: str) -> None:
    print(json.dumps(data, indent=2) if getattr(args, "json", False) else text)


def _fail(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps({"success": False, "error": {"code": "backup_failed",
                                                      "message": message}}))
    else:
        print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)
