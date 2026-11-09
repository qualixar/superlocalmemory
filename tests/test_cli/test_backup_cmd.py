# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm backup`: encryption status, the recovery key, and decrypting a download.

The recovery key is read from stdin, never from the command line, so it does
not end up in shell history or the process list.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from superlocalmemory.cli import backup_cmd
from superlocalmemory.infra import backup_crypto as bc
from superlocalmemory.infra import backup_keys as bk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_infra"))
from _backup_key_env import key_env  # noqa: E402,F401


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    backup_cmd.add_backup_parser(sub)
    return parser.parse_args(argv)


def _run(argv: list[str], monkeypatch, stdin: str = "") -> int:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    try:
        backup_cmd.cmd_backup(_parse(argv))
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


@pytest.fixture
def encrypted(key_env) -> dict:
    key, _ = bk.ensure_backup_key(origin="backup", legacy_uploads=True)
    plain = key_env / "memory-1.db"
    plain.write_bytes(bc.SQLITE_MAGIC + b"payload" * 100)
    enc = key_env / ("memory-1.db" + bc.ENCRYPTED_SUFFIX)
    bc.encrypt_file(plain, enc, key)
    return {"key": key, "plain": plain, "enc": enc}


def test_status_reports_the_pending_recovery_key(encrypted, monkeypatch, capsys) -> None:
    assert _run(["backup", "status"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert bc.key_id_hex(encrypted["key"]) in out
    assert "slm backup recovery-key" in out
    assert bk.format_recovery_key(encrypted["key"]) not in out


def test_recovery_key_prints_the_key_and_clears_the_notice(encrypted, monkeypatch, capsys) -> None:
    assert _run(["backup", "recovery-key"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert bk.format_recovery_key(encrypted["key"]) in out
    assert bk.encryption_status()["recovery_key_pending"] is False


def test_recovery_key_json(encrypted, monkeypatch, capsys) -> None:
    assert _run(["backup", "recovery-key", "--json"], monkeypatch) == 0
    data = json.loads(capsys.readouterr().out)
    assert bk.parse_recovery_key(data["recovery_key"]) == encrypted["key"]


def test_decrypt_with_the_local_key(encrypted, monkeypatch, capsys) -> None:
    out = encrypted["plain"].parent / "restored.db"
    assert _run(["backup", "decrypt", str(encrypted["enc"]), "-o", str(out)], monkeypatch) == 0
    assert out.read_bytes() == encrypted["plain"].read_bytes()


def test_decrypt_with_a_recovery_key_from_stdin(encrypted, key_env, monkeypatch, capsys) -> None:
    recovery = bk.format_recovery_key(encrypted["key"])
    (key_env / ".credentials.json").unlink()
    bk.state_path().unlink()
    out = key_env / "restored.db"
    code = _run(["backup", "decrypt", str(encrypted["enc"]), "-o", str(out),
                 "--recovery-key-stdin"], monkeypatch, stdin=recovery + "\n")
    assert code == 0
    assert out.read_bytes() == encrypted["plain"].read_bytes()
    assert recovery not in capsys.readouterr().out


def test_decrypt_with_a_wrong_recovery_key_fails_cleanly(encrypted, key_env, monkeypatch, capsys) -> None:
    out = key_env / "restored.db"
    code = _run(["backup", "decrypt", str(encrypted["enc"]), "-o", str(out),
                 "--recovery-key-stdin"], monkeypatch,
                stdin=bk.format_recovery_key(bytes(32)) + "\n")
    assert code == 1
    assert "encrypted with key" in capsys.readouterr().err
    assert not out.exists()


def test_import_recovery_key_from_stdin(key_env, monkeypatch) -> None:
    key = bytes(range(32))
    assert _run(["backup", "recovery-key", "--import"], monkeypatch,
                stdin=bk.format_recovery_key(key)) == 0
    assert bk.load_backup_key() == key


def test_no_option_accepts_a_key_on_the_command_line() -> None:
    with pytest.raises(SystemExit):
        _parse(["backup", "decrypt", "f", "--recovery-key", "SLMBK1-AAAA"])


def test_restoring_on_a_new_computer_does_not_start_the_setup_wizard(monkeypatch) -> None:
    from superlocalmemory.cli import setup_wizard

    def wizard_would_run() -> bool:
        raise AssertionError("first-use setup ran before `slm backup`")

    monkeypatch.setattr(setup_wizard, "is_setup_complete", wizard_would_run)
    setup_wizard.check_first_use("backup")
