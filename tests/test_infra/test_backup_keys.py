# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Backup key management: one key per machine, kept in the credential store,
shown to the user as a recovery key, never logged, never silently replaced."""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import base64
import json
import logging
import threading

import pytest
from _backup_key_env import files_containing, key_env  # noqa: F401

from superlocalmemory.infra import backup_crypto as bc
from superlocalmemory.infra import backup_keys as bk

_KEY = bytes(range(32))


def _needles(key: bytes) -> list[bytes]:
    return [
        key,
        key.hex().encode(),
        base64.urlsafe_b64encode(key).rstrip(b"="),
        bk.format_recovery_key(key).encode(),
        bk.format_recovery_key(key).replace("-", "").encode(),
    ]


# ---- recovery key encoding --------------------------------------------------


def test_recovery_key_round_trips() -> None:
    text = bk.format_recovery_key(_KEY)
    assert text.startswith("SLMBK1-")
    assert all(len(g) == 4 for g in text.split("-")[1:])
    assert bk.parse_recovery_key(text) == _KEY


def test_recovery_key_parse_tolerates_case_spaces_and_lookalikes() -> None:
    text = bk.format_recovery_key(_KEY)
    sloppy = "  " + text.lower().replace("-", " ") + "\n"
    assert bk.parse_recovery_key(sloppy) == _KEY
    body = text.split("-", 1)[1].replace("O", "0").replace("I", "1")
    assert bk.parse_recovery_key("SLMBK1-" + body) == _KEY


def test_mistyped_recovery_key_is_reported_as_a_typo() -> None:
    text = bk.format_recovery_key(_KEY)
    groups = text.split("-")
    groups[3] = "AAAA" if groups[3] != "AAAA" else "BBBB"
    with pytest.raises(bk.RecoveryKeyFormatError, match="mistyped"):
        bk.parse_recovery_key("-".join(groups))


@pytest.mark.parametrize("bad", ["", "SLMBK1-ABCD", "hello world", "SLMBK1-" + "!" * 56])
def test_malformed_recovery_key_is_rejected(bad: str) -> None:
    with pytest.raises(bk.RecoveryKeyFormatError):
        bk.parse_recovery_key(bad)


# ---- key creation and storage -----------------------------------------------


def test_key_is_created_once_and_kept_in_the_credential_store(key_env) -> None:
    key, created = bk.ensure_backup_key(origin="connect")
    again, created_again = bk.ensure_backup_key(origin="backup")
    assert created is True and created_again is False
    assert again == key and len(key) == 32
    store = json.loads((key_env / ".credentials.json").read_text(encoding="utf-8"))
    assert bk.CREDENTIAL_NAME in store
    assert bk.load_backup_key() == key


def test_key_is_never_written_next_to_backups_or_into_state(key_env) -> None:
    backups = key_env / "backups"
    backups.mkdir()
    (backups / "memory-20260101-000000.db").write_bytes(b"SQLite format 3\x00")
    key, _ = bk.ensure_backup_key(origin="backup")
    assert files_containing(backups, _needles(key)) == []
    state = bk.state_path()
    assert state.exists()
    assert files_containing(state.parent, _needles(key)) == [key_env / ".credentials.json"]


def test_missing_key_with_a_recorded_key_id_fails_closed(key_env) -> None:
    key, _ = bk.ensure_backup_key(origin="backup")
    import superlocalmemory.infra.cloud_backup as cb

    cb._delete_credential(bk.CREDENTIAL_NAME)
    with pytest.raises(bk.BackupKeyUnavailableError) as info:
        bk.ensure_backup_key(origin="backup")
    assert bc.key_id_hex(key) in str(info.value)
    assert "recovery-key --import" in str(info.value)
    assert cb._get_credential(bk.CREDENTIAL_NAME) is None  # no silent replacement


def test_store_failure_raises_and_records_nothing(key_env, monkeypatch) -> None:
    import superlocalmemory.infra.cloud_backup as cb

    monkeypatch.setattr(cb, "_store_credential", lambda k, v: False)
    with pytest.raises(bk.BackupKeyUnavailableError):
        bk.ensure_backup_key(origin="backup")
    assert not bk.state_path().exists()


def test_store_that_does_not_persist_is_detected(key_env, monkeypatch) -> None:
    import superlocalmemory.infra.cloud_backup as cb

    monkeypatch.setattr(cb, "_store_credential", lambda k, v: True)
    with pytest.raises(bk.BackupKeyUnavailableError):
        bk.ensure_backup_key(origin="backup")


def test_corrupt_stored_key_is_not_replaced(key_env) -> None:
    import superlocalmemory.infra.cloud_backup as cb

    cb._store_credential(bk.CREDENTIAL_NAME, "garbage")
    with pytest.raises(bk.BackupKeyUnavailableError):
        bk.ensure_backup_key(origin="backup")
    assert cb._get_credential(bk.CREDENTIAL_NAME) == "garbage"


def test_concurrent_first_use_converges_on_one_key(key_env) -> None:
    results: list[bytes] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(bk.ensure_backup_key(origin="backup")[0])
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(results)) == 1
    assert bk.load_backup_key() == results[0]


# ---- recovery key reveal, notices and import ---------------------------------


def test_upgrade_notice_is_pending_until_the_key_is_viewed(key_env) -> None:
    key, _ = bk.ensure_backup_key(origin="backup", legacy_uploads=True)
    status = bk.encryption_status()
    assert status["enabled"] is True
    assert status["key_id"] == bc.key_id_hex(key)
    assert status["recovery_key_pending"] is True
    assert any("slm backup recovery-key" in n for n in status["notices"])
    assert any("not encrypted" in n for n in status["notices"])

    shown = bk.reveal_recovery_key()
    assert bk.parse_recovery_key(shown["recovery_key"]) == key
    after = bk.encryption_status()
    assert after["recovery_key_pending"] is False
    assert any("not encrypted" in n for n in after["notices"])  # legacy note stays


def test_status_without_a_key(key_env) -> None:
    status = bk.encryption_status()
    assert status["enabled"] is False and status["key_id"] is None


def test_reveal_without_a_key_raises(key_env) -> None:
    with pytest.raises(bk.BackupKeyUnavailableError):
        bk.reveal_recovery_key()


def test_import_installs_a_recovery_key_on_a_new_machine(key_env) -> None:
    result = bk.install_recovery_key(bk.format_recovery_key(_KEY))
    assert result["key_id"] == bc.key_id_hex(_KEY)
    assert bk.load_backup_key() == _KEY
    assert bk.encryption_status()["recovery_key_pending"] is False


def test_import_refuses_to_replace_a_different_key(key_env) -> None:
    key, _ = bk.ensure_backup_key(origin="backup")
    with pytest.raises(bk.BackupKeyError, match="different"):
        bk.install_recovery_key(bk.format_recovery_key(_KEY))
    assert bk.load_backup_key() == key
    # Importing the same key again is harmless.
    bk.install_recovery_key(bk.format_recovery_key(key))


def test_import_repairs_the_fail_closed_state(key_env) -> None:
    key, _ = bk.ensure_backup_key(origin="backup")
    import superlocalmemory.infra.cloud_backup as cb

    cb._delete_credential(bk.CREDENTIAL_NAME)
    bk.install_recovery_key(bk.format_recovery_key(key))
    assert bk.ensure_backup_key(origin="backup") == (key, False)


def test_no_key_material_reaches_the_logs(key_env, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    key, _ = bk.ensure_backup_key(origin="backup", legacy_uploads=True)
    bk.encryption_status()
    bk.reveal_recovery_key()
    with pytest.raises(bk.RecoveryKeyFormatError):
        bk.parse_recovery_key(bk.format_recovery_key(key)[:-2] + "ZZ")
    logged = caplog.text.encode()
    assert caplog.records, "creation should be logged (without the key)"
    for needle in _needles(key):
        assert needle not in logged
