# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Cloud backup uploads only ciphertext, and every backup can be restored.

All HTTP is mocked: GitHub through ``httpx`` and Google Drive through a fake
service object. The bytes handed to each mock are captured and checked.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
from _backup_key_env import key_env  # noqa: F401

from superlocalmemory.infra import backup_crypto as bc
from superlocalmemory.infra import backup_keys as bk
from superlocalmemory.infra import cloud_backup as cb
from superlocalmemory.infra import cloud_backup_crypto as cbc

PLANTED = "ghp_PLANTEDcredentialDoNotUpload0123456789"
TS = "20261003-101500-cloud-sync"


# ---- fixtures ----------------------------------------------------------------


def _make_db(path: Path, rows: list[str]) -> Path:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE facts (content TEXT)")
    conn.executemany("INSERT INTO facts VALUES (?)", [(r,) for r in rows])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def backup_set(key_env) -> list[Path]:
    backups = key_env / "backups"
    backups.mkdir()
    memory = _make_db(backups / f"memory-{TS}.db",
                      [f"my github token is {PLANTED}", "x" * 5000])
    learning = _make_db(backups / f"learning-{TS}.db", [f"note: {PLANTED}"])
    return [memory, learning]


def _destinations_db(data: Path, rows: list[tuple[str, str, str, str | None]]) -> Path:
    db = data / "memory.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE backup_destinations (id TEXT PRIMARY KEY, destination_type TEXT, "
        "display_name TEXT, credentials_ref TEXT, config TEXT, created_at TEXT, "
        "enabled INTEGER, last_sync_at TEXT, last_sync_status TEXT, last_sync_error TEXT)"
    )
    for dest_id, kind, config, status in rows:
        conn.execute(
            "INSERT INTO backup_destinations VALUES (?, ?, ?, '', ?, '2026-01-01', 1, ?, ?, '')",
            (dest_id, kind, kind, config, "2026-09-01" if status else None, status),
        )
    conn.commit()
    conn.close()
    return db


class _Resp:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> Any:
        return self._payload


@pytest.fixture
def github(monkeypatch, key_env) -> dict[str, Any]:
    captured: dict[str, Any] = {"assets": {}, "release_bodies": [], "posts": 0}
    cb._store_credential("github_pat", "ghp_testtoken_for_mock_only")

    def post(url, **kw):
        captured["posts"] += 1
        if url.endswith("/releases"):
            captured["release_bodies"].append(json.dumps(kw.get("json")))
            return _Resp(201, {"upload_url": "https://uploads.invalid/assets{?name}"})
        content = kw["content"]
        data = content if isinstance(content, bytes) else content.read()
        captured["assets"][kw["params"]["name"]] = data
        return _Resp(201, {})

    def get(url, **kw):
        if url.endswith("/user"):
            return _Resp(200, {"login": "tester"})
        if url.endswith("/commits"):
            return _Resp(200, [{"sha": "1"}])
        if url.endswith("/releases"):
            return _Resp(200, [])
        return _Resp(200, {})

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "put", lambda *a, **k: _Resp(201, {}))
    monkeypatch.setattr(httpx, "delete", lambda *a, **k: _Resp(204, {}))
    return captured


class _Call:
    def __init__(self, result: Any) -> None:
        self._result = result

    def execute(self) -> Any:
        return self._result


class _FakeFiles:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.captured = captured

    def list(self, **kw):
        return _Call({"files": []})

    def create(self, body=None, media_body=None, fields=None):
        if media_body is not None:
            self.captured["uploads"][body["name"]] = media_body
        return _Call({"id": "folder-or-file"})

    def update(self, fileId=None, media_body=None):  # noqa: N803 - Drive API name
        return _Call({})


@pytest.fixture
def drive(monkeypatch, key_env) -> dict[str, Any]:
    captured: dict[str, Any] = {"uploads": {}}
    files = _FakeFiles(captured)
    service = types.SimpleNamespace(files=lambda: files)
    monkeypatch.setattr(cb, "_get_drive_service", lambda: service)

    class MediaFileUpload:
        def __init__(self, filename, mimetype=None, resumable=False):
            self.data = Path(filename).read_bytes()
            self.mimetype = mimetype

    http_mod = types.ModuleType("googleapiclient.http")
    http_mod.MediaFileUpload = MediaFileUpload
    pkg = sys.modules.get("googleapiclient") or types.ModuleType("googleapiclient")
    monkeypatch.setitem(sys.modules, "googleapiclient", pkg)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http_mod)
    return captured


def _decrypt_bytes(blob: bytes, key: bytes) -> bytes:
    out = io.BytesIO()
    bc.decrypt_stream(io.BytesIO(blob), out, key)
    return out.getvalue()


# ---- uploads are ciphertext ------------------------------------------------------


def test_github_assets_are_ciphertext_without_the_planted_credential(github, backup_set) -> None:
    assert cb.sync_to_github(backup_set, {"full_repo": "tester/slm-backup"}) is True
    key = bk.load_backup_key()
    assert set(github["assets"]) == {p.name + bc.ENCRYPTED_SUFFIX for p in backup_set}
    for original in backup_set:
        blob = github["assets"][original.name + bc.ENCRYPTED_SUFFIX]
        assert blob.startswith(bc.MAGIC)
        assert PLANTED.encode() not in blob
        assert bc.SQLITE_MAGIC not in blob
        assert _decrypt_bytes(blob, key) == original.read_bytes()
    assert all(PLANTED not in body for body in github["release_bodies"])


def test_drive_uploads_are_ciphertext_without_the_planted_credential(drive, backup_set) -> None:
    for path in backup_set:
        assert cb.sync_to_google_drive(path, {"folder": "SLM-Backup"}) is True
    key = bk.load_backup_key()
    assert set(drive["uploads"]) == {p.name + bc.ENCRYPTED_SUFFIX for p in backup_set}
    for original in backup_set:
        media = drive["uploads"][original.name + bc.ENCRYPTED_SUFFIX]
        assert media.mimetype == "application/octet-stream"
        assert PLANTED.encode() not in media.data
        assert _decrypt_bytes(media.data, key) == original.read_bytes()


def test_no_upload_happens_when_the_key_cannot_be_saved(github, drive, backup_set, monkeypatch) -> None:
    real_store = cb._store_credential

    def refuse_backup_key(name, value):
        return False if name == bk.CREDENTIAL_NAME else real_store(name, value)

    monkeypatch.setattr(cb, "_store_credential", refuse_backup_key)
    assert cb.sync_to_github(backup_set, {"full_repo": "tester/slm-backup"}) is False
    assert cb.sync_to_google_drive(backup_set[0], {}) is False
    assert github["posts"] == 0  # not even an empty release
    assert drive["uploads"] == {}


def test_temporary_ciphertext_is_removed_after_upload(github, backup_set, tmp_path, monkeypatch) -> None:
    import tempfile

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert cb.sync_to_github(backup_set, {"full_repo": "tester/slm-backup"}) is True
    assert list(scratch.iterdir()) == []


# ---- upgrade path and connect ------------------------------------------------------


def test_first_backup_after_upgrade_creates_the_key_and_reports_it_once(github, backup_set, key_env) -> None:
    db = _destinations_db(key_env, [("d1", "github", json.dumps({"full_repo": "tester/slm-backup"}), "success")])
    assert bk.load_backup_key() is None

    first = cb.sync_all_destinations(db)
    key = bk.load_backup_key()
    assert key is not None and first["synced"] == 1
    assert first["encryption"]["recovery_key_created"] is True
    assert "slm backup recovery-key" in first["encryption"]["message"]
    assert bk.format_recovery_key(key) not in json.dumps(first)
    status = bk.encryption_status()
    assert status["recovery_key_pending"] is True
    assert status["legacy_plaintext_uploads"] is True

    second = cb.sync_all_destinations(db)
    assert second["encryption"]["recovery_key_created"] is False
    assert bk.load_backup_key() == key


def test_connect_shows_the_recovery_key_once(github, key_env) -> None:
    _destinations_db(key_env, [])
    first = cb.connect_github("ghp_testtoken_for_mock_only", "slm-backup")
    assert first["status"] == "connected"
    shown = first["encryption"]["recovery_key"]
    assert bk.parse_recovery_key(shown) == bk.load_backup_key()
    assert bk.encryption_status()["recovery_key_pending"] is False
    assert bk.encryption_status()["legacy_plaintext_uploads"] is False

    second = cb.connect_github("ghp_testtoken_for_mock_only", "slm-backup")
    assert "recovery_key" not in second["encryption"]
    assert "slm backup recovery-key" in second["encryption"]["message"]


def test_connect_page_text_carries_the_key_only_when_created() -> None:
    key = bytes(range(32))
    created = {"encryption": {"recovery_key": bk.format_recovery_key(key)}}
    assert bk.format_recovery_key(key) in cbc.recovery_key_page_text(created)
    assert "recovery-key" in cbc.recovery_key_page_text({"encryption": {"enabled": True}})
    assert cbc.recovery_key_page_text({}) == ""


def test_disconnect_keeps_the_backup_key(github, key_env) -> None:
    _destinations_db(key_env, [])
    result = cb.connect_github("ghp_testtoken_for_mock_only", "slm-backup")
    key = bk.load_backup_key()
    cb.remove_destination(result["destination_id"])
    assert bk.load_backup_key() == key


# ---- restore -----------------------------------------------------------------------


@pytest.fixture
def uploaded(github, backup_set, key_env) -> dict[str, Any]:
    cb.sync_to_github(backup_set, {"full_repo": "tester/slm-backup"})
    name = backup_set[0].name + bc.ENCRYPTED_SUFFIX
    downloaded = key_env / "downloads" / name
    downloaded.parent.mkdir()
    downloaded.write_bytes(github["assets"][name])
    return {"file": downloaded, "original": backup_set[0], "key": bk.load_backup_key()}


def test_restore_with_the_local_key(uploaded, key_env) -> None:
    out = key_env / "restored" / "memory.db"
    result = cbc.restore_backup_file(uploaded["file"], out)
    assert result["format"] == "encrypted" and result["key_source"] == "this computer"
    assert out.read_bytes() == uploaded["original"].read_bytes()


def test_default_output_name_drops_the_suffix(uploaded) -> None:
    result = cbc.restore_backup_file(uploaded["file"])
    assert Path(result["output"]).name == uploaded["original"].name


def test_restore_on_a_new_machine_with_the_recovery_key(uploaded, key_env) -> None:
    recovery = bk.format_recovery_key(uploaded["key"])
    (key_env / ".credentials.json").unlink()
    bk.state_path().unlink()  # a fresh computer has neither
    assert bk.load_backup_key() is None
    with pytest.raises(bk.BackupKeyUnavailableError, match="recovery key"):
        cbc.restore_backup_file(uploaded["file"], key_env / "nokey.db")
    out = key_env / "restored.db"
    result = cbc.restore_backup_file(uploaded["file"], out, recovery_key=recovery)
    assert result["key_source"] == "recovery key"
    assert out.read_bytes() == uploaded["original"].read_bytes()


def test_wrong_recovery_key_gives_a_clear_error_and_no_file(uploaded, key_env) -> None:
    wrong = bk.format_recovery_key(bytes(32))
    out = key_env / "restored.db"
    with pytest.raises(bc.BackupKeyMismatchError, match="different|recovery key"):
        cbc.restore_backup_file(uploaded["file"], out, recovery_key=wrong)
    assert not out.exists()
    with pytest.raises(bk.RecoveryKeyFormatError):
        cbc.restore_backup_file(uploaded["file"], out, recovery_key="SLMBK1-AAAA")
    assert not out.exists()


def test_tampered_download_gives_a_clear_error_and_no_file(uploaded, key_env) -> None:
    blob = bytearray(uploaded["file"].read_bytes())
    blob[len(blob) // 2] ^= 0x40
    uploaded["file"].write_bytes(bytes(blob))
    out = key_env / "restored.db"
    with pytest.raises(bc.BackupIntegrityError):
        cbc.restore_backup_file(uploaded["file"], out)
    assert not out.exists()


def test_a_4118_plaintext_backup_still_restores(key_env) -> None:
    legacy = _make_db(key_env / f"memory-{TS}.db", ["old plaintext backup"])
    out = key_env / "restored.db"
    result = cbc.restore_backup_file(legacy, out)
    assert result["format"] == "plaintext"
    assert out.read_bytes() == legacy.read_bytes()


def test_unknown_file_is_rejected(key_env) -> None:
    junk = key_env / "junk.bin"
    junk.write_bytes(b"definitely not a backup")
    with pytest.raises(bc.BackupFormatError):
        cbc.restore_backup_file(junk, key_env / "out.db")


def test_no_key_material_in_logs_across_sync_connect_and_restore(github, backup_set, key_env, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    db = _destinations_db(key_env, [("d1", "github", json.dumps({"full_repo": "tester/slm-backup"}), "success")])
    cb.sync_all_destinations(db)
    cb.connect_github("ghp_testtoken_for_mock_only", "slm-backup")
    key = bk.load_backup_key()
    name = backup_set[0].name + bc.ENCRYPTED_SUFFIX
    enc = key_env / name
    enc.write_bytes(github["assets"][name])
    cbc.restore_backup_file(enc, key_env / "r.db", recovery_key=bk.format_recovery_key(key))
    text = caplog.text
    assert bk.format_recovery_key(key) not in text
    assert key.hex() not in text
    assert bk.format_recovery_key(key).replace("-", "") not in text.replace("-", "")


def test_httpx_streams_an_open_file_with_its_content_length(tmp_path) -> None:
    """The GitHub asset upload passes the open file; GitHub needs Content-Length."""
    payload = tmp_path / "x.slmenc"
    payload.write_bytes(b"\x00\x01ciphertext" * 100_000)
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["length"] = request.headers.get("content-length")
        seen["chunked"] = request.headers.get("transfer-encoding")
        seen["body"] = request.read()
        return httpx.Response(201)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client, open(payload, "rb") as fh:
        client.post("https://uploads.invalid/a", content=fh)
    assert seen["length"] == str(payload.stat().st_size)
    assert seen["chunked"] is None
    assert seen["body"] == payload.read_bytes()


def test_cloud_backup_never_routes_ciphertext_through_the_redaction_gate() -> None:
    """Redacting ciphertext would corrupt it; uploads bypass the text gate by design.

    The byte-exact decrypt in the upload tests proves nothing altered the
    payload; this pins that no cloud backup module imports the gate at all.
    """
    infra = Path(cb.__file__).parent
    for name in ("cloud_backup.py", "cloud_backup_crypto.py", "cloud_backup_github.py"):
        assert "outbound_http" not in (infra / name).read_text(encoding="utf-8"), name
