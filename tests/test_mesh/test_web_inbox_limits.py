"""What a web app may take from, and put into, the mesh: redaction, caps, names, schema."""

from __future__ import annotations

import sqlite3
import time

import pytest

from superlocalmemory.mesh import broker_security, broker_web
from tests.test_mesh.conftest import make_peer, rows

WEB_REF = "w_" + "cd" * 12
FAKE_KEY = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 3


@pytest.fixture()
def web(broker) -> str:
    assert broker.ensure_web_peer(WEB_REF, app="notes", display_name="Notes",
                                  connection_id="c" * 32)["ok"]
    return WEB_REF


def test_mail_taken_by_a_web_app_is_secret_redacted_in_both_views(broker, web) -> None:
    local = make_peer(broker, "sess-1")
    assert broker.send_message(local, web, f"the key is {FAKE_KEY} ok")["ok"]
    (msg,) = broker.claim_web_inbox(web)
    assert FAKE_KEY not in msg["content"] and "REDACTED" in msg["content"]
    assert FAKE_KEY not in str(msg["envelope"])


def test_a_local_reader_still_sees_the_text_unchanged(broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    broker.send_message(a, b, f"key {FAKE_KEY}")
    conn = broker._conn()
    try:
        from superlocalmemory.mesh import broker_inbox
        got = broker_inbox.query_inbox(conn, b, "", "default", direct_only=True)
    finally:
        conn.close()
    assert FAKE_KEY in got[0]["content"]


def test_a_claim_takes_at_most_twenty_messages_and_leaves_the_rest_unread(broker, web) -> None:
    local = make_peer(broker, "sess-1")
    for i in range(30):
        assert broker.send_message(local, web, f"m{i}")["ok"] is True or i >= 20
    first = broker.claim_web_inbox(web)
    second = broker.claim_web_inbox(web)
    assert len(first) == broker_web.CLAIM_MAX_MESSAGES == 20
    assert len(second) > 0 and not {m["id"] for m in first} & {m["id"] for m in second}


def test_a_claim_stops_at_the_content_budget_but_always_returns_one(broker, web, monkeypatch) -> None:
    monkeypatch.setattr(broker_web, "CLAIM_MAX_BYTES", 10)
    local = make_peer(broker, "sess-1")
    for text in ("x" * 8, "y" * 8, "z" * 8):
        broker.send_message(local, web, text)
    assert len(broker.claim_web_inbox(web)) == 1
    assert len(broker.claim_web_inbox(web)) == 1


@pytest.mark.parametrize("text", ["a\x00b", "a\x1b[31mb", "a\rb", "a\x7fb", "a\x85b"])
def test_a_web_send_with_control_characters_is_refused(broker, web, text) -> None:
    local = make_peer(broker, "sess-1")
    out = broker.web_send(web, "notes", local, text)
    assert out["ok"] is False and "control" in out["error"]
    assert rows(broker, "SELECT * FROM mesh_messages") == []


def test_a_web_send_keeps_newlines_and_tabs(broker, web) -> None:
    local = make_peer(broker, "sess-1")
    assert broker.web_send(web, "notes", local, "line1\n\tline2")["ok"]


def test_expired_unread_mail_does_not_fill_a_recipients_inbox(broker) -> None:
    sender = broker.ensure_web_peer("w_" + "11" * 12, app="a", display_name="A",
                                    connection_id="c" * 32)
    assert sender["ok"]
    local = make_peer(broker, "sess-1")
    for i in range(broker_web.broker_inbox.MAX_UNREAD_DIRECT):
        assert broker.send_message(local, local, f"old{i}")["ok"]
    conn = broker._conn()
    try:
        conn.execute("UPDATE mesh_messages SET expires_at='2000-01-01T00:00:00+00:00'")
        conn.commit()
    finally:
        conn.close()
    assert broker.web_send("w_" + "11" * 12, "a", local, "fresh")["ok"] is True


def test_peer_names_drop_format_characters_and_are_capped(broker) -> None:
    name = "Evil‮App​" + "n" * 100
    assert broker.ensure_web_peer(WEB_REF, app="notes", display_name=name,
                                  connection_id="c" * 32)["ok"]
    shown = rows(broker, "SELECT display_name FROM mesh_peer_profiles WHERE peer_id=?",
                 (WEB_REF,))[0][0]
    assert "‮" not in shown and "​" not in shown
    assert shown.startswith("EvilApp") and len(shown) == 64


def test_the_names_shown_for_gateway_apps_are_cleaned_too() -> None:
    from superlocalmemory.remote_connections import peer_names
    peer_names.set_names("c" * 32, {"a1": "My‮Notes"})
    try:
        assert peer_names.display_name("c" * 32, "a1", "w_abcdefgh") == "MyNotes"
    finally:
        peer_names.set_names("c" * 32, {})


def test_a_wait_poll_with_nothing_unread_does_not_take_the_write_lock(broker, web) -> None:
    taken: list[str] = []
    real = broker._write_with_retry

    def spy(fn, *a, **k):
        taken.append("write")
        return real(fn, *a, **k)

    broker._write_with_retry = spy
    found, timed_out = broker.wait_web_inbox(web, timeout_s=1)
    assert found == [] and timed_out is True and taken == []
    local = make_peer(broker, "sess-1")
    broker.send_message(local, web, "now there is one")
    taken.clear()
    found, _ = broker.wait_web_inbox(web, timeout_s=1)
    assert len(found) == 1 and taken == ["write"]


class _Conn:
    """A connection whose ALTER fails with a chosen error."""

    def __init__(self, real, error):
        self.real, self.error = real, error

    def execute(self, sql, *a):
        if sql.startswith("ALTER TABLE mesh_peer_profiles"):
            raise sqlite3.OperationalError(self.error)
        return self.real.execute(sql, *a)

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_a_failing_connection_ref_alter_other_than_a_duplicate_column_is_not_hidden(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "m.db")
    conn.row_factory = sqlite3.Row
    from tests.test_mesh.conftest import init_mesh_schema
    init_mesh_schema(str(tmp_path / "m.db"))
    conn.execute("CREATE TABLE mesh_peer_profiles (peer_id TEXT PRIMARY KEY, kind TEXT, "
                 "app_name TEXT, display_name TEXT, authorization_ref TEXT, muted INTEGER, "
                 "retired_at TEXT, updated_at TEXT)")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        broker_security.apply_security_schema(_Conn(conn, "database is locked"))


def test_the_connection_ref_column_is_added_once_and_reapplying_is_quiet(broker) -> None:
    conn = broker._conn()
    try:
        broker_security.apply_security_schema(conn)
        broker_security.apply_security_schema(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(mesh_peer_profiles)")]
    finally:
        conn.close()
    assert cols.count("connection_ref") == 1
