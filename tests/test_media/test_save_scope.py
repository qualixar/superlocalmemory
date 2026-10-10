"""A picture or page is saved with the same scope rules as typed text."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from superlocalmemory.access import rbac
from superlocalmemory.media.ingest import MediaReceipt
from superlocalmemory.server.routes import media as routes
from tests.test_media._real_roles import Roles
from tests.test_media.test_ingest import env, png, root, save, store  # noqa: F401  (fixtures)

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode()
URLS = ("/api/v3/media/remember", "/api/v3/documents")


def _config(default: str):
    return SimpleNamespace(pii_redaction=False, scope=SimpleNamespace(default_scope=default))


def test_a_picture_save_carries_scope_and_shared_with(env):
    assert save(env, scope="shared", shared_with=("p2",)).status == "stored"
    sent = env.runtime.requests[-1]
    assert sent.scope == "shared" and tuple(sent.shared_with) == ("p2",)


def test_a_picture_save_without_scope_uses_the_configured_default(env):
    env.config = _config("global")
    save(env)
    assert env.runtime.requests[-1].scope == "global"


def test_a_picture_save_is_personal_when_nothing_is_configured(env):
    save(env)
    assert env.runtime.requests[-1].scope == "personal"


def test_a_global_picture_memory_is_visible_to_another_profile_like_text(env, tmp_path):
    """The picture's memory row carries the scope, so the text visibility clause finds it for B."""
    from superlocalmemory.storage.database import _scope_where

    env.config = _config("global")
    save(env, profile_id="A")
    admission = env.runtime.requests[-1]
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memories (memory_id, profile_id, scope, shared_with)")
    conn.execute("INSERT INTO memories VALUES ('pic', 'A', ?, ?)", (admission.scope, "[]"))
    conn.execute("INSERT INTO memories VALUES ('txt', 'A', 'global', '[]')")
    where, params = _scope_where("B", include_global=True)
    seen = {r[0] for r in conn.execute(f"SELECT memory_id FROM memories WHERE {where}", params)}
    assert seen == {"pic", "txt"}
    where, params = _scope_where("B")
    assert not list(conn.execute(f"SELECT memory_id FROM memories WHERE {where}", params))


@pytest.fixture
def roles(tmp_path):
    return Roles(tmp_path)


@pytest.fixture
def sent(monkeypatch):
    seen: list[dict] = []

    def fake(inp, **kw):
        seen.append(kw)
        return MediaReceipt("stored", media_id="m" * 32, memory_id="mem1")

    monkeypatch.setattr(routes, "remember_media", fake)
    monkeypatch.setattr(routes, "submit_document", fake)
    return seen


def _post(roles, role, url, **extra):
    return roles.client.post(url, json={"base64": PNG, "content": "x", **extra}, headers=roles.headers(role))


@pytest.mark.parametrize("url", URLS)
def test_a_save_without_scope_is_personal_and_needs_no_share(roles, sent, url):
    assert _post(roles, "member", url).status_code == 200
    assert sent[-1]["scope"] == "personal"


@pytest.mark.parametrize("url", URLS)
def test_scope_and_shared_with_are_passed_through(roles, sent, url):
    assert _post(roles, "member", url, scope="shared", shared_with=["p2"]).status_code == 200
    assert sent[-1]["scope"] == "shared" and list(sent[-1]["shared_with"]) == ["p2"]


@pytest.mark.parametrize("url", URLS)
def test_the_configured_default_applies_and_is_checked_like_text(roles, sent, url, monkeypatch):
    roles.client.app.state.engine._config = _config("global")
    assert _post(roles, "member", url).status_code == 200
    assert sent[-1]["scope"] == "global"
    no_share = {**rbac._ROLE_PERMISSIONS, rbac.Role.MEMBER: frozenset({rbac.Permission.READ, rbac.Permission.WRITE})}
    monkeypatch.setattr(rbac, "_ROLE_PERMISSIONS", no_share)
    sent.clear()
    assert _post(roles, "member", url).status_code == 403
    assert sent == []


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("scope", ["shared", "global"])
def test_a_member_without_share_cannot_save_shared_or_global(roles, sent, url, scope, monkeypatch):
    no_share = {**rbac._ROLE_PERMISSIONS, rbac.Role.MEMBER: frozenset({rbac.Permission.READ, rbac.Permission.WRITE})}
    monkeypatch.setattr(rbac, "_ROLE_PERMISSIONS", no_share)
    assert _post(roles, "member", url, scope=scope, shared_with=["p2"]).status_code == 403
    assert _post(roles, "member", url, scope="personal").status_code == 200
    assert len(sent) == 1


@pytest.mark.parametrize("url", URLS)
def test_an_unknown_scope_is_refused(roles, sent, url):
    assert _post(roles, "owner", url, scope="everyone").status_code == 422
    assert sent == []
