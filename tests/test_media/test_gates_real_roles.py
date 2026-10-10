"""Who may do what on the image, document, feature and folder routes, with the real role engine."""

from __future__ import annotations

import base64

import pytest

from superlocalmemory.media.gc import GcReport
from superlocalmemory.media.ingest import MediaReceipt
from superlocalmemory.server.routes import features, media
from tests.test_media._real_roles import Roles

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode()
ROLES = ("owner", "admin", "member", "viewer")


@pytest.fixture
def roles(tmp_path):
    return Roles(tmp_path)


@pytest.fixture
def calls(monkeypatch):
    seen: list[tuple] = []

    def fake(inp, **kw):
        seen.append((inp, kw))
        return MediaReceipt("stored", media_id="m" * 32, memory_id="mem1")

    monkeypatch.setattr(media, "remember_media", fake)
    monkeypatch.setattr(media, "submit_document", fake)
    return seen


def _post(roles, role, url, body):
    return roles.client.post(url, json=body, headers=roles.headers(role))


@pytest.mark.parametrize("url", ["/api/v3/media/remember", "/api/v3/documents"])
@pytest.mark.parametrize("role,code", [("owner", 200), ("admin", 200), ("member", 403), ("viewer", 403)])
def test_naming_a_file_needs_manage(roles, calls, tmp_path, url, role, code):
    f = tmp_path / "pic.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\nxx")
    assert _post(roles, role, url, {"path": str(f), "content": "x"}).status_code == code
    assert len(calls) == (1 if code == 200 else 0)


@pytest.mark.parametrize("url", ["/api/v3/media/remember", "/api/v3/documents"])
@pytest.mark.parametrize("role,code", [("owner", 200), ("admin", 200), ("member", 200), ("viewer", 403)])
def test_pasted_data_still_needs_only_write(roles, calls, url, role, code):
    assert _post(roles, role, url, {"base64": PNG, "content": "x"}).status_code == code


def test_download_url_still_needs_only_write(roles, calls):
    body = {"download_url": "https://example.com/a.png", "content": "x"}
    assert _post(roles, "member", "/api/v3/media/remember", body).status_code == 200
    assert _post(roles, "viewer", "/api/v3/media/remember", body).status_code == 403


def test_without_any_users_the_owner_can_name_a_file(tmp_path, monkeypatch):
    from tests.test_media._real_roles import Roles as R

    solo = R(tmp_path, with_users=False)
    seen: list = []
    monkeypatch.setattr(media, "remember_media", lambda inp, **kw: seen.append(inp) or MediaReceipt("stored"))
    f = tmp_path / "pic.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\nxx")
    assert solo.client.post("/api/v3/media/remember", json={"path": str(f)}).status_code == 200
    assert len(seen) == 1


def test_unknown_and_forbidden_profiles_look_the_same(roles):
    """A caller with no role on a profile gets one answer whether or not the profile exists."""
    codes = {roles.client.get(f"/api/v3/media?profile_id={pid}", headers=roles.headers("outsider")).status_code
             for pid in ("p2", "nope")}
    assert codes == {403}


def test_an_outsider_cannot_probe_profile_names_on_writes(roles, calls):
    body = {"base64": PNG, "content": "x"}
    codes = {_post(roles, "outsider", "/api/v3/media/remember", {**body, "profile_id": p}).status_code
             for p in ("p2", "nope")}
    assert codes == {403}
    assert calls == []


def test_the_owner_still_gets_404_for_an_unknown_profile(roles):
    assert roles.client.get("/api/v3/media?profile_id=nope").status_code == 404


@pytest.mark.parametrize("role,code", [("owner", 200), ("admin", 200), ("member", 403), ("viewer", 403)])
@pytest.mark.parametrize("path", ["/api/v3/features/media/enable", "/api/v3/features/media/disable"])
def test_turning_features_on_or_off_needs_manage(roles, monkeypatch, path, role, code):
    monkeypatch.setattr("superlocalmemory.server.write_identity.require_write_actor", lambda *a, **k: "x")
    fake = lambda *a, **k: {"enabled": True}  # noqa: E731
    monkeypatch.setattr(features.feat, "enable_media", fake)
    monkeypatch.setattr(features.feat, "disable_media", fake)
    monkeypatch.setattr(features, "_media_view", lambda status, root: {})
    r = roles.client.post(path, json={"yes": True}, headers=roles.headers(role))
    assert r.status_code in ({200, 202} if code == 200 else {code})


@pytest.mark.parametrize("dry_run,role,code", [
    (True, "admin", 200), (True, "member", 403), (True, "viewer", 403),
    (False, "owner", 200), (False, "admin", 200), (False, "member", 403), (False, "viewer", 403)])
def test_the_sweep_gates(roles, monkeypatch, dry_run, role, code):
    monkeypatch.setattr(media, "run_gc", lambda profile, dry: GcReport(dry_run=dry))
    r = roles.client.post("/api/v3/media/gc", json={"dry_run": dry_run}, headers=roles.headers(role))
    assert r.status_code == code


def test_deleting_strays_asks_for_manage(roles, monkeypatch):
    asked: list = []
    import superlocalmemory.server.rbac_enforce as enforce

    real = enforce.require_manage
    monkeypatch.setattr(enforce, "require_manage", lambda *a, **k: asked.append(k) or real(*a, **k))
    monkeypatch.setattr(media, "run_gc", lambda profile, dry: GcReport(dry_run=dry))
    roles.client.post("/api/v3/media/gc", json={"dry_run": True}, headers=roles.headers("admin"))
    assert asked == []
    roles.client.post("/api/v3/media/gc", json={"dry_run": False}, headers=roles.headers("admin"))
    assert len(asked) == 1


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("role,code", [("owner", 200), ("admin", 200), ("member", 403), ("viewer", 403)])
def test_repairing_the_picture_index_needs_manage(roles, monkeypatch, dry_run, role, code):
    from superlocalmemory.media.repair import RepairReport

    monkeypatch.setattr(media, "run_repair", lambda profile, dry: RepairReport(dry_run=dry))
    r = roles.client.post("/api/v3/media/repair", json={"dry_run": dry_run}, headers=roles.headers(role))
    assert r.status_code == code
