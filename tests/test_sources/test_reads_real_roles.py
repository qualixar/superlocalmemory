# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every picture, document, job and folder READ route, with the real role engine:
who may read, and that one profile's ids answer nothing to another profile."""

from __future__ import annotations

import pytest

from superlocalmemory import sources
from superlocalmemory.media import open_media_store
from tests.test_media._real_roles import PW, Roles

A_TITLE = "A-private-lease-title"
A_FOLDER = "a-secret-vault"
DOC = "d" * 32


@pytest.fixture
def world(tmp_path, env):
    roles = Roles(tmp_path)
    store = open_media_store(create=True, data_root=env.data)
    pic = store.insert_item(profile_id="default", kind="image", source_sha256="a" * 64, mime="image/png",
                            bytes=1, origin="tool", thumb_webp=b"RIFFxxxxWEBPa-pixels")
    store.insert_document(document_id=DOC, profile_id="default", sha256="b" * 64, title=A_TITLE,
                          mime="application/pdf", bytes=1, source_relpath="", origin="user")
    job = store.enqueue_job("default", "document", 0, {"document_id": DOC})
    store.close()
    vault = tmp_path / A_FOLDER
    vault.mkdir()
    preview = sources.add_source(vault, profile_id="default")
    sources.confirm_source(preview.source_id, profile_id="default")
    yield roles, {"pic": pic, "job": job, "src": preview.source_id}


def _only_in(roles: Roles, profile: str) -> dict[str, str]:
    """Headers of a user who holds a role on ``profile`` and on no other."""
    user = roles.rbac.create_user(f"only-{profile}", PW, f"only {profile}")
    roles.rbac.set_membership(profile, user["user_id"], "member")
    return {"X-SLM-User-Session": roles.rbac.create_session(user["user_id"])}


def _id_routes(ids: dict[str, str]) -> list[str]:
    return [f"/api/v3/media/{ids['pic']}/thumb", f"/api/v3/media/{ids['pic']}/thumb?format=json",
            f"/api/v3/jobs/{ids['job']}", f"/api/v3/sources/{ids['src']}/report"]


LISTS = ["/api/v3/media", "/api/v3/documents", "/api/v3/documents/lint", "/api/v3/sources"]


def _with_profile(url: str, profile: str) -> str:
    return f"{url}{'&' if '?' in url else '?'}profile_id={profile}"


@pytest.mark.parametrize("role", ["owner", "admin", "member", "viewer"])
def test_every_role_may_read_its_profile(world, role):
    roles, ids = world
    for url in [*_id_routes(ids), *LISTS]:
        reply = roles.client.get(_with_profile(url, "default"), headers=roles.headers(role))
        assert reply.status_code == 200, (role, url, reply.status_code)


def test_a_user_with_no_role_on_the_profile_is_refused_everywhere(world):
    roles, ids = world
    for url in [*_id_routes(ids), *LISTS]:
        for profile in ("default", "nope"):  # an unknown profile looks the same
            reply = roles.client.get(_with_profile(url, profile), headers=roles.headers("outsider"))
            assert reply.status_code == 403, (url, profile, reply.status_code)


def test_suggestions_need_manage_with_the_real_roles(world, monkeypatch):
    from superlocalmemory.sources import picker

    monkeypatch.setattr(picker, "suggestions", lambda: [])
    roles, _ = world
    codes = {r: roles.client.get("/api/v3/sources/suggestions", headers=roles.headers(r)).status_code
             for r in ("owner", "admin", "member", "viewer")}
    assert codes == {"owner": 200, "admin": 200, "member": 403, "viewer": 403}


def test_the_features_overview_is_readable_by_every_role(world):
    roles, _ = world
    for role in ("owner", "admin", "member", "viewer"):
        assert roles.client.get("/api/v3/features", headers=roles.headers(role)).status_code == 200


def test_profile_b_user_cannot_read_profile_a_by_id(world):
    """The other profile's ids answer 404 (as its own member) or 403 (naming it), and carry no data."""
    roles, ids = world
    only_b = _only_in(roles, "p2")
    for url in _id_routes(ids):
        as_b = roles.client.get(_with_profile(url, "p2"), headers=only_b)
        naming_a = roles.client.get(_with_profile(url, "default"), headers=only_b)
        assert as_b.status_code == 404, (url, as_b.status_code)
        assert naming_a.status_code == 403, (url, naming_a.status_code)
        for reply in (as_b, naming_a):
            assert A_FOLDER not in reply.text and A_TITLE not in reply.text and b"a-pixels" not in reply.content


def test_profile_b_lists_hold_none_of_profile_a(world):
    roles, _ = world
    only_b = _only_in(roles, "p2")
    for url in LISTS:
        reply = roles.client.get(_with_profile(url, "p2"), headers=only_b)
        assert reply.status_code == 200, url
        for secret in (A_FOLDER, A_TITLE, DOC):
            assert secret not in reply.text, (url, secret)
        assert not any(isinstance(v, list) and v for v in reply.json().values()), url
        assert roles.client.get(_with_profile(url, "default"), headers=only_b).status_code == 403


def test_the_owner_of_profile_a_still_reads_it(world):
    """The control: the same routes do answer for the profile that owns the data."""
    roles, ids = world
    for url in _id_routes(ids):
        assert roles.client.get(_with_profile(url, "default")).status_code == 200, url
    assert A_TITLE in roles.client.get("/api/v3/documents?profile_id=default").text
