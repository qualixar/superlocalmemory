"""Who may connect a folder, and which folders may be connected, with the real role engine."""

from __future__ import annotations

import os

import pytest

from tests.test_media._real_roles import Roles

ADD = "/api/v3/sources"


@pytest.fixture
def roles(tmp_path):
    return Roles(tmp_path)


def _add(roles, role, path, **extra):
    return roles.client.post(ADD, json={"path": str(path), **extra}, headers=roles.headers(role))


@pytest.mark.parametrize("role,code", [("owner", 200), ("admin", 200), ("member", 403), ("viewer", 403)])
def test_adding_a_folder_needs_manage(roles, env, role, code):
    assert _add(roles, role, env.root).status_code == code


@pytest.mark.parametrize("role,code", [("owner", 202), ("admin", 202), ("member", 403), ("viewer", 403)])
def test_confirming_a_folder_needs_manage(roles, env, role, code):
    sid = _add(roles, "owner", env.root).json()["source_id"]
    r = roles.client.post(f"{ADD}/{sid}/confirm", headers=roles.headers(role))
    assert r.status_code == code


def test_without_any_users_the_owner_connects_a_folder(tmp_path, env):
    (tmp_path / "solo").mkdir()
    solo = Roles(tmp_path / "solo", with_users=False)
    sid = solo.client.post(ADD, json={"path": str(env.root)}).json()["source_id"]
    assert solo.client.post(f"{ADD}/{sid}/confirm").status_code == 202


def test_a_preview_belongs_to_the_profile_that_asked(roles, env):
    sid = _add(roles, "admin", env.root, profile_id="p2").json()["source_id"]
    wrong = roles.client.post(f"{ADD}/{sid}/confirm", headers=roles.headers("admin"))
    assert wrong.status_code == 404
    right = roles.client.post(f"{ADD}/{sid}/confirm?profile_id=p2", headers=roles.headers("admin"))
    assert right.status_code == 202


def test_an_outsider_cannot_probe_profile_names(roles, env):
    codes = {_add(roles, "outsider", env.root, profile_id=p).status_code for p in ("p2", "nope")}
    assert codes == {403}


def _refused(resp):
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["code"] == "data_root"
    assert "own data" in resp.json()["detail"]["message"]


def test_the_data_folder_cannot_be_connected(roles, env):
    _refused(_add(roles, "owner", env.data))


def test_a_folder_inside_the_data_folder_cannot_be_connected(roles, env):
    (env.data / "media" / "ab").mkdir(parents=True)
    _refused(_add(roles, "owner", env.data / "media"))
    _refused(_add(roles, "owner", env.data / "media" / "ab"))


def test_a_link_into_the_data_folder_cannot_be_connected(roles, env, tmp_path):
    (env.data / "media").mkdir()
    link = env.root / "inside"
    os.symlink(env.data / "media", link)
    _refused(_add(roles, "owner", link))


def test_a_parent_of_the_data_folder_cannot_be_connected(roles, env, tmp_path):
    _refused(_add(roles, "owner", tmp_path))


def test_a_folder_that_later_turns_out_to_hold_the_data_folder_is_refused_at_confirm(roles, env):
    from superlocalmemory.sources import api

    sid = _add(roles, "owner", env.root).json()["source_id"]
    api._pending[sid] = api._Pending(env.data, "default", "folder", api._pending[sid].include_types, 0.0)
    _refused(roles.client.post(f"{ADD}/{sid}/confirm", headers=roles.headers("owner")))
