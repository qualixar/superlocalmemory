"""Who may open the folder dialog or see suggestions, with the real role engine."""

from __future__ import annotations

import pytest

from superlocalmemory.sources import picker
from tests.test_media._real_roles import Roles

PICK = "/api/v3/sources/pick-folder"
SUGG = "/api/v3/sources/suggestions"


@pytest.fixture(autouse=True)
def one_dialog(monkeypatch):
    monkeypatch.setattr(picker.sys, "platform", "linux")
    monkeypatch.setattr(picker.shutil, "which", lambda t: "/bin/zenity" if t == "zenity" else None)
    monkeypatch.setattr(picker, "_run", lambda argv, t: (0, "/picked\n"))


@pytest.mark.parametrize("role,code", [("owner", 200), ("admin", 200), ("member", 403), ("viewer", 403)])
def test_both_routes_need_manage_with_the_real_roles(tmp_path, env, role, code):
    roles = Roles(tmp_path)
    assert roles.client.post(PICK, headers=roles.headers(role)).status_code == code
    assert roles.client.get(SUGG, headers=roles.headers(role)).status_code == code


def test_without_any_users_the_owner_may_pick(tmp_path, env):
    (tmp_path / "solo").mkdir()
    solo = Roles(tmp_path / "solo", with_users=False)
    assert solo.client.post(PICK).status_code == 200


