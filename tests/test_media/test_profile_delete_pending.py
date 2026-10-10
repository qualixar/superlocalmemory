"""A profile delete whose picture move fails still completes, and the move is finished later."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from superlocalmemory.server.routes import helpers, profiles
from superlocalmemory.storage import pending_media_moves as pending
from superlocalmemory.storage import profile_fold_sidecars as sidecars
from tests.test_media.test_profile_delete_order import _owners, root  # noqa: F401  (fixture)


@pytest.fixture
def broken_move(monkeypatch):
    real = sidecars.move_media
    state = {"broken": True}

    def move(*args, **kw):
        if state["broken"]:
            raise sidecars.SidecarFoldError("injected")
        return real(*args, **kw)

    monkeypatch.setattr(sidecars, "move_media", move)
    return state


def _request():
    request = MagicMock()
    request.app.state = SimpleNamespace()
    return request


def _delete(monkeypatch, name, stored):
    monkeypatch.setattr(profiles, "DB_PATH", helpers.DB_PATH)
    monkeypatch.setattr(profiles, "_load_profiles_json", lambda: stored)
    monkeypatch.setattr(profiles, "_save_profiles_json", lambda cfg: stored.update(cfg))
    runtime = SimpleNamespace(snapshot=SimpleNamespace(profile_id="default"))
    with patch.object(profiles, "sync_profiles", return_value=[{"profile_id": "default"}, {"profile_id": name}]), \
            patch.object(profiles, "get_profile_runtime", return_value=runtime), \
            patch.object(profiles, "authorize_route_mutation", return_value=MagicMock()), \
            patch("superlocalmemory.server.rbac_enforce.require_manage", lambda *a, **k: None):
        return asyncio.run(profiles.delete_profile(name, _request()))


def test_a_failed_picture_move_is_recorded_and_the_delete_still_succeeds(root, broken_move, monkeypatch):
    stored = {"profiles": {"alice": {}, "default": {}}}
    out = _delete(monkeypatch, "alice", stored)
    assert out["success"] and out["pictures_pending"] is True
    assert "alice" not in stored["profiles"]
    assert pending.pending(root) == [{"from": "alice", "to": "default"}]
    assert _owners(root) == ["alice"]


def test_the_retry_at_start_moves_the_rows_and_clears_the_record(root, broken_move):
    helpers.delete_profile_from_db("alice")
    assert pending.retry(root)  # still broken: stays recorded
    broken_move["broken"] = False
    assert pending.retry(root) == []
    assert _owners(root) == ["default"]
    assert pending.pending(root) == []
    assert not (root / "pending_media_moves.json").exists()


def test_a_profile_cannot_be_created_under_a_name_with_a_pending_move(root, broken_move, monkeypatch):
    helpers.delete_profile_from_db("alice")
    monkeypatch.setattr(profiles, "DB_PATH", helpers.DB_PATH)
    monkeypatch.setattr(profiles, "sync_profiles", lambda: [{"profile_id": "default"}])
    with pytest.raises(HTTPException) as refused:
        asyncio.run(profiles.create_profile(SimpleNamespace(profile_name="alice"), _request()))
    assert refused.value.status_code == 409 and "alice" in refused.value.detail
    broken_move["broken"] = False  # the retry before creation finishes the move
    assert pending.creation_blocker(root, "alice") == ""
    assert _owners(root) == ["default"]


def test_a_name_that_still_owns_rows_is_refused_even_without_a_record(root):
    assert "alice" in pending.creation_blocker(root, "alice")
    assert pending.creation_blocker(root, "bob") == ""
