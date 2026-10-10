"""A picture shared with a profile shows its thumbnail there, by the rules text is shown by."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.media import open_media_store
from tests.test_media._real_roles import Roles

BASE = dict(kind="image", mime="image/png", bytes=1, origin="tool", thumb_webp=b"RIFFxxxxWEBP")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    roles = Roles(tmp_path)
    for pid in ("A", "B", "C"):
        roles.db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))
    store = open_media_store(create=True, data_root=tmp_path / "slm")
    yield roles, store
    store.close()


def _picture(world, owner, scope, shared_with=()):
    roles, store = world
    memory = f"mem-{owner}-{scope}-{len(shared_with)}"
    roles.db.execute("INSERT INTO memories (memory_id, profile_id, scope, shared_with, content) VALUES (?, ?, ?, ?, 'p')",
                     (memory, owner, scope, json.dumps(list(shared_with))))
    return store.insert_item(profile_id=owner, source_sha256=memory.ljust(64, "0")[:64], anchor_memory_id=memory, **BASE)


def _get(world, media_id, profile, suffix=""):
    roles, _ = world
    return roles.client.get(f"/api/v3/media/{media_id}/thumb{suffix}?profile_id={profile}")


def test_a_picture_shared_with_a_profile_shows_there_and_nowhere_else(world):
    pic = _picture(world, "B", "shared", ["A"])
    assert _get(world, pic, "A").status_code == 200
    assert _get(world, pic, "B").status_code == 200
    assert _get(world, pic, "C").status_code == 404


def test_a_global_picture_shows_to_every_profile(world):
    pic = _picture(world, "B", "global")
    assert all(_get(world, pic, p).status_code == 200 for p in ("A", "B", "C"))


def test_a_personal_picture_stays_private(world):
    pic = _picture(world, "B", "personal")
    assert _get(world, pic, "A").status_code == 404 and _get(world, pic, "B").status_code == 200


def test_a_profile_name_inside_another_name_does_not_match(world):
    roles, _ = world
    roles.db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('AB', 'AB')")
    pic = _picture(world, "B", "shared", ["AB"])
    assert _get(world, pic, "A").status_code == 404 and _get(world, pic, "AB").status_code == 200


def test_the_json_form_follows_the_same_rule(world):
    pic = _picture(world, "B", "shared", ["A"])
    assert _get(world, pic, "A", "").status_code == 200
    roles, _ = world
    got = roles.client.get(f"/api/v3/media/{pic}/thumb?format=json&profile_id=A")
    assert got.status_code == 200 and got.json()["mime"] == "image/webp"
    assert roles.client.get(f"/api/v3/media/{pic}/thumb?format=json&profile_id=C").status_code == 404


def test_a_picture_without_a_memory_is_not_shown_across_profiles(world):
    roles, store = world
    pic = store.insert_item(profile_id="B", source_sha256="e" * 64, **BASE)
    assert _get(world, pic, "A").status_code == 404
