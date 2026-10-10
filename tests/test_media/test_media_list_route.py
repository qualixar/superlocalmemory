"""The saved-images list: newest first, paged, local only, one profile, nothing private in the answer."""

from __future__ import annotations

import base64

import pytest

from superlocalmemory.media import open_media_store

from .test_media_routes import LOCAL, REMOTE, make

STAMP = "2026-01-01T00:00:0%d.000000Z"


def _row(store, n, *, profile="default", kind="image", state="active", thumb=b"RIFFxxxxWEBP"):
    sha = f"{n:02d}" * 32
    mid = store.insert_item(profile_id=profile, kind=kind, source_sha256=sha, mime="image/png", bytes=10 + n,
                            origin="tool", thumb_webp=thumb, original_relpath="secret/path.png",
                            exif_json={"Make": "cam"}, width=4, height=3)
    with store._write() as conn:
        conn.execute("UPDATE media_items SET created_at = ?, state = ? WHERE media_id = ?",
                     (STAMP % n, state, mid))
    return mid


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    s = open_media_store(create=True, data_root=tmp_path)
    yield s
    s.close()


def test_newest_first_with_only_safe_fields(monkeypatch, store):
    ids = [_row(store, n) for n in (1, 2, 3)]
    c, _ = make(monkeypatch)
    r = c.get("/api/v3/media")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert [i["media_id"] for i in body["items"]] == ids[::-1] and body["next_cursor"] is None
    first = body["items"][0]
    assert first["has_thumb"] is True and first["width"] == 4 and first["bytes"] == 13
    text = r.text
    assert "original_relpath" not in text and "exif_json" not in text and "secret/path" not in text
    assert "thumb_webp" not in first and "source_sha256" not in first


def test_paging_has_no_overlap_and_no_gap(monkeypatch, store):
    ids = [_row(store, n) for n in range(1, 6)][::-1]
    c, _ = make(monkeypatch)
    seen, cursor, sizes = [], "", []
    while True:
        r = c.get("/api/v3/media", params={"limit": 2, "cursor": cursor})
        assert r.status_code == 200
        body = r.json()
        sizes.append(len(body["items"]))
        seen += [i["media_id"] for i in body["items"]]
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert sizes == [2, 2, 1] and seen == ids


def test_tombstoned_pages_and_other_profiles_are_left_out(monkeypatch, store):
    mine = _row(store, 1)
    _row(store, 2, state="tombstoned")
    _row(store, 3, kind="page")
    _row(store, 4, profile="other")
    c, _ = make(monkeypatch)
    assert [i["media_id"] for i in c.get("/api/v3/media").json()["items"]] == [mine]


def test_an_image_without_a_thumbnail_says_so(monkeypatch, store):
    _row(store, 1, thumb=None)
    c, _ = make(monkeypatch)
    assert c.get("/api/v3/media").json()["items"][0]["has_thumb"] is False


@pytest.mark.parametrize("cursor", ["!!!", "bm90LWEtY3Vyc29y", base64.urlsafe_b64encode(b"x|NOTANID").decode()])
def test_a_bad_cursor_is_400(monkeypatch, store, cursor):
    c, _ = make(monkeypatch)
    r = c.get("/api/v3/media", params={"cursor": cursor})
    assert r.status_code == 400 and r.json()["detail"] == "Bad cursor."


def test_limit_is_bounded(monkeypatch, store):
    c, _ = make(monkeypatch)
    assert c.get("/api/v3/media", params={"limit": 0}).status_code == 422
    assert c.get("/api/v3/media", params={"limit": 201}).status_code == 422


def test_remote_callers_are_refused(monkeypatch, store):
    c, _ = make(monkeypatch, client=REMOTE)
    assert c.get("/api/v3/media").status_code == 403


def test_feature_off_is_an_empty_list_and_creates_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "empty"))
    c, _ = make(monkeypatch)
    r = c.get("/api/v3/media")
    assert r.status_code == 200 and r.json() == {"items": [], "next_cursor": None}
    assert not (tmp_path / "empty" / "media.db").exists()


def test_the_thumbnail_route_still_answers(monkeypatch, store):
    mid = _row(store, 1)
    c, _ = make(monkeypatch)
    assert c.get(f"/api/v3/media/{mid}/thumb").status_code == 200
