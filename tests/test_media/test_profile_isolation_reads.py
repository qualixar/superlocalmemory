"""One profile's picture and document ids cannot be read from another profile on any read route."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.media import open_media_store
from tests.test_media._real_roles import Roles


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    roles = Roles(tmp_path)
    for pid in ("A", "B"):
        roles.db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))
    store = open_media_store(create=True, data_root=tmp_path / "slm")
    pic = store.insert_item(profile_id="A", kind="image", source_sha256="a" * 64, mime="image/png", bytes=1,
                            origin="tool", thumb_webp=b"RIFFxxxxWEBP")
    doc = store.insert_document(document_id="d" * 32, profile_id="A", sha256="b" * 64, title="t",
                                mime="application/pdf", bytes=1, source_relpath="bb/x.pdf", origin="user")
    job = store.enqueue_job("A", "document", 0, {"document_id": "d" * 32})
    yield roles, pic, "d" * 32, job
    store.close()


def test_profile_b_cannot_read_profile_a_items(world):
    roles, pic, doc, job = world
    get = roles.client.get
    assert get(f"/api/v3/media/{pic}/thumb?profile_id=A").status_code == 200
    assert get(f"/api/v3/media/{pic}/thumb?profile_id=B").status_code == 404
    assert get(f"/api/v3/media/{pic}/thumb?format=json&profile_id=B").status_code == 404
    assert get(f"/api/v3/jobs/{job}?profile_id=A").status_code == 200
    assert get(f"/api/v3/jobs/{job}?profile_id=B").status_code == 404
    assert get("/api/v3/media?profile_id=B").json()["items"] == []
    assert [i["media_id"] for i in get("/api/v3/media?profile_id=A").json()["items"]] == [pic]
    roles.client.app.state.canonical_remember_runtime = SimpleNamespace(ready=True)
    assert roles.client.delete(f"/api/v3/documents/{doc}?profile_id=B").status_code == 404
    listed = get("/api/v3/documents?profile_id=B").json()
    assert doc not in str(listed)
