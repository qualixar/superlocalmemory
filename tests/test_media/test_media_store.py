"""The images-and-documents store: created only on request, tolerant of newer files."""

from __future__ import annotations

import json
import sqlite3
import stat
import sys
import threading

import pytest

from superlocalmemory.media import MediaStoreReadOnly, media_db_exists, media_db_path, open_media_store
from superlocalmemory.media.schema import MEDIA_SCHEMA_VERSION
from tests.helpers.env_capabilities import NO_VECTOR_SEARCH_REASON, vector_search_available

pytestmark = pytest.mark.skipif(not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON)

DIM = 768


def vec(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "slm"


@pytest.fixture()
def store(root):
    s = open_media_store(create=True, data_root=root)
    yield s
    s.close()


def item(profile="default", sha="a" * 64, **kw):
    base = dict(profile_id=profile, kind="image", source_sha256=sha, mime="image/png",
                bytes=10, origin="tool")
    base.update(kw)
    return base


def test_absent_file_is_not_created_by_a_read(root):
    assert open_media_store(data_root=root) is None
    assert not media_db_exists(root)
    assert not root.exists() or not media_db_path(root).exists()


def test_create_makes_schema_wal_and_private_file(root):
    s = open_media_store(create=True, data_root=root)
    try:
        path = media_db_path(root)
        assert path.exists()
        if sys.platform != "win32":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        s.close()
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("media_schema", "media_spaces", "media_items", "media_vector_rows", "documents",
              "doc_pages", "jobs", "sources", "source_files", "source_links"):
        assert t in tables
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("SELECT value FROM media_schema WHERE key='version'").fetchone()[0] == str(
        MEDIA_SCHEMA_VERSION)
    conn.close()


def test_reopen_is_idempotent(root):
    open_media_store(create=True, data_root=root).close()
    s = open_media_store(data_root=root)
    assert s is not None
    s.close()


def test_newer_schema_opens_read_only(root, caplog):
    open_media_store(create=True, data_root=root).close()
    conn = sqlite3.connect(media_db_path(root))
    conn.execute("UPDATE media_schema SET value='99' WHERE key='version'")
    conn.commit()
    conn.close()
    s = open_media_store(data_root=root)
    try:
        assert s.read_only
        assert s.list_items("default") == []
        with pytest.raises(MediaStoreReadOnly):
            s.insert_item(**item())
    finally:
        s.close()


def test_item_crud_and_sha_lookup(store):
    mid = store.insert_item(**item(width=3, height=4))
    got = store.get_item(mid)
    assert got["source_sha256"] == "a" * 64 and got["state"] == "active" and got["width"] == 3
    assert store.find_by_sha("default", "a" * 64)["media_id"] == mid
    assert store.find_by_sha("other", "a" * 64) is None
    store.insert_item(**item(sha="b" * 64, kind="page"))
    assert len(store.list_items("default")) == 2
    assert len(store.list_items("default", kind="page")) == 1
    assert store.count_and_bytes("default") == (2, 20)
    store.set_state(mid, "tombstoned")
    assert len(store.list_items("default")) == 1
    assert len(store.list_items("default", state="tombstoned")) == 1
    assert store.count_and_bytes("default") == (1, 10)


@pytest.mark.parametrize("key", ["GPSLatitude", "gpsinfo", "GPS", 34853, "34853"])
def test_gps_in_exif_is_refused(store, key):
    with pytest.raises(ValueError):
        store.insert_item(**item(exif_json={"Make": "x", key: "1"}))


def test_exif_without_gps_is_stored(store):
    mid = store.insert_item(**item(exif_json={"Make": "x"}))
    assert '"Make"' in store.get_item(mid)["exif_json"]


def test_vectors_wrong_dim_and_knn_by_profile(store):
    sid = store.ensure_active_space("m", "r1")
    assert store.active_space()["space_id"] == sid
    assert store.ensure_active_space("m", "r1") == sid
    a = store.insert_item(**item(sha="1" * 64))
    b = store.insert_item(**item(sha="2" * 64))
    c = store.insert_item(**item(profile="other", sha="3" * 64))
    with pytest.raises(ValueError):
        store.put_vector(a, sid, "default", [1.0, 2.0])
    store.put_vector(a, sid, "default", vec(0))
    store.put_vector(b, sid, "default", vec(1))
    store.put_vector(c, sid, "other", vec(0))
    hits = store.knn(vec(0), "default", 5)
    assert [h[0] for h in hits][0] == a
    assert {h[0] for h in hits} == {a, b}
    assert hits[0][1] <= hits[1][1]
    assert [h[0] for h in store.knn(vec(0), "other", 5)] == [c]
    store.delete_vectors(a)
    assert {h[0] for h in store.knn(vec(0), "default", 5)} == {b}


def test_new_space_demotes_the_old_one(store):
    first = store.ensure_active_space("m", "r1")
    second = store.ensure_active_space("m", "r2")
    assert first != second and store.active_space()["space_id"] == second


def test_jobs_lifecycle(store):
    jid = store.enqueue_job("default", "document", total=4)
    job = store.claim_job("w1", lease_s=60)
    assert job["job_id"] == jid and job["state"] == "running"
    assert store.claim_job("w2", lease_s=60) is None
    assert store.progress_job(jid, "w1", 2)
    assert store.get_job(jid)["done"] == 2
    assert store.renew_lease(jid, "w1", 30)
    assert not store.renew_lease(jid, "w2", 30)
    assert store.finish_job(jid, "w1", "done")
    assert store.get_job(jid)["state"] == "done"
    assert store.list_jobs("default", states=["done"])[0]["job_id"] == jid
    assert store.list_jobs("default", states=["queued"]) == []
    assert store.claim_job("w1", kinds=["gc"]) is None


def test_expired_lease_is_reclaimed(store):
    jid = store.enqueue_job("default", "gc")
    assert store.claim_job("w1", lease_s=-5)["job_id"] == jid
    assert store.claim_job("w2", lease_s=60)["lease_owner"] == "w2"


def test_purge_jobs_drops_old_finished_jobs_only(store):
    old = store.enqueue_job("default", "gc")
    store.claim_job("w")
    store.finish_job(old, "w", "done")
    live = store.enqueue_job("default", "gc")
    with store._write() as conn:
        conn.execute("UPDATE jobs SET updated_at='2000-01-01T00:00:00.000000Z' WHERE job_id=?", (old,))
    assert store.purge_jobs(older_than_days=30) == 1
    assert store.get_job(old) is None and store.get_job(live) is not None


def test_two_threads_claim_one_job(store):
    store.enqueue_job("default", "gc")
    got, barrier = [], threading.Barrier(2)

    def worker(name):
        barrier.wait()
        got.append(store.claim_job(name))

    ts = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(1 for g in got if g) == 1


def _fill(store, profile, sha):
    sid = store.ensure_active_space("m", "r")
    mid = store.insert_item(**item(profile=profile, sha=sha))
    store.put_vector(mid, sid, profile, vec(0))
    store.enqueue_job(profile, "gc")
    with store._write() as c:
        c.execute("INSERT INTO documents(document_id,profile_id,sha256,title,mime,state,created_at,updated_at)"
                  " VALUES (?,?,?,?,?,'ready','t','t')",
                  (f"d-{profile}", profile, sha, "t", "application/pdf"))
        c.execute("INSERT INTO doc_pages(document_id,page_no,text_origin) VALUES (?,1,'none')",
                  (f"d-{profile}",))
        c.execute("INSERT INTO sources VALUES (?,?,'folder','/x','x','[]','active',0,1,'t',NULL,'{}')",
                  (f"s-{profile}", profile))
        c.execute("INSERT INTO source_files(source_id,relpath,size,mtime_ns,state,updated_at) "
                  "VALUES (?, 'a', 1, 1, 'pending', 't')", (f"s-{profile}",))
        c.execute("INSERT INTO source_links VALUES (?, 'a', 't', 'wikilink', NULL)", (f"s-{profile}",))
    return mid


def _counts(store, profile):
    with store._write() as c:
        return {t: c.execute(f"SELECT COUNT(*) FROM {t} WHERE profile_id=?", (profile,)).fetchone()[0]
                for t in ("media_items", "documents", "jobs", "sources", "media_vector_rows")}


def test_delete_profile_rows_only_touches_that_profile(store):
    _fill(store, "p1", "1" * 64)
    other = _fill(store, "p2", "2" * 64)
    assert store.delete_profile_rows("p1") > 0
    assert all(v == 0 for v in _counts(store, "p1").values())
    assert all(v == 1 for v in _counts(store, "p2").values())
    assert store.knn(vec(0), "p1", 5) == []
    assert [h[0] for h in store.knn(vec(0), "p2", 5)] == [other]
    with store._write() as c:
        assert c.execute("SELECT COUNT(*) FROM source_files WHERE source_id='s-p1'").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM doc_pages WHERE document_id='d-p1'").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM source_links WHERE source_id='s-p1'").fetchone()[0] == 0


def test_move_profile_rows_moves_everything(store):
    mid = _fill(store, "p1", "1" * 64)
    assert store.move_profile_rows("p1", "default") > 0
    assert all(v == 0 for v in _counts(store, "p1").values())
    assert store.get_item(mid)["profile_id"] == "default"
    assert [h[0] for h in store.knn(vec(0), "default", 5)] == [mid]
    assert store.knn(vec(0), "p1", 5) == []


def test_profile_delete_sidecar_moves_media_and_makes_nothing_when_absent(root):
    from superlocalmemory.storage import profile_fold_sidecars as sidecars

    root.mkdir()
    assert sidecars.move_media(root, "gone", "default") == 0
    assert not media_db_path(root).exists()
    s = open_media_store(create=True, data_root=root)
    mid = s.insert_item(**item(profile="gone"))
    s.close()
    assert sidecars.move_media(root, "gone", "default") == 1
    s = open_media_store(data_root=root)
    try:
        assert s.get_item(mid)["profile_id"] == "default"
    finally:
        s.close()


def test_exif_keeps_only_known_plain_keys(store):
    mid = store.insert_item(**item(exif_json={"Make": "x", "MakerNote": "blob", "Model": {"a": 1}}))
    kept = json.loads(store.get_item(mid)["exif_json"])
    assert kept == {"Make": "x"}


def test_space_id_must_fully_match(store):
    from superlocalmemory.media.store import vec_table

    with pytest.raises(ValueError):
        vec_table("a" * 32 + "\n")


def test_old_owner_cannot_finish_a_reclaimed_job(store):
    jid = store.enqueue_job("default", "gc")
    store.claim_job("w1", lease_s=-5)
    assert store.claim_job("w2", lease_s=60)["lease_owner"] == "w2"
    assert store.finish_job(jid, "w1", "done") is False
    assert store.progress_job(jid, "w1", 5) is False
    job = store.get_job(jid)
    assert job["state"] == "running" and job["lease_owner"] == "w2" and job["done"] == 0
    assert store.finish_job(jid, "w2", "done") is True


def test_move_resolves_source_root_collisions(store):
    with store._write() as c:
        for pid, sid in (("p1", "s1"), ("default", "s2")):
            c.execute("INSERT INTO sources VALUES (?,?,'folder','/same','x','[]','active',0,1,'t',NULL,'{}')",
                      (sid, pid))
    store.move_profile_rows("p1", "default")
    with store._write() as c:
        rows = dict(c.execute("SELECT source_id, state FROM sources").fetchall())
    assert rows == {"s1": "removed", "s2": "active"}


def test_move_skips_orphan_vector_rows(store):
    mid = _fill(store, "p1", "1" * 64)
    sid = store.active_space()["space_id"]
    with store._write() as c:
        c.execute(f"DELETE FROM media_vec_{sid}")
    store.move_profile_rows("p1", "default")
    with store._write() as c:
        assert c.execute("SELECT COUNT(*) FROM media_vector_rows").fetchone()[0] == 0
    assert store.get_item(mid)["profile_id"] == "default"


def test_sidecar_preflight_refuses_a_read_only_media_db_before_anything(root):
    from superlocalmemory.storage import profile_fold_sidecars as sidecars

    open_media_store(create=True, data_root=root).close()
    conn = sqlite3.connect(media_db_path(root))
    conn.execute("UPDATE media_schema SET value='99' WHERE key='version'")
    conn.commit()
    conn.close()
    with pytest.raises(sidecars.SidecarFoldError):
        sidecars.check_media(root)
    with pytest.raises(sidecars.SidecarFoldError):
        sidecars.move_media(root, "a", "default")


def test_sidecar_preflight_refuses_an_unopenable_media_db(root, monkeypatch):
    from superlocalmemory import media
    from superlocalmemory.storage import profile_fold_sidecars as sidecars

    open_media_store(create=True, data_root=root).close()

    def boom(**_):
        raise ImportError("sqlite_vec")

    monkeypatch.setattr(media, "open_media_store", boom)
    with pytest.raises(sidecars.SidecarFoldError):
        sidecars.check_media(root)
    with pytest.raises(sidecars.SidecarFoldError):
        sidecars.move_media(root, "a", "default")


def test_sidecar_preflight_passes_when_absent_or_fine(root):
    from superlocalmemory.storage import profile_fold_sidecars as sidecars

    root.mkdir()
    sidecars.check_media(root)
    open_media_store(create=True, data_root=root).close()
    sidecars.check_media(root)


def test_two_hashes_remote_flag_and_preassigned_id(store):
    mid = "c" * 32
    got_id = store.insert_item(**item(media_id=mid, stored_sha256="b" * 64))
    assert got_id == mid
    row = store.get_item(mid)
    assert row["stored_sha256"] == "b" * 64 and row["remote_ok"] == 0
    store.insert_item(**item(sha="d" * 64, remote_ok=1))
    assert [r["remote_ok"] for r in store.list_items("default")] == [0, 1]


def test_phash_candidates_are_per_profile_and_active_only(store):
    a = store.insert_item(**item(sha="1" * 64, phash="f" * 16))
    store.insert_item(**item(sha="2" * 64, phash=None))
    store.insert_item(**item(profile="other", sha="3" * 64, phash="0" * 16))
    gone = store.insert_item(**item(sha="4" * 64, phash="e" * 16))
    store.set_state(gone, "tombstoned")
    assert store.phash_candidates("default") == [(a, "f" * 16)]


def test_vector_count_is_per_profile(store):
    space = store.ensure_active_space("m", "r", DIM)
    assert store.vector_count("default") == 0
    store.insert_item(**item(media_id="1" * 32, sha="1" * 64))
    store.put_vector("1" * 32, space, "default", vec(0))
    assert store.vector_count("default") == 1
    assert store.vector_count("other") == 0


def test_memory_ids_of_gives_anchors_and_page_memories(store):
    store.insert_item(**item(media_id="2" * 32, sha="2" * 64, anchor_memory_id="m-anchor"))
    store.insert_item(**item(media_id="3" * 32, sha="3" * 64, kind="page", document_id="d", page_no=1,
                             origin="document"))
    store.insert_item(**item(media_id="4" * 32, sha="4" * 64, anchor_memory_id="m-gone"))
    store.set_state("4" * 32, "tombstoned")
    with store._write() as conn:
        conn.execute("INSERT INTO doc_pages(document_id, page_no, media_id, memory_ids_json, text_origin)"
                     " VALUES ('d', 1, ?, '[\"p1\",\"p2\"]', 'ocr')", ("3" * 32,))
    assert store.memory_ids_of(["2" * 32, "3" * 32, "4" * 32, "5" * 32]) == {
        "2" * 32: ["m-anchor"], "3" * 32: ["p1", "p2"]}
    assert store.memory_ids_of([]) == {}
