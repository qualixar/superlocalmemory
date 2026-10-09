"""Pictures through the real picture path (a throwaway media.db and a fake image worker)."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from tests.test_media.test_ingest import FakeClient, png


@pytest.fixture
def pics(env, monkeypatch):
    """The folder environment with the real ``remember_media`` and a writer that records facts."""
    monkeypatch.setenv("SLM_DATA_DIR", str(env.data))
    monkeypatch.setattr("superlocalmemory.runtimes.worker_client.media_embedder", lambda: FakeClient())
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE atomic_facts(fact_id, memory_id, lifecycle DEFAULT 'active', profile_id DEFAULT 'default')")
    env.runtime._db = db
    remember, archive = env.runtime.remember, env.runtime.archive_fact

    def recording(admission, actor, deadline_ms=0, accept_after_ms=0):
        known = len(env.runtime.saved)
        out = remember(admission, actor, deadline_ms, accept_after_ms)
        if len(env.runtime.saved) > known:
            row = env.runtime.saved[-1]
            db.execute("INSERT INTO atomic_facts(fact_id, memory_id) VALUES (?, ?)", (row["fid"], row["mid"]))
        return out

    def archiving(profile_id, fact_id, *, idempotency_key=None):
        db.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (fact_id,))
        return archive(profile_id, fact_id, idempotency_key=idempotency_key)

    env.runtime.remember, env.runtime.archive_fact = recording, archiving
    env.db = db
    return env


def media_rows(env):
    from superlocalmemory.media import open_media_store

    store = open_media_store(data_root=env.data)
    try:
        return {r["media_id"]: r["state"] for r in store._read().execute("SELECT media_id, state FROM media_items")}
    finally:
        store.close()


def rewrite(path, data, seconds):
    path.write_bytes(data)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, 1_000_000_000_000_000_000 + seconds * 10**9))


def live(env, sid, relpath):
    entries = json.loads(env.files(sid)[relpath]["memory_ids_json"])
    return [e for e in entries if e.get("m") and not e.get("sup")]


def test_a_picture_edited_back_to_earlier_content_is_visible_again(pics):
    env = pics
    path = env.write("p.png", png("a"))
    sid = env.add_and_confirm()
    env.scan(sid)
    first = env.files(sid)["p.png"]["media_id"]
    assert first and media_rows(env) == {first: "active"}
    rewrite(path, png("b"), 1)
    env.scan(sid)
    rewrite(path, png("a"), 2)
    env.scan(sid)
    row = env.files(sid)["p.png"]
    [now] = live(env, sid, "p.png")
    archived = {r[0] for r in env.db.execute("SELECT memory_id FROM atomic_facts WHERE lifecycle = 'archived'")}
    assert row["state"] == "indexed" and row["reason"] is None and row["media_id"]
    assert now["m"] not in archived and len(archived) == 2
    states = media_rows(env)
    assert states[row["media_id"]] == "active" and list(states.values()).count("active") == 1


def test_purging_a_deleted_picture_erases_its_row_and_file(pics):
    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.media.erasure import MediaErasureOwner

    env = pics
    owner = MediaErasureOwner(env.db, data_root=env.data)

    def eraser(profile_id, fact_ids, subject_id):  # the media part of the real erasure service
        ctx = OperationContext(operation_id="op", profile_id=profile_id, subject_id=subject_id,
                               fact_ids=tuple(fact_ids))
        return {"erasure_complete": int(owner.erase(ctx).erased)}

    env.host.eraser = eraser
    path = env.write("p.png", png("a"))
    sid = env.add_and_confirm()
    env.scan(sid)
    stored = [p for p in (env.data / "media").rglob("*.png")]
    assert stored and len(media_rows(env)) == 1
    path.unlink()
    env.scan(sid)
    assert list(media_rows(env).values()) == ["tombstoned"]  # hidden at once, kept for the grace period
    env.host.purge_after_s = -1.0
    env.scan(sid)
    assert media_rows(env) == {} and not [p for p in (env.data / "media").rglob("*.png")]
    assert "p.png" not in env.files(sid)
