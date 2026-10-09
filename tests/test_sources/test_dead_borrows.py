"""A copy that borrowed another source's picture is saved afresh once that picture is gone."""

from __future__ import annotations

import json

from superlocalmemory import sources
from tests.test_sources.test_real_picture import media_rows, pics, png  # noqa: F401


def test_a_copy_in_another_source_becomes_its_own_owner(pics, tmp_path):
    env = pics
    env.write("p.png", png("a"))
    env.write("keep.canvas", "{}")
    sid1 = env.add_and_confirm()
    env.scan(sid1)
    root2 = tmp_path / "vault2"
    root2.mkdir()
    (root2 / "p.png").write_bytes(png("a"))
    (root2 / "k.canvas").write_text("{}")
    prev = sources.add_source(root2, profile_id="default")
    sources.confirm_source(prev.source_id)
    sid2 = prev.source_id
    env.scan(sid2)
    assert env.files(sid2)["p.png"]["reason"] == "shared"
    (env.root / "p.png").unlink()
    env.scan(sid1)
    env.scan(sid2)
    row = env.files(sid2)["p.png"]
    archived = {r[0] for r in env.db.execute("SELECT memory_id FROM atomic_facts WHERE lifecycle='archived'")}
    own = [e["m"] for e in json.loads(row["memory_ids_json"]) if e.get("m")]
    assert row["state"] == "indexed" and row["reason"] is None and row["media_id"]
    assert own and not set(own) & archived
    assert media_rows(env)[row["media_id"]] == "active"


def test_borrow_of_a_retention_archived_user_memory_stays_put(pics, tmp_path):
    env = pics
    from superlocalmemory.media.ingest import MediaInput, remember_media
    env.write("keep.canvas", "{}")
    sid = env.add_and_confirm()
    env.scan(sid)
    mine = tmp_path / "mine.png"
    mine.write_bytes(png("a"))
    rec = remember_media(MediaInput(path=mine, file_name="mine.png"), profile_id="default",
                         actor_id="user", runtime=env.runtime, config=None)
    env.write("p.png", png("a"))
    env.scan(sid)
    assert env.files(sid)["p.png"]["reason"] == "shared"
    env.db.execute("UPDATE atomic_facts SET lifecycle='archived' WHERE memory_id=?", (rec.memory_id,))
    saves_before = len(env.runtime.saved)
    outs = [env.scan(sid) for _ in range(3)]
    m = env.store()
    n = m._read().execute("SELECT n FROM source_save_counters WHERE relpath='p.png'").fetchone()
    m.close()
    row = env.files(sid)["p.png"]
    assert row["reason"] == "shared" and row["state"] == "indexed"
    assert [o.changed for o in outs] == [0, 0, 0]
    assert len(env.runtime.saved) == saves_before
    assert n is None or n[0] == 1
