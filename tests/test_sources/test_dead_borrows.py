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
