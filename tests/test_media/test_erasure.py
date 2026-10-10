"""Erasing a memory also erases the image it was saved with."""

from __future__ import annotations

import sqlite3

from superlocalmemory.core.transactions.concrete_owners import build_erasure_service_for_db
from superlocalmemory.core.transactions.erasure import ErasureService
from superlocalmemory.core.transactions.owners import OperationContext
from superlocalmemory.media import files, media_db_path
from superlocalmemory.media.erasure import MediaErasureOwner

from tests.test_media._erase_support import MemDb, add_image, cached, make_root


def _erase(db, profile, fact_ids, root=None):
    svc = ErasureService({"media": MediaErasureOwner(db, data_root=root)})
    ctx = OperationContext(operation_id="op1", profile_id=profile, subject_id="x", fact_ids=tuple(fact_ids))
    removed = svc.remove(db, ctx)
    return svc.finalize(db, ctx, subject_type="entity", subject_id="x", remove_result=removed)


def _rows(store, table="media_items"):
    return store._read().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_erasing_the_anchor_memory_removes_row_vector_file_and_cached_text(tmp_path):
    root, db, store = make_root(tmp_path)
    db.add_memory("m1", fact_id="f1")
    sha, rel = add_image(root, store, media_id="a" * 32, memory_id="m1")
    assert cached(root, sha)
    receipt = _erase(db, "p1", ["f1"], root)
    assert [p.owner for p in receipt.proofs] == ["media"]
    assert receipt.proofs[0].erased and receipt.all_erased
    assert _rows(store) == 0 and _rows(store, "media_vector_rows") == 0
    assert not (files.media_root(root) / rel).exists()
    assert not cached(root, sha)
    store.close()


def test_a_shared_original_stays_until_the_last_row_goes(tmp_path):
    root, db, store = make_root(tmp_path)
    db.add_memory("m1", fact_id="f1")
    db.add_memory("m2", fact_id="f2")
    sha, rel = add_image(root, store, media_id="a" * 32, memory_id="m1", data=b"same")
    add_image(root, store, media_id="b" * 32, memory_id="m2", data=b"same")
    _erase(db, "p1", ["f1"], root)
    assert _rows(store) == 1
    assert (files.media_root(root) / rel).exists()
    assert cached(root, sha)  # still used by the other row
    _erase(db, "p1", ["f2"], root)
    assert _rows(store) == 0
    assert not (files.media_root(root) / rel).exists()
    assert not cached(root, sha)
    store.close()


def test_another_profiles_image_is_untouched(tmp_path):
    root, db, store = make_root(tmp_path)
    db.add_memory("m1", profile="p1", fact_id="f1")
    db.add_memory("m1b", profile="p2", fact_id="f9")
    add_image(root, store, media_id="a" * 32, profile="p1", memory_id="m1", data=b"one")
    sha2, rel2 = add_image(root, store, media_id="b" * 32, profile="p2", memory_id="m1b", data=b"two")
    # a same-named anchor id in the other profile must not match either
    add_image(root, store, media_id="c" * 32, profile="p2", memory_id="m1", data=b"three")
    _erase(db, "p1", ["f1"], root)
    left = {r["media_id"] for r in store.list_items("p2")}
    assert left == {"b" * 32, "c" * 32}
    assert (files.media_root(root) / rel2).exists()
    store.close()


def test_no_media_db_is_a_no_op_and_creates_nothing(tmp_path):
    root = tmp_path / "slm"
    root.mkdir()
    db = MemDb(root)
    db.add_memory("m1", fact_id="f1")
    before = sorted(p.name for p in root.rglob("*"))
    receipt = _erase(db, "p1", ["f1"], root)
    assert receipt.proofs[0].owner == "media" and receipt.proofs[0].erased
    assert not media_db_path(root).exists()
    assert sorted(p.name for p in root.rglob("*")) == before


def test_unknown_data_root_is_a_no_op(tmp_path):
    class Bare:
        def execute(self, sql, params=()):
            return []

    owner = MediaErasureOwner(Bare())
    ctx = OperationContext(operation_id="o", profile_id="p1", subject_id="x", fact_ids=("f1",))
    assert owner.erase(ctx).erased and owner.prove_erased(ctx).erased


def test_facts_with_no_image_change_nothing(tmp_path):
    root, db, store = make_root(tmp_path)
    db.add_memory("m1", fact_id="f1")
    db.add_memory("m2", fact_id="f2")
    sha, rel = add_image(root, store, media_id="a" * 32, memory_id="m2")
    receipt = _erase(db, "p1", ["f1"], root)
    assert receipt.all_erased and _rows(store) == 1
    assert (files.media_root(root) / rel).exists()
    store.close()


def test_an_unremovable_file_is_reported_as_residue(tmp_path, monkeypatch):
    root, db, store = make_root(tmp_path)
    db.add_memory("m1", fact_id="f1")
    sha, rel = add_image(root, store, media_id="a" * 32, memory_id="m1")
    monkeypatch.setattr("superlocalmemory.media.files.remove_original", lambda *a, **k: False)
    receipt = _erase(db, "p1", ["f1"], root)
    assert not receipt.proofs[0].erased and not receipt.all_erased
    store.close()


def test_the_real_erasure_service_has_a_media_owner(tmp_path):
    root, db, store = make_root(tmp_path)
    store.close()
    svc = build_erasure_service_for_db(db)
    assert "media" in svc._owners


def test_erasing_a_whole_profile_takes_rows_with_no_anchor_too(tmp_path):
    from superlocalmemory.media.erasure import erase_profile

    root, db, store = make_root(tmp_path)
    sha, rel = add_image(root, store, media_id="a" * 32, memory_id="gone")
    sha2, rel2 = add_image(root, store, media_id="b" * 32, profile="p2", memory_id="x", data=b"o")
    out = erase_profile(root, "p1")
    assert out["items"] == 1
    assert [r["media_id"] for r in store.list_items("p2")] == ["b" * 32]
    assert not (files.media_root(root) / rel).exists()
    assert (files.media_root(root) / rel2).exists()
    store.close()
    assert erase_profile(tmp_path / "none", "p1")["items"] == 0
    assert not (tmp_path / "none").exists()
