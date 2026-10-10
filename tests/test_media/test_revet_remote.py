"""Pictures held back from web apps by the 4.1.25 upgrade are vetted again from the text kept in their memory.

Up to 4.1.24 a picture counted as clean when nothing had been counted, so the upgrade holds every
such picture back. Where the picture's text is still stored in its memory, the same full-text scan
the save path uses reads it again, and the picture is offered to web apps only when that scan is clean.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from superlocalmemory.media import open_media_store
from superlocalmemory.media.ingest import _segments
from superlocalmemory.media.labels import NO_TEXT
from tests.helpers.env_capabilities import NO_VECTOR_SEARCH_REASON, vector_search_available

pytestmark = pytest.mark.skipif(not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON)

KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"


def saved_text(note: str, ocr: str) -> str:
    """The content ``remember_media`` hands the writer for a picture."""
    return "".join(text for text, _origin in _segments(note, ocr))


class Library:
    """A data folder with a media.db and a memory.db, the way an upgraded install looks."""

    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.store = open_media_store(create=True, data_root=root)
        self.mem = sqlite3.connect(root / "memory.db")
        self.mem.execute("CREATE TABLE IF NOT EXISTS memories (memory_id TEXT PRIMARY KEY, profile_id TEXT,"
                         " content TEXT, metadata_json TEXT DEFAULT '{}')")
        self.mem.commit()
        self.n = 0
        self.m = 0

    def memory(self, content, profile="p1", memory_id=None):
        self.m += 1
        memory_id = memory_id or f"mem{self.m}"
        self.mem.execute("INSERT INTO memories(memory_id, profile_id, content) VALUES (?,?,?)",
                         (memory_id, profile, content))
        self.mem.commit()
        return memory_id

    def picture(self, text=None, *, profile="p1", ok=0, anchor=True, kind="image", state="active", **kw):
        self.n += 1
        media_id = f"{self.n:032x}"
        anchor_id = self.memory(text, profile) if (anchor and text is not None) else None
        fields = dict(media_id=media_id, profile_id=profile, kind=kind, source_sha256=media_id * 2,
                      mime="image/png", bytes=10, origin="tool", remote_ok=ok,
                      anchor_memory_id=anchor_id, state=state)
        self.store.insert_item(**{**fields, **kw})
        return media_id

    def flag(self, media_id):
        return self.store.get_item(media_id)["remote_ok"]

    def close(self):
        self.store.close()
        self.mem.close()


@pytest.fixture()
def lib(tmp_path):
    library = Library(tmp_path / "slm")
    yield library
    library.close()


def revet(lib, **kw):
    from superlocalmemory.media.revet import revet_remote_flags

    return revet_remote_flags(data_root=lib.root, **kw)


def test_a_held_back_picture_whose_text_is_clean_becomes_shareable(lib):
    mid = lib.picture(saved_text("my cat", "Opening hours 9 to 5, Main Street"))

    report = revet(lib)

    assert lib.flag(mid) == 1
    assert report.cleared == 1 and report.held == 0


def test_a_picture_whose_text_holds_a_credential_stays_held_back(lib):
    raw = lib.picture(saved_text("", f"token {KEY}"))
    redacted = lib.picture(saved_text("", "token [REDACTED:API_KEY]"))

    report = revet(lib)

    assert lib.flag(raw) == 0 and lib.flag(redacted) == 0
    assert report.cleared == 0 and report.held == 2


@pytest.mark.parametrize("ocr", ["write to bob@example.com", "write to [PII:EMAIL]"])
def test_personal_data_in_the_text_keeps_the_picture_held_back(lib, ocr):
    mid = lib.picture(saved_text("", ocr))

    revet(lib)

    assert lib.flag(mid) == 0


def test_a_picture_with_no_stored_text_stays_held_back(lib):
    mid = lib.picture(NO_TEXT)

    report = revet(lib)

    assert lib.flag(mid) == 0 and report.held == 1


def test_a_picture_whose_text_may_have_been_cut_stays_held_back(lib):
    """Saving keeps at most 8,000 characters of a picture's text; the rest was never stored."""
    mid = lib.picture(saved_text("", "word " * 1600))      # 8,000 characters

    revet(lib)

    assert lib.flag(mid) == 0


def test_only_the_picture_text_is_scanned_not_the_persons_own_note(lib):
    mid = lib.picture(saved_text("call me on bob@example.com", "Opening hours 9 to 5"))

    revet(lib)

    assert lib.flag(mid) == 1


def test_a_picture_without_its_memory_yet_is_tried_again_later(lib):
    mid = lib.picture(None, anchor=False)

    first = revet(lib)
    lib.store.fill_anchor(mid, lib.memory(saved_text("", "Opening hours 9 to 5")))
    second = revet(lib)

    assert first.waiting == 1 and first.cleared == 0
    assert lib.flag(mid) == 1 and second.cleared == 1


def test_a_memory_that_belongs_to_another_profile_is_not_used(lib):
    other = lib.memory(saved_text("", "Opening hours"), profile="p2")
    mid = lib.picture(None, anchor=False)
    lib.store.fill_anchor(mid, other)

    revet(lib)

    assert lib.flag(mid) == 0


def test_a_picture_already_cleared_or_removed_is_left_alone(lib):
    ok = lib.picture(saved_text("", f"token {KEY}"), ok=1)       # vetted by the new rule: never touched
    gone = lib.picture(saved_text("", "Opening hours"), state="tombstoned")

    revet(lib)

    assert lib.flag(ok) == 1 and lib.flag(gone) == 0


def test_a_picture_is_read_once_and_the_work_resumes_where_it_stopped(lib, monkeypatch):
    from superlocalmemory.media import revet as revet_mod

    ids = [lib.picture(saved_text("", f"line {i} of the sign")) for i in range(3)]
    scanned = []
    real = revet_mod.scan_sensitive
    monkeypatch.setattr(revet_mod, "scan_sensitive", lambda text: scanned.append(text) or real(text))

    first = revet(lib, max_items=2)
    assert first.cleared == 2 and first.more is True
    lib.store.close()
    lib.store = open_media_store(data_root=lib.root)      # the daemon restarted
    second = revet(lib)

    assert second.cleared == 1 and second.more is False
    assert len(scanned) == 3, "no picture was read twice"
    assert [lib.flag(i) for i in ids] == [1, 1, 1]
    assert revet(lib).cleared == 0 and len(scanned) == 3


def test_a_document_page_is_vetted_from_the_text_of_all_its_parts(lib):
    clean = lib.picture(None, anchor=False, kind="page", document_id="d" * 32, page_no=1, origin="document")
    held = lib.picture(None, anchor=False, kind="page", document_id="d" * 32, page_no=2, origin="document")
    for page, parts in ((1, ["[Page 1]\nFirst part of a page.", "[Page 1]\nSecond part."]),
                        (2, ["[Page 2]\nFirst part.", f"[Page 2]\nPassword is {KEY}"])):
        ids = [lib.memory(p) for p in parts]
        with lib.store._write() as conn:
            conn.execute("INSERT INTO doc_pages(document_id, page_no, media_id, memory_ids_json, text_origin)"
                         " VALUES (?,?,?,?, 'text_layer')",
                         ("d" * 32, page, clean if page == 1 else held, json.dumps(ids)))

    revet(lib)

    assert lib.flag(clean) == 1 and lib.flag(held) == 0


def test_a_page_with_no_stored_text_stays_held_back(lib):
    mid = lib.picture(None, anchor=False, kind="page", document_id="d" * 32, page_no=1, origin="document")

    revet(lib)

    assert lib.flag(mid) == 0


def test_a_library_with_nothing_to_vet_or_no_memory_database_is_a_no_op(tmp_path):
    from superlocalmemory.media.revet import revet_remote_flags

    assert revet_remote_flags(data_root=tmp_path / "nothing").cleared == 0
    library = Library(tmp_path / "slm")
    library.mem.close()
    (tmp_path / "slm" / "memory.db").unlink()
    mid = library.picture(None, anchor=False)
    assert revet_remote_flags(data_root=library.root).cleared == 0
    assert library.flag(mid) == 0
    library.store.close()


def test_an_older_media_database_gets_the_progress_column(tmp_path):
    root = tmp_path / "slm"
    store = open_media_store(create=True, data_root=root)
    store.close()
    db = sqlite3.connect(root / "media.db")
    db.execute("ALTER TABLE media_items DROP COLUMN remote_checked")
    db.commit()
    db.close()

    store = open_media_store(data_root=root)
    try:
        cols = {r[1] for r in store._read().execute("PRAGMA table_info(media_items)")}
    finally:
        store.close()

    assert "remote_checked" in cols


# -- the daemon runs it once in the background after an upgrade --------------------------------

def test_the_housekeeping_loop_vets_in_batches_and_survives_a_failure(lib):
    from superlocalmemory.server import media_housekeeping as hk

    mids = [lib.picture(saved_text("", f"sign {i}")) for i in range(5)]
    calls = {"n": 0}

    def vet(**kw):
        from superlocalmemory.media.revet import revet_remote_flags

        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk busy")
        return revet_remote_flags(**kw)

    async def drive():
        task = asyncio.create_task(hk.run(lib.root, fill=lambda **kw: 0, revet=vet, revet_batch=2,
                                          interval_s=0.01, first_delay_s=0.0))
        await asyncio.sleep(0.6)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())

    assert [lib.flag(m) for m in mids] == [1] * 5
    assert calls["n"] >= 4, "batches of two, after one failed pass"
