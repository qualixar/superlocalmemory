"""A document the user saves is never merged into a folder-owned one, so removing the folder copy keeps it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.documents.status import remove_document
from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from tests.test_documents.pdfs import make_pdf
from tests.test_documents.support import pdf_input

CFG = SimpleNamespace(pii_redaction=False)
FOLDER = {"origin": "folder", "source_id": "s1", "relpath": "a.pdf", "version": "v1"}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    s = open_media_store(create=True, data_root=tmp_path / "slm")
    yield s
    s.close()


def go(store, **kw):
    return submit_document(pdf_input(("same",)), profile_id="p1", actor_id="a", config=CFG, store=store, **kw)


def test_user_save_after_folder_save_gets_its_own_document(store):
    folder = go(store, folder=FOLDER)
    mine = go(store)
    assert mine.status == "processing" and mine.document_id != folder.document_id
    assert store.get_document(folder.document_id)["origin"] == "folder"
    assert store.get_document(mine.document_id)["origin"] == "user"
    assert remove_document(folder.document_id, "p1", store=store)
    assert store.get_document(mine.document_id)["state"] != "tombstoned"


def test_folder_save_still_reuses_the_users_document(store):
    mine = go(store)
    folder = go(store, folder=FOLDER)
    assert folder.status == "duplicate" and folder.document_id == mine.document_id


def test_a_repeat_user_save_still_dedupes_against_the_users_document(store):
    go(store, folder=FOLDER)
    first = go(store)
    again = go(store)
    assert again.status == "duplicate" and again.document_id == first.document_id
