# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The document model and region provenance in a document job (WS-8, D-19): fake models, no download."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from superlocalmemory.media import regions as region_mod
from superlocalmemory.runtimes import media_packs
from superlocalmemory.runtimes.worker_client import MediaWorkerError
from superlocalmemory.trust import why
from tests.test_documents.support import FakeClient, Runtime, fake_script, make_service, pdf_input

CFG = SimpleNamespace(pii_redaction=False)
LINE = [{"text": "Scanned receipt text here", "bbox": [0.05, 0.05, 0.4, 0.1]}]


class RegionClient(FakeClient):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.asked = []

    def ocr_image(self, path, *, wait_cold=True, regions=False):
        self.asked.append(regions)
        reply = super().ocr_image(path, wait_cold=wait_cold)
        return {**reply, "regions": list(LINE)} if regions else reply


class DocModel:
    """The document pack's worker client: PaddleOCR-VL at the boundary (a page is one region)."""

    def __init__(self, text="Scanned receipt text from the big model", exc=None, empty=False):
        self.text, self.exc, self.empty, self.calls = text, exc, empty, 0

    def parse_document(self, path, *, wait_cold=True):
        self.calls += 1
        if self.exc:
            raise self.exc
        text = "" if self.empty else self.text
        return {"engine": "paddleocr_vl", "text": text,
                "regions": [{"text": text, "bbox": [0, 0, 1, 1], "label": "page"}] if text else []}


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    for name in ("SLM_FEATURE_MEDIA_DOCUMENT_PACK", "SLM_FEATURE_MEDIA_REGIONS", "SLM_4230_OFF"):
        monkeypatch.delenv(name, raising=False)
    return r


@pytest.fixture()
def store(root):
    s = open_media_store(create=True, data_root=root)
    yield s
    s.close()


@pytest.fixture()
def doc_model(monkeypatch):
    holder = SimpleNamespace(client=DocModel())
    monkeypatch.setenv("SLM_FEATURE_MEDIA_DOCUMENT_PACK", "on")
    monkeypatch.setattr(media_packs, "installed", lambda spec, data_root=None: True)
    monkeypatch.setattr(media_packs, "pack_client", lambda *a, **k: holder.client)
    return holder


def _run(store, tmp_path, client, pages=("", "")):
    receipt = submit_document(pdf_input(("one", "two")), content="", profile_id="p1", actor_id="a",
                              config=CFG, store=store)
    runtime = Runtime()
    service = make_service(store, runtime, client, fake_script(tmp_path, list(pages)), config=CFG)
    assert service.process_next() is True
    return receipt, runtime


def _page_text(runtime):
    return {r.metadata["_slm_source"]["page"]: r.content for r in runtime.requests
            if "page" in r.metadata.get("_slm_source", {})}


def test_everything_off_the_standard_reader_reads_and_asks_for_no_boxes(store, tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_MEDIA_REGIONS", "off")  # on by default; this is a person's own off
    client = RegionClient(ocr={1: "Scanned receipt text here"})
    receipt, runtime = _run(store, tmp_path, client)
    assert client.asked == [False, False]
    assert "Scanned receipt text here" in _page_text(runtime)[1]
    assert store._read().execute("SELECT COUNT(*) FROM media_regions").fetchone()[0] == 0


def test_the_document_model_reads_scanned_pages_when_that_part_is_on(store, tmp_path, doc_model):
    client = RegionClient(ocr={1: "standard reader text"})
    receipt, runtime = _run(store, tmp_path, client)
    assert "from the big model" in _page_text(runtime)[1] and doc_model.client.calls == 2
    assert client.kinds("ocr") == []                      # the standard reader was never needed
    doc = store.get_document(receipt.document_id)
    assert doc["state"] == "ready" and doc["pages_ocr"] == 2


@pytest.mark.parametrize("model", [DocModel(exc=MediaWorkerError("stopped")), DocModel(exc=RuntimeError("boom")),
                                   DocModel(empty=True)])
def test_a_document_model_failure_falls_back_to_the_standard_reader(store, tmp_path, doc_model, model, caplog):
    doc_model.client = model
    client = RegionClient(ocr={1: "Scanned receipt text here", 2: "second page words"})
    with caplog.at_level("WARNING"):
        receipt, runtime = _run(store, tmp_path, client)
    assert "Scanned receipt text here" in _page_text(runtime)[1]          # saved by the standard reader
    assert store.get_document(receipt.document_id)["state"] == "ready"
    assert "boom" not in caplog.text


def test_no_document_client_falls_back_quietly(store, tmp_path, doc_model):
    doc_model.client = None
    client = RegionClient(ocr={1: "Scanned receipt text here"})
    _, runtime = _run(store, tmp_path, client)
    assert "Scanned receipt text here" in _page_text(runtime)[1]


def test_the_kill_switch_puts_the_standard_reader_back(store, tmp_path, doc_model, monkeypatch):
    monkeypatch.setenv("SLM_4230_OFF", "1")
    client = RegionClient(ocr={1: "Scanned receipt text here"})
    _run(store, tmp_path, client)
    assert doc_model.client.calls == 0


def test_page_and_box_are_kept_for_a_scanned_page_and_shown_in_words(store, tmp_path, root, monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_MEDIA_REGIONS", "on")
    client = RegionClient(ocr={1: "Scanned receipt text here"})
    receipt, runtime = _run(store, tmp_path, client)
    assert client.asked == [True, True]
    rows = store._read().execute("SELECT page_no, snippet, engine FROM media_regions ORDER BY page_no").fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [(1, "Scanned receipt text here", "rapidocr")]
    memory = next(r for r in runtime.requests if r.metadata["_slm_source"].get("page") == 1)
    conn = sqlite3.connect(root / "memory.db")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT, profile_id TEXT, content TEXT, created_at TEXT, "
                 "memory_id TEXT)")
    memory_id = store.page_memory_ids(receipt.document_id, 1)[0]
    placed = why._regions(conn, memory_id, "Scanned receipt text here")
    assert [p["where"] for p in placed] == ["from page 1, top left"] and memory.profile_id == "p1"


def test_the_document_model_reports_the_whole_page_as_its_one_region(store, tmp_path, root, doc_model, monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_MEDIA_REGIONS", "on")
    receipt, runtime = _run(store, tmp_path, RegionClient())
    conn = sqlite3.connect(root / "memory.db")
    memory_id = store.page_memory_ids(receipt.document_id, 2)[0]
    placed = why._regions(conn, memory_id, "Scanned receipt text from the big model")
    assert [p["where"] for p in placed] == ["from page 2"]
    assert region_mod.place_words(2, [0, 0, 1, 1]) == "page 2"


def test_erasing_a_document_erases_its_page_regions(store, tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_MEDIA_REGIONS", "on")
    _run(store, tmp_path, RegionClient(ocr={1: "Scanned receipt text here"}))
    ids = store.all_item_ids("p1")
    store.erase_items(ids)
    assert store._read().execute("SELECT COUNT(*) FROM media_regions").fetchone()[0] == 0


def test_the_document_models_page_region_is_not_kept_while_regions_is_off(store, tmp_path, doc_model, monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_MEDIA_REGIONS", "off")  # on by default; this is a person's own off
    _run(store, tmp_path, RegionClient())
    assert store._read().execute("SELECT COUNT(*) FROM media_regions").fetchone()[0] == 0
