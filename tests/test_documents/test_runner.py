"""The job runner: pages become memories with their source; the person's words stay apart."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from superlocalmemory.documents import runner as runner_mod
from superlocalmemory.documents.chunking import chunk_text
from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from tests.test_documents.pdfs import make_pdf
from tests.test_documents.support import KEY, FakeClient, Runtime, fake_script, make_service, pdf_input


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    return r


@pytest.fixture()
def store(root):
    s = open_media_store(create=True, data_root=root)
    yield s
    s.close()


CFG = SimpleNamespace(pii_redaction=False)


def submit(store, inp=None, *, content="", config=CFG, **kw):
    r = submit_document(inp or pdf_input(("one", "two", "three")), content=content, profile_id="p1",
                        actor_id="actor-1", config=config, store=store, **kw)
    assert r.status == "processing", r.reason
    return r


def run(store, tmp_path, pages, *, client=None, runtime=None, config=CFG, **spec):
    client, runtime = client or FakeClient(), runtime or Runtime()
    service = make_service(store, runtime, client, fake_script(tmp_path, pages, **spec), config=config)
    return service, client, runtime


def test_pages_become_memories_with_their_source(store, tmp_path, root):
    receipt = submit(store, pdf_input(("one", "two", "three"), file_name="scan.pdf"), content=f"my note {KEY}")
    long_text = f"Invoice number 42 total due, key {KEY} end"
    service, client, runtime = run(store, tmp_path, [long_text, "", ""], client=FakeClient(ocr={2: "Scanned receipt text here"}),
                                   title="Quarterly", created="2024-01-31")
    assert service.process_next() is True and service.process_next() is False
    doc = store.get_document(receipt.document_id)
    assert doc["state"] == "ready" and doc["page_count"] == 3 and doc["title"] == "Quarterly"
    assert (doc["pages_text_layer"], doc["pages_ocr"], doc["pages_empty"]) == (1, 1, 1)
    pages = {r["page_no"]: r for r in store.get_pages(receipt.document_id)}
    assert [pages[n]["text_origin"] for n in (1, 2, 3)] == ["text_layer", "ocr", "none"]
    assert json.loads(pages[3]["memory_ids_json"]) == [] and pages[3]["char_count"] == 0
    one, two = (r for r in runtime.requests if r.source_type == "document" and "page" in r.metadata["_slm_source"])
    assert one.metadata["_slm_source"] == {"type": "document", "document_id": receipt.document_id, "page": 1}
    assert one.idempotency_key == f"doc:{receipt.document_id}:1:1" and "Invoice number 42" in one.content
    assert KEY not in one.content and "[REDACTED" in one.content          # page text is derived: key stripped
    assert "Scanned receipt text here" in two.content and "my note" not in one.content + two.content
    words = [r for r in runtime.requests if "page" not in r.metadata["_slm_source"]]
    assert len(words) == 1 and KEY in words[0].content and "Quarterly" in words[0].content   # words verbatim
    assert words[0].metadata["_slm_source"] == {"type": "document", "document_id": receipt.document_id}
    assert client.kinds("ocr") == [2, 3] and client.kinds("embed") == [1, 2, 3]
    item = store.get_item(pages[1]["media_id"])
    assert (item["kind"], item["document_id"], item["page_no"], item["origin"]) == ("page", receipt.document_id, 1, "document")
    assert item["anchor_memory_id"] == "mem1" and item["thumb_webp"] == b"RIFFthumb" and item["remote_ok"] == 0
    assert store.get_item(pages[2]["media_id"])["remote_ok"] == 1
    job = store.get_job(receipt.job_id)
    assert (job["state"], job["done"], job["total"]) == ("done", 3, 3)
    assert not list((root / "media" / "tmp").iterdir())


def test_no_title_and_no_words_means_no_document_memory(store, tmp_path):
    receipt = submit(store)
    service, _, runtime = run(store, tmp_path, ["a" * 40, "b" * 40, "c" * 40])
    service.process_next()
    assert len(runtime.requests) == 3 and all("page" in r.metadata["_slm_source"] for r in runtime.requests)
    assert store.get_document(receipt.document_id)["state"] == "ready"


def test_a_file_name_gives_the_title_and_a_document_memory(store, tmp_path):
    submit(store, pdf_input(("x",), file_name="Tax return 2024.pdf"))
    service, _, runtime = run(store, tmp_path, ["a" * 40])
    service.process_next()
    assert any("Tax return 2024" in r.content for r in runtime.requests if "page" not in r.metadata["_slm_source"])


def test_long_page_is_cut_at_paragraph_boundaries(store, tmp_path):
    receipt = submit(store, pdf_input(("x",)))
    text = "\n\n".join(f"Paragraph {i} " + "word " * 400 for i in range(40))
    assert len(text) > 60_000
    service, _, runtime = run(store, tmp_path, [text])
    service.process_next()
    reqs = [r for r in runtime.requests if "page" in r.metadata["_slm_source"]]
    assert len(reqs) >= 3 and all(len(r.content) <= 24_000 for r in reqs)
    assert [r.idempotency_key for r in reqs] == [f"doc:{receipt.document_id}:1:{i}" for i in range(1, len(reqs) + 1)]
    assert [r.metadata["_slm_source"]["part"] for r in reqs] == list(range(1, len(reqs) + 1))
    body = "".join(r.content.split("\n", 1)[1] for r in reqs)
    assert body.replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")
    page = store.get_pages(receipt.document_id)[0]
    assert len(json.loads(page["memory_ids_json"])) == len(reqs)


def test_chunking_prefers_paragraphs_then_sentences_then_a_hard_cut():
    paras = ["a" * 10_000, "b" * 10_000, "c" * 10_000]
    assert chunk_text("\n\n".join(paras), 24_000) == ["a" * 10_000 + "\n\n" + "b" * 10_000, "c" * 10_000]
    sentences = ("Hello world. " * 3000)
    parts = chunk_text(sentences, 24_000)
    assert all(len(p) <= 24_000 for p in parts) and all(p.endswith(".") for p in parts)
    hard = chunk_text("x" * 50_000, 24_000)
    assert [len(p) for p in hard] == [24_000, 24_000, 2_000]
    assert chunk_text("short", 24_000) == ["short"] and chunk_text("  \n ", 24_000) == []


def test_progress_is_recorded_page_by_page(store, tmp_path):
    receipt = submit(store)
    seen = []
    runtime = Runtime(on_remember=lambda adm: seen.append(store.get_job(receipt.job_id)["done"]))
    service, _, _ = run(store, tmp_path, ["a" * 40, "b" * 40, "c" * 40], runtime=runtime)
    service.process_next()
    assert seen == [0, 1, 2]


def test_ocr_engine_missing_leaves_a_page_row_but_no_memory(store, tmp_path):
    receipt = submit(store, pdf_input(("x",)))
    service, _, runtime = run(store, tmp_path, [""], client=FakeClient(engine="none"))
    service.process_next()
    page = store.get_pages(receipt.document_id)[0]
    assert page["text_origin"] == "none" and runtime.requests == []
    assert store.get_item(page["media_id"])["remote_ok"] == 0


def test_pii_on_redacts_page_text_and_words(store, tmp_path):
    cfg = SimpleNamespace(pii_redaction=True)
    submit(store, pdf_input(("x",)), content="mail bob@example.com", config=cfg)
    service, _, runtime = run(store, tmp_path, ["write to amy@example.org please now"], config=cfg)
    service.process_next()
    assert runtime.requests and all("@" not in r.content for r in runtime.requests)


def test_a_document_removed_before_it_runs_is_cancelled(store, tmp_path):
    receipt = submit(store)
    store.tombstone_document(receipt.document_id)
    service, client, runtime = run(store, tmp_path, ["a" * 40])
    service.process_next()
    assert store.get_job(receipt.job_id)["state"] == "cancelled" and runtime.requests == [] and client.calls == []


def test_unusable_runtime_leaves_the_job_waiting(store, tmp_path):
    receipt = submit(store)
    service = make_service(store, None, FakeClient(), fake_script(tmp_path, ["a" * 40]))
    assert service.process_next() is False
    assert store.get_job(receipt.job_id)["state"] == "queued"


def test_real_pdf_end_to_end(store, tmp_path, pdf_python):
    inp = pdf_input(("The quick brown fox jumps over the lazy dog", ""), title="Real", created="D:20240131000000Z")
    receipt = submit(store, inp)
    runtime, client = Runtime(), FakeClient()
    client.ocr = {}
    service = make_service(store, runtime, client, runner_mod.PARSE_SCRIPT)
    service._deps = service._deps.__class__(**{**service._deps.__dict__, "python_supplier": lambda: pdf_python})
    service.process_next()
    doc = store.get_document(receipt.document_id)
    assert doc["state"] == "ready" and doc["page_count"] == 2 and doc["title"] == "Real"
    assert any("quick brown fox" in r.content for r in runtime.requests)
    assert (doc["pages_text_layer"], doc["pages_empty"]) == (1, 1)


def _scoped_config(default):
    return SimpleNamespace(pii_redaction=False, scope=SimpleNamespace(default_scope=default))


def test_pages_are_saved_with_the_scope_the_document_was_sent_with(store, tmp_path, root):
    submit(store, scope="shared", shared_with=("p2",))
    service, _client, runtime = run(store, tmp_path, ["alpha text", "beta text", "gamma text"])
    assert service.process_next() is True
    sent = [r for r in runtime.requests if "page" in r.metadata["_slm_source"]]
    assert sent and all(r.scope == "shared" and tuple(r.shared_with) == ("p2",) for r in sent)


def test_pages_use_the_configured_default_scope_when_none_was_sent(store, tmp_path, root):
    cfg = _scoped_config("global")
    submit(store, config=cfg)
    service, _client, runtime = run(store, tmp_path, ["alpha text", "beta text", "gamma text"], config=cfg)
    assert service.process_next() is True
    assert runtime.requests and all(r.scope == "global" for r in runtime.requests)


def _admitting(request):
    from superlocalmemory.core.engine_ingestion import content_passes_admission

    if not content_passes_admission(request.content):
        raise AssertionError("ingestion produced no queryable facts")


def test_a_one_word_title_still_saves_its_document_memory(store, tmp_path):
    submit(store, pdf_input(("x",), file_name="report.pdf"))
    service, _, runtime = run(store, tmp_path, ["a" * 40], runtime=Runtime(on_remember=_admitting))
    service.process_next()
    anchors = [r for r in runtime.requests if "page" not in r.metadata["_slm_source"]]
    assert len(anchors) == 1 and anchors[0].content.startswith("report")
    from superlocalmemory.media.labels import DOCUMENT
    from superlocalmemory.retrieval.media_rerank import strip_labels

    assert DOCUMENT in anchors[0].content and strip_labels(anchors[0].content) == "report"
