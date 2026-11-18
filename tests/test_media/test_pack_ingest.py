# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The multimodal pack inside saving a picture (WS-8, D-19): captions, fail-open, regions, flags off = today.

Fake models only: a fake pack client stands at the worker boundary; no model is downloaded.
"""

from __future__ import annotations

import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from superlocalmemory.media import captions, ingest, open_media_store, regions
from superlocalmemory.media.labels import CAPTION
from superlocalmemory.runtimes import media_packs
from superlocalmemory.runtimes.worker_client import MediaWorkerError, MediaWorkerWarming
from superlocalmemory.trust import origin_class as oc
from superlocalmemory.trust import why

from .test_ingest import FakeClient, Runtime, SpyCache, png, save

KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
BOXES = [{"text": "Invoice 42 total", "bbox": [0.05, 0.05, 0.45, 0.12]},
         {"text": "Thank you for shopping", "bbox": [0.3, 0.9, 0.7, 0.96]}]


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    for name in ("SLM_FEATURE_MEDIA_CAPTIONS", "SLM_FEATURE_MEDIA_REGIONS", "SLM_FEATURE_MEDIA_DOCUMENT_PACK",
                 "SLM_4230_OFF"):
        monkeypatch.delenv(name, raising=False)
    return r


@pytest.fixture()
def store(root):
    s = open_media_store(create=True, data_root=root)
    yield s
    s.close()


@pytest.fixture()
def env(root, store):
    return SimpleNamespace(client=FakeClient(), runtime=Runtime(), cache=SpyCache(), store=store,
                           config=SimpleNamespace(pii_redaction=False), root=root)


class PackClient:
    """What the caption worker's client does, at the boundary."""

    def __init__(self, text="a red bicycle leaning on a wall", exc: Exception | None = None):
        self.text, self.exc, self.calls = text, exc, []

    def caption_image(self, path, *, wait_cold=True):
        self.calls.append((str(path), wait_cold))
        if self.exc:
            raise self.exc
        return self.text


class RegionClient(FakeClient):
    """The picture client when it is asked for boxes (``regions=True``); plain otherwise."""

    def __init__(self, boxes=BOXES, **kw):
        super().__init__(**kw)
        self.boxes, self.regions_asked = boxes, []

    def ocr_image(self, path, *, wait_cold=True, regions=False):
        self.regions_asked.append(regions)
        reply = super().ocr_image(path, wait_cold=wait_cold)
        return {**reply, "regions": list(self.boxes)} if regions else reply


@pytest.fixture()
def packs_on(monkeypatch):
    """Both flags on and a model 'installed'; the caption client is whatever the test sets on the holder."""
    holder = SimpleNamespace(client=PackClient(), asked=[])
    monkeypatch.setenv("SLM_FEATURE_MEDIA_CAPTIONS", "on")
    monkeypatch.setattr(media_packs, "installed", lambda spec, data_root=None: True)

    def fake_pack_client(pack_id, data_root=None, *, env=None):
        holder.asked.append(pack_id)
        return holder.client

    monkeypatch.setattr(media_packs, "pack_client", fake_pack_client)
    return holder


# -- flags off = today, byte for byte ---------------------------------------------------

def test_everything_off_the_admission_and_the_receipt_are_what_they_were(env, monkeypatch):
    asked = []
    monkeypatch.setattr(media_packs, "pack_client", lambda *a, **k: asked.append(a) or PackClient())
    env.client.ocr = ("rapidocr", "Invoice 42 total")
    r = save(env, content="receipt from the cafe")
    assert env.runtime.requests[0].content == "receipt from the cafe\n\n[Text in image]\nInvoice 42 total"
    assert asked == []                                              # no pack worker was even asked for
    assert "description" not in ingest.receipt_dict(r)
    assert set(ingest.receipt_dict(r)) == {"status", "media_id", "memory_id", "reason", "duplicate_of",
                                           "near_duplicate_of", "extracted_text_preview"}


def test_ocr_cache_key_and_payload_are_unchanged_with_the_part_off(env):
    from superlocalmemory.cache.keys import CacheKey, params_hash

    sha = "ab" * 32
    assert ingest._ocr_key(sha, True) == CacheKey(
        sha, "ocr.auto", "2", params_hash=params_hash({"redaction": True, "lang": "auto"}))
    assert ingest._ocr_key(sha, True, True) != ingest._ocr_key(sha, True)
    out = ingest._Ocr("rapidocr", "x", 0, 0, scanned=True)
    assert b"regions" not in ingest._cache_payload(out)             # same bytes as before the pack
    assert b'"regions"' in ingest._cache_payload(ingest._Ocr("rapidocr", "x", regions=[{"bbox": [0, 0, 1, 1]}]))


def test_the_ocr_call_is_the_same_call_with_regions_off(env):
    env.client = RegionClient()
    save(env)
    assert env.client.regions_asked == [False]


# -- captions -------------------------------------------------------------------------

def test_a_caption_joins_the_memory_under_its_own_label_and_the_receipt(env, packs_on):
    env.client.ocr = ("rapidocr", "Invoice 42 total")
    r = save(env, content="receipt from the cafe")
    content = env.runtime.requests[0].content
    assert content == ("receipt from the cafe\n\n[Text in image]\nInvoice 42 total\n\n"
                       "[Description written by a model]\na red bicycle leaning on a wall")
    assert CAPTION.strip() in content and r.description == "a red bicycle leaning on a wall"
    assert ingest.receipt_dict(r)["description"] == "a red bicycle leaning on a wall"
    assert packs_on.asked == ["captions"]
    assert packs_on.client.calls[0][1] is True        # a person on this computer waits out a cold start


def test_a_caption_is_a_models_words_so_its_origin_is_model_generated(env, packs_on):
    """Every picture memory is saved through the ``media`` channel, whose origin row is ``model_generated`` (D-19)."""
    save(env)
    adm = env.runtime.requests[0]
    assert adm.source_type == "media"
    classified = oc.classify(oc.Signals(channel="media", transform="verbatim", source_type=adm.source_type))
    assert classified.origin_class == oc.MODEL_GENERATED and classified.basis.endswith("media.model")


def test_a_caption_alone_is_enough_words_for_a_picture_without_text(env, packs_on):
    save(env)
    assert env.runtime.requests[0].content == ("[Image without text]\n\n"
                                               "[Description written by a model]\na red bicycle leaning on a wall")


def test_a_remote_caller_does_not_wait_for_a_cold_caption_model(env, packs_on):
    save(env, inp=ingest.MediaInput(base64=__import__("base64").b64encode(png()).decode(), remote=True))
    assert packs_on.client.calls[0][1] is False


@pytest.mark.parametrize("failure", [MediaWorkerError("the worker stopped"), MediaWorkerWarming("starting"),
                                     RuntimeError("boom"), MemoryError()])
def test_a_caption_failure_still_saves_the_picture_exactly_as_without_it(env, packs_on, failure, caplog):
    packs_on.client = PackClient(exc=failure)
    env.client.ocr = ("rapidocr", "Invoice 42 total")
    with caplog.at_level("WARNING"):
        r = save(env, content="receipt")
    assert r.status == "stored" and r.description == ""
    assert env.runtime.requests[0].content == "receipt\n\n[Text in image]\nInvoice 42 total"
    assert any("picture description skipped" in rec.message for rec in caplog.records)   # the reason is logged
    assert "boom" not in caplog.text and "Invoice" not in caplog.text            # the kind only, never content


def test_no_pack_client_means_no_caption_and_no_error(env, packs_on):
    packs_on.client = None
    r = save(env)
    assert r.status == "stored" and env.runtime.requests[0].content == "[Image without text]"


def test_a_caption_goes_through_the_same_redaction_as_text_read_from_a_picture(env, packs_on):
    packs_on.client = PackClient(text=f"a sign that says {KEY}")
    env.config.pii_redaction = True
    save(env)
    content = env.runtime.requests[0].content
    assert KEY not in content and "[Description written by a model]" in content


def test_an_empty_or_messy_caption_is_dropped_or_tidied(env, packs_on):
    packs_on.client = PackClient(text="  \n ")
    save(env)
    assert CAPTION.strip() not in env.runtime.requests[0].content
    assert captions.tidy("a\x00b \n  c\x1f") == "ab c"
    assert len(captions.tidy("x" * 900)) == captions.MAX_CAPTION_CHARS


def test_the_kill_switch_stops_captions_at_once(env, packs_on, monkeypatch):
    monkeypatch.setenv("SLM_4230_OFF", "1")
    save(env)
    assert packs_on.asked == [] and env.runtime.requests[0].content == "[Image without text]"


def test_describe_is_empty_when_the_part_is_off_whatever_the_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    assert captions.describe(tmp_path, tmp_path / "x.png") == ""
    assert captions.describe(tmp_path, tmp_path / "x.png", client=PackClient()) == "a red bicycle leaning on a wall"


# -- regions ---------------------------------------------------------------------------

@pytest.fixture()
def regions_on(monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_MEDIA_REGIONS", "on")


def test_the_boxes_of_the_text_are_kept_with_the_picture(env, regions_on):
    env.client = RegionClient(ocr=("rapidocr", "Invoice 42 total\nThank you for shopping"))
    r = save(env)
    assert env.client.regions_asked == [True]
    rows = env.store._read().execute("SELECT * FROM media_regions WHERE media_id = ?", (r.media_id,)).fetchall()
    assert len(rows) == 2 and {x["page_no"] for x in rows} == {0}
    assert {x["engine"] for x in rows} == {"rapidocr"} and rows[0]["profile_id"] == "p1"
    assert {x["snippet"] for x in rows} == {"Invoice 42 total", "Thank you for shopping"}
    assert env.store.get_item(r.media_id)["anchor_memory_id"] == "mem1"
    assert env.runtime.requests[0].content == "[Text in image]\nInvoice 42 total\nThank you for shopping"


def test_a_snippet_never_holds_more_than_the_memory_does(env, regions_on):
    env.config.pii_redaction = True
    env.client = RegionClient(boxes=[{"text": f"key {KEY}", "bbox": [0, 0, 0.5, 0.1]}], ocr=("rapidocr", f"key {KEY}"))
    r = save(env)
    row = env.store._read().execute("SELECT snippet FROM media_regions WHERE media_id = ?", (r.media_id,)).fetchone()
    assert KEY not in row[0]


def test_malformed_boxes_are_dropped_and_never_fail_the_save(env, regions_on):
    bad = [{"text": "x", "bbox": [0.5, 0.5, 0.4, 0.6]}, {"text": "y", "bbox": "no"}, "junk",
           {"text": "z", "bbox": [0, 0, float("nan"), 1]}, {"text": "ok", "bbox": [0.1, 0.1, 0.2, 0.2]}]
    env.client = RegionClient(boxes=bad, ocr=("rapidocr", "ok"))
    r = save(env)
    n = env.store._read().execute("SELECT COUNT(*) FROM media_regions WHERE media_id = ?", (r.media_id,)).fetchone()[0]
    assert r.status == "stored" and n == 1


def test_a_failure_while_keeping_boxes_never_blocks_the_save(env, regions_on, monkeypatch):
    env.client = RegionClient(ocr=("rapidocr", "Invoice 42 total"))
    monkeypatch.setattr(type(env.store), "put_regions", lambda *a, **k: (_ for _ in ()).throw(sqlite3.Error("x")))
    r = save(env)
    assert r.status == "stored" and r.media_id


def test_erasing_the_picture_erases_its_regions_and_a_profile_move_carries_them(env, regions_on):
    from superlocalmemory.media.schema import PROFILE_TABLES

    env.client = RegionClient(ocr=("rapidocr", "Invoice 42 total"))
    r = save(env)
    assert "media_regions" in PROFILE_TABLES
    env.store.erase_items([r.media_id])
    assert env.store._read().execute("SELECT COUNT(*) FROM media_regions").fetchone()[0] == 0


def test_the_kill_switch_stops_regions(env, regions_on, monkeypatch):
    monkeypatch.setenv("SLM_4230_OFF", "1")
    env.client = RegionClient(ocr=("rapidocr", "Invoice 42 total"))
    r = save(env)
    assert env.client.regions_asked == [False]
    assert env.store._read().execute("SELECT COUNT(*) FROM media_regions").fetchone()[0] == 0 and r.media_id


# -- region provenance in the answer to "why" ---------------------------------------------

def _memory_db(root, fact="Invoice 42 total"):
    """A memory.db next to media.db with one fact whose memory is the picture's anchor."""
    conn = sqlite3.connect(root / "memory.db")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT, profile_id TEXT, content TEXT, created_at TEXT, "
                 "memory_id TEXT)")
    conn.execute("INSERT INTO atomic_facts VALUES ('f1', 'p1', ?, '2026-11-18', 'mem1')", (fact,))
    conn.commit()
    return conn


def test_why_says_where_on_the_picture_the_fact_was_read(env, regions_on):
    env.client = RegionClient(ocr=("rapidocr", "Invoice 42 total\nThank you for shopping"))
    save(env)
    conn = _memory_db(env.root)
    placed = why._regions(conn, "mem1", "Invoice 42 total")
    assert [p["where"] for p in placed] == ["from top left of the picture"]
    again = why._regions(conn, "mem1", "Thank you for shopping")
    assert [p["where"] for p in again] == ["from bottom of the picture"]
    assert why._regions(conn, "mem1", "something unrelated about bicycles") == []


def test_why_has_no_regions_key_when_the_part_is_off(env):
    env.client = RegionClient(ocr=("rapidocr", "Invoice 42 total"))
    save(env)
    assert why._regions(_memory_db(env.root), "mem1", "Invoice 42 total") is None


def test_why_lines_show_the_places(env):
    out = why.lines({"fact_id": "f1", "created_at": "x", "preview": "p",
                     "origin": {"words": "w", "parents": []}, "firewall": {"words": "fw", "approvals": []},
                     "signature": {"words": "s"}, "truth": {"slots": [], "disputes": []},
                     "compiler": {"versions": [], "proposals": [], "pages": []},
                     "deliveries": {"shown": 0, "considered": 0, "recent": []},
                     "source_regions": [{"where": "from page 3, top left"}]})
    assert "Where the text was read" in out and "  from page 3, top left" in out


# -- the words for a place -----------------------------------------------------------------

@pytest.mark.parametrize("page,box,words", [
    (3, [0.05, 0.05, 0.3, 0.1], "page 3, top left"),
    (3, [0.4, 0.05, 0.6, 0.1], "page 3, top"),
    (3, [0.3, 0.05, 0.5, 0.1], "page 3, top"),       # centre at 0.4: not yet "left"
    (3, [0.1, 0.4, 0.3, 0.5], "page 3, left"),
    (1, [0.7, 0.4, 0.95, 0.5], "page 1, right"),
    (2, [0.4, 0.4, 0.6, 0.6], "page 2, middle"),
    (2, [0.7, 0.8, 0.95, 0.95], "page 2, bottom right"),
    (4, [0.0, 0.0, 1.0, 1.0], "page 4"),
    (0, [0.0, 0.0, 1.0, 1.0], "the whole picture"),
    (0, [0.05, 0.8, 0.3, 0.95], "bottom left of the picture"),
])
def test_place_words(page, box, words):
    assert regions.place_words(page, box) == words


def test_a_fact_that_quotes_no_region_shows_none():
    rows = [{"page_no": 1, "x0": 0, "y0": 0, "x1": 0.5, "y1": 0.1, "label": "", "snippet": "Invoice 42 total"}]
    assert regions.match_fact(rows, "the weather today") == []
    assert regions.match_fact(rows, "the invoice 42 total is due")[0]["page"] == 1
    assert len(regions.match_fact(rows * 9, "invoice 42 total")) == 3


def test_no_media_database_no_regions(tmp_path):
    conn = sqlite3.connect(tmp_path / "memory.db")
    assert regions.for_fact(conn, "mem1", "text") == [] and regions.media_db_next_to(conn) is None


def test_the_lookup_is_read_only(env, regions_on):
    env.client = RegionClient(ocr=("rapidocr", "Invoice 42 total"))
    save(env)
    before = hashlib.sha256((env.root / "media.db").read_bytes()).hexdigest()
    regions.for_fact(_memory_db(env.root), "mem1", "Invoice 42 total")
    assert hashlib.sha256((env.root / "media.db").read_bytes()).hexdigest() == before


def test_describe_redacts_the_description_itself_before_returning_it(tmp_path):
    out = captions.describe(tmp_path, tmp_path / "x.png", client=PackClient(text="a card 4111 on a desk"),
                            redact=lambda t: t.replace("4111", "[REDACTED]"))
    assert out == "a card [REDACTED] on a desk"
    assert captions.describe(tmp_path, tmp_path / "x.png", client=PackClient(text="x"),
                             redact=lambda t: (_ for _ in ()).throw(ValueError("no"))) == ""
