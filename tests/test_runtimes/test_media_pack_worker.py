# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The pack models at the worker boundary (WS-8): a real worker process in fake mode, no model downloaded.

The real classes (Florence-2, PaddleOCR-VL) are not loaded here; ``test_pack_models_native.py`` does that
when the weights are on the machine.
"""

from __future__ import annotations

import json

import pytest

from superlocalmemory.runtimes import media_image_ops, media_models, pack_models
from superlocalmemory.runtimes.worker_client import MediaWorkerClient, MediaWorkerError, shared_pack_client


@pytest.fixture()
def picture(tmp_path):
    path = tmp_path / "pic.png"
    path.write_bytes(b"\x89PNG-test-bytes")
    return path


def _pack_client(stub_env, role, **kw):
    return MediaWorkerClient(stub_env, model_id="fake:8", revision="", role=role, **kw)


def test_caption_round_trip_in_its_own_worker(stub_env, picture):
    c = _pack_client(stub_env, "caption")
    try:
        text = c.caption_image(picture)
        assert text.startswith("A test picture ") and c.is_warm()
    finally:
        c.stop()


def test_a_test_can_choose_the_caption(stub_env, picture):
    (picture.parent / (picture.name + ".caption.txt")).write_text("a red bicycle by a wall", encoding="utf-8")
    c = _pack_client(stub_env, "caption")
    try:
        assert c.caption_image(picture) == "a red bicycle by a wall"
    finally:
        c.stop()


def test_a_caption_worker_refuses_document_work_and_embeddings(stub_env, picture):
    c = _pack_client(stub_env, "caption")
    try:
        with pytest.raises(MediaWorkerError):
            c.parse_document(picture)
    finally:
        c.stop()


def test_a_worker_without_the_caption_model_refuses_a_caption(stub_env, picture):
    c = MediaWorkerClient(stub_env, model_id="fake:8", revision="")  # the embedding role
    try:
        with pytest.raises(MediaWorkerError):
            c.caption_image(picture)
    finally:
        c.stop()


def test_document_page_text_and_regions(stub_env, picture):
    (picture.parent / (picture.name + ".ocr.txt")).write_text("Total due 40", encoding="utf-8")
    (picture.parent / (picture.name + ".regions.json")).write_text(
        json.dumps([{"text": "Total due 40", "bbox": [0.1, 0.1, 0.5, 0.2]}]), encoding="utf-8")
    c = _pack_client(stub_env, "document")
    try:
        reply = c.parse_document(picture)
        assert reply["text"] == "Total due 40" and reply["engine"] == "fake"
        assert reply["regions"][0]["bbox"] == [0.1, 0.1, 0.5, 0.2]
    finally:
        c.stop()


def test_ocr_regions_only_when_asked_and_the_default_reply_is_unchanged(stub_env, picture):
    (picture.parent / (picture.name + ".ocr.txt")).write_text("hello", encoding="utf-8")
    (picture.parent / (picture.name + ".regions.json")).write_text(
        json.dumps([{"text": "hello", "bbox": [0, 0, 1, 1]}]), encoding="utf-8")
    c = MediaWorkerClient(stub_env, model_id="fake:8", revision="")
    try:
        assert c.ocr_image(picture) == {"engine": "fake", "text": "hello"}     # flag off: exactly as before
        assert c.ocr_image(picture, regions=True)["regions"] == [{"text": "hello", "bbox": [0, 0, 1, 1]}]
    finally:
        c.stop()


def test_a_pack_worker_loads_only_from_a_local_folder(stub_env, tmp_path, monkeypatch, picture):
    """Real mode with no folder: a plain refusal that names no path, and the client never downloads."""
    monkeypatch.delenv("SLM_MEDIA_WORKER_FAKE", raising=False)
    c = MediaWorkerClient(stub_env, model_id="florence-community/Florence-2-base", revision="r", role="caption",
                          model_dir=tmp_path / "nothing-here")
    try:
        with pytest.raises(MediaWorkerError) as seen:
            c.caption_image(picture)
        assert "turn the pack on again" in str(seen.value) and str(tmp_path) not in str(seen.value)
    finally:
        c.stop()


def test_pack_clients_are_kept_apart_from_the_embedding_clients(stub_env, tmp_path):
    from superlocalmemory.runtimes import worker_client

    c = shared_pack_client(stub_env, "fake:8", "", "caption", tmp_path, idle_s=5.0)
    assert c is shared_pack_client(stub_env, "fake:8", "", "caption", tmp_path, idle_s=5.0)
    assert c in worker_client.live_pack_clients() and c not in worker_client.live_clients()
    assert c.idle_s == 5.0 and c.role == "caption"


def test_a_pack_worker_goes_idle_and_gives_its_memory_back(stub_env, picture):
    import time

    c = _pack_client(stub_env, "caption", idle_s=0.4)
    c.caption_image(picture)
    assert c.pid is not None
    deadline = time.time() + 15
    while c.pid is not None and time.time() < deadline:
        time.sleep(0.1)
    assert c.pid is None


# -- memory numbers ---------------------------------------------------------------

def test_pack_models_have_memory_numbers_and_the_embedding_models_are_untouched():
    assert media_models.rss_limit_mb_for("florence-community/Florence-2-base") == 4000  # measured 3,199 MB
    assert media_models.load_mb_for("PaddlePaddle/PaddleOCR-VL-1.6") == 6000
    assert media_models.rss_limit_mb_for(media_models.EG2_REPO) == 4500  # unchanged
    assert media_models.rss_limit_mb_for("fake:8") == media_models.DEFAULT_RSS_LIMIT_MB
    assert media_models.profile_for("florence-community/Florence-2-base") is None  # no evidence floor


def test_the_watchdog_limit_moves_only_while_a_pack_is_on_and_its_model_is_here(tmp_path, monkeypatch):
    from superlocalmemory.runtimes import media_packs

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", raising=False)
    monkeypatch.delenv("SLM_FEATURE_MEDIA_DOCUMENT_PACK", raising=False)
    before = media_models.watchdog_limit_mb(1600)
    assert before == 4500  # the flag is on by default, but no model is downloaded: nothing moves
    spec = media_packs.DOCUMENTS
    media_packs.model_dir(spec, tmp_path).mkdir(parents=True)
    (media_packs.pack_dir(spec, tmp_path) / media_packs.READY_FILE).write_text(
        json.dumps({"repo": spec.repo, "revision": spec.revision}), encoding="utf-8")
    assert media_models.watchdog_limit_mb(1600) == 7500  # on and installed (measured 5,948 MB)
    monkeypatch.setenv("SLM_FEATURE_MEDIA_DOCUMENT_PACK", "off")  # a person's own off puts it back
    assert media_models.watchdog_limit_mb(1600) == before
    monkeypatch.setenv("SLM_FEATURE_MEDIA_DOCUMENT_PACK", "on")
    monkeypatch.setenv("SLM_4230_OFF", "1")  # the kill switch puts it back
    assert media_models.watchdog_limit_mb(1600) == before


# -- helpers that run inside the worker ---------------------------------------------

def test_normalised_regions_are_fractions_and_bad_boxes_are_dropped():
    box = [[10, 20], [110, 20], [110, 60], [10, 60]]
    got = pack_models.normalised_regions(
        [(box, "line one"), ([[0, 0]], "broken"), (box, "  "), ("not a box", "x")], (200, 100))
    assert got == [{"text": "line one", "bbox": [0.05, 0.2, 0.55, 0.6]}]
    assert pack_models.normalised_regions([(box, "x")], (0, 0)) == []
    off = [[-50, -50], [400, -50], [400, 500], [-50, 500]]  # clamped to the picture
    assert pack_models.normalised_regions([(off, "x")], (200, 100))[0]["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_clean_text_keeps_words_only():
    assert pack_models.clean_text("a\x00 red\n\tbike  ", 100) == "a red\n bike"
    assert pack_models.clean_text("x" * 500, 10) == "x" * 10


def test_ocr_ops_report_boxes_only_when_given_a_list(monkeypatch):
    class _Result:
        txts = ("one", "two")
        boxes = [[[0, 0], [1, 0], [1, 1], [0, 1]], [[2, 2], [3, 2], [3, 3], [2, 3]]]

    class _Module:
        __file__ = "/x/rapidocr/__init__.py"

        @staticmethod
        def RapidOCR(params):
            return lambda path: _Result

    monkeypatch.setattr(media_image_ops.importlib, "import_module", lambda name: _Module)
    monkeypatch.setattr(media_image_ops, "_local_ocr_params", lambda module: {"x": "y"})
    found: list = []
    assert media_image_ops._rapidocr("p", found) == "one\ntwo" and len(found) == 2
    assert media_image_ops._rapidocr("p") == "one\ntwo"
    reply = media_image_ops.ocr_image("p", engine="rapidocr", found=found)
    assert reply == {"engine": "rapidocr", "text": "one\ntwo"}


# -- how the native classes are called (fakes stand in for transformers and torch) ----------

class _Batch(dict):
    def to(self, device):
        self["device"] = device
        return self


class _Img:
    closed = False

    def close(self):
        self.closed = True


class _FakeModel:
    device = "cpu"

    def __init__(self, reply):
        self.reply, self.kwargs, self.moved = reply, None, None

    def to(self, device):
        self.moved = device
        return self

    def eval(self):
        return self

    def generate(self, **kw):
        self.kwargs = kw
        return self.reply


def _fake_libs(monkeypatch, record):
    class _Processor:
        def __init__(self):
            self.template_args = None

        def __call__(self, text=None, images=None, return_tensors=None):
            record["processor_call"] = (text, return_tensors)
            return _Batch(input_ids=[[1, 2]])

        def batch_decode(self, ids, **kw):
            record["decode"] = kw
            return ["  a red\x00 bicycle \n by a wall  "]

        def apply_chat_template(self, messages, **kw):
            record["template"] = (messages, kw)
            return _Batch(input_ids=[[1, 2]])

    class _Auto:
        @staticmethod
        def from_pretrained(path, **kw):
            record["processor_load"] = (path, kw)
            return _Processor()

    def loader(kind, reply):
        class _Cls:
            @staticmethod
            def from_pretrained(path, **kw):
                record[kind] = (path, kw)
                return _FakeModel(reply)
        return _Cls

    transformers = type("T", (), {
        "AutoProcessor": _Auto,
        "Florence2ForConditionalGeneration": loader("florence", [[7, 8, 9]]),
        "PaddleOCRVLForConditionalGeneration": loader("paddle", [[1, 2, 5, 6]]),
    })
    import contextlib

    torch = type("Torch", (), {"float32": "f32", "no_grad": staticmethod(contextlib.nullcontext)})
    monkeypatch.setattr(pack_models.importlib, "import_module",
                        lambda name: {"torch": torch, "transformers": transformers}[name])


def test_florence_is_loaded_by_its_native_class_from_a_local_folder_without_remote_code(monkeypatch, picture):
    seen: dict = {}
    _fake_libs(monkeypatch, seen)
    state = pack_models.load_caption("/models/f2", "cpu")
    path, kwargs = seen["florence"]
    assert path == "/models/f2" and kwargs == {"local_files_only": True, "dtype": "f32"}
    assert "trust_remote_code" not in kwargs and "trust_remote_code" not in seen["processor_load"][1]
    assert seen["processor_load"][1] == {"local_files_only": True} and state[0].moved == "cpu"


def test_a_caption_asks_for_the_caption_task_and_comes_back_as_plain_words(monkeypatch, tmp_path):
    seen: dict = {}
    _fake_libs(monkeypatch, seen)
    monkeypatch.setattr(pack_models, "_open_rgb", lambda path: _Img())
    state = pack_models.load_caption("/models/f2", "cpu")
    assert pack_models.caption(state, str(tmp_path / "p.png")) == "a red bicycle \n by a wall"
    assert seen["processor_call"] == ("<CAPTION>", "pt")
    assert state[0].kwargs["max_new_tokens"] == pack_models.CAPTION_MAX_NEW_TOKENS
    assert state[0].kwargs["do_sample"] is False and seen["decode"] == {"skip_special_tokens": True}


def test_paddleocr_vl_is_loaded_by_its_native_class_and_a_page_is_one_region(monkeypatch, tmp_path):
    seen: dict = {}
    _fake_libs(monkeypatch, seen)
    monkeypatch.setattr(pack_models, "_open_rgb", lambda path: _Img())
    state = pack_models.load_document("/models/pd", "cpu")
    assert seen["paddle"][0] == "/models/pd" and seen["paddle"][1] == {"local_files_only": True, "dtype": "f32"}
    out = pack_models.parse_document(state, str(tmp_path / "p.png"))
    messages, kw = seen["template"]
    assert messages[0]["content"][1] == {"type": "text", "text": "OCR:"} and kw["add_generation_prompt"] is True
    assert out["engine"] == "paddleocr_vl" and out["regions"] == [
        {"text": out["text"][:400], "bbox": [0.0, 0.0, 1.0, 1.0], "label": "page"}]
    assert pack_models.parse_document.__doc__ and "page itself" in pack_models.parse_document.__doc__


def test_page_reading_gets_room_and_a_caption_keeps_the_short_deadline(tmp_path):
    """Measured 19 Nov: one A4 page through PaddleOCR-VL took ~160 s on a busy CPU, past the 120 s
    every worker had; page reading runs in background jobs, so its pack gets more room."""
    from types import SimpleNamespace

    from superlocalmemory.runtimes import media_packs, worker_client

    assert media_packs.PACK_REQUEST_TIMEOUT_S["documents"] >= 300
    assert media_packs.PACK_REQUEST_TIMEOUT_S["captions"] == 120.0
    managed = SimpleNamespace(root=tmp_path / "m")
    client = worker_client.shared_pack_client(managed, "x/doc", "r1", "document", tmp_path, 60.0, 600.0)
    try:
        assert client.request_timeout_s == 600.0
    finally:
        worker_client._PACK_CLIENTS.clear()
