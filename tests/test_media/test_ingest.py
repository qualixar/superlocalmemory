"""Saving an image: strip, store, read its text, keep one memory that points at it."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.media import ingest, open_media_store
from superlocalmemory.media.ingest import MediaInput, remember_media
from superlocalmemory.runtimes.worker_client import MediaWorkerWarming

PNG = b"\x89PNG\r\n\x1a\n"
KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
DIM = 8


def png(tag: str = "a", size: int = 64) -> bytes:
    return PNG + (tag.encode() * size)[:size]


class FakeClient:
    model_id, revision, dim = "fake:test", "r1", DIM

    def __init__(self, ocr=("rapidocr", ""), phash=None, warm=True):
        self.ocr, self.phash, self.warm = ocr, phash, warm
        self.calls: list[str] = []
        self.prepare_exc: Exception | None = None

    def is_warm(self):
        return self.warm

    def warm_up(self):
        self.calls.append("warm_up")

    def prepare_image(self, path, out_dir, *, wait_cold=True):
        self.calls.append("prepare")
        if self.prepare_exc:
            raise self.prepare_exc
        src = Path(path).read_bytes()
        out = Path(out_dir)
        stored = out / "stored.png"
        stored.write_bytes(b"STRIPPED" + src)
        thumb = out / "thumb.webp"
        thumb.write_bytes(b"RIFFthumb")
        ph = self.phash(src) if self.phash else hashlib.sha256(src).hexdigest()[:16]
        return {"mime": "image/png", "width": 3, "height": 2, "exif": {"Make": "Cam"},
                "phash": ph, "stored_path": str(stored), "stored_ext": "png", "thumb_path": str(thumb)}

    def ocr_image(self, path, *, wait_cold=True):
        self.calls.append("ocr")
        engine, text = self.ocr
        return {"engine": engine, "text": text}

    def embed_images(self, paths, *, wait_cold=True):
        self.calls.append("embed")
        return [[1.0] + [0.0] * (DIM - 1) for _ in paths]


class Runtime:
    def __init__(self, fail=False):
        self.requests, self.fail = [], fail

    def remember(self, admission, actor, *, deadline_ms, accept_after_ms):
        if self.fail:
            raise RuntimeError("writer down")
        self.requests.append(admission)
        n = len(self.requests)
        return SimpleNamespace(payload={"status": "queryable", "operation_id": f"op{n}",
                                        "fact_ids": [f"f{n}"], "memory_id": f"mem{n}"})


class SpyCache:
    def __init__(self):
        self.data, self.gets, self.puts = {}, [], []

    def get(self, key):
        self.gets.append(key)
        return self.data.get(key.as_tuple())

    def put(self, key, payload, *, kind):
        self.puts.append((key, kind))
        self.data[key.as_tuple()] = bytes(payload)


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


@pytest.fixture()
def env(root, store):
    return SimpleNamespace(client=FakeClient(), runtime=Runtime(), cache=SpyCache(), store=store,
                           config=SimpleNamespace(pii_redaction=False), root=root)


def save(env, data=None, *, content="", **kw):
    inp = kw.pop("inp", None) or MediaInput(base64=base64.b64encode(data if data is not None else png()).decode())
    args = dict(content=content, profile_id="p1", actor_id="actor-1", runtime=env.runtime, config=env.config,
                client=env.client, store=env.store, cache=env.cache)
    args.update(kw)
    return remember_media(inp, **args)


def test_happy_path(env):
    env.client.ocr = ("rapidocr", "Invoice 42 total")
    r = save(env, content="receipt from the cafe")
    assert r.status == "stored" and r.memory_id == "mem1" and r.media_id
    row = env.store.get_item(r.media_id)
    assert row["source_sha256"] == hashlib.sha256(png()).hexdigest()
    assert row["stored_sha256"] == hashlib.sha256(b"STRIPPED" + png()).hexdigest()
    assert row["phash"] and row["anchor_memory_id"] == "mem1" and row["thumb_webp"] == b"RIFFthumb"
    assert row["origin"] == "tool" and row["kind"] == "image" and row["state"] == "active"
    assert row["width"] == 3 and '"Make"' in row["exif_json"]
    orig = env.root / "media" / row["original_relpath"]
    assert orig.read_bytes() == b"STRIPPED" + png() and stat.S_IMODE(orig.stat().st_mode) == 0o600
    assert row["original_relpath"] == f"{row['stored_sha256'][:2]}/{row['stored_sha256']}.png"
    adm = env.runtime.requests[0]
    assert adm.content == "receipt from the cafe\n\n[Text in image]\nInvoice 42 total"
    assert adm.source_type == "media" and adm.profile_id == "p1"
    assert adm.metadata["_slm_source"] == {"type": "media", "media_id": r.media_id, "origin": "tool"}
    assert r.extracted_text_preview == "Invoice 42 total"
    assert env.store.knn([1.0] + [0.0] * (DIM - 1), "p1", 1)[0][0] == r.media_id
    assert not list((env.root / "media" / "tmp").iterdir())


def test_image_without_words_or_text_still_gets_an_anchor(env):
    r = save(env)
    assert r.status == "stored" and env.runtime.requests[0].content.strip()


def test_words_survive_verbatim_but_a_key_in_the_image_text_is_redacted(env):
    env.client.ocr = ("rapidocr", f"sticky note {KEY}")
    save(env, content=f"my own key is {KEY}")
    text = env.runtime.requests[0].content
    head, tail = text.split("[Text in image]")
    assert KEY in head and KEY not in tail and "[REDACTED:" in tail


def test_pii_on_redacts_both_parts(env):
    env.config = SimpleNamespace(pii_redaction=True)
    env.client.ocr = ("rapidocr", "write to amy@example.org")
    save(env, content="mail bob@example.com")
    assert "@" not in env.runtime.requests[0].content


def test_exact_duplicate_is_merged_near_duplicate_is_reported(env):
    first = save(env)
    again = save(env)
    assert again.status == "duplicate" and again.duplicate_of == first.media_id and again.media_id == first.media_id
    assert len(env.runtime.requests) == 1 and len(env.store.list_items("p1")) == 1
    env.client.phash = lambda b: "0000000000000001"
    base = save(env, png("b"))
    env.client.phash = lambda b: "0000000000000003"
    near = save(env, png("c"))
    assert near.status == "stored" and near.near_duplicate_of == base.media_id
    env.client.phash = lambda b: "ffffffffffffffff"
    assert save(env, png("d")).near_duplicate_of is None
    assert save(env, png(), profile_id="p2").status == "stored"


def test_feature_off_refuses_and_creates_nothing(root):
    r = remember_media(MediaInput(base64=base64.b64encode(png()).decode()), profile_id="p1", actor_id="a",
                       runtime=Runtime(), config=SimpleNamespace(pii_redaction=False))
    assert r.status == "refused" and "turned off" in r.reason
    assert not (root / "media.db").exists() and not (root / "media").exists()


@pytest.mark.parametrize("data,reason", [(b"GIF99a" + b"x" * 20, "supported"), (b"", "empty"),
                                         (b"%PDF-1.4" + b"x" * 30, "supported")])
def test_bad_content_is_refused(env, data, reason):
    r = save(env, data)
    assert r.status == "refused" and reason in r.reason
    assert env.runtime.requests == [] and env.client.calls == []


def test_bad_base64_and_oversize_and_missing_path(env, monkeypatch, tmp_path):
    assert save(env, inp=MediaInput(base64="not base64 !!")).status == "refused"
    assert save(env, inp=MediaInput()).status == "refused"
    assert save(env, inp=MediaInput(path=tmp_path / "missing.png")).status == "refused"
    assert save(env, inp=MediaInput(path=tmp_path)).status == "refused"
    monkeypatch.setattr(ingest, "MAX_BASE64_BYTES", 10)
    assert "too large" in save(env).reason
    monkeypatch.setattr(ingest, "MAX_FILE_BYTES", 10)
    big = tmp_path / "big.png"
    big.write_bytes(png())
    assert "too large" in save(env, inp=MediaInput(path=big)).reason
    assert env.client.calls == []


def test_a_file_path_works_and_is_never_logged(env, tmp_path, caplog):
    p = tmp_path / "secret-name-photo.png"
    p.write_bytes(png())
    with caplog.at_level(logging.DEBUG):
        r = save(env, inp=MediaInput(path=p), content="words")
    assert r.status == "stored"
    assert "secret-name-photo" not in caplog.text and "words" not in caplog.text


def test_quota_refuses(env, monkeypatch):
    save(env)
    monkeypatch.setattr(ingest, "QUOTA_BYTES", 100)
    r = save(env, png("z", 70))
    assert r.status == "refused" and "full" in r.reason and env.client.calls.count("prepare") == 1


def test_cold_worker_stores_nothing_and_leaves_no_temp_files(env):
    env.client.prepare_exc = MediaWorkerWarming("starting")
    r = save(env)
    assert r.status == "warming" and "try again" in r.reason
    assert env.runtime.requests == [] and env.store.list_items("p1") == []
    assert not (env.root / "media" / "tmp").exists() or not list((env.root / "media" / "tmp").iterdir())


def test_worker_that_never_warms_gives_a_warming_receipt(env, monkeypatch):
    monkeypatch.setenv("SLM_MEDIA_COLD_WAIT_S", "1")
    env.client.warm = False
    r = save(env)
    assert r.status == "warming" and "prepare" not in env.client.calls


def test_save_failure_leaves_no_row_and_no_original(env):
    env.runtime.fail = True
    r = save(env)
    assert r.status == "refused" and env.store.list_items("p1") == []
    media = env.root / "media"
    assert [p for p in media.rglob("*") if p.is_file()] == []


def test_row_failure_keeps_the_memory_and_the_file_and_logs_ids_only(env, monkeypatch, caplog):
    def boom(**kw):
        raise RuntimeError("db gone")

    monkeypatch.setattr(env.store, "insert_item", boom)
    env.client.ocr = ("rapidocr", "private words in the picture")
    with caplog.at_level(logging.WARNING):
        r = save(env, content="my private note")
    assert r.status == "stored" and r.memory_id == "mem1" and r.media_id is None
    assert len(env.runtime.requests) == 1
    assert [p for p in (env.root / "media").rglob("*.png")]
    assert "private" not in caplog.text and "mem1" in caplog.text


def test_remote_ok_only_for_clean_real_ocr(env):
    env.client.ocr = ("none", "")
    assert env.store.get_item(save(env, png("n")).media_id)["remote_ok"] == 0
    env.client.ocr = ("rapidocr", f"has {KEY}")
    assert env.store.get_item(save(env, png("s")).media_id)["remote_ok"] == 0
    env.client.ocr = ("rapidocr", "just a menu")
    assert env.store.get_item(save(env, png("c")).media_id)["remote_ok"] == 1
    env.config = SimpleNamespace(pii_redaction=True)
    env.client.ocr = ("rapidocr", "call amy@example.org")
    assert env.store.get_item(save(env, png("p")).media_id)["remote_ok"] == 0


def test_ocr_result_is_cached_by_stored_hash_and_redaction_flag(env):
    env.client.ocr = ("rapidocr", f"text {KEY}")
    a = save(env, png("a"))
    assert env.client.calls.count("ocr") == 1 and len(env.cache.puts) == 1
    key = env.cache.puts[0][0]
    stored_sha = env.store.get_item(a.media_id)["stored_sha256"]
    assert key.content_sha256 == stored_sha and key.deriver_id.startswith("ocr.")
    # same stored bytes again (another profile): no second OCR, same redaction outcome
    b = save(env, png("a"), profile_id="p2")
    assert env.client.calls.count("ocr") == 1 and b.status == "stored"
    assert "[REDACTED:" in env.runtime.requests[1].content and KEY not in env.runtime.requests[1].content
    assert env.store.get_item(b.media_id)["remote_ok"] == 0
    env.config = SimpleNamespace(pii_redaction=True)
    save(env, png("a"), profile_id="p3")
    assert env.client.calls.count("ocr") == 2
    assert env.cache.puts[1][0].params_hash != key.params_hash


def test_a_missing_ocr_engine_is_not_cached(env):
    env.client.ocr = ("none", "")
    save(env)
    assert env.cache.puts == []


def test_a_file_that_grows_after_the_size_check_is_still_refused(env, monkeypatch, tmp_path):
    big = tmp_path / "grows.png"
    big.write_bytes(png())
    monkeypatch.setattr(ingest, "MAX_FILE_BYTES", 10)
    real = Path.stat

    def small(self, *a, **k):
        r = real(self, *a, **k)
        return os.stat_result((r.st_mode, r.st_ino, r.st_dev, r.st_nlink, r.st_uid, r.st_gid, 5,
                               int(r.st_atime), int(r.st_mtime), int(r.st_ctime))) if self == big else r

    monkeypatch.setattr(Path, "stat", small)
    r = save(env, inp=MediaInput(path=big))
    assert r.status == "refused" and "too large" in r.reason and env.client.calls == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX fifos")
def test_a_fifo_is_refused_without_blocking(env, tmp_path, monkeypatch):
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    monkeypatch.setattr(Path, "is_file", lambda self: True)  # as if swapped in after the check
    fd = os.open(fifo, os.O_RDWR)  # keeps open() from blocking if the code under test does reach it
    try:
        r = save(env, inp=MediaInput(path=fifo))
    finally:
        os.close(fd)
    assert r.status == "refused" and env.client.calls == []


def test_a_download_link_goes_through_the_same_checks(env, monkeypatch):
    from superlocalmemory.core import media_fetch

    seen = {}

    def fake_fetch(link, **kw):
        seen.update(link=link, **kw)
        return media_fetch.FetchedMedia(png(), "https://img.example.com/a.png", "text/html")

    monkeypatch.setattr(media_fetch, "fetch_media", fake_fetch)
    r = save(env, inp=MediaInput(download_url="https://img.example.com/a.png?k=1"))
    assert r.status == "stored" and seen["remote"] is False
    monkeypatch.setattr(media_fetch, "fetch_media",
                        lambda link, **kw: media_fetch.FetchedMedia(b"<html>nope</html>", "u", "image/png"))
    assert "not supported" in save(env, inp=MediaInput(download_url="https://img.example.com/b")).reason


def test_a_refused_download_is_a_refused_receipt_without_the_link(env, monkeypatch):
    from superlocalmemory.core import media_fetch

    def boom(link, **kw):
        raise media_fetch.MediaFetchRefused("That link points to a private or reserved network address.")

    monkeypatch.setattr(media_fetch, "fetch_media", boom)
    r = save(env, inp=MediaInput(download_url="https://10.0.0.1/a.png?k=SECRET"))
    assert r.status == "refused" and "private" in r.reason and "SECRET" not in r.reason
    assert env.runtime.requests == []


# -- off is not the same as not ready ------------------------------------------

def _turn_on(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "features.json").write_text('{"schema": 1, "media": {"enabled": true}}', encoding="utf-8")


def _fake_env(monkeypatch, state):
    import importlib

    module = importlib.import_module("superlocalmemory.runtimes.media_env")

    fake = SimpleNamespace(root=Path("."), status=lambda: SimpleNamespace(state=state, step="x"))
    monkeypatch.setattr(module, "media_env", lambda root=None: fake)


def _bare(inp):
    return remember_media(inp, profile_id="p1", actor_id="a", runtime=Runtime(),
                          config=SimpleNamespace(pii_redaction=False))


@pytest.mark.parametrize("state,words", [("unsupported", "can't be set up"), ("installing", "still being set up"),
                                         ("not_installed", "still being set up"), ("failed", "did not finish")])
def test_on_but_not_ready_says_so_not_turned_off(root, monkeypatch, state, words):
    _turn_on(root)
    _fake_env(monkeypatch, state)
    r = _bare(MediaInput(base64=base64.b64encode(png()).decode()))
    assert r.status == "refused" and words in r.reason and "turned off" not in r.reason


def test_off_still_says_turned_off(root):
    r = _bare(MediaInput(base64=base64.b64encode(png()).decode()))
    assert "turned off" in r.reason


def test_bad_input_is_reported_as_bad_input_while_off(root, tmp_path):
    assert "could not be found" in _bare(MediaInput(path=tmp_path / "nope.png")).reason
    assert "not supported" in _bare(MediaInput(base64=base64.b64encode(b"%PDF-1.4" + b"x" * 30).decode())).reason
