"""The picture worker's memory cap is enforced while a request runs, not only after it.

A growing footprint is simulated by replacing the shared memory reader; the worker is the real
fake-mode process, so a stop really kills a process and a restart really starts another.
"""

from __future__ import annotations

import importlib.util
import itertools
import struct
import sys
import time
import zlib
from pathlib import Path

import psutil
import pytest

from superlocalmemory.infra import proc_memory
from superlocalmemory.runtimes import media_image_ops, worker_client, worker_limits
from superlocalmemory.runtimes.worker_client import MediaWorkerError

WORKER = Path(worker_client.WORKER_PATH)


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _growing(start: float, step: float):
    counter = itertools.count()
    return lambda pid: start + step * next(counter)


def _spawns(client, monkeypatch) -> list[int]:
    seen: list[int] = []
    real = client._spawn

    def counted() -> None:
        real()
        seen.append(client._proc.pid)

    monkeypatch.setattr(client, "_spawn", counted)
    return seen


# -- the watchdog during a request ------------------------------------------------

def test_a_footprint_that_grows_during_a_request_stops_the_worker_before_it_finishes(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=500, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    pid = c.pid
    monkeypatch.setattr(proc_memory, "process_memory_mb", _growing(100.0, 60.0))
    started = time.monotonic()
    with pytest.raises(MediaWorkerError, match="too much memory"):
        c._request("sleep", seconds=20, retry=False)
    assert time.monotonic() - started < 10, "the request ran to its end instead of being stopped"
    assert c.pid is None and not _alive(pid)


def test_an_over_cap_request_is_failed_not_replayed_on_a_fresh_worker(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=500, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    spawned = _spawns(c, monkeypatch)
    monkeypatch.setattr(proc_memory, "process_memory_mb", _growing(100.0, 80.0))
    with pytest.raises(MediaWorkerError, match="too much memory"):
        c._request("sleep", seconds=20)  # retry allowed: a memory stop must still not replay it
    assert spawned == [], "the oversized request was sent to a new worker"


def test_the_next_request_after_a_memory_stop_starts_a_fresh_worker(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=500, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    old = c.pid
    monkeypatch.setattr(proc_memory, "process_memory_mb", _growing(100.0, 80.0))
    with pytest.raises(MediaWorkerError):
        c._request("sleep", seconds=20, retry=False)
    monkeypatch.setattr(proc_memory, "process_memory_mb", lambda pid: 50.0)
    assert len(c.embed_texts(["again"], prompt="Document")) == 1
    assert c.pid and c.pid != old


def test_a_footprint_under_the_cap_is_left_alone(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=500, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    pid = c.pid
    monkeypatch.setattr(proc_memory, "process_memory_mb", lambda p: 499.0)
    assert c._request("sleep", seconds=0.4)["ok"]
    assert c.pid == pid


def test_no_cap_means_no_watching(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=0, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    monkeypatch.setattr(proc_memory, "process_memory_mb", lambda p: 10 ** 6)
    assert c._request("sleep", seconds=0.3)["ok"]
    assert c.pid is not None


def test_an_unreadable_footprint_is_not_a_reason_to_stop(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=500, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    monkeypatch.setattr(proc_memory, "process_memory_mb", lambda p: 0.0)  # 0.0 = unknown
    assert c._request("sleep", seconds=0.3)["ok"]


def test_a_loading_worker_gets_more_room_than_a_serving_one():
    from types import SimpleNamespace
    stub = SimpleNamespace(root=Path("."), weights_dir=lambda: Path("w"))
    c = worker_client.MediaWorkerClient(stub, model_id="fake:768", revision="", rss_limit_mb=1000)
    assert c._load_cap_mb() == max(1000, 1500) * worker_client.LOAD_CAP_FACTOR
    assert worker_client.MediaWorkerClient(stub, model_id="fake:768", revision="", rss_limit_mb=0)._load_cap_mb() == 0


def test_a_runaway_during_the_load_exchange_is_stopped(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=500, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    monkeypatch.setattr(proc_memory, "process_memory_mb", lambda p: 10 ** 6)
    with pytest.raises(MediaWorkerError, match="too much memory"):
        c._roundtrip({"cmd": "sleep", "seconds": 20}, 30, cap_mb=c._load_cap_mb())
    assert c.pid is None


def test_the_stop_message_is_plain_language(client_factory, monkeypatch):
    c = client_factory(rss_limit_mb=500, request_timeout_s=30)
    c.watch_interval_s = 0.05
    c.embed_texts(["warm"], prompt="Document")
    monkeypatch.setattr(proc_memory, "process_memory_mb", _growing(100.0, 200.0))
    with pytest.raises(MediaWorkerError) as info:
        c._request("sleep", seconds=20, retry=False)
    text = str(info.value)
    assert "Traceback" not in text and "/" not in text and "MB" not in text.split("memory")[0]


# -- input guards before anything is sent -----------------------------------------

def _png_header(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    return b"\x89PNG\r\n\x1a\n" + chunk + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


def test_a_text_longer_than_the_worker_takes_is_refused_without_starting_it(client_factory):
    c = client_factory()
    with pytest.raises(MediaWorkerError, match="too long"):
        c.embed_texts(["ok", "x" * (worker_limits.MAX_TEXT_CHARS + 1)], prompt="Document")
    assert c.pid is None


def test_too_many_texts_in_total_are_split_not_refused(client_factory):
    c = client_factory()
    assert len(c.embed_texts(["a"] * (worker_client.MAX_TEXTS + 3), prompt="Document")) == worker_client.MAX_TEXTS + 3


def test_a_file_over_the_size_limit_is_refused_without_starting_the_worker(client_factory, tmp_path):
    big = tmp_path / "big.png"
    with open(big, "wb") as fh:
        fh.truncate(worker_limits.MAX_FILE_BYTES + 1)  # sparse: no real disk used
    c = client_factory()
    for call in (lambda: c.embed_images([big]), lambda: c.prepare_image(big, tmp_path),
                 lambda: c.ocr_image(big)):
        with pytest.raises(MediaWorkerError, match="too large"):
            call()
    assert c.pid is None


def test_a_picture_with_too_many_pixels_is_refused_without_starting_the_worker(client_factory, tmp_path):
    huge = tmp_path / "huge.png"
    huge.write_bytes(_png_header(20000, 20000) + b"\0" * 64)  # 400 million pixels in a tiny file
    c = client_factory()
    with pytest.raises(MediaWorkerError, match="too large"):
        c.embed_images([huge])
    assert c.pid is None


def test_a_normal_picture_passes_the_guard(client_factory, tmp_path):
    ok = tmp_path / "ok.png"
    ok.write_bytes(_png_header(800, 600) + b"\0" * 64)
    assert len(client_factory().embed_images([ok])[0]) == 768


def test_a_missing_file_is_left_to_the_worker(client_factory, tmp_path):
    with pytest.raises(MediaWorkerError):
        client_factory().embed_images([tmp_path / "gone.png"])


@pytest.mark.parametrize("data,expected", [
    (_png_header(300, 200), 60000),
    (b"GIF89a" + struct.pack("<HH", 640, 480) + b"\0" * 8, 307200),
    (b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\0" + b"\0" * 9
     + b"\xff\xc0" + struct.pack(">HBHH", 11, 8, 500, 400) + b"\0" * 8, 200000),
    (b"RIFF\0\0\0\0WEBPVP8X" + struct.pack("<I", 10) + b"\0" * 4
     + (1000 - 1).to_bytes(3, "little") + (700 - 1).to_bytes(3, "little"), 700000),
    (b"RIFF\0\0\0\0WEBPVP8L" + struct.pack("<I", 5) + b"\x2f"
     + struct.pack("<I", (300 - 1) | ((200 - 1) << 14)), 60000),
    (b"RIFF\0\0\0\0WEBPVP8 " + struct.pack("<I", 10) + b"\0\0\0" + b"\x9d\x01\x2a"
     + struct.pack("<HH", 320, 240), 76800),
    (b"not an image at all", None),
    (b"", None),
])
def test_image_pixels_reads_the_header_only(tmp_path, data, expected):
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    assert worker_limits.image_pixels(path) == expected


def test_the_limits_match_what_the_worker_and_image_ops_enforce():
    spec = importlib.util.spec_from_file_location("mm_worker_limits_check", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert worker_limits.MAX_TEXT_CHARS == mod.MAX_TEXT_CHARS
    assert worker_limits.MAX_FILE_BYTES == mod.MAX_FILE_BYTES
    assert worker_limits.MAX_PIXELS == media_image_ops.MAX_PIXELS


# -- the hard limit set when the worker starts --------------------------------------

def test_the_hard_limit_is_above_the_cap_and_only_set_on_linux():
    assert worker_limits.data_limit_env(0, platform="linux") == {}
    assert worker_limits.data_limit_env(4500, platform="darwin") == {}
    assert worker_limits.data_limit_env(4500, platform="win32") == {}
    env = worker_limits.data_limit_env(4500, platform="linux")
    limit = int(env[worker_limits.DATA_LIMIT_ENV])
    assert limit > 4500 and limit < 4500 * 3


class FakeResource:
    RLIMIT_DATA = 2
    RLIM_INFINITY = -1

    def __init__(self, current=(-1, -1), fail=False):
        self.limits = {self.RLIMIT_DATA: current}
        self.fail = fail

    def getrlimit(self, which):
        return self.limits[which]

    def setrlimit(self, which, pair):
        if self.fail:
            raise ValueError("not allowed")
        self.limits[which] = pair


def _worker_module():
    spec = importlib.util.spec_from_file_location("mm_worker_under_test", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_worker_sets_its_own_data_limit_at_start_on_linux():
    mod, res = _worker_module(), FakeResource()
    assert mod._apply_memory_limit({mod.DATA_LIMIT_ENV: "6000"}, platform="linux", resource_module=res) == 6000
    assert res.limits[res.RLIMIT_DATA] == (6000 * 1024 * 1024,) * 2


def test_the_worker_never_raises_a_limit_that_is_already_lower():
    mod, res = _worker_module(), FakeResource(current=(1000 * 1024 * 1024, 1000 * 1024 * 1024))
    assert mod._apply_memory_limit({mod.DATA_LIMIT_ENV: "6000"}, platform="linux", resource_module=res) == 1000


@pytest.mark.parametrize("env,platform", [({}, "linux"), ({"SLM_MEDIA_WORKER_DATA_LIMIT_MB": "6000"}, "darwin"),
                                           ({"SLM_MEDIA_WORKER_DATA_LIMIT_MB": "junk"}, "linux"),
                                           ({"SLM_MEDIA_WORKER_DATA_LIMIT_MB": "0"}, "linux"),
                                           ({"SLM_MEDIA_WORKER_DATA_LIMIT_MB": "-5"}, "linux")])
def test_the_worker_sets_nothing_without_a_usable_value_on_linux(env, platform):
    mod, res = _worker_module(), FakeResource()
    assert mod._apply_memory_limit(env, platform=platform, resource_module=res) == 0
    assert res.limits[res.RLIMIT_DATA] == (-1, -1)


def test_a_refused_limit_does_not_stop_the_worker_starting():
    mod = _worker_module()
    assert mod._apply_memory_limit({mod.DATA_LIMIT_ENV: "6000"}, platform="linux",
                                   resource_module=FakeResource(fail=True)) == 0


def test_the_client_hands_the_limit_to_the_worker_it_starts(client_factory, monkeypatch):
    monkeypatch.setattr(worker_limits.sys, "platform", "linux")
    c = client_factory(rss_limit_mb=1600)
    env = c._worker_env()
    assert int(env[worker_limits.DATA_LIMIT_ENV]) > 1600
    assert client_factory(rss_limit_mb=0)._worker_env().get(worker_limits.DATA_LIMIT_ENV) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_DATA is enforced on Linux")
def test_a_real_linux_worker_reports_the_limit_it_was_given(client_factory):
    c = client_factory(rss_limit_mb=1600)
    c.embed_texts(["x"], prompt="Document")
    assert c._request("ping")["data_limit_mb"] == int(c._worker_env()[worker_limits.DATA_LIMIT_ENV])
