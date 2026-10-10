"""The client: lazy start, idle kill, restart, timeouts, memory limit, factory."""

import os
import subprocess
import threading
import time

import psutil
import pytest

from superlocalmemory.runtimes import features, worker_client
from superlocalmemory.runtimes.worker_client import MediaWorkerError, MediaWorkerWarming


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_nothing_runs_until_the_first_call(client_factory):
    c = client_factory()
    assert c.pid is None and not c.is_warm()
    vecs = c.embed_texts(["hi"], prompt="Document")
    assert len(vecs) == 1 and len(vecs[0]) == 768 and c.pid and c.is_warm()
    assert c.dim == 768


def test_images_round_trip(client_factory, tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert len(client_factory().embed_images([f])[0]) == 768


def test_fake_models_are_refused_outside_test_isolation(stub_env, monkeypatch):
    monkeypatch.delenv("SLM_TEST_ISOLATION")
    with pytest.raises(ValueError):
        worker_client.MediaWorkerClient(stub_env, model_id="fake:768", revision="")


def test_idle_kill_then_transparent_restart(client_factory):
    c = client_factory(idle_s=0.5)
    c.embed_texts(["a"], prompt="Document")
    pid = c.pid
    deadline = time.time() + 15
    while c.pid is not None and time.time() < deadline:
        time.sleep(0.1)
    assert c.pid is None and not alive(pid)
    c.embed_texts(["a"], prompt="Document")
    assert c.pid and c.pid != pid


def test_idle_default_comes_from_env(stub_env, monkeypatch):
    monkeypatch.setenv("SLM_MEDIA_WORKER_IDLE_S", "42")
    c = worker_client.MediaWorkerClient(stub_env, model_id="fake:768", revision="")
    assert c.idle_s == 42
    monkeypatch.delenv("SLM_MEDIA_WORKER_IDLE_S")
    assert worker_client.MediaWorkerClient(stub_env, model_id="fake:768", revision="").idle_s == 1800


def test_crash_between_requests_is_retried_once(client_factory):
    c = client_factory()
    c.embed_texts(["a"], prompt="Document")
    psutil.Process(c.pid).kill()
    time.sleep(0.2)
    assert len(c.embed_texts(["b"], prompt="Document")) == 1


def test_crash_mid_request_is_retried_once(client_factory):
    c = client_factory(request_timeout_s=20)
    c.embed_texts(["a"], prompt="Document")
    pid = c.pid
    threading.Timer(0.5, lambda: psutil.Process(pid).kill()).start()
    reply = c._request("sleep", seconds=3)  # killed during the sleep, restarted, replayed
    assert reply["ok"] and c.pid != pid


def test_a_second_failure_raises(client_factory, monkeypatch):
    c = client_factory()
    c.embed_texts(["a"], prompt="Document")
    monkeypatch.setattr(worker_client, "WORKER_PATH", worker_client.WORKER_PATH.with_name("missing.py"))
    c.stop()
    with pytest.raises(MediaWorkerError):
        c.embed_texts(["a"], prompt="Document")


def test_hung_worker_times_out_and_is_gone(client_factory):
    c = client_factory(request_timeout_s=0.5)
    c.embed_texts(["a"], prompt="Document")
    pid = c.pid
    with pytest.raises(MediaWorkerError, match="timed out"):
        c._request("sleep", seconds=30, retry=False)
    assert c.pid is None and not alive(pid)
    assert len(c.embed_texts(["a"], prompt="Document")) == 1


def test_rss_over_the_limit_stops_the_worker(client_factory):
    c = client_factory(rss_limit_mb=1)
    c.embed_texts(["a"], prompt="Document")
    assert c.pid is None
    assert len(c.embed_texts(["a"], prompt="Document")) == 1  # restarts lazily


def test_the_cap_is_judged_on_the_shared_memory_reader(client_factory, monkeypatch):
    from superlocalmemory.infra import proc_memory

    c = client_factory(rss_limit_mb=5000)
    monkeypatch.setattr(proc_memory, "process_memory_mb", lambda pid: 3650.0)  # footprint, not the 215 MB RSS
    c.embed_texts(["a"], prompt="Document")
    assert c.pid is not None  # under its cap: stays up
    monkeypatch.setattr(proc_memory, "process_memory_mb", lambda pid: 5200.0)
    c.embed_texts(["a"], prompt="Document")
    assert c.pid is None  # over its cap: stopped


def test_rss_limit_comes_from_env(stub_env, monkeypatch):
    monkeypatch.setenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", "77")
    assert worker_client.MediaWorkerClient(stub_env, model_id="fake:768", revision="").rss_limit_mb == 77


def test_four_threads_are_serialised_and_all_answered(client_factory):
    c = client_factory()
    out, errs = [], []

    def go(i):
        try:
            out.append(c.embed_texts([f"t{i}"], prompt="Document")[0])
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert not errs and len(out) == 4 and len({tuple(v) for v in out}) == 4


def test_stop_is_idempotent_and_leaves_no_child(client_factory):
    c = client_factory()
    c.embed_texts(["a"], prompt="Document")
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    pid = c.pid
    c.stop()
    c.stop()
    assert pid in before and not alive(pid) and c.pid is None
    assert pid not in {p.pid for p in psutil.Process().children(recursive=True)
                       if p.status() != psutil.STATUS_ZOMBIE}


def test_cold_ingest_can_answer_warming_and_warms_in_background(client_factory, tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    c = client_factory()
    with pytest.raises(MediaWorkerWarming):
        c.embed_images([f], wait_cold=False)
    deadline = time.time() + 20
    while not c.is_warm() and time.time() < deadline:
        time.sleep(0.1)
    assert c.is_warm()
    assert len(c.embed_images([f], wait_cold=False)) == 1


def test_embed_query_never_spawns_inside_a_recall(client_factory):
    c = client_factory()
    assert c.embed_query("hello") is None
    deadline = time.time() + 20
    while not c.is_warm() and time.time() < deadline:
        time.sleep(0.1)
    vec = c.embed_query("hello")
    assert vec is not None and len(vec) == 768


def test_embed_query_returns_none_when_busy(client_factory):
    c = client_factory(request_timeout_s=20)
    c.embed_texts(["a"], prompt="Document")
    t = threading.Thread(target=lambda: c._request("sleep", seconds=2))
    t.start()
    time.sleep(0.3)
    started = time.time()
    assert c.embed_query("q", wait_s=0.2) is None
    assert time.time() - started < 1.5
    t.join(10)


def test_model_load_takes_the_ram_reservation(client_factory, monkeypatch):
    from contextlib import contextmanager
    seen = []

    @contextmanager
    def fake(name, **kw):
        seen.append(name)
        yield

    monkeypatch.setattr(worker_client.ram_lock, "ram_reservation", fake)
    client_factory().embed_texts(["a"], prompt="Document")
    assert seen == ["media-model-load"]


def test_no_memory_for_the_model_is_a_plain_error(client_factory, monkeypatch):
    def refuse(*a, **k):
        raise RuntimeError("free 1MB < required")

    monkeypatch.setattr(worker_client.ram_lock, "ram_reservation", refuse)
    c = client_factory()
    with pytest.raises(MediaWorkerError):
        c.embed_texts(["a"], prompt="Document")
    assert c.pid is None


# -- factory ------------------------------------------------------------------

@pytest.fixture
def no_spawn(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(subprocess, "Popen", boom)


def test_factory_returns_none_when_media_is_off(stub_env, tmp_path, no_spawn):
    assert worker_client.media_embedder(env=stub_env, data_root=tmp_path) is None


def test_factory_returns_none_when_env_is_not_ready(tmp_path, no_spawn):
    from tests.test_runtimes.conftest import StubEnv
    features._write_features(tmp_path, {"schema": 1, "media": {"enabled": True}})
    assert worker_client.media_embedder(env=StubEnv(tmp_path, "installing"), data_root=tmp_path) is None


def test_factory_returns_a_lazy_client_and_registers_the_stop_hook(stub_env, tmp_path, no_spawn, monkeypatch):
    features._write_features(tmp_path, {"schema": 1, "media": {"enabled": True}})
    hooks = []
    monkeypatch.setattr(worker_client, "register_media_stop_hook", hooks.append)
    c = worker_client.media_embedder(env=stub_env, data_root=tmp_path, model_id="fake:768", revision="")
    assert c is not None and c.pid is None and hooks == [c.stop]
    assert worker_client.media_embedder(env=stub_env, data_root=tmp_path, model_id="fake:768", revision="") is c
    worker_client._CLIENTS.clear()


def test_the_health_monitor_counts_and_may_kill_the_media_worker():
    from superlocalmemory.core.health_monitor import HealthMonitor
    cmd = "/x/venv/bin/python -i -I /site/superlocalmemory/runtimes/multimodal_worker.py"
    assert any(i in cmd for i in HealthMonitor._WORKER_IDENTIFIERS)
    assert HealthMonitor._EMBEDDING_IDENTIFIER not in cmd


def test_the_factory_takes_the_image_model_from_the_space_plan(stub_env, tmp_path, no_spawn, monkeypatch):
    features._write_features(tmp_path, {"schema": 1, "media": {"enabled": True}})
    monkeypatch.setattr(worker_client, "register_media_stop_hook", lambda f: None)
    monkeypatch.setenv("SLM_MEDIA_SPACE_MODE", "paired")
    worker_client._CLIENTS.clear()
    paired = worker_client.media_embedder(env=stub_env, data_root=tmp_path)
    assert paired.model_id == "nomic-ai/nomic-embed-vision-v1.5" and paired.role == "image"
    monkeypatch.setenv("SLM_MEDIA_SPACE_MODE", "separate")
    separate = worker_client.media_embedder(env=stub_env, data_root=tmp_path)
    assert separate is not paired and separate.role == ""
    worker_client._CLIENTS.clear()
