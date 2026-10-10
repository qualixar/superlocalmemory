# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Switching the embedding model on a REAL daemon, over HTTP, with recall served.

4.1.21 re-embedded the whole store inside engine start-up while every recall
and remember waited behind the profile-runtime barrier. 4.1.22 re-indexes in
the background. Everything here runs the composed path: a real daemon
subprocess on a private port and data folder, and a hermetic OpenAI-compatible
stub embedding server that serves several "models" of different widths.

The stub maps one probe query to a DIFFERENT fact in each model's space, so
every recall says which space answered it: before the swap the probe must
find ALPHA (old space), after it BETA (new space), never an error, a 503, or
anything else in between.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    REPO_ROOT,
    RealDaemon,
    _child_env,
    _foreign_daemon_pids,
    _reserve_private_port,
)

pytest.importorskip("sqlite_vec")

from tests.helpers.env_capabilities import (
    NO_VECTOR_SEARCH_REASON,
    vector_search_available,
)

# The fixtures build stores with sqlite-vec loaded. An interpreter whose sqlite3
# cannot load extensions (the python.org builds on macOS) has the package but
# cannot use it, which an import check does not see.
pytestmark = pytest.mark.skipif(
    not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON,
)

RUN = uuid.uuid4().hex[:6].upper()
#: model -> (dimension, which marker the probe query lands on)
MODELS = {"stub-old": (768, "ALPHAMARK"), "stub-new": (384, "BETAMARK"),
          "stub-third": (512, "ALPHAMARK"), "stub-flaky": (256, "BETAMARK")}
PROBE = "zzqprobe"
ALPHA = f"Kestrel fixture ALPHAMARK{RUN} note about harbour tide tables."
BETA = f"Kestrel fixture BETAMARK{RUN} note about mountain weather stations."
FILLERS = 60
DELAY: dict[str, float] = {}
EMBEDDED: dict[str, int] = {}
_LOCK = threading.Lock()


def _hash_vector(key: str, dim: int) -> list[float]:
    raw = [b for i in range(0, dim, 32)
           for b in hashlib.sha256(f"{i}:{key}".encode()).digest()][:dim]
    values = [(b - 127.5) / 127.5 for b in raw]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def stub_vector(model: str, text: str) -> list[float]:
    dim, probe_lands_on = MODELS[model]
    if PROBE in text:
        key = probe_lands_on
    elif "ALPHAMARK" in text:
        key = "ALPHAMARK"
    elif "BETAMARK" in text:
        key = "BETAMARK"
    else:
        key = text
    return _hash_vector(f"{model}|{key}", dim)


class _Models(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        model = str(request.get("model"))
        texts = request.get("input") or []
        texts = [texts] if isinstance(texts, str) else texts
        is_probe = texts == ["SuperLocalMemory re-index probe"]
        if model not in MODELS or (model == "stub-flaky" and not is_probe):
            self.send_response(500)
            self.end_headers()
            return
        time.sleep(DELAY.get(model, 0.0))
        with _LOCK:
            EMBEDDED[model] = EMBEDDED.get(model, 0) + len(texts)
        body = json.dumps({"data": [{"index": i, "embedding": stub_vector(model, str(t))}
                                    for i, t in enumerate(texts)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Lane:
    def __init__(self, root: Path, models_url: str) -> None:
        self.root, self.models_url = root, models_url
        self.data_root = root / "data"
        self.data_root.mkdir()
        self.port = _reserve_private_port()
        assert self.port not in PRODUCTION_PORTS
        self.foreign = _foreign_daemon_pids()
        config = {"mode": "a", "active_profile": "default", "daemon_port": self.port,
                  "daemon_enable_legacy_port": False, "mesh_enabled": False,
                  "scale_auto_promote_enabled": False,
                  "embedding": {"provider": "openai", "api_endpoint": f"{models_url}/v1",
                                "model_name": "stub-old", "dimension": 768, "api_key": ""}}
        (self.data_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.daemon = self.start()

    def start(self) -> RealDaemon:
        env = _child_env(self.data_root, self.port, self.root / "home", self.root / "cache")
        env.update({"SLM_REINDEX_BATCH": "4", "SLM_REINDEX_PAUSE_S": "0"})
        log = self.root / f"daemon-{time.monotonic_ns()}.log"
        with log.open("wb") as handle:
            proc = subprocess.Popen(
                [sys.executable, "-m", "superlocalmemory.server.unified_daemon", "--start",
                 f"--port={self.port}"], stdout=handle, stderr=handle, env=env,
                cwd=str(REPO_ROOT), start_new_session=True)
        daemon = RealDaemon(proc, self.port, self.data_root, log, env)
        daemon.wait_ready()
        return daemon

    def kill(self) -> None:
        os.killpg(self.daemon.proc.pid, signal.SIGKILL)
        self.daemon.proc.wait(timeout=30)
        for name in ("daemon.json", "daemon.pid", "daemon.port"):
            (self.data_root / name).unlink(missing_ok=True)

    def rows(self, sql: str, args: tuple = ()) -> list[tuple]:
        conn = sqlite3.connect(f"file:{self.data_root / 'memory.db'}?mode=ro", uri=True,
                               timeout=30)
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def vec_dimension(self, table: str = "fact_embeddings") -> int | None:
        import re

        found = self.rows("SELECT sql FROM sqlite_master WHERE name = ?", (table,))
        if not found:
            return None
        return int(re.search(r"float\[(\d+)\]", found[0][0]).group(1))

    def status(self) -> dict:
        code, body = self.daemon.request("GET", "/api/v3/embedding/reindex")
        assert code == 200, body
        return body

    def switch(self, model: str) -> tuple[int, dict]:
        return self.daemon.request("POST", "/api/v3/embedding/reindex", {
            "model_name": model, "dimension": MODELS[model][0], "provider": "openai",
            "api_endpoint": f"{self.models_url}/v1"})

    def wait_job(self, job_id: int, states: tuple, timeout: float = 240.0) -> dict:
        deadline = time.monotonic() + timeout
        job: dict = {}
        while time.monotonic() < deadline:
            job = self.status()["job"] or {}
            if job.get("job_id") == job_id and job.get("state") in states:
                return job
            time.sleep(0.1)
        raise AssertionError(f"job {job_id} never reached {states}: {job}\n"
                             f"{self.daemon._log_tail()}")

    def probe_top(self) -> str:
        payload = self.daemon.recall(PROBE, "default")
        results = payload.get("results") or []
        return str(results[0].get("content", "")) if results else ""

    def wait_embedded(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            facts = self.rows("SELECT COUNT(*) FROM atomic_facts")[0][0]
            vectors = self.rows("SELECT COUNT(*) FROM embedding_metadata")[0][0] \
                if self.rows("SELECT 1 FROM sqlite_master WHERE name='embedding_metadata'") else 0
            if facts and vectors >= facts:
                return
            time.sleep(0.3)
        raise AssertionError("memories were never embedded")


def _which(content: str) -> str:
    return "ALPHA" if "ALPHAMARK" in content else "BETA" if "BETAMARK" in content else content


@pytest.fixture()
def lane(tmp_path_factory):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    lane = Lane(tmp_path_factory.mktemp("embed-switch"),
                f"http://127.0.0.1:{server.server_address[1]}")
    try:
        for i, text in enumerate([ALPHA, BETA] + [
                f"Kestrel filler record {i} {RUN}: inventory code Q{i:03d} for depot {i % 7}."
                for i in range(FILLERS)]):
            lane.daemon.remember(text, "default", f"embed-switch-{RUN}-{i}")
        lane.wait_embedded()
        yield lane
    finally:
        try:
            _stop_and_reap(lane)
        finally:
            server.shutdown()
            server.server_close()


def _stop_and_reap(lane: Lane) -> None:
    """Stop the daemon; every model worker it started must be gone with it.

    4.1.21's shutdown closed the engine built at start-up. After a switch (or
    any hot reconfigure) that engine is already closed and the LIVE one was
    never closed, so its model workers lived on until their parent watchdog
    noticed. A clean stop ends them itself, within its own worker timeouts.
    Only this daemon's own descendant pids are looked at; any left are killed
    after the assertion so the run stays clean.
    """
    import psutil

    try:
        kids = psutil.Process(lane.daemon.proc.pid).children(recursive=True)
    except psutil.Error:
        kids = []
    lane.daemon.stop(lane.foreign)
    _gone, alive = psutil.wait_procs(kids, timeout=3)
    for proc in alive:
        proc.kill()
    assert not alive, f"model workers outlived the daemon stop: {[p.pid for p in alive]}"


def _before_any_switch_the_probe_finds_alpha_in_the_old_space(lane):
    assert lane.vec_dimension() == 768
    assert _which(lane.probe_top()) == "ALPHA"
    assert lane.status()["live"] == "stub-old::768"


def _recall_is_served_throughout_a_switch_and_flips_once_at_activation(lane):
    DELAY["stub-new"] = 0.15
    seen: list[tuple[float, int, str]] = []
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            started = time.monotonic()
            code, payload = lane.daemon.request("GET", "/recall",
                                                params={"q": PROBE, "profile_id": "default"})
            top = (payload.get("results") or [{}])[0].get("content", "") if code == 200 else ""
            seen.append((time.monotonic() - started, code, _which(str(top))))

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    try:
        code, body = lane.switch("stub-new")
        assert code == 202, body
        job_id = body["job"]["job_id"]
        lane.wait_job(job_id, ("running", "catching_up"))
        # A memory saved while the job runs must land in the new space.
        lane.daemon.remember(f"Kestrel GAMMA{RUN} saved during the switch.", "default",
                             f"embed-switch-{RUN}-gamma")
        conflict, refused = lane.switch("stub-third")
        assert conflict == 409 and refused["job_id"] == job_id, refused
        job = lane.wait_job(job_id, ("activated", "failed"))
        assert job["state"] == "activated", job
        time.sleep(1.5)  # recalls after the swap
    finally:
        stop.set()
        thread.join(30)
    codes = {code for _t, code, _w in seen}
    answers = [w for _t, code, w in seen]
    assert codes == {200}, f"recall was not served throughout: {sorted(codes)}"
    assert set(answers) <= {"ALPHA", "BETA"}, f"a recall mixed spaces: {set(answers)}"
    flip = answers.index("BETA")
    assert "ALPHA" not in answers[flip:], "the old space answered after the new one"
    assert answers.count("ALPHA") >= 3 and answers.count("BETA") >= 3, answers
    print(f"\nrecalls during switch: {len(seen)}, max latency "
          f"{max(t for t, _c, _w in seen):.3f}s, activation stats {job.get('stats', {})}")
    assert lane.vec_dimension() == 384
    assert {r[0] for r in lane.rows("SELECT model_name FROM embedding_metadata")} == {"stub-new"}
    gamma = lane.rows("SELECT embedding FROM atomic_facts WHERE content LIKE ?",
                      (f"%GAMMA{RUN}%",))
    assert gamma and len(gamma[0][0]) == 384 * 4, "the memory saved mid-switch has no new vector"
    health = lane.daemon.request("GET", "/health")[1]
    assert health["embedding_reindex"]["state"] == "activated"


def _a_failing_model_fails_the_job_and_leaves_the_space_live(lane):
    code, body = lane.switch("stub-flaky")
    assert code == 202, body
    job = lane.wait_job(body["job"]["job_id"], ("failed", "activated"))
    assert job["state"] == "failed" and job["error"], job
    assert lane.vec_dimension() == 384
    assert lane.status()["live"] == "stub-new::384"
    assert not lane.rows("SELECT 1 FROM sqlite_master WHERE name = 'reembed_next_vec'")
    assert _which(lane.probe_top()) == "BETA"


def _a_killed_daemon_keeps_the_old_space_and_resumes_the_job(lane):
    DELAY["stub-third"] = 0.6
    EMBEDDED.pop("stub-third", None)
    code, body = lane.switch("stub-third")
    assert code == 202, body
    job_id = body["job"]["job_id"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and (lane.status()["job"] or {}).get("done", 0) < 24:
        time.sleep(0.2)
    # Erase a memory mid-job: it must not survive in the staged space.
    victim = lane.rows("SELECT fact_id FROM atomic_facts WHERE content LIKE ?",
                       ("%inventory code Q007%",))[0][0]
    code, _ = lane.daemon.request("DELETE", f"/api/memories/{victim}")
    assert code == 200
    assert not lane.rows("SELECT 1 FROM reembed_next_map WHERE fact_id = ?", (victim,))
    lane.kill()
    state = lane.rows("SELECT state, cursor, done FROM embedding_reindex_jobs WHERE job_id = ?",
                      (job_id,))[0]
    assert state[0] in ("running", "catching_up") and state[1] > 0, state
    done_before_kill = state[2]
    assert lane.vec_dimension() == 384, "a killed job changed the live space"
    assert lane.rows("PRAGMA quick_check")[0][0] == "ok"
    lane.daemon = lane.start()
    assert _which(lane.probe_top()) == "BETA", "after restart the old space must still answer"
    DELAY["stub-third"] = 0.0
    job = lane.wait_job(job_id, ("activated", "failed"))
    assert job["state"] == "activated", job
    total = lane.rows("SELECT COUNT(*) FROM atomic_facts")[0][0]
    # Resuming re-does at most one batch (4) plus a probe per start (2); starting
    # over would re-embed everything done before the kill (>= 24).
    assert EMBEDDED["stub-third"] < total + done_before_kill / 2, (
        f"the job restarted from zero: {EMBEDDED['stub-third']} texts for {total} memories, "
        f"{done_before_kill} done before the kill")
    assert lane.vec_dimension() == 512
    assert _which(lane.probe_top()) == "ALPHA"
    for table in ("fact_embeddings", "reembed_prev_map", "embedding_metadata"):
        if table == "fact_embeddings":
            continue
        assert not lane.rows(f"SELECT 1 FROM {table} WHERE fact_id = ?", (victim,)), table


def _rollback_restores_the_previous_answers(lane):
    code, body = lane.daemon.request("POST", "/api/v3/embedding/reindex/rollback", {})
    assert code == 202, body
    job = lane.wait_job(body["job"]["job_id"], ("activated", "failed"))
    assert job["state"] == "activated" and job["kind"] == "rollback", job
    assert job["copied"] > 0, "unchanged vectors were re-embedded instead of copied"
    assert lane.vec_dimension() == 384
    assert _which(lane.probe_top()) == "BETA"
    assert lane.status()["previous"] == "stub-third::512"


def _an_erased_memory_leaves_nothing_in_the_previous_space(lane):
    victim = lane.rows("SELECT fact_id FROM atomic_facts WHERE content LIKE ?",
                       ("%inventory code Q011%",))[0][0]
    vec_rowid = lane.rows("SELECT vec_rowid FROM reembed_prev_map WHERE fact_id = ?",
                          (victim,))[0][0]
    code, _ = lane.daemon.request("DELETE", f"/api/memories/{victim}")
    assert code == 200
    assert not lane.rows("SELECT 1 FROM reembed_prev_map WHERE fact_id = ?", (victim,))
    assert not lane.rows("SELECT 1 FROM reembed_prev_vec_rowids WHERE rowid = ?", (vec_rowid,)), (
        "the erased memory's vector survived in the previous space")


def _forget_previous_frees_the_previous_space(lane):
    code, body = lane.daemon.request("POST", "/api/v3/embedding/reindex/forget-previous", {})
    assert code == 200 and body["freed_vectors"] > 0, body
    assert not lane.rows("SELECT 1 FROM sqlite_master WHERE name LIKE 'reembed_prev%'")
    assert lane.status()["previous_vectors_kept"] is False
    code, body = lane.daemon.request("POST", "/api/v3/embedding/reindex/rollback", {})
    assert code == 409, body


def _a_model_named_in_config_by_hand_is_reindexed_in_the_background(lane):
    """Engine start used to re-embed right there; now it serves the stored space."""
    _stop_and_reap(lane)
    config_path = lane.data_root / "config.json"
    config = json.loads(config_path.read_text())
    config["embedding"].update({"model_name": "stub-old", "dimension": 768})
    config_path.write_text(json.dumps(config))
    DELAY["stub-old"] = 0.3
    lane.daemon = lane.start()
    assert lane.vec_dimension() == 384, "start-up rebuilt the vectors instead of serving them"
    assert _which(lane.probe_top()) == "BETA", "start-up paired the new model with old vectors"
    job = lane.status()["job"]
    assert job["to"] == "stub-old::768" and job["state"] in sp_active(), job
    assert json.loads(config_path.read_text())["embedding"]["model_name"] == "stub-new", (
        "config.json must name the model that is live until the switch is done")
    DELAY["stub-old"] = 0.0
    done = lane.wait_job(job["job_id"], ("activated", "failed"))
    assert done["state"] == "activated", done
    assert lane.vec_dimension() == 768 and _which(lane.probe_top()) == "ALPHA"


def sp_active() -> tuple:
    from superlocalmemory.storage.embedding_spaces import ACTIVE_STATES
    return ACTIVE_STATES


#: One daemon carries its state from step to step, so the steps run in this
#: order inside ONE test: a parallel run cannot split them across workers.
STEPS = (
    _before_any_switch_the_probe_finds_alpha_in_the_old_space,
    _recall_is_served_throughout_a_switch_and_flips_once_at_activation,
    _a_failing_model_fails_the_job_and_leaves_the_space_live,
    _a_killed_daemon_keeps_the_old_space_and_resumes_the_job,
    _rollback_restores_the_previous_answers,
    _an_erased_memory_leaves_nothing_in_the_previous_space,
    _forget_previous_frees_the_previous_space,
    _a_model_named_in_config_by_hand_is_reindexed_in_the_background,
)


def test_a_model_switch_on_a_real_daemon_end_to_end(lane):
    for step in STEPS:
        print(f"\n-- {step.__name__}", flush=True)
        step(lane)
