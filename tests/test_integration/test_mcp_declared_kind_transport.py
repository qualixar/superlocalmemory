# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A kind declared through the tool interface reaches the database confirmed.

4.1.21 lost it: the tool's ``remember`` put the declared kind into free-form
metadata under the reserved ``_slm_memory_kind`` key and never sent the
request's own ``kind`` field. The daemon (correctly) strips every reserved
``_slm_*`` key a caller sends, so 0 of 9 declared kinds survived, while the
same request over plain HTTP with a top-level ``kind`` worked.

Everything here runs the composed path, never an engine mock: a REAL daemon
subprocess on a private port with its own data folder, a REAL ``slm mcp``
stdio child speaking JSON-RPC to it, and the database read back directly.

Covered: all nine kinds confirmed (source ``caller``) on the first queryable
facts; kind-filtered recall finds each save at once; a retry with the same
key returns the same memory; the same key with a different kind is refused
and changes nothing; named-profile routing; the tool's fallback path through
the daemon proxy; confirmed rules/decisions in session start context; a
re-classification run leaves confirmed kinds alone; a daemon restart; and a
forged ``_slm_memory_kind`` in ordinary metadata confirms nothing, over HTTP
and through the proxy.
"""
from __future__ import annotations

import hashlib
import json
import math
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
    _RpcClient,
)

KINDS = ("semantic", "episodic", "status", "opinion", "rule", "decision",
         "procedure", "prospective", "correction")
RUN = uuid.uuid4().hex[:6].upper()
PROFILE = "kestrel-two"
MCP_TOOLS = "remember,recall,session_init,update_memory,review_correction"


def _content(kind: str) -> str:
    # Deliberately neutral wording: nothing in it reads as a rule, a plan or a
    # decision, so only the caller's declaration can give it its kind.
    token = f"QK{KINDS.index(kind)}{RUN}"
    return (f"Fixture record {token}: the assigned verification code is "
            f"C54{KINDS.index(kind)} for synthetic-kestrel.")


def _vector(text: str, dim: int = 768) -> list[float]:
    """Deterministic unit vector per text (a stand-in embedding model).

    Mirrors ``test_source_fidelity_e2e.py``'s stub: a hash, not a real model,
    so enrichment has something to embed on any machine, offline.
    """
    raw = [b for i in range(0, dim, 32)
           for b in hashlib.sha256(f"{i}:{text}".encode()).digest()][:dim]
    values = [(b - 127.5) / 127.5 for b in raw]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class _StubEmbeddings(BaseHTTPRequestHandler):
    """A local OpenAI-compatible embedding endpoint: the one thing this file's
    daemon needs from a model. 4.1.22 made a fresh install default to the
    zero-setup mode (Mode A) instead of assuming a local Ollama; in that mode
    the real embedder is a sentence-transformers model that has to download
    on first use, which a HOME with no HF cache and no network can never do,
    so enrichment stayed "queryable" forever. Pointing the embedding config at
    this stub keeps the test hermetic (no network, no Ollama) while still
    running the real daemon's enrichment path end to end.
    """

    def log_message(self, *_args) -> None:  # keep test output clean
        return

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        texts = request.get("input") or []
        texts = [texts] if isinstance(texts, str) else texts
        body = json.dumps({"data": [{"index": i, "embedding": _vector(str(t))}
                                    for i, t in enumerate(texts)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def stub_embedder():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubEmbeddings)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _start_daemon(
    data_root: Path, port: int, root: Path, models_url: str | None = None,
) -> RealDaemon:
    # Pinned to Mode A (the now-default fresh-install mode) so this never
    # depends on, or reaches, a real local Ollama -- only the embedding
    # model is swapped for the hermetic stub; the rules-based extraction
    # backend Mode A uses needs no LLM at all. ``models_url`` is optional:
    # a caller that only needs the daemon itself (not hermetic enrichment)
    # gets the previous, config-free default untouched -- this file's own
    # callers (below) always pass it.
    if models_url is not None:
        config = {
            "mode": "a", "active_profile": "default", "daemon_port": port,
            "daemon_enable_legacy_port": False, "mesh_enabled": False,
            "scale_auto_promote_enabled": False,
            "embedding": {"provider": "openai", "api_endpoint": f"{models_url}/v1",
                          "model_name": "stub-embed", "dimension": 768, "api_key": ""},
        }
        (data_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    env = _child_env(data_root, port, root / "home", root / "cache")
    log = root / f"daemon-{time.monotonic_ns()}.log"
    with log.open("wb") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
             "--start", f"--port={port}"],
            stdout=handle, stderr=handle, env=env, cwd=str(REPO_ROOT),
            start_new_session=True,
        )
    daemon = RealDaemon(proc, port, data_root, log, env)
    daemon.wait_ready()
    return daemon


class _Lane:
    """One daemon plus one MCP stdio child, restartable on the same store."""

    def __init__(self, root: Path, models_url: str | None = None) -> None:
        self.root = root
        self.models_url = models_url
        self.data_root = root / "data"
        self.data_root.mkdir()
        self.port = _reserve_private_port()
        assert self.port not in PRODUCTION_PORTS
        self.foreign = _foreign_daemon_pids()
        self.daemon = _start_daemon(self.data_root, self.port, root, models_url)
        self.daemon.precreate_profiles((PROFILE,))
        self.mcp = None
        self.saved: dict[str, dict] = {}

    def open_mcp(self) -> None:
        self.daemon.wait_health_fast()
        env = dict(self.daemon.env)
        env.update({"SLM_MCP_TOOLS": MCP_TOOLS,
                    "SLM_DISABLE_WARMUP_SIDE_EFFECTS": "1"})
        stderr_path = self.root / f"mcp-{time.monotonic_ns()}.log"
        with stderr_path.open("wb") as stderr_log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "superlocalmemory.cli.main", "mcp"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr_log, env=env, cwd=str(REPO_ROOT),
            )
        client = _RpcClient(proc, stderr_path)
        hello = client.call("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "kind-transport", "version": "1.0"},
        })
        assert "result" in hello, hello
        client.notify("notifications/initialized")
        self.mcp = client

    def close_mcp(self) -> None:
        if self.mcp is not None:
            self.mcp.close()
            self.mcp = None

    def tool(self, name: str, **arguments) -> dict:
        reply = self.mcp.call("tools/call", {"name": name, "arguments": arguments})
        assert "error" not in reply, reply
        result = reply["result"]
        body = json.loads(result["content"][0]["text"])
        if body.get("success") is False:
            assert result.get("isError") is True, reply
            assert body.get("code") in {"CONFLICT", "IDEMPOTENCY_CONFLICT"}, reply
            assert body.get("retryable") is False, reply
        else:
            assert result.get("isError") is not True, reply
        if result.get("structuredContent") is not None:
            assert result["structuredContent"] == body, reply
        return body

    def restart(self) -> None:
        self.close_mcp()
        self.daemon.stop(self.foreign)
        self.daemon = _start_daemon(self.data_root, self.port, self.root, self.models_url)
        self.open_mcp()

    def rows(self, sql: str, args: tuple = ()) -> list[tuple]:
        conn = sqlite3.connect(f"file:{self.data_root / 'memory.db'}?mode=ro",
                               uri=True, timeout=30)
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def kind_of(self, fact_id: str) -> tuple:
        found = self.rows("SELECT memory_kind, memory_kind_source FROM atomic_facts "
                          "WHERE fact_id=?", (fact_id,))
        assert found, f"fact {fact_id} is not in the database"
        return found[0]

    def run_child(self, script: str) -> dict:
        """Run ``script`` in a child that talks to this daemon like a client."""
        done = subprocess.run(
            [sys.executable, "-c", script], env=self.daemon.env, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=180,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def lane(tmp_path_factory, stub_embedder):
    lane = _Lane(tmp_path_factory.mktemp("kind-transport"), stub_embedder)
    try:
        lane.open_mcp()
        for kind in KINDS:
            out = lane.tool("remember", content=_content(kind), kind=kind,
                            tags="metadata-only-qtag,zeta", project="synthetic-kestrel",
                            idempotency_key=f"fixture-declared-{kind}-{RUN}")
            assert out["success"] is True, out
            assert out["fact_ids"], out
            lane.saved[kind] = out
        yield lane
    finally:
        lane.close_mcp()
        lane.daemon.stop(lane.foreign)


def test_all_nine_declared_kinds_are_confirmed_on_the_first_queryable_facts(lane):
    got = {kind: [lane.kind_of(fid) for fid in lane.saved[kind]["fact_ids"]]
           for kind in KINDS}
    want = {kind: [(kind, "caller")] * len(lane.saved[kind]["fact_ids"]) for kind in KINDS}
    assert got == want


def test_kind_filtered_recall_finds_each_save_right_after_it(lane):
    misses = []
    for kind in KINDS:
        token = f"QK{KINDS.index(kind)}{RUN}"
        out = lane.tool("recall", query=f"Fixture record {token} verification code",
                        kind=kind, limit=10)
        ids = {item.get("fact_id") for item in out.get("results", [])}
        if not ids & set(lane.saved[kind]["fact_ids"]):
            misses.append(kind)
    assert misses == []


def test_a_retry_with_the_same_key_returns_the_same_confirmed_memory(lane):
    again = lane.tool("remember", content=_content("decision"), kind="decision",
                      tags="metadata-only-qtag,zeta", project="synthetic-kestrel",
                      idempotency_key=f"fixture-declared-decision-{RUN}")
    assert again["success"] is True, again
    assert again["fact_ids"] == lane.saved["decision"]["fact_ids"]
    assert lane.kind_of(again["fact_ids"][0]) == ("decision", "caller")


def test_the_same_key_with_a_different_kind_is_refused_and_changes_nothing(lane):
    # The kind is part of what was asked. A key reused for a different kind is
    # a different request: it must neither keep the old kind while reporting
    # success, nor quietly confirm the new one.
    out = lane.tool("remember", content=_content("decision"), kind="rule",
                    tags="metadata-only-qtag,zeta", project="synthetic-kestrel",
                    idempotency_key=f"fixture-declared-decision-{RUN}")
    assert out["success"] is False, out
    assert out["retryable"] is False, out
    assert "idempotency" in out["error"].lower(), out
    assert lane.kind_of(lane.saved["decision"]["fact_ids"][0]) == ("decision", "caller")


def test_the_same_words_saved_again_with_another_kind_and_no_key_are_a_new_request(lane):
    # Without a caller key the tool derives one from the request. The kind is
    # part of that request, so the derived key differs: the second save is a
    # new request, not an idempotency conflict over a key the caller never set
    # (the same answer HTTP, which keys such a save afresh, has always given).
    words = f"Fixture record QKX{RUN}: the shared verification code is C549."
    first = lane.tool("remember", content=words, kind="status")
    second = lane.tool("remember", content=words, kind="opinion")
    assert first["success"] is True and second["success"] is True, (first, second)
    assert first["operation_id"] != second["operation_id"], (first, second)
    assert lane.kind_of(first["fact_ids"][0]) == ("status", "caller")


def test_identical_words_declared_as_another_kind_do_not_silently_keep_the_first(lane):
    # Identical words are one fact. Its first confirmed kind is kept, and the
    # second save says so instead of reporting a kind it did not record.
    words = f"Fixture record QKY{RUN}: the shared verification code is C548."
    first = lane.tool("remember", content=words, kind="status")
    second = lane.tool("remember", content=words, kind="opinion")
    assert second["success"] is True, (first, second)
    assert second["kind_conflict"] == {"kept": "status", "requested": "opinion"}, second
    assert lane.kind_of(second["fact_ids"][0]) == ("status", "caller")


def test_a_named_profile_keeps_the_declared_kind(lane):
    out = lane.tool("remember", content=f"Fixture record QKP{RUN}: routed code C550.",
                    kind="procedure", profile_id=PROFILE,
                    idempotency_key=f"fixture-routed-{RUN}")
    assert out["success"] is True, out
    rows = lane.rows("SELECT profile_id, memory_kind, memory_kind_source FROM atomic_facts "
                     "WHERE fact_id=?", (out["fact_ids"][0],))
    assert rows == [(PROFILE, "procedure", "caller")]
    found = lane.tool("recall", query=f"Fixture record QKP{RUN} routed code",
                      kind="procedure", profile_id=PROFILE)
    assert out["fact_ids"][0] in {r.get("fact_id") for r in found.get("results", [])}


def _daemon_log_tail(lane) -> str:
    logs = sorted(lane.root.glob("daemon-*.log"))
    lines = logs[-1].read_text(errors="replace").splitlines() if logs else []
    return "\n".join(line for line in lines if "huggingface" not in line.lower())[-2500:]


def test_editing_a_confirmed_memory_carries_its_kind_once_the_edit_is_applied(lane):
    # update_memory takes no kind. An edit is a proposed correction: until a
    # person applies it, its successor is deliberately NOT confirmed (a
    # proposed rule must never be loaded as a standing one) and the original
    # keeps its kind; applying it carries the confirmed kind over. Runs before
    # enrichment starts competing for the writer.
    fid = lane.saved["procedure"]["fact_ids"][0]
    out = lane.tool("update_memory", fact_id=fid,
                    content=f"Fixture record QK6{RUN}: the code is now C560.")
    assert out.get("success") is True, (out, _daemon_log_tail(lane))
    successor = out["successor_fact_id"]
    assert lane.kind_of(fid) == ("procedure", "caller")
    assert lane.kind_of(successor)[1] not in ("caller", "user")
    case = out["correction_case"]
    applied = lane.tool("review_correction", case_id=case["case_id"], action="apply",
                        expected_version=case["version"])
    assert applied.get("success") is True, applied
    assert lane.kind_of(successor) == ("procedure", "caller")


_FALLBACK_CHILD = r"""
import asyncio, json
from superlocalmemory.cli import daemon as d
from superlocalmemory.mcp import tools_core
real = d.is_daemon_running
calls = {"n": 0}
def first_probe_misses():
    # The tool's first look at the daemon misses (a daemon still starting);
    # it must then save through the proxy path, never a local writer.
    calls["n"] += 1
    return False if calls["n"] == 1 else real()
d.is_daemon_running = first_probe_misses
tools = {}
class S:
    def tool(self, *a, **k):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco
tools_core.register_core_tools(S(), lambda: None)
out = asyncio.run(tools["remember"](CONTENT, kind=KIND, idempotency_key=KEY, **EXTRA))
print(json.dumps({"out": out, "probes": calls["n"]}))
"""


def _fallback(lane, content: str, kind: str, key: str, **extra) -> dict:
    script = (f"CONTENT={content!r}\nKIND={kind!r}\nKEY={key!r}\nEXTRA={extra!r}\n"
              + _FALLBACK_CHILD)
    return lane.run_child(script)


def test_the_tools_fallback_path_through_the_proxy_keeps_the_kind(lane):
    got = _fallback(lane, f"Fixture record QKF{RUN}: fallback code C551.", "rule",
                    f"fixture-fallback-{RUN}")
    assert got["probes"] >= 2, got  # the fallback branch really ran
    assert got["out"]["success"] is True, got
    assert lane.kind_of(got["out"]["fact_ids"][0]) == ("rule", "caller")


def test_the_fallback_path_keeps_the_kind_on_a_named_profile(lane):
    got = _fallback(lane, f"Fixture record QKG{RUN}: routed fallback C552.", "prospective",
                    f"fixture-fallback-routed-{RUN}", profile_id=PROFILE)
    assert got["out"]["success"] is True, got
    rows = lane.rows("SELECT profile_id, memory_kind, memory_kind_source FROM atomic_facts "
                     "WHERE fact_id=?", (got["out"]["fact_ids"][0],))
    assert rows == [(PROFILE, "prospective", "caller")]


def test_forged_metadata_over_http_confirms_no_kind(lane):
    code, out = lane.daemon.request("POST", "/remember", {
        "content": f"Fixture record QKH{RUN}: forged code C553.",
        "idempotency_key": f"fixture-forged-http-{RUN}",
        "metadata": {"_slm_memory_kind": "rule"},
    })
    assert code == 200, out
    kind, source = lane.kind_of(out["fact_ids"][0])
    assert source not in ("caller", "user"), (kind, source)


def test_forged_metadata_through_the_proxy_confirms_no_kind(lane):
    # Anything that reaches the daemon through the tool's proxy with the
    # reserved key in its metadata (and no declared kind) confirms nothing.
    got = lane.run_child(
        "import json\n"
        "from superlocalmemory.mcp._daemon_proxy import choose_pool\n"
        f"out = choose_pool().store('Fixture record QKM{RUN}: forged proxy code C554.', "
        f"{{'_slm_memory_kind': 'rule', 'kind_hint': 'rule', "
        f"'idempotency_key': 'fixture-forged-proxy-{RUN}'}})\n"
        "print(json.dumps(out))\n"
    )
    assert got.get("ok") is True, got
    kind, source = lane.kind_of(got["fact_ids"][0])
    assert source not in ("caller", "user"), (kind, source)


def test_confirmed_rules_and_decisions_are_loaded_at_session_start(lane):
    # Loaded as STANDING facts (because they are confirmed), not merely as
    # recall hits: on a store this small every memory is a recall hit.
    out = lane.tool("session_init", query="unrelated starting topic")
    context = str(out.get("context", ""))
    for kind in ("rule", "decision"):
        marker = (f"fact_id={lane.saved[kind]['fact_ids'][0]}, "
                  f"source_type=standing-{kind}")
        assert marker in context, (kind, context[-1500:])


def _wait_run(lane, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        code, last = lane.daemon.request("GET", "/api/memory-kinds/status")
        assert code == 200, last
        runs = [r for r in [last.get("active_run") or {}, *(last.get("recent_runs") or [])]
                if r.get("run_id") == run_id]
        if runs and runs[0].get("status") in ("completed", "failed", "cancelled", "paused"):
            return runs[0]
        time.sleep(0.5)
    raise AssertionError(f"classification run {run_id} did not finish: {last}")


def test_a_reclassification_run_leaves_confirmed_kinds_alone(lane):
    code, out = lane.daemon.request("POST", "/api/memory-kinds/settings",
                                    {"enabled": True, "backend": "rules"})
    assert code == 200, out
    code, run = lane.daemon.request("POST", "/api/memory-kinds/backfill", {"mode": "refresh"})
    assert code == 200, run
    finished = _wait_run(lane, run["run_id"])
    assert finished["status"] == "completed", finished
    for kind in KINDS:
        for fid in lane.saved[kind]["fact_ids"]:
            assert lane.kind_of(fid) == (kind, "caller"), (kind, fid)


def test_every_fact_of_each_memory_carries_the_kind_once_enriched(lane):
    # Whatever enrichment adds to a memory inherits the caller's kind; nothing
    # it produced may carry a machine label instead. Enrichment speed depends
    # on machine load, which this test does not own: it waits a bounded time,
    # requires that enrichment really finished for at least one memory, and
    # checks the facts of every memory whose enrichment finished.
    done: dict[str, str] = {}
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline and len(done) < len(KINDS):
        for kind in KINDS:
            state = lane.rows("SELECT state FROM ingestion_operations WHERE operation_id=?",
                              (lane.saved[kind]["operation_id"],))
            if state and state[0][0] == "complete":
                done[kind] = state[0][0]
        time.sleep(1.0)
    assert done, "enrichment finished for none of the nine memories"
    stray = []
    for kind in done:
        fid = lane.saved[kind]["fact_ids"][0]
        memory = lane.rows("SELECT memory_id FROM atomic_facts WHERE fact_id=?", (fid,))[0][0]
        for other, mk, src in lane.rows(
                "SELECT fact_id, memory_kind, memory_kind_source FROM atomic_facts "
                "WHERE memory_id=?", (memory,)):
            if (mk, src) != (kind, "caller"):
                stray.append((kind, other, mk, src))
    assert stray == []


def test_declared_kinds_survive_a_daemon_restart(lane):
    lane.restart()
    for kind in KINDS:
        for fid in lane.saved[kind]["fact_ids"]:
            assert lane.kind_of(fid) == (kind, "caller"), (kind, fid)


def test_kind_filtered_recall_still_finds_a_memory_after_the_restart(lane):
    # Was intermittent (2 of 4 runs) while the kind filter only ran after
    # fusion; kind-filtered recall now also searches inside the kind
    # (retrieval/kind_scope; deterministic case: test_kind_recall_is_not_crowded_out).
    # Runs after the restart above (file order).
    out = lane.tool("recall", query=_content("decision"), kind="decision")
    # By now enrichment has split the memory into facts of its own, and recall
    # may show any of them: what must be found is the MEMORY, under its kind.
    def memory_of(fact_id: str) -> str:
        return lane.rows("SELECT memory_id FROM atomic_facts WHERE fact_id=?", (fact_id,))[0][0]

    found = {memory_of(item["fact_id"]) for item in out.get("results", [])}
    assert memory_of(lane.saved["decision"]["fact_ids"][0]) in found, out


def test_a_retry_with_the_same_key_after_a_restart_returns_the_same_memory(lane):
    # Runs after the restart above (file order).
    again = lane.tool("remember", content=_content("decision"), kind="decision",
                      tags="metadata-only-qtag,zeta", project="synthetic-kestrel",
                      idempotency_key=f"fixture-declared-decision-{RUN}")
    assert again.get("fact_ids") == lane.saved["decision"]["fact_ids"], again
