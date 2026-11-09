# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every MCP tool, once, over real stdio, against an isolated Mode A daemon.

Opt-in (slow; starts a daemon and a stdio MCP server): ::

    SLM_E2E=1 pytest tests/e2e/test_every_mcp_tool_end_to_end.py -m slow -s

or, for the full report without pytest: ::

    ~/.superlocalmemory/venv/bin/python -m tests.e2e.test_every_mcp_tool_end_to_end \
        --out ./slm-e2e-sweep

Nothing here touches the user's daemon, data or configuration: HOME, SLM_DATA_DIR,
the port and the caches are all under a temporary root. Models are read from the
user's HuggingFace cache through a symlink (offline; never written).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    import pytest
except ImportError:  # the runner script uses this module without pytest installed
    pytest = None

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PRODUCTION_PORTS = {8765, 8767}
PYTHON = os.environ.get("SLM_E2E_PYTHON") or sys.executable
CALL_TIMEOUT = 30.0

if pytest is not None:
    pytestmark = [
        pytest.mark.slow,
        pytest.mark.skipif(os.environ.get("SLM_E2E") != "1", reason="opt-in: set SLM_E2E=1"),
    ]


# --------------------------------------------------------------------------- isolation


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in PRODUCTION_PORTS:
            return port


def isolated_env(root: Path, port: int, profile: str | None = None) -> dict:
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
    home = root / "home"
    hf = root / "cache" / "hf"
    home.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    hf.mkdir(parents=True, exist_ok=True)
    real_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if real_hub.exists() and not (hf / "hub").exists():
        (hf / "hub").symlink_to(real_hub)
    real_mod = Path.home() / ".cache" / "huggingface" / "modules"
    if real_mod.exists() and not (hf / "modules").exists():
        shutil.copytree(real_mod, hf / "modules")
    env.update({
        "HOME": str(home), "PYTHONPATH": str(SRC),
        "SLM_DATA_DIR": str(root / "data"), "SLM_DAEMON_PORT": str(port),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(hf), "XDG_CACHE_HOME": str(root / "cache"),
        "SENTENCE_TRANSFORMERS_HOME": str(root / "cache" / "st"), "CI": "1",
        "SLM_DISABLE_LEGACY_PORT": "1", "SLM_NON_INTERACTIVE": "1",
        "SLM_DISABLE_HF_DOWNLOAD": "1", "TOKENIZERS_PARALLELISM": "false",
        "NO_PROXY": "127.0.0.1,localhost", "SLM_RATE_LIMIT_WRITE": "100000",
        "SLM_AGENT_ID": "claude_code",
    })
    if profile:
        env["SLM_MCP_PROFILE"] = profile
    return env


def slm_cli(env: dict, *argv: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, "-m", "superlocalmemory.cli.main", *argv], env=env,
                          capture_output=True, text=True, timeout=timeout, cwd=str(REPO))


class Daemon:
    def __init__(self, root: Path, port: int) -> None:
        self.root, self.port = root, port
        self.env = isolated_env(root, port)
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        log = (self.root / "daemon.log").open("ab")
        self.proc = subprocess.Popen(
            [PYTHON, "-m", "superlocalmemory.server.unified_daemon", "--start",
             f"--port={self.port}"], stdout=log, stderr=log, env=self.env, cwd=str(REPO),
            start_new_session=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            assert self.proc.poll() is None, self.log_tail()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health",
                                            timeout=3) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.5)
        raise AssertionError("daemon did not start\n" + self.log_tail())

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)  # our own process group only
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=20)

    def log_tail(self) -> str:
        p = self.root / "daemon.log"
        return p.read_text(errors="replace", encoding="utf-8")[-4000:] if p.exists() else ""


# --------------------------------------------------------------------------- stdio client


def _unwrap(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def parse(result) -> object:
    text = "".join(getattr(b, "text", "") for b in result.content)
    try:
        return json.loads(text)
    except Exception:
        return text


async def with_session(env: dict, fn):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=PYTHON, args=["-m", "superlocalmemory.cli.main", "mcp"], env=env, cwd=str(REPO))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


# --------------------------------------------------------------------------- the sweep

SEED = [
    "Atlas ferry leaves pier nine at dawn every Tuesday",
    "The Qualixar project uses AGPL licensing for SuperLocalMemory",
    "Varun prefers dark mode and short answers in the terminal",
    "Deploy runbook: run the migration, then restart the daemon, then smoke test",
    "Postgres is the primary database for the billing service",
    "The billing service runs on port 7412 behind nginx",
    "Priya owns the on-call rotation for the checkout team",
    "Release 4.2.3 fixes the recall latency regression in Mode A",
    "The ferry timetable changes in winter: pier nine closes at five",
    "Coffee order for the standup: two flat whites and one oat cappuccino",
    "Kubernetes cluster is named orca and lives in the eu-west region",
    "Always run the test suite before tagging a release",
    "The mobile app crash was caused by a null avatar url",
    "Marketing launch is planned for the second week of November",
    "Bounded loops stop when an independent gate passes",
    "The design review for the dashboard is on Thursday at three",
    "Ledger reconciliation fails when currency codes are lowercase",
    "Mesh peers share lock state through the local daemon",
    "Tokens must be masked as REDACTED in every log line",
    "The quarterly roadmap prioritises memory kinds and saved views",
]

# Tools that must not run live against the sandbox daemon in the generic pass.
SKIP = {
    "switch_profile": "run explicitly last (changes active profile)",
    "set_mode": "run explicitly (mode change)",
}


def default_for(name: str, spec: dict, ctx: dict):
    """A minimal valid value for one JSON-schema property."""
    if "default" in spec and spec["default"] is not None:
        return spec["default"]
    if "enum" in spec:
        return spec["enum"][0]
    t = spec.get("type")
    if t is None and "anyOf" in spec:
        for alt in spec["anyOf"]:
            if alt.get("type") != "null":
                return default_for(name, alt, ctx)
        return None
    if t == "string":
        n = name.lower()
        if n in ("query", "q", "text", "prompt", "question"):
            return "ferry pier nine"
        if n in ("content", "summary", "note", "message", "description", "body"):
            return "e2e sweep ferry note"
        if n in ("fact_id", "memory_id", "id"):
            return ctx["fact_id"]
        if "id" in n:
            return ctx.get(n, "e2e-" + n)
        if n in ("path", "file_path"):
            return str(ctx["tmpfile"])
        if n == "name":
            return "e2e_" + str(int(time.time()))
        return "e2e"
    if t == "integer":
        return spec.get("minimum", 1) if spec.get("minimum", 1) > 0 else 1
    if t == "number":
        return 0.5
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return None


def build_args(tool, ctx: dict, overrides: dict) -> dict:
    if tool.name in overrides:
        o = overrides[tool.name]
        return o(ctx) if callable(o) else dict(o)
    schema = tool.input_schema or {}
    props = schema.get("properties", {})
    args = {}
    for key in schema.get("required", []):
        args[key] = default_for(key, props.get(key, {}), ctx)
    return args


def _id_of(result) -> str | None:
    """Pull a fact/memory id out of a remember/search/list_recent result."""
    found: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for k in ("fact_id", "memory_id", "id"):
                v = o.get(k)
                if isinstance(v, str) and v:
                    found.append(v)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(result)
    return found[0] if found else None


def _nonempty(res) -> bool:
    if not isinstance(res, dict):
        return bool(res)
    for k in ("results", "memories", "facts", "items", "hits", "recent", "matches"):
        v = res.get(k)
        if isinstance(v, list):
            return len(v) > 0
        if isinstance(v, dict):
            return bool(v)
    d = res.get("data")
    if isinstance(d, dict):
        return _nonempty(d)
    if isinstance(d, list):
        return len(d) > 0
    return bool(res.get("count") or res.get("total") or res.get("result_count"))


async def call(session, name: str, args: dict, rec: list, expect_nonempty: bool = False,
               note: str = "", expect_error: bool = False):
    t0 = time.monotonic()
    row = {"tool": name, "args": json.dumps(args, default=str)[:160], "ok": True,
           "error": "", "ms": 0, "nonempty": None, "note": note, "result": ""}
    res = None
    try:
        r = await asyncio.wait_for(session.call_tool(name, args), CALL_TIMEOUT)
        res = parse(r)
        if r.is_error:
            row["ok"], row["error"] = False, str(res)[:200]
        elif isinstance(res, dict) and res.get("success") is False:
            row["ok"] = False
            row["error"] = str(res.get("error") or res.get("message") or res)[:200]
        if (row["ok"] and isinstance(res, dict) and res.get("error")
                and res.get("success") is not True and res.get("ok") is not True):
            row["ok"], row["error"] = False, "in-band error with no success flag: " + str(res.get("error"))[:160]
        if expect_nonempty:
            row["nonempty"] = _nonempty(res)
            if not row["nonempty"] and row["ok"]:
                row["ok"], row["error"] = False, "EMPTY result for a query matching seeded memories: " + str(res)[:140]
    except asyncio.TimeoutError:
        row["ok"], row["error"] = False, f"TIMEOUT >{CALL_TIMEOUT}s"
    except BaseException as exc:  # noqa: BLE001
        row["ok"], row["error"] = False, f"{type(_unwrap(exc)).__name__}: {_unwrap(exc)}"[:200]
    row["ms"] = int((time.monotonic() - t0) * 1000)
    row["result"] = str(res)[:1500 if name.startswith("slm_loop") else 300] if res is not None else ""
    if expect_error and not row["ok"] and not row["error"].startswith("TIMEOUT"):
        row["ok"], row["verdict"] = True, "refused as designed"
    else:
        row["verdict"] = "pass" if row["ok"] else "FAIL"
    rec.append(row)
    return res


def _ids_in(res) -> list[str]:
    found: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("fact_id", "memory_id", "id") and isinstance(v, str) and v:
                    found.append(v)
                elif k in ("fact_ids", "memory_ids", "ids") and isinstance(v, list):
                    found.extend(x for x in v if isinstance(x, str))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(res)
    return list(dict.fromkeys(found))


async def sweep(session, rec: list, listing: dict, home: Path) -> None:
    tools = (await session.list_tools()).tools
    by_name = {t.name: t for t in tools}
    listing["names"] = sorted(by_name)
    work = home / "work"  # inside the isolated HOME: the code graph refuses paths outside it
    (work / "repo").mkdir(parents=True, exist_ok=True)
    (work / "repo" / "ferry.py").write_text(
        "def board(pier):\n    return depart(pier)\n\ndef depart(pier):\n    return pier\n")
    (work / "doc.txt").write_text("SLM e2e document. The ferry leaves pier nine at dawn.\n")
    repo, doc = str(work / "repo"), str(work / "doc.txt")
    subprocess.run(["git", "init", "-q", repo], check=False, capture_output=True)
    subprocess.run(["git", "-C", repo, "add", "."], check=False, capture_output=True)
    subprocess.run(["git", "-C", repo, "-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit", "-qm", "init"], check=False, capture_output=True)

    # ---- seed ~20 memories
    ids: list[str] = []
    first_seed = None
    for text in SEED:
        r = await call(session, "remember", {"content": text, "tags": "e2e"}, rec, note="seed")
        if first_seed is None:
            first_seed = r
        ids.extend(_ids_in(r))
    listing["first_remember_result"] = json.dumps(first_seed, default=str)[:400]
    listing["seeded_ids"] = len(ids)
    await asyncio.sleep(10)  # let async ingest / enrichment settle
    c: dict = {"fid": ids[0] if ids else "missing", "fid2": ids[1] if len(ids) > 1 else "missing",
         "fid3": ids[2] if len(ids) > 2 else "missing"}

    async def go(name, args=None, nonempty=False, note="", expect_error=False):
        if name not in by_name:
            rec.append({"tool": name, "args": "", "ok": False, "error": "NOT LISTED by tools/list",
                        "ms": 0, "nonempty": None, "note": note})
            return None
        return await call(session, name, args if args is not None else {}, rec,
                          expect_nonempty=nonempty, note=note, expect_error=expect_error)

    # ---- retrieval at every depth
    for depth in (0, 1, 2):
        r = await go("recall", {"query": "ferry pier nine", "depth": depth}, True, f"depth={depth}")
    await go("recall", {"query": "billing service port", "limit": 5}, True, "default depth")
    await go("search", {"query": "ferry pier nine"}, True)
    await go("search", {"query": "billing database", "depth": 2}, True, "depth=2")
    r = await go("list_recent", {"limit": 5}, True)
    if c["fid"] == "missing":
        got = _ids_in(r)
        if got:
            c["fid"] = got[0]
    await go("recall_trace", {"query": "ferry pier nine"}, True)
    await go("fetch", {"fact_ids": [c["fid"]]}, True)
    await go("expand", {"memory_id": c["fid"]}, True)
    await go("session_init", {"project_path": repo, "query": "ferry pier nine"}, True)
    await go("get_status")
    await go("health")
    await go("get_version")
    await go("get_mode")
    await go("get_memory_summary")
    await go("get_memory_summary", {"kind": "day", "target": "today"}, note="explicit day")
    await go("memory_used")

    # ---- writes on a seeded memory
    await go("update_memory", {"fact_id": c["fid2"], "content": "Atlas ferry leaves pier nine at dawn (updated)"})
    await go("report_feedback", {"fact_id": c["fid"], "feedback": "relevant", "query": "ferry"})
    await go("report_outcome", {"memory_ids": c["fid"], "outcome": "success"})
    await go("observe", {"content": "e2e observation: user asked about the ferry"})
    await go("set_memory_kind", {"fact_id": c["fid"], "kind": "fact"})
    await go("memory_kinds_status")
    await go("review_memory_kinds")
    await go("confirm_memory_kinds", {"items": [{"fact_id": c["fid"], "kind": "fact"}]})
    await go("core_memory", {"action": "list"})

    # ---- mesh
    for n in ("mesh_summary", "mesh_peers", "mesh_inbox", "mesh_events", "mesh_status"):
        await go(n)
    await go("mesh_summary", {"summary": "e2e sweep session"}, note="with summary")
    await go("mesh_send", {"to": "broadcast", "message": "e2e hello"})
    await go("mesh_state", {"key": "e2e", "value": "1", "action": "set"})
    await go("mesh_state", {"key": "e2e", "action": "get"}, note="get")
    await go("mesh_lock", {"file_path": "/tmp/e2e-lock", "action": "query"})
    await go("mesh_wait", {"timeout_s": 1})

    # ---- optimize
    comp = await go("slm_compress", {"content": "alpha   beta\n\n\n\n   gamma\t\tdelta  epsilon   \n" * 300})
    ccr = None
    if isinstance(comp, dict):
        ccr = comp.get("ccr_id") or (comp.get("data") or {}).get("ccr_id")
    await go("slm_retrieve", {"ccr_id": ccr or "missing"}, note="id from slm_compress" if ccr else "NO ccr_id returned")
    await go("slm_cache_set", {"key": "e2e-key", "value": "e2e-value"})
    await go("slm_cache_get", {"key": "e2e-key"}, note="expects hit")
    await go("slm_optimize_stats")

    # ---- bounded loops (trivial gate: the seeded memory satisfies the query)
    lr = await go("slm_loop_run", {"name": "e2e-loop", "gate_query": "ferry pier nine", "max_iterations": 2,
                                    "max_wallclock_s": 10.0, "poll_interval_s": 0.5})
    await go("slm_loop_history", {"name": "e2e-loop"})
    run_id = None
    if isinstance(lr, dict):
        run_id = lr.get("run_id") or (lr.get("data") or {}).get("run_id")
    await go("slm_loop_show", {"run_id": run_id or "missing"}, note="id from slm_loop_run" if run_id else "NO run_id returned")

    # ---- views, corrections, assertions, skills, brain
    await go("manage_view", {"action": "create", "name": "e2e-view", "query": "ferry"})
    await go("manage_view", {"action": "rename", "name": "e2e-view", "new_name": "e2e-view2"}, note="rename")
    await go("manage_view", {"action": "rename", "name": "e2e-view2", "new_name": "e2e-view"}, note="rename back")
    await go("run_view", {"name": "e2e-view"}, True)
    await go("run_view", {}, note="no name: list views")
    await go("manage_view", {"action": "delete", "name": "e2e-view"}, note="delete")
    await go("list_corrections")
    lc = await go("list_corrections")
    case = (lc or {}).get("corrections", [{}])[0] if isinstance(lc, dict) and lc.get("corrections") else {}
    if case:
        await go("review_correction", {"case_id": case["case_id"], "action": "reject", "expected_version": case["version"]}, note="real case from update_memory")
    await go("review_correction", {"case_id": "e2e-none", "action": "reject", "expected_version": 1},
             note="unknown case: error expected", expect_error=True)
    await go("get_assertions")
    await go("reinforce_assertion", {"assertion_id": "e2e-none"}, note="unknown id: error expected", expect_error=True)
    await go("contradict_assertion", {"assertion_id": "e2e-none"}, note="unknown id: error expected", expect_error=True)
    await go("skill_health")
    await go("skill_lineage")
    await go("evolve_skill", {"skill_name": "e2e-skill"}, note="feature off by default", expect_error=True)
    await go("get_brain_evidence_status")
    await go("record_agent_experience", {"payload": {"experience_id": "e2e-exp-1", "profile_id": "default", "occurred_at": "2026-10-13T00:00:00+00:00", "task_class": "code", "project_scope": "project-digest", "route": {"harness": "claude", "provider": "anthropic", "model": "m", "effort": "high", "machine": "m"}, "verification": {"authority": "deterministic_gate", "evidence_digest": "a" * 64}, "producer_claim": "success", "terminal_status": "succeeded"}})
    rct = await go("record_cognitive_turn", {"payload": {"receipt_id": "e2e-turn-1", "task_id": "e2e-task-1", "profile_id": "default", "project_scope": "project-digest", "query_digest": "b" * 64, "fact_decisions": {c["fid"]: "used"}, "state": "open"}})
    rid = None
    if isinstance(rct, dict):
        rid = "e2e-turn-1" if rct.get("success") else None
    await go("finalize_cognitive_turn", {"receipt_id": rid or "e2e-turn-1", "outcome": {"authority": "deterministic_gate", "receipt_digest": "c" * 64, "reference": "gate-1"}},
             note="id from record_cognitive_turn" if rid else "NO receipt_id returned")
    await go("observe_bounded_loop_evidence", {"workspace": repo})
    await go("observe_bounded_loop_execution_learning", {"workspace": repo})
    await go("consolidate_cognitive")
    await go("get_soft_prompts")
    await go("log_tool_event", {"tool_name": "Read", "event_type": "invoke"})
    await go("settle_session_outcomes", {"session_id": "e2e-session"})

    # ---- maintenance / governance (non-destructive modes)
    for n in ("run_maintenance", "get_lifecycle_status", "get_retention_stats", "get_behavioral_patterns",
              "audit_trail", "consistency_check", "get_learned_patterns", "backup_status", "get_attribution",
              "list_failed_operations", "list_graph_stats"):
        await go(n)
    await go("resolve_operation", {"operation_id": "e2e-none", "action": "cancel"}, note="unknown id: error expected", expect_error=True)
    await go("correct_pattern", {"pattern_id": "e2e-none", "correction": "x"}, note="unknown id: error expected", expect_error=True)
    await go("compact_memories", {"dry_run": True})
    await go("quantize", {"dry_run": True})
    await go("reap_processes", {"dry_run": True})
    await go("set_retention_policy", {"cold_after_days": 90, "archive_after_days": 365})
    await go("set_mode", {"mode": "a"})
    await go("build_graph")

    # ---- code graph (tiny temp repo, never the user's code)
    await go("build_code_graph", {"repo_path": repo})
    await go("update_code_graph", {"repo_path": repo, "changed_files": "ferry.py"})
    await go("get_blast_radius", {"changed_files": "ferry.py"})
    await go("get_review_context", {"changed_files": "ferry.py"})
    await go("query_graph", {"pattern": "callers_of", "target": "depart"})
    await go("semantic_search_code", {"query": "ferry departure"})
    await go("find_large_functions")
    await go("list_flows")
    await go("get_flow", {"flow_name": "board"}, note="unknown flow", expect_error=True)
    await go("get_affected_flows", {"changed_files": "ferry.py"})
    await go("list_communities")
    await go("get_community", {"community_id": "0"}, note="unknown community", expect_error=True)
    await go("get_architecture_overview")
    await go("detect_changes")
    await go("refactor_preview", {"action": "rename", "target": "depart", "new_name": "leave"})
    await go("apply_refactor", {"action": "rename", "target": "depart", "new_name": "leave", "dry_run": True})
    await go("link_memory_to_code", {"fact_id": c["fid"], "code_entity": "board"})
    await go("code_memory_search", {"code_entity": "board"})
    await go("code_entity_history", {"code_entity": "board"})
    await go("enrich_blast_radius", {"changed_files": "ferry.py"})
    await go("code_stale_check", note="bridge off by default", expect_error=True)
    await go("prestage_context", {"query": "ferry pier nine"}, True)

    # ---- media (only listed while images are on)
    if "remember_media" in by_name:
        await go("media_status")
        await go("media_upload_link")
        await go("remember_document", {"path": doc})
        await go("get_media", {"media_id": "missing"}, note="unknown id", expect_error=True)
        await go("remember_media", {"path": doc}, note="non-image file", expect_error=True)

    # ---- destructive / last
    await go("forget", {"dry_run": True})
    await go("delete_memory", {"fact_id": c["fid3"]})
    await go("switch_profile", {"profile_id": "default"})
    await go("close_session", {})

    called = {r["tool"] for r in rec}
    listing["never_called"] = sorted(set(by_name) - called)
    for name in listing["never_called"]:  # generic fallback so no tool is skipped
        await go(name, build_args(by_name[name], {"fact_id": c["fid"], "tmpfile": doc}, {}),
                 note="generic fallback")
    shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- Claude Code hooks

#: (hook name as the plugin's hooks.json runs it, stdin payload builder, text that must appear
#: in stdout, or None when only a clean exit and parseable output are required)
JSON_HOOKS = {"auto_recall", "user_prompt_submit", "post_tool_outcome", "post_tool_async", "user_prompt_rehash", "stop_outcome"}


def hook_cases(repo: str) -> list[tuple[str, dict, str | None]]:
    base = {"session_id": "e2e-hook-session", "cwd": repo, "transcript_path": "",
            "permission_mode": "default"}
    prompt = "When does the ferry leave pier nine?"
    tool = {"tool_name": "Write", "tool_input": {"file_path": f"{repo}/ferry.py", "content": "x"},
            "tool_response": {"success": True}}
    return [
        ("mandate", {**base, "hook_event_name": "SessionStart", "source": "startup"}, None),
        ("start", {**base, "hook_event_name": "SessionStart", "source": "startup"}, None),
        ("checkpoint", {**base, "hook_event_name": "PostToolUse", **tool}, None),
        ("post_tool_outcome", {**base, "hook_event_name": "PostToolUse", **tool}, None),
        ("post_tool_async", {**base, "hook_event_name": "PostToolUse", **tool}, None),
        ("user_prompt_rehash", {**base, "hook_event_name": "UserPromptSubmit", "prompt": prompt}, None),
        ("topic_shift", {**base, "hook_event_name": "UserPromptSubmit", "prompt": prompt}, None),
        ("auto_recall", {**base, "hook_event_name": "UserPromptSubmit", "prompt": prompt}, "pier"),
        ("user_prompt_submit", {**base, "hook_event_name": "UserPromptSubmit", "prompt": prompt}, None),
        ("before_web", {**base, "hook_event_name": "PreToolUse", "tool_name": "WebSearch",
                        "tool_input": {"query": "ferry pier nine timetable"}}, None),
        ("stop", {**base, "hook_event_name": "Stop", "stop_hook_active": False}, None),
        ("stop_outcome", {**base, "hook_event_name": "Stop", "stop_hook_active": False}, None),
    ]


def run_hooks(env: dict, repo: str) -> list[dict]:
    """Invoke each hook the way ``slm-run hook <name>`` does: ``slm hook <name>`` with JSON on stdin."""
    out = []
    for name, payload, must_contain in hook_cases(repo):
        t0 = time.monotonic()
        row = {"hook": name, "exit": None, "ms": 0, "stdout": "", "stderr": "", "ok": False, "why": ""}
        try:
            r = subprocess.run([PYTHON, "-m", "superlocalmemory.cli.main", "hook", name], env=env,
                               input=json.dumps(payload), capture_output=True, text=True, timeout=60,
                               cwd=str(REPO))
            row.update(exit=r.returncode, stdout=r.stdout[:600], stderr=r.stderr[-300:])
            why = []
            if r.returncode != 0:
                why.append(f"exit {r.returncode}")
            text = r.stdout.strip()
            if text and name in JSON_HOOKS:
                try:
                    json.loads(text)
                except ValueError:
                    why.append("stdout is not JSON")
            if must_contain and must_contain.lower() not in r.stdout.lower():
                why.append(f"stdout does not contain {must_contain!r} for a prompt matching a seeded memory")
            row["ok"], row["why"] = not why, "; ".join(why)
        except subprocess.TimeoutExpired:
            row["why"] = "TIMEOUT >60s"
        row["ms"] = int((time.monotonic() - t0) * 1000)
        out.append(row)
    return out


def run_sweep(out_dir: Path, profile: str | None = None, keep: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir / "root"
    port = free_port()
    daemon = Daemon(root, port)
    slm_cli(daemon.env, "mode", "a")
    daemon.start()
    rec: list = []
    listing: dict = {"port": port}
    hooks: list = []
    try:
        env_full = isolated_env(root, port, None)
        env_whole = isolated_env(root, port, "whole")

        async def list_only(s):
            return [t.name for t in (await s.list_tools()).tools]

        listing["default_profile_tools"] = asyncio.run(with_session(env_full, list_only))
        asyncio.run(with_session(env_whole, lambda s: sweep(s, rec, listing, root / "home")))
        hook_repo = root / "home" / "work" / "hookrepo"
        hook_repo.mkdir(parents=True, exist_ok=True)
        hooks = run_hooks(env_full, str(hook_repo))
    finally:
        daemon.stop()
        listing["daemon_log_tail"] = daemon.log_tail()[-1500:]
    return {"rows": rec, "listing": listing, "hooks": hooks, "root": str(root)}


def test_every_mcp_tool_end_to_end(tmp_path) -> None:
    out = run_sweep(tmp_path)
    failures = [r for r in out["rows"] if not r["ok"]]
    bad_hooks = [h for h in out["hooks"] if not h["ok"]]
    assert not failures and not bad_hooks, json.dumps([failures, bad_hooks], indent=1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "slm-e2e-sweep"))
    a = ap.parse_args()
    res = run_sweep(Path(a.out))
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"calls": len(res["rows"]),
                      "failures": sum(not r["ok"] for r in res["rows"])}))
