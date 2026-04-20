# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.3, §4.4

"""Tests for UserPromptSubmit + PostToolUse async hook handlers.

Validates LLD-01 §6.3 / §6.4 test matrix:
  - UserPromptSubmit returns Claude Code April-2026 envelope shape on hit.
  - UserPromptSubmit returns ``{}`` (empty JSON) on miss / error / bad input.
  - UserPromptSubmit NEVER raises to Claude Code.
  - PostToolUse returns ``{"async": true}`` unconditionally.
  - PostToolUse swallows all errors.
  - PostToolUse uses stdlib urllib (verified via import / grep).
  - Both hooks use stdlib only — no httpx, no fastapi, no torch.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

from superlocalmemory.core import context_cache as cc
from superlocalmemory.core import security_primitives as sp
from superlocalmemory.hooks import user_prompt_hook, post_tool_async_hook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    slm_home = tmp_path / ".superlocalmemory"
    slm_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sp, "_install_token_path",
                        lambda: slm_home / ".install_token")
    monkeypatch.setenv("HOME", str(tmp_path))
    # Force ContextCache and read_entry_fast defaults to look inside our
    # sandbox without relying on Path.home() at call time.
    monkeypatch.setattr(cc, "CACHE_DB_DEFAULT",
                         slm_home / "active_brain_cache.db")
    return slm_home


@pytest.fixture
def seeded_cache(home: Path) -> Path:
    """Seed a cache DB with one entry so hooks can exercise the hit path."""
    cache = cc.ContextCache(db_path=home / "active_brain_cache.db",
                              home_dir=home)
    try:
        cache.upsert(cc.CacheEntry(
            session_id="sess-hit",
            topic_sig="abcd1234deadbeef",
            content="seeded memory bullet",
            fact_ids=["f1"],
            provenance="tool_observation",
            computed_at=int(time.time()),
        ))
    finally:
        cache.close()
    return home / "active_brain_cache.db"


def _run_hook(hook_main, stdin_text: str,
              monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    """Invoke ``hook_main()`` with mocked stdin/stdout. Returns (rc, stdout)."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    rc = hook_main()
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# UserPromptSubmit — envelope shape
# ---------------------------------------------------------------------------


def test_user_prompt_returns_envelope_on_hit(
    home: Path, seeded_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Compute the topic_sig the hook will derive, then re-seed with it.
    from superlocalmemory.core.topic_signature import compute_topic_signature
    prompt = "please refactor the context cache writer into smaller functions"
    sig = compute_topic_signature(prompt)
    cache = cc.ContextCache(db_path=home / "active_brain_cache.db",
                              home_dir=home)
    try:
        cache.upsert(cc.CacheEntry(
            session_id="sess-envelope",
            topic_sig=sig,
            content="prior work: split ContextCache into writer/reader",
            fact_ids=["f42"],
            provenance="tool_observation",
            computed_at=int(time.time()),
        ))
    finally:
        cache.close()

    payload = json.dumps({"session_id": "sess-envelope", "prompt": prompt})
    rc, out = _run_hook(user_prompt_hook.main, payload, monkeypatch)
    assert rc == 0
    parsed = json.loads(out)
    assert "hookSpecificOutput" in parsed
    inner = parsed["hookSpecificOutput"]
    assert inner["hookEventName"] == "UserPromptSubmit"
    assert "prior work" in inner["additionalContext"]
    # SEC-v2-01: injected context must be wrapped in untrusted-boundary
    # markers so the downstream LLM can refuse embedded instructions.
    ac = inner["additionalContext"]
    assert "[BEGIN UNTRUSTED SLM CONTEXT" in ac
    assert "[END UNTRUSTED SLM CONTEXT]" in ac
    # The real content must sit between the markers, not before/after.
    begin_idx = ac.index("[BEGIN UNTRUSTED SLM CONTEXT")
    end_idx = ac.index("[END UNTRUSTED SLM CONTEXT]")
    assert begin_idx < ac.index("prior work") < end_idx


def test_user_prompt_returns_empty_on_miss(
    home: Path, seeded_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({"session_id": "sess-never",
                           "prompt": "totally unrelated prompt text here"})
    rc, out = _run_hook(user_prompt_hook.main, payload, monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_empty_on_broken_payload(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc, out = _run_hook(user_prompt_hook.main,
                         "this is not json at all",
                         monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_empty_on_missing_fields(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No session_id — must not attempt the lookup.
    rc, out = _run_hook(user_prompt_hook.main,
                         json.dumps({"prompt": "anything"}),
                         monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_empty_on_missing_db(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No cache DB created — hook must fail-open, not crash.
    rc, out = _run_hook(user_prompt_hook.main,
                         json.dumps({"session_id": "x", "prompt": "hello"}),
                         monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with NO home directory at all, hook must return a valid JSON
    # and exit 0.
    monkeypatch.setenv("HOME", "/definitely/not/a/real/path/xyz123")
    rc, out = _run_hook(user_prompt_hook.main,
                         json.dumps({"session_id": "s", "prompt": "p"}),
                         monkeypatch)
    assert rc == 0
    # Always valid JSON.
    json.loads(out)


def test_user_prompt_under_budget(
    home: Path, seeded_cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ballpark wall-clock budget check for the Python fallback path."""
    payload = json.dumps({"session_id": "sess-hit",
                           "prompt": "refactor context cache module"})
    # Warm-up
    _run_hook(user_prompt_hook.main, payload, monkeypatch)
    start = time.perf_counter()
    for _ in range(5):
        _run_hook(user_prompt_hook.main, payload, monkeypatch)
    avg = (time.perf_counter() - start) / 5
    # Budget is 50 ms p95 in prod; we allow a relaxed 100 ms in CI noise.
    assert avg < 0.1, f"avg {avg*1000:.2f} ms"


# ---------------------------------------------------------------------------
# UserPromptSubmit — stdlib-only (static grep)
# ---------------------------------------------------------------------------


def test_user_prompt_hook_has_no_heavy_imports() -> None:
    """LLD-01 R8 / R9: no httpx, no torch, no fastapi, no urllib even."""
    src = Path(user_prompt_hook.__file__).read_text(encoding="utf-8")
    for banned in ("import httpx", "from httpx",
                   "import torch", "from torch",
                   "import fastapi", "from fastapi",
                   "import requests", "from requests",
                   "sentence_transformers", "sqlalchemy",
                   "anthropic"):
        assert banned not in src, f"banned import found: {banned}"


# ---------------------------------------------------------------------------
# PostToolUse async hook
# ---------------------------------------------------------------------------


def test_post_tool_async_returns_async_true(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No token file, no daemon → must still emit {"async": true}.
    payload = json.dumps({
        "session_id": "sess-1", "tool_name": "Read",
        "tool_input": {"file_path": "/x/y.txt"},
        "tool_response": "hello",
    })
    rc, out = _run_hook(post_tool_async_hook.main, payload, monkeypatch)
    assert rc == 0
    assert json.loads(out) == {"async": True}


def test_post_tool_async_swallows_bad_payload(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc, out = _run_hook(post_tool_async_hook.main,
                         "not valid json", monkeypatch)
    assert rc == 0
    assert json.loads(out) == {"async": True}


def test_post_tool_async_swallows_daemon_down(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Create token so the hook attempts to POST, then route the POST to a
    # dead port. Must still emit {"async": true}.
    sp.ensure_install_token()
    monkeypatch.setattr(post_tool_async_hook, "DAEMON_URL",
                         "http://127.0.0.1:1")  # reserved port
    rc, out = _run_hook(
        post_tool_async_hook.main,
        json.dumps({"session_id": "s", "tool_name": "Bash",
                     "tool_input": {"command": "ls"},
                     "tool_response": "file1\nfile2"}),
        monkeypatch,
    )
    assert rc == 0
    assert json.loads(out) == {"async": True}


def test_post_tool_async_sends_install_token(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = sp.ensure_install_token()
    captured: dict = {}

    def fake_urlopen(req, timeout=0.5):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["token"] = req.headers.get("X-slm-hook-token") or \
                            req.headers.get("X-Slm-Hook-Token")
        captured["body"] = req.data
        # Return a minimal response-like object.
        class _R:
            def read(self) -> bytes:
                return b"{}"
            def close(self) -> None:
                return None
        return _R()

    # Patch urllib inside the module.
    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    rc, out = _run_hook(
        post_tool_async_hook.main,
        json.dumps({"session_id": "s", "tool_name": "Read",
                     "tool_input": {"file_path": "/x"},
                     "tool_response": "some content"}),
        monkeypatch,
    )
    assert rc == 0
    assert json.loads(out) == {"async": True}
    assert captured["token"] == token
    # Body was JSON
    assert isinstance(captured["body"], bytes)
    body = json.loads(captured["body"].decode("utf-8"))
    assert body["session_id"] == "s"
    assert body["tool_name"] == "Read"


# ---------------------------------------------------------------------------
# PostToolUse — stdlib-only (static grep for httpx)
# ---------------------------------------------------------------------------


def test_post_tool_async_uses_urllib_not_httpx() -> None:
    src = Path(post_tool_async_hook.__file__).read_text(encoding="utf-8")
    for banned in ("import httpx", "from httpx",
                   "import requests", "from requests",
                   "import torch", "import fastapi"):
        assert banned not in src, f"banned import found: {banned}"
    # And urllib MUST be present.
    assert "urllib" in src


# ---------------------------------------------------------------------------
# Dispatch integration — slm hook user_prompt_submit / post_tool_async
# ---------------------------------------------------------------------------


def test_hook_dispatcher_knows_user_prompt_submit(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.hooks import hook_handlers
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    with pytest.raises(SystemExit) as excinfo:
        hook_handlers.handle_hook("user_prompt_submit")
    assert excinfo.value.code == 0


def test_hook_dispatcher_knows_post_tool_async(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.hooks import hook_handlers
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    with pytest.raises(SystemExit) as excinfo:
        hook_handlers.handle_hook("post_tool_async")
    assert excinfo.value.code == 0
    assert json.loads(buf.getvalue()) == {"async": True}


# ---------------------------------------------------------------------------
# Defensive-branch coverage — various malformed payloads
# ---------------------------------------------------------------------------


def test_user_prompt_empty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, out = _run_hook(user_prompt_hook.main, "", monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_non_dict_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, out = _run_hook(user_prompt_hook.main, "[1, 2, 3]", monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_non_string_session(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, out = _run_hook(user_prompt_hook.main,
                         json.dumps({"session_id": 42, "prompt": "hi"}),
                         monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_non_string_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, out = _run_hook(user_prompt_hook.main,
                         json.dumps({"session_id": "s", "prompt": 123}),
                         monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_empty_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, out = _run_hook(user_prompt_hook.main,
                         json.dumps({"session_id": "s", "prompt": ""}),
                         monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_user_prompt_swallows_sig_error(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If compute_topic_signature raises, hook emits {} and exits 0."""
    import superlocalmemory.core.topic_signature as _ts

    def boom(_text: str) -> str:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(_ts, "compute_topic_signature", boom)
    rc, out = _run_hook(user_prompt_hook.main,
                         json.dumps({"session_id": "s", "prompt": "hello"}),
                         monkeypatch)
    assert rc == 0
    assert json.loads(out) == {}


def test_post_tool_async_empty_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, out = _run_hook(post_tool_async_hook.main, "", monkeypatch)
    assert rc == 0
    assert json.loads(out) == {"async": True}


def test_post_tool_async_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc, out = _run_hook(post_tool_async_hook.main, "[1, 2]", monkeypatch)
    assert rc == 0
    assert json.loads(out) == {"async": True}


def test_post_tool_async_without_token(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Token file absent → hook must still emit the envelope and never touch
    # the network.
    token_path = home / ".install_token"
    if token_path.exists():
        token_path.unlink()
    called = {"count": 0}

    def fake_urlopen(*args, **kwargs):  # pragma: no cover — must not run
        called["count"] += 1
        raise AssertionError("urlopen should not be called without token")

    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    rc, out = _run_hook(
        post_tool_async_hook.main,
        json.dumps({"session_id": "s", "tool_name": "Read"}),
        monkeypatch,
    )
    assert rc == 0
    assert json.loads(out) == {"async": True}
    assert called["count"] == 0


def test_post_tool_async_summarize_string_input(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sp.ensure_install_token()
    captured: dict = {}

    def fake_urlopen(req, timeout=0.5):  # noqa: ARG001
        captured["body"] = req.data

        class _R:
            def read(self) -> bytes:
                return b"{}"
            def close(self) -> None:
                return None
        return _R()

    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    rc, out = _run_hook(
        post_tool_async_hook.main,
        json.dumps({"session_id": "s", "tool_name": "Bash",
                     "tool_input": "plain string input",
                     "tool_response": None}),
        monkeypatch,
    )
    assert rc == 0
    body = json.loads(captured["body"])
    assert body["input_summary"] == "plain string input"
    assert body["output_summary"] == ""


def test_post_tool_async_truncates_large_output(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sp.ensure_install_token()
    captured: dict = {}

    def fake_urlopen(req, timeout=0.5):  # noqa: ARG001
        captured["body"] = req.data
        class _R:
            def read(self) -> bytes:
                return b"{}"
            def close(self) -> None:
                return None
        return _R()

    import urllib.request as _ur
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    big = "y" * 20000
    rc, out = _run_hook(
        post_tool_async_hook.main,
        json.dumps({"session_id": "s", "tool_name": "X",
                     "tool_input": {"arg": "a" * 5000},
                     "tool_response": big}),
        monkeypatch,
    )
    assert rc == 0
    body = json.loads(captured["body"])
    assert len(body["output_summary"]) <= 4000
    assert len(body["input_summary"]) <= 2000
