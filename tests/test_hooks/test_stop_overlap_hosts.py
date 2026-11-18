# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3 open item: Codex and Hermes send the reply overlap too, as Claude Code's
Stop does (Codex Stop carries ``last_assistant_message``; Hermes ``post_llm_call``
carries ``assistant_response``)."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

from superlocalmemory.hooks import codex_handlers, stop_outcome_hook

PLUGIN = Path(__file__).resolve().parents[2] / "plugin-src" / "hermes"


def _capture(monkeypatch):
    from superlocalmemory.hooks import event_delivery as ed

    sent: list = []
    monkeypatch.setattr(ed, "active_profile", lambda: "default")
    monkeypatch.setattr(ed, "deliver", lambda events, **kw: sent.append(list(events)) or 1)
    monkeypatch.setattr(stop_outcome_hook, "overlap_events",
                        lambda payload, sid, pid: [{"kind": "overlap", "sid": sid}]
                        if payload.get("last_assistant_message") else [])
    monkeypatch.setattr(stop_outcome_hook, "_cleanup_session_state", lambda sid: None)
    return sent


def test_settle_sends_the_overlap_before_the_finalize(monkeypatch) -> None:
    sent = _capture(monkeypatch)
    out = stop_outcome_hook.settle({"session_id": "s1", "last_assistant_message": "done"})
    assert out == "finalize_1"
    (events,) = sent
    assert events[0] == {"kind": "overlap", "sid": "s1"} and len(events) == 2
    assert stop_outcome_hook.settle({"session_id": ""}) == "no_session"


def test_the_installed_codex_stop_settles_its_turn(monkeypatch) -> None:
    sent = _capture(monkeypatch)
    payload = {"session_id": "codex-1", "cwd": "/tmp", "last_assistant_message": "fixed it"}
    monkeypatch.setattr(codex_handlers, "_codex_payload", lambda: payload)
    monkeypatch.setattr(codex_handlers, "_apply_codex_session", lambda p: "codex-1")
    bodies: list = []
    codex_handlers.hook_codex_stop(lambda deadline: bodies.append(deadline))
    assert len(bodies) == 1
    assert sent and sent[0][0] == {"kind": "overlap", "sid": "codex-1"}


def test_a_failing_checkpoint_still_settles(monkeypatch) -> None:
    sent = _capture(monkeypatch)
    monkeypatch.setattr(codex_handlers, "_codex_payload",
                        lambda: {"session_id": "codex-2", "last_assistant_message": "x"})
    monkeypatch.setattr(codex_handlers, "_apply_codex_session", lambda p: "codex-2")

    def broken(deadline):
        raise RuntimeError("checkpoint failed")

    codex_handlers.hook_codex_stop(broken)
    assert sent


def test_overlap_only_never_finalizes(monkeypatch) -> None:
    sent = _capture(monkeypatch)
    assert stop_outcome_hook.overlap_only({"session_id": "h1", "last_assistant_message": "a"}) \
        == "overlap_1"
    assert sent == [[{"kind": "overlap", "sid": "h1"}]]
    assert stop_outcome_hook.overlap_only({"session_id": "h1"}) == "no_overlap"
    assert stop_outcome_hook.overlap_only({}) == "no_session"


def _hermes(mode="local"):
    spec = importlib.util.spec_from_file_location("slm_hermes_overlap", PLUGIN / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    class Context:
        def call_mcp(self, _server, tool, arguments, timeout=0):
            return {"ok": True}

        def get_config(self, key, default):
            return default

    plugin = module.SlmHermesPlugin(Context())
    plugin._mode = lambda: mode
    return module, plugin


def test_hermes_sends_the_reply_to_the_local_slm_only(monkeypatch) -> None:
    module, plugin = _hermes()
    runs: list = []
    done = threading.Event()

    def fake_run(argv, **kw):
        runs.append((argv, json.loads(kw["input"])))
        done.set()

    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/slm")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    plugin.post_llm_call(session_id="h-7", user_message="q", assistant_response="the answer")
    assert done.wait(5)
    (argv, payload), = runs
    assert argv == ["/bin/slm", "hook", "overlap"]
    assert payload == {"session_id": "h-7", "last_assistant_message": "the answer"}


def test_hermes_sends_nothing_remote_or_without_a_reply(monkeypatch) -> None:
    for mode, reply, sid in (("remote", "an answer", "h"), ("local", "  ", "h"),
                             ("local", "an answer", "")):
        module, plugin = _hermes(mode)
        calls: list = []
        monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/slm")
        monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: calls.append(a))
        plugin._overlap({"session_id": sid, "assistant_response": reply})
        assert calls == []
