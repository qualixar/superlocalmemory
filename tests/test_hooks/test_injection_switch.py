"""Issue #297 (question 5): ``injection.enabled`` and ``auto_invoke.enabled`` are real, settable switches.

``injection.enabled`` was saved to config.json but read by nothing, and ``slm config set`` refused
both keys (``DISALLOWED_KEY``) although ``AutoInvokeConfig`` documents disabling it there. With
``injection.enabled`` false, SLM adds no memories to an agent's context on its own: no session-start
context (Claude and Codex), no per-prompt recall, no before-web recall, no session_init or
prestage memories. Explicit recall and search still work.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.cli.commands import cmd_config, cmd_session_context
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.hooks import hook_handlers, rules_engine


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return tmp_path


def _set(key: str, value: str, *, as_json: bool = False) -> None:
    cmd_config(Namespace(action="set", key=key, value=value, json=as_json))


def _injection_off(data_dir: Path) -> None:
    (data_dir / "config.json").write_text(json.dumps({"injection": {"enabled": False}}))


# -- slm config set --------------------------------------------------------------------------------

@pytest.mark.parametrize("key,section", [("injection.enabled", "injection"),
                                         ("auto_invoke.enabled", "auto_invoke")])
def test_config_set_turns_the_switch_off_and_the_loader_sees_it(data_dir, key, section):
    _set(key, "false")
    raw = json.loads((data_dir / "config.json").read_text())
    assert raw[section]["enabled"] is False
    assert getattr(SLMConfig.load(), section).enabled is False
    _set(key, "true")
    assert getattr(SLMConfig.load(), section).enabled is True


@pytest.mark.parametrize("key", ["injection.enabled", "auto_invoke.enabled"])
def test_config_set_refuses_a_non_boolean(data_dir, key, capsys):
    with pytest.raises(SystemExit) as stop:
        _set(key, "sometimes", as_json=True)
    assert stop.value.code == 1
    assert "INVALID_VALUE" in capsys.readouterr().out
    assert not (data_dir / "config.json").exists()


# -- one switch, every automatic surface ----------------------------------------------------------

def test_the_rules_say_no_automatic_recall_when_injection_is_off(data_dir):
    _injection_off(data_dir)
    rules = rules_engine.load()
    assert not rules.should_recall("session_start")
    assert not rules.should_recall("every_prompt")
    assert not rules.should_recall("before_web")
    assert not rules.per_prompt_recall_on()


def test_the_rules_keep_automatic_recall_on_by_default(data_dir):
    rules = rules_engine.load()
    assert rules.should_recall("session_start") and rules.should_recall("before_web")


@pytest.mark.parametrize("action,module", [("before_web", "before_web_hook"),
                                           ("user_prompt_submit", "user_prompt_hook"),
                                           ("auto_recall", "auto_recall_hook")])
def test_recall_hooks_inject_nothing_when_injection_is_off(data_dir, monkeypatch, capsys,
                                                           action, module):
    _injection_off(data_dir)
    ran: list[str] = []
    import importlib

    mod = importlib.import_module(f"superlocalmemory.hooks.{module}")
    monkeypatch.setattr(mod, "main", lambda: ran.append(action) or 0)
    monkeypatch.setattr("superlocalmemory.hooks.payload_keys.normalize_stdin", lambda: None)
    with pytest.raises(SystemExit):
        hook_handlers.handle_hook(action)
    assert ran == []
    assert capsys.readouterr().out == "{}"


def test_before_web_still_runs_when_injection_is_on(data_dir, monkeypatch):
    ran: list[str] = []
    from superlocalmemory.hooks import before_web_hook

    monkeypatch.setattr(before_web_hook, "main", lambda: ran.append("ran") or 0)
    monkeypatch.setattr("superlocalmemory.hooks.payload_keys.normalize_stdin", lambda: None)
    try:
        hook_handlers.handle_hook("before_web")
    except SystemExit:
        pass
    assert ran == ["ran"]


def _session_context(monkeypatch, data_dir) -> list[str]:
    (data_dir / "memory.db").write_bytes(b"")
    reads: list[str] = []

    class _NoRead:
        def __enter__(self):
            reads.append("read")
            raise RuntimeError("stop here")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("superlocalmemory.storage.memory_write.memory_read", lambda *_a, **_k: _NoRead())
    cmd_session_context(Namespace(json=False, full=False, project="p", query=""))
    return reads


def test_session_start_context_is_not_read_when_injection_is_off(data_dir, monkeypatch, capsys):
    _injection_off(data_dir)
    assert _session_context(monkeypatch, data_dir) == []
    assert capsys.readouterr().out == ""


def test_session_start_context_is_not_read_when_session_start_recall_is_off(data_dir, monkeypatch):
    (data_dir / "config.json").write_text(json.dumps(
        {"rules_version": 2, "rules": {"auto_recall": {"on_session_start": False}}}))
    assert _session_context(monkeypatch, data_dir) == []


def test_session_start_context_is_read_by_default(data_dir, monkeypatch):
    assert _session_context(monkeypatch, data_dir) == ["read"]


def test_prestage_returns_no_memories_when_injection_is_off(data_dir):
    from superlocalmemory.mcp.tools_context import prestage_context

    _injection_off(data_dir)
    called: list[str] = []
    out = prestage_context("anything", recall_fn=lambda *a, **k: called.append("x") or [])
    assert called == [] and out["memories"] == [] and out.get("error") == "injection_disabled"


def test_explicit_recall_is_not_a_switchable_injection(data_dir):
    """The switch is for what SLM adds on its own; an explicit recall is not gated."""
    _injection_off(data_dir)
    assert "recall" not in hook_handlers.INJECTING_ACTIONS
    assert sys.modules["superlocalmemory.hooks.hook_handlers"] is hook_handlers


def test_auto_invoke_change_says_a_restart_is_needed(data_dir, capsys):
    _set("auto_invoke.enabled", "false", as_json=True)
    assert json.loads(capsys.readouterr().out)["data"]["restart_required"] is True
    _set("injection.enabled", "false", as_json=True)
    assert "restart_required" not in json.loads(capsys.readouterr().out)["data"]
