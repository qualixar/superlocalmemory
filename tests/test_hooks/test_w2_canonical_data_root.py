# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""W2 contracts for canonical hook and process-semaphore state paths."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from superlocalmemory.core import ram_lock
from superlocalmemory.hooks import (
    _outcome_common,
    auto_recall_hook,
    claude_code_hooks,
    topic_shift_hook,
)

_ALIASES = ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")


def _select_only(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    root: Path,
) -> None:
    for name in _ALIASES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(alias, str(root))


@pytest.mark.parametrize("alias", _ALIASES)
def test_hot_hook_durable_paths_honor_every_alias(
    alias: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / alias.lower()
    _select_only(monkeypatch, alias, selected)

    assert _outcome_common.slm_home() == selected.resolve()
    assert _outcome_common.memory_db_path() == selected.resolve() / "memory.db"
    assert auto_recall_hook._get_queue_db_path() == (
        selected.resolve() / "recall_queue.db"
    )


def test_topic_state_is_namespaced_by_selected_root_in_one_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    _select_only(monkeypatch, "SLM_DATA_DIR", first)
    first_path = topic_shift_hook.state_path("same-session")
    _select_only(monkeypatch, "SLM_DATA_DIR", second)
    second_path = topic_shift_hook.state_path("same-session")

    assert first_path != second_path
    assert Path(first_path).parent.resolve() == Path(second_path).parent.resolve()
    assert Path(first_path).parent.resolve() == Path(tempfile.gettempdir()).resolve()


def test_topic_log_follows_root_after_environment_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setattr(topic_shift_hook, "_LOG_ENABLED", True)

    _select_only(monkeypatch, "SLM_DATA_DIR", first)
    topic_shift_hook._log_decision(
        "session-a", ["alpha"], [], -1, False, "alpha prompt"
    )
    _select_only(monkeypatch, "SLM_DATA_DIR", second)
    topic_shift_hook._log_decision(
        "session-b", ["beta"], [], -1, False, "beta prompt"
    )

    assert (first / "logs" / "topic-shift.log").exists()
    assert (second / "logs" / "topic-shift.log").exists()


def test_claude_hook_state_and_temp_markers_switch_roots_without_moving_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    original_settings = claude_code_hooks.CLAUDE_SETTINGS

    _select_only(monkeypatch, "SLM_DATA_DIR", first)
    first_version_dir = claude_code_hooks._version_dir()
    first_defs = claude_code_hooks._hook_definitions(include_gate=True)
    first_namespace = claude_code_hooks._root_namespace()

    _select_only(monkeypatch, "SLM_DATA_DIR", second)
    second_version_dir = claude_code_hooks._version_dir()
    second_defs = claude_code_hooks._hook_definitions(include_gate=True)
    second_namespace = claude_code_hooks._root_namespace()

    first_gate = first_defs["PreToolUse"][0]["hooks"][0]["command"]
    second_gate = second_defs["PreToolUse"][0]["hooks"][0]["command"]
    first_start = first_defs["SessionStart"][1]["hooks"][0]["command"]
    second_start = second_defs["SessionStart"][1]["hooks"][0]["command"]
    first_init = first_defs["PostToolUse"][0]["hooks"][0]["command"]
    second_init = second_defs["PostToolUse"][0]["hooks"][0]["command"]
    first_stop = first_defs["Stop"][0]["hooks"][0]["command"]
    second_stop = second_defs["Stop"][0]["hooks"][0]["command"]

    assert first_version_dir == first.resolve() / "hooks"
    assert second_version_dir == second.resolve() / "hooks"
    assert first_gate != second_gate
    assert first_start != second_start
    assert all(
        first_namespace in command
        for command in (first_gate, first_start, first_init, first_stop)
    )
    assert all(
        second_namespace in command
        for command in (second_gate, second_start, second_init, second_stop)
    )
    assert claude_code_hooks.CLAUDE_SETTINGS == original_settings
    assert claude_code_hooks.CLAUDE_SETTINGS == Path.home() / ".claude" / "settings.json"


def test_claude_hook_metadata_writes_only_to_selected_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected"
    legacy = tmp_path / "legacy"
    hook = tmp_path / "hook"
    settings = tmp_path / ".claude" / "settings.json"
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SL_MEMORY_PATH", str(legacy))
    monkeypatch.setenv("SLM_HOME", str(hook))
    monkeypatch.setattr(claude_code_hooks, "CLAUDE_SETTINGS", settings)

    installed = claude_code_hooks.install_hooks(include_gate=True)

    assert installed["success"] is True
    assert (selected / "hooks" / ".version").read_text() == (
        claude_code_hooks.HOOKS_VERSION
    )
    assert settings.exists()
    assert not legacy.exists()
    assert not hook.exists()

    removed = claude_code_hooks.remove_hooks()

    assert removed["success"] is True
    assert not (selected / "hooks" / ".version").exists()
    assert (selected / "hooks" / ".hooks-disabled").exists()


def test_ram_semaphore_switches_roots_in_one_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Memory:
        available = 16 * 1024 * 1024 * 1024

    monkeypatch.setattr(ram_lock.psutil, "virtual_memory", lambda: _Memory())
    first = tmp_path / "first"
    second = tmp_path / "second"

    _select_only(monkeypatch, "SLM_DATA_DIR", first)
    with ram_lock.ram_reservation("first", required_mb=0):
        assert (first / "ram_lock.sem").exists()

    _select_only(monkeypatch, "SLM_DATA_DIR", second)
    with ram_lock.ram_reservation("second", required_mb=0):
        assert (second / "ram_lock.sem").exists()


def test_primary_alias_wins_for_every_w2_durable_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary"
    legacy = tmp_path / "legacy"
    hook = tmp_path / "hook"
    monkeypatch.setenv("SLM_DATA_DIR", str(primary))
    monkeypatch.setenv("SL_MEMORY_PATH", str(legacy))
    monkeypatch.setenv("SLM_HOME", str(hook))

    assert _outcome_common.memory_db_path() == primary.resolve() / "memory.db"
    assert auto_recall_hook._get_queue_db_path() == primary.resolve() / "recall_queue.db"
    assert claude_code_hooks._version_dir() == primary.resolve() / "hooks"


def test_auto_recall_opens_queue_in_selected_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.core import recall_queue

    selected = tmp_path / "selected"
    _select_only(monkeypatch, "SLM_DATA_DIR", selected)
    observed: dict[str, object] = {}

    class _Queue:
        def __init__(self, path: Path) -> None:
            observed["path"] = path

        def enqueue(self, **payload: object) -> str:
            observed["enqueue"] = payload
            return "request-1"

        def poll_result(self, request_id: str, timeout_s: float) -> dict:
            observed["poll"] = (request_id, timeout_s)
            return {"ok": True, "results": [{"content": "root-local"}]}

        def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(recall_queue, "RecallQueue", _Queue)
    monkeypatch.setattr(auto_recall_hook, "_detect_mode", lambda: "A")

    result = auto_recall_hook._do_recall(
        "canonical queue",
        limit=7,
        session_id="session-1",
    )

    assert result == [{"content": "root-local"}]
    assert observed["path"] == selected.resolve() / "recall_queue.db"
    assert observed["closed"] is True
