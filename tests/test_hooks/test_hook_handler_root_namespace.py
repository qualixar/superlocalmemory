# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Root-isolation contracts for the stdlib lifecycle hook handler."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from superlocalmemory.hooks import claude_code_hooks, hook_handlers


def _select_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    monkeypatch.setenv("SL_MEMORY_PATH", str(root.parent / "wrong-legacy"))
    monkeypatch.setenv("SLM_HOME", str(root.parent / "wrong-hook"))


def _handler_paths() -> tuple[str, str, str]:
    return tuple(
        os.fspath(path)
        for path in (
            hook_handlers._MARKER,
            hook_handlers._START_TIME,
            hook_handlers._ACTIVITY_LOG,
        )
    )


def test_lifecycle_temp_paths_change_with_root_in_one_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    _select_root(monkeypatch, first)
    first_paths = _handler_paths()
    _select_root(monkeypatch, second)
    second_paths = _handler_paths()

    assert first_paths != second_paths
    assert all(hook_handlers._TMP == str(Path(path).parent) for path in first_paths)
    assert all(hook_handlers._TMP == str(Path(path).parent) for path in second_paths)


def test_handler_and_installer_use_identical_gate_marker_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_root(monkeypatch, tmp_path / "selected")
    marker, start, _activity = _handler_paths()

    assert marker == claude_code_hooks._marker_path()
    assert start == claude_code_hooks._start_marker_path()


def test_explicit_marker_overrides_remain_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "explicit-marker"
    start = tmp_path / "explicit-start"
    activity = tmp_path / "explicit-activity"
    monkeypatch.setattr(hook_handlers, "_MARKER", str(marker))
    monkeypatch.setattr(hook_handlers, "_START_TIME", str(start))
    monkeypatch.setattr(hook_handlers, "_ACTIVITY_LOG", str(activity))
    monkeypatch.setattr(hook_handlers.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(
        hook_handlers.subprocess,
        "run",
        lambda *a, **k: type("Result", (), {"stdout": ""})(),
    )

    hook_handlers._hook_start()

    assert start.exists()
    assert activity.exists()
    assert not marker.exists()


def test_starting_second_root_does_not_remove_first_root_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hook_handlers.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(
        hook_handlers.subprocess,
        "run",
        lambda *a, **k: type("Result", (), {"stdout": ""})(),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    _select_root(monkeypatch, first)
    first_marker, first_start, first_activity = _handler_paths()
    hook_handlers._hook_start()

    _select_root(monkeypatch, second)
    second_marker, second_start, second_activity = _handler_paths()
    hook_handlers._hook_start()

    assert Path(first_start).exists()
    assert Path(first_activity).exists()
    assert Path(second_start).exists()
    assert Path(second_activity).exists()
    assert not Path(first_marker).exists()
    assert not Path(second_marker).exists()
