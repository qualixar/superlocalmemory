# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for per-user daemon port discovery in lifecycle hooks.

On a shared host each user runs their own daemon on a different port
(written to ``~/.superlocalmemory/daemon.port`` at startup). The hooks
must read that file rather than hard-coding ``8765``, which on a
multi-user machine may belong to another user's daemon.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.hooks import hook_handlers, post_tool_async_hook


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    slm_home = tmp_path / ".superlocalmemory"
    slm_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path), 1))
    return slm_home


# ---------------------------------------------------------------------------
# hook_handlers._daemon_url
# ---------------------------------------------------------------------------

def test_hook_handlers_reads_port_file(home: Path) -> None:
    (home / "daemon.port").write_text("8766")
    assert hook_handlers._daemon_url() == "http://127.0.0.1:8766"


def test_hook_handlers_defaults_when_no_port_file(home: Path) -> None:
    assert hook_handlers._daemon_url() == "http://127.0.0.1:8765"


def test_hook_handlers_defaults_on_garbage_port_file(home: Path) -> None:
    (home / "daemon.port").write_text("not-a-number")
    assert hook_handlers._daemon_url() == "http://127.0.0.1:8765"


# ---------------------------------------------------------------------------
# post_tool_async_hook._port_file_url and fallback path
# ---------------------------------------------------------------------------

def test_async_hook_reads_port_file(home: Path) -> None:
    (home / "daemon.port").write_text("8766")
    assert post_tool_async_hook._port_file_url() == "http://127.0.0.1:8766"


def test_async_hook_defaults_when_no_port_file(home: Path) -> None:
    assert post_tool_async_hook._port_file_url() == "http://127.0.0.1:8765"


def test_async_hook_env_unset_falls_back_to_port_file(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLM_HOOK_DAEMON_URL", raising=False)
    (home / "daemon.port").write_text("8766")
    assert post_tool_async_hook._sanitised_daemon_url() == "http://127.0.0.1:8766"


def test_async_hook_non_loopback_env_falls_back_to_port_file(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S8-SEC-02: a hostile non-loopback URL must be refused — and the
    # fallback must still honour the per-user port, not hard-coded 8765.
    monkeypatch.setenv("SLM_HOOK_DAEMON_URL", "http://evil.example.com:9999")
    (home / "daemon.port").write_text("8766")
    assert post_tool_async_hook._sanitised_daemon_url() == "http://127.0.0.1:8766"


def test_async_hook_loopback_env_with_port_is_preserved(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLM_HOOK_DAEMON_URL", "http://127.0.0.1:8888")
    assert post_tool_async_hook._sanitised_daemon_url() == "http://127.0.0.1:8888"
