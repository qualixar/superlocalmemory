# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""MCP surfaces a live daemon's unknown-profile 404 as an answer (4.1.14 audit).

Collapsing the 404 into DAEMON_UNAVAILABLE made a refusing daemon
indistinguishable from a dead one — and the remember path retried a
hopeless 404 three times before misreporting it.
"""
from __future__ import annotations

import pytest

from superlocalmemory.cli.daemon import DaemonNotFound
from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy


def _boom_404(*args, **kwargs):
    raise DaemonNotFound(
        404, "unknown_profile",
        "profile 'ghost' does not exist; per-request routing never creates "
        "a profile implicitly",
        "/remember",
    )


def test_proxy_recall_surfaces_unknown_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.daemon_request", _boom_404,
    )
    proxy = DaemonPoolProxy(port=18773)

    result = proxy.recall("ghost query", profile_id="ghost")

    assert result["ok"] is False
    assert result["code"] == "unknown_profile"
    assert result["retryable"] is False
    assert "ghost" in result["error"]


def test_proxy_store_surfaces_unknown_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.daemon_request", _boom_404,
    )
    proxy = DaemonPoolProxy(port=18773)

    result = proxy.store("ghost content", {"profile_id": "ghost"})

    assert result["ok"] is False
    assert result["code"] == "unknown_profile"
    assert result["retryable"] is False


def test_daemon_not_found_carries_code_and_message() -> None:
    exc = DaemonNotFound(404, "unknown_profile", "gone", "/recall")
    assert exc.code == "unknown_profile"
    assert exc.status == 404
    assert "gone" in str(exc)


def test_proxy_still_reports_dead_daemon(monkeypatch) -> None:
    def _boom_down(*args, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.daemon_request", _boom_down,
    )
    proxy = DaemonPoolProxy(port=18773)

    result = proxy.recall("anything")

    assert result["ok"] is False
    assert result["code"] == "DAEMON_UNAVAILABLE"
    assert result["retryable"] is True
