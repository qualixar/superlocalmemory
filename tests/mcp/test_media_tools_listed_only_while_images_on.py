# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4

"""The image and document tools are listed by default only while images are on."""

from __future__ import annotations

import importlib
import sys

import pytest

MEDIA_TOOLS = frozenset({"remember_media", "get_media", "remember_document", "media_status"})


def _load(monkeypatch: pytest.MonkeyPatch, images_on: bool | Exception):
    for key in ("SLM_MCP_ALL_TOOLS", "SLM_MCP_TOOLS", "SLM_MCP_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLM_MCP_EMBEDDED", "1")
    monkeypatch.setenv("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
    monkeypatch.setenv("SLM_MCP_MESH_TOOLS", "1")

    from superlocalmemory.runtimes import features

    def fake(*_a, **_k):
        if isinstance(images_on, Exception):
            raise images_on
        return images_on

    monkeypatch.setattr(features, "media_enabled", fake)
    profiles = importlib.reload(importlib.import_module("superlocalmemory.mcp.profiles"))
    sys.modules.pop("superlocalmemory.mcp.server", None)
    server = importlib.import_module("superlocalmemory.mcp.server")
    return server, profiles


@pytest.fixture(autouse=True)
def _restore_modules():
    yield
    sys.modules.pop("superlocalmemory.mcp.server", None)
    importlib.reload(importlib.import_module("superlocalmemory.mcp.profiles"))


def test_images_off_lists_no_media_tool_and_keeps_the_counts(monkeypatch):
    server, profiles = _load(monkeypatch, False)
    assert not (MEDIA_TOOLS & server._ESSENTIAL_TOOLS)
    assert not (MEDIA_TOOLS & profiles._PROFILE_DEFINITIONS["full"])
    assert len(server._ESSENTIAL_TOOLS) == 56
    assert len(profiles._PROFILE_DEFINITIONS["full"]) == 56
    assert len(profiles._PROFILE_DEFINITIONS["power"]) == 68
    assert server._ESSENTIAL_TOOLS == profiles._PROFILE_DEFINITIONS["full"]


def test_images_on_lists_the_four_tools_in_default_and_full(monkeypatch):
    server, profiles = _load(monkeypatch, True)
    assert MEDIA_TOOLS <= server._ESSENTIAL_TOOLS
    assert MEDIA_TOOLS <= profiles._PROFILE_DEFINITIONS["full"]
    assert len(server._ESSENTIAL_TOOLS) == 60
    assert len(profiles._PROFILE_DEFINITIONS["full"]) == 60
    assert server._ESSENTIAL_TOOLS == profiles._PROFILE_DEFINITIONS["full"]


def test_images_on_leaves_core_unchanged(monkeypatch):
    _server, profiles = _load(monkeypatch, True)
    assert not (MEDIA_TOOLS & profiles._PROFILE_DEFINITIONS["core"])
    assert len(profiles._PROFILE_DEFINITIONS["core"]) == 18


def test_a_failing_check_means_images_off_and_never_raises(monkeypatch):
    server, profiles = _load(monkeypatch, RuntimeError("features file unreadable"))
    assert not (MEDIA_TOOLS & server._ESSENTIAL_TOOLS)
    assert not (MEDIA_TOOLS & profiles._PROFILE_DEFINITIONS["full"])
