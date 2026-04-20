"""LLD-05 §12.6 — CLI tests for ``slm context prestage`` and ``slm connect``."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def fake_recall():
    def _fn(q, l, p):
        if "topics" in q:
            return [{"name": "Qualixar", "score": 0.9}]
        if "entities" in q:
            return [{"name": "SLM", "mentions": 10}]
        if "decisions" in q:
            return [{"text": "ship v3.4.22"}]
        if "memories" in q:
            return [{"text": "mem one"}]
        return []
    return _fn


def test_connect_orchestrates_all_detected(tmp_path, fake_recall, monkeypatch,
                                           capsys):
    # Force every adapter active.
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    monkeypatch.setenv("SLM_ANTIGRAVITY_FORCE", "1")
    monkeypatch.setenv("SLM_COPILOT_FORCE", "1")
    (tmp_path / ".github").mkdir()

    from superlocalmemory.cli.context_commands import (
        build_default_adapters,
        cmd_connect_cross_platform,
    )
    from superlocalmemory.hooks.ide_connector import CrossPlatformConnector

    adapters = build_default_adapters(
        base_dir=tmp_path, recall_fn=fake_recall,
        sync_log_db=tmp_path / "memory.db",
    )
    connector = CrossPlatformConnector(adapters)
    results = connector.connect()
    # At least one adapter should have written.
    assert any(v == "wrote" for v in results.values())


def test_connect_disable_specific_adapter(tmp_path, fake_recall, monkeypatch,
                                           capsys):
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    from superlocalmemory.cli.context_commands import build_default_adapters
    from superlocalmemory.hooks.ide_connector import CrossPlatformConnector

    adapters = build_default_adapters(
        base_dir=tmp_path, recall_fn=fake_recall,
        sync_log_db=tmp_path / "memory.db",
    )
    connector = CrossPlatformConnector(adapters)
    connector.connect()
    ok = connector.disable("cursor_project")
    assert ok is True
    # Disabling a non-existent name returns False.
    assert connector.disable("does_not_exist") is False


def test_context_prestage_markdown_and_json(tmp_path, monkeypatch, fake_recall,
                                             capsys):
    from superlocalmemory.cli.context_commands import cmd_context
    monkeypatch.chdir(tmp_path)
    # Patch recall-fn discovery.
    monkeypatch.setattr(
        "superlocalmemory.cli.context_commands._get_recall_fn",
        lambda: fake_recall,
    )
    from superlocalmemory.hooks import context_payload as cp
    monkeypatch.setattr(cp, "_now_iso", lambda: "2026-04-18T00:00:00+00:00")

    # Markdown mode
    args = Namespace(subcommand="prestage", query="what about Qualixar?",
                     limit=5, profile_id="default", json=False, tool=False)
    cmd_context(args)
    out = capsys.readouterr().out
    assert "SLM Context" in out
    assert "Qualixar" in out

    # JSON mode
    args2 = Namespace(subcommand="prestage", query="q", limit=5,
                      profile_id="default", json=True, tool=False)
    cmd_context(args2)
    out2 = capsys.readouterr().out
    data = json.loads(out2)
    assert "topics" in data and "entities" in data


def test_context_prestage_tool_mode(tmp_path, monkeypatch, fake_recall, capsys):
    from superlocalmemory.cli.context_commands import cmd_context
    monkeypatch.setattr(
        "superlocalmemory.cli.context_commands._get_recall_fn",
        lambda: lambda q, l, p: [
            {"id": "m1", "text": "memory content", "score": 0.9}
        ],
    )
    args = Namespace(subcommand="prestage", query="q", limit=3,
                     profile_id="default", json=False, tool=True)
    cmd_context(args)
    data = json.loads(capsys.readouterr().out)
    assert "memories" in data


def test_context_unknown_subcommand(capsys):
    from superlocalmemory.cli.context_commands import cmd_context
    cmd_context(Namespace(subcommand="unknown", query="", limit=5,
                          profile_id="default", json=False, tool=False))
    assert "Unknown subcommand" in capsys.readouterr().out


def test_cross_platform_dry_run(tmp_path, monkeypatch, fake_recall, capsys):
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "superlocalmemory.cli.context_commands._get_recall_fn",
        lambda: fake_recall,
    )
    from superlocalmemory.cli.context_commands import (
        cmd_connect_cross_platform,
    )
    args = Namespace(disable=None, dry_run=True, json=False,
                     cross_platform=True)
    cmd_connect_cross_platform(args)
    out = capsys.readouterr().out
    assert "cursor_project" in out


def test_cross_platform_json_output(tmp_path, monkeypatch, fake_recall, capsys):
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "superlocalmemory.cli.context_commands._get_recall_fn",
        lambda: fake_recall,
    )
    from superlocalmemory.cli.context_commands import (
        cmd_connect_cross_platform,
    )
    args = Namespace(disable=None, dry_run=False, json=True,
                     cross_platform=True)
    cmd_connect_cross_platform(args)
    data = json.loads(capsys.readouterr().out)
    assert "results" in data


def test_cross_platform_disable_cli(tmp_path, monkeypatch, fake_recall, capsys):
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "superlocalmemory.cli.context_commands._get_recall_fn",
        lambda: fake_recall,
    )
    from superlocalmemory.cli.context_commands import (
        cmd_connect_cross_platform,
    )
    args = Namespace(disable="cursor_project", dry_run=False, json=False,
                     cross_platform=True)
    cmd_connect_cross_platform(args)
    out = capsys.readouterr().out
    assert "cursor_project" in out
