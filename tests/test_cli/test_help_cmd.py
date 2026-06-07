# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for ``slm help-optimize`` CLI handler."""

from argparse import Namespace

import pytest


def test_help_full_renders(capsys):
    from superlocalmemory.cli.help_cmd import cmd_help_optimize

    cmd_help_optimize(Namespace(topic=None, no_pager=True))
    out = capsys.readouterr().out
    assert "SLM v3.6 Optimize" in out
    assert "slm optimize" in out
    assert "Claude Code" in out
    assert "AI Reliability Engineering" in out


def test_help_topic_cache(capsys):
    from superlocalmemory.cli.help_cmd import cmd_help_optimize

    cmd_help_optimize(Namespace(topic="cache", no_pager=True))
    out = capsys.readouterr().out
    assert "slm cache" in out
    assert "TTL" in out


def test_help_topic_safety(capsys):
    from superlocalmemory.cli.help_cmd import cmd_help_optimize

    cmd_help_optimize(Namespace(topic="safety", no_pager=True))
    out = capsys.readouterr().out
    assert "Aggressive" in out
    assert "DO NOT use for" in out


def test_help_topic_agents(capsys):
    from superlocalmemory.cli.help_cmd import cmd_help_optimize

    cmd_help_optimize(Namespace(topic="agents", no_pager=True))
    out = capsys.readouterr().out
    assert "ANTHROPIC_BASE_URL" in out
    assert "127.0.0.1:8765" in out


def test_help_unknown_topic_exits_1():
    from superlocalmemory.cli.help_cmd import cmd_help_optimize

    with pytest.raises(SystemExit) as exc:
        cmd_help_optimize(Namespace(topic="nonexistent", no_pager=True))
    assert exc.value.code == 1
