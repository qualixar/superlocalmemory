# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""F3: `slm forget --dry-run` previews all memories; a bare `slm forget`
(no query, no --dry-run) must NEVER mass-delete — it refuses with exit 2.

These tests exercise the argument-handling contract only; the refusal path
returns before any engine/model initialization, so they stay fast.
"""

from __future__ import annotations

import argparse

import pytest

from superlocalmemory.cli.commands import cmd_forget


def _args(**kw) -> argparse.Namespace:
    base = {"query": None, "dry_run": False, "json": False, "yes": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_bare_forget_without_query_refuses_text(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cmd_forget(_args(query=None, dry_run=False, json=False))
    assert exc.value.code == 2
    out = capsys.readouterr().out.lower()
    assert "query is required" in out
    assert "--dry-run" in out


def test_bare_forget_without_query_refuses_json(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cmd_forget(_args(query=None, dry_run=False, json=True))
    assert exc.value.code == 2
    assert "QUERY_REQUIRED" in capsys.readouterr().out
