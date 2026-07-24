"""Super-help (`slm help`) completeness + topics (v3.8.2 UX-7).

Guarantees:
1. Every top-level command registered in the CLI parser appears in the
   grouped `slm help` overview — so adding a command without documenting it
   fails CI (drift guard).
2. Focused topics (modes / config / self-heal) render without error.
3. The no-topic overview lists the groups + the self-healing section.
"""
from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path

import superlocalmemory.cli.commands as cmds


def _top_level_commands() -> set[str]:
    """Command names registered directly on the top-level subparser (`sub`).

    Nested subparsers use other variables (db_sub, session_sub, …), so the
    ``\\bsub\\.add_parser`` anchor selects only top-level commands.
    """
    main_src = Path(cmds.__file__).with_name("main.py").read_text()
    return set(re.findall(r"\bsub\.add_parser\(\s*[\"']([a-z0-9\-]+)[\"']",
                          main_src))


def test_super_help_covers_every_command():
    top = _top_level_commands()
    assert top, "parsed no top-level commands from main.py"
    missing = top - cmds.all_help_commands()
    assert not missing, f"slm help omits commands: {sorted(missing)}"


def test_help_topics_render_without_error():
    for topic in ("modes", "config", "self-heal", "health", "mode"):
        cmds.cmd_help(Namespace(topic=topic))  # must not raise


def test_help_unknown_topic_is_graceful(capsys):
    cmds.cmd_help(Namespace(topic="does-not-exist"))
    out = capsys.readouterr().out
    assert "No help topic" in out


def test_help_overview_lists_groups_and_selfheal(capsys):
    cmds.cmd_help(Namespace(topic=None))
    out = capsys.readouterr().out
    assert "command overview" in out
    assert "self-healing" in out.lower()
    assert "doctor --fix" in out
    # A representative command from each of a few groups is present.
    for cmd in ("remember", "recall", "doctor", "setup", "mesh", "loop"):
        assert re.search(rf"\b{cmd}\b", out), f"{cmd} missing from overview"
