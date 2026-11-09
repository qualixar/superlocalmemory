# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Every environment variable the deployment guide names must be read by SLM.

Issue #142: the guide told users to set ``SLM_DAEMON_URL`` on client hosts,
but nothing in the package reads it, so a remote setup silently fell back to
a local daemon. A documented knob that does nothing is worse than none.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOC = _REPO / "docs" / "distributed-deployment.md"
_SRC = _REPO / "src" / "superlocalmemory"

# Named in the guide only to say "this is the wrong name".
_NAMED_AS_WRONG = frozenset({"SLM_MCP_WS_PORT"})


def _source_text() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in _SRC.rglob("*.py")
    )


def test_every_documented_env_var_is_read_somewhere_in_the_package() -> None:
    names = set(re.findall(r"\bSLM_[A-Z0-9_]+", _DOC.read_text(encoding="utf-8")))
    source = _source_text()
    # A trailing underscore is a documented prefix family (SLM_CURSOR_*);
    # the prefix itself must appear in the source.
    unread = sorted(n for n in names - _NAMED_AS_WRONG if n not in source)
    assert not unread, f"documented but never read by SLM: {unread}"


def test_remote_clients_are_pointed_at_a_route_that_exists() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "SLM_DAEMON_URL" not in text
    assert "mcp-remote" in text
