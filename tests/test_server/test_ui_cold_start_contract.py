"""Regression contracts found by the V3.7 live dashboard witness."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_daemon_lifespan_bootstraps_dashboard_install_token() -> None:
    """A fresh daemon must create the credential used by Browser UI calls."""
    source = (ROOT / "src/superlocalmemory/server/unified_daemon.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    lifespan = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    body = ast.get_source_segment(source, lifespan) or ""
    assert "ensure_install_token()" in body


def test_timeline_accepts_dashboard_date_grouping() -> None:
    """The UI's chronological timeline request must pass API validation."""
    from superlocalmemory.server.routes.timeline import VALID_GROUP_BY

    assert "date" in VALID_GROUP_BY
