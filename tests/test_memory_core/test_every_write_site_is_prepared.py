# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Every place that builds a durable write request must prepare its text.

A new ``IngestionRequest(...)`` or ``RememberRequest(...)`` site fails here until
its author either calls ``prepare_for_save`` or records why it is exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"

_SITES: dict[tuple[str, str], str] = {
    ("server/unified_daemon.py", "remember"): "prepared",
    ("server/unified_daemon.py", "enqueue"): "prepared",
    ("server/routes/ingest.py", "ingest"): "prepared",
    ("server/routes/data_io.py", "import_memories"): "prepared",
    ("daemon/materializer.py", "legacy_item"): "prepared",
    ("core/engine_ingestion.py", "canonical_store"): "prepared",
    ("core/engine_ingestion.py", "canonical_store_fact"): "prepared",
    ("core/remember_runtime.py", "_handle_admission"): "exempt: replay of an already admitted request",
    ("storage/own_fact_repair.py", "_queue_enrichment"): "exempt: repair of text that is already stored",
}
_CALLS = {"IngestionRequest", "RememberRequest"}


def _visit(node, fn, source, rel, found) -> None:
    """Attribute each request call to its innermost enclosing function."""
    for child in ast.iter_child_nodes(node):
        inner = child if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) else fn
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in _CALLS
            and fn is not None
        ):
            found[(rel, fn.name)] = ast.get_source_segment(source, fn) or ""
        _visit(child, inner, source, rel, found)


def _found() -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for path in _SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if not any(name + "(" in source for name in _CALLS):
            continue
        tree = ast.parse(source)
        rel = path.relative_to(_SRC).as_posix()
        _visit(tree, None, source, rel, found)
    return found


def test_every_write_site_is_mapped_and_prepared() -> None:
    found = _found()
    for site, body in found.items():
        assert site in _SITES, (
            f"{site} builds a write request: call prepare_for_save on the "
            "text first, then add the site to _SITES."
        )
    for site, kind in _SITES.items():
        assert site in found, f"{site} no longer builds a write request; update _SITES"
        if kind == "prepared":
            assert "prepare_for_save" in found[site], (
                f"{site} must call prepare_for_save before building the request"
            )
