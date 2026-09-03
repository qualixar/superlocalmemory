# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Coverage guard: every migration ships repair or a justification (4.1.14 #133).

A completed migration whose end-state no longer holds can only be
reconciled by a module-supplied ``repair(conn)`` hook — automatic replay
is deliberately disabled. A module with neither is a future #128 Bug 4.
The single sanctioned alternative is ``REPAIR_NOT_APPLICABLE`` with a
non-empty reason (destructive rebuilds such as M002, where replay would
duplicate rows and drop the live table).
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_MIGRATIONS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src" / "superlocalmemory" / "storage" / "migrations"
)


def _module_facts(path: pathlib.Path) -> tuple[bool, bool, bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assigns = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            assigns.add(node.target.id)
    return (
        "repair" in funcs,
        "REPAIR_NOT_APPLICABLE" in assigns,
        "verify" in funcs,
    )


def _migration_modules() -> list[pathlib.Path]:
    return sorted(
        path for path in _MIGRATIONS_DIR.glob("M*.py")
        if path.name != "_repair_util.py"
    )


def test_every_migration_has_repair_or_justification() -> None:
    offenders = []
    for path in _migration_modules():
        has_repair, has_justification, _ = _module_facts(path)
        if not has_repair and not has_justification:
            offenders.append(path.name)
    assert not offenders, (
        "migrations without repair() or REPAIR_NOT_APPLICABLE: "
        + ", ".join(offenders)
    )


def test_justifications_are_non_empty() -> None:
    import importlib

    for path in _migration_modules():
        _, has_justification, _ = _module_facts(path)
        if not has_justification:
            continue
        module = importlib.import_module(
            f"superlocalmemory.storage.migrations.{path.stem}"
        )
        reason = getattr(module, "REPAIR_NOT_APPLICABLE", "")
        assert isinstance(reason, str) and reason.strip(), path.name


def test_repair_modules_keep_verify() -> None:
    """A repair without its own verify() cannot satisfy the framework gate."""
    offenders = []
    for path in _migration_modules():
        has_repair, _, has_verify = _module_facts(path)
        if has_repair and not has_verify:
            offenders.append(path.name)
    assert not offenders, (
        "repair() without verify(): " + ", ".join(offenders)
    )
