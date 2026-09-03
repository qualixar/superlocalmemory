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


#: Modules whose repair path rebuilds tables or mutates rows, with the
#: test that proves crash-resume round-trip safety for each. A repair
#: that only re-applies additive DDL needs no entry; anything matching
#: the destructive patterns below must name its proof here, or the guard
#: fails the suite. (4.1.14 #133 audit.)
ROUND_TRIP_COVERED = {
    "M021_ingestion_log_profile": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m021_interrupted_rebuild_resumes_from_old"
    ),
    "M023_mesh_profile_isolation": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m023_dropped_column_returns_with_backfill"
    ),
    "M026_rbac_memberships_fk": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m026_old_shape_gains_fk_preserving_grants"
    ),
    "M027_transferable_patterns_profile": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m027_old_shape_rebuilds_preserving_rows"
    ),
    "M020_model_state_integrity": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m020_backfill_fills_only_empty_hashes"
    ),
    "M022_entity_aliases_profile": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m022_backfill_uses_parent_profile_orphans_default"
    ),
    "M028_fact_entity_associations": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m028_dropped_state_table_returns"
    ),
    "M046_prospective_memory_has_its_own_name": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m046_old_value_converts_and_rows_survive"
    ),
    "M047_fisher_vectors_are_stored_like_every_other_vector": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m047_text_vectors_convert_to_buffers"
    ),
    "M049_a_schema_version_marker_is_one_row": (
        "tests/test_migrations/test_repair_rebuild_roundtrip.py::"
        "test_m049_duplicate_versions_collapse_to_one"
    ),
}


def _repair_body_strings(path: pathlib.Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == "apply" or node.name == "repair"
        ):
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(
                    child.value, str
                ):
                    chunks.append(child.value)
    return "\n".join(chunks)


def test_destructive_repairs_name_round_trip_proof() -> None:
    """Rebuild/row-mutating repairs must cite their crash-resume test."""
    import re

    pattern = re.compile(
        r"\bDROP\s+TABLE\b|\bRENAME\s+TO\b|\bUPDATE\s+\w+|\bDELETE\s+FROM\b|"
        r"\bINSERT\s+INTO\b",
        re.IGNORECASE,
    )
    offenders = []
    for path in _migration_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        funcs = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if "repair" not in funcs:
            continue
        if pattern.search(_repair_body_strings(path)):
            if path.stem not in ROUND_TRIP_COVERED:
                offenders.append(path.name)
    assert not offenders, (
        "destructive repair() without named round-trip proof "
        "(add to ROUND_TRIP_COVERED with its test): " + ", ".join(offenders)
    )
