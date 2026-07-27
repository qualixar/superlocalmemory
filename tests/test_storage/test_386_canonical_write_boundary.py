# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""AST release gate for canonical ``memory.db`` ownership.

This is intentionally an AST rule, not a grep.  It resolves import aliases
(``import sqlite3 as sql`` and ``from sqlite3 import connect``), detects both
positional and keyword connection arguments, and accepts only provably
read-only URI opens outside bootstrap/migration/coordinator code.
"""

from __future__ import annotations

import ast
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from superlocalmemory.storage.read_connection import ReadConnectionFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "superlocalmemory"

# The release gate protects *client/query* surfaces.  Storage, migrations and
# lifecycle workers have their own ownership tests while they are migrated to
# the coordinator.  This is deliberately not a path-wide suppression: each
# bridge below is named with its owning function and its write responsibility.
CLIENT_QUERY_SURFACE_ROOTS = ("cli/", "mcp/", "server/")
AUDITED_MUTATION_BRIDGES = {
    ("cli/ingest_cmd.py", "_write_tool_events"): (
        "temporary ingestion writer; must move to daemon/coordinator before "
        "the coordinator feature gate is removed"
    ),
    ("server/routes/entity.py", "recompile_entity"): (
        "manage-authorised recompilation command; reads the entity before "
        "delegating its mutation to EntityCompiler"
    ),
}

# Names and call forms that establish the *canonical memory.db*, rather than
# a coincidentally named local, test, pending, learning, cache, or evolution
# database.  Generic names such as ``db_path`` are intentionally absent: a
# broad marker list produced false positives and hid the real ownership story.
CANONICAL_GLOBAL_NAMES = {"DB_PATH", "MEMORY_DB"}


@dataclass(frozen=True)
class RawConnectViolation:
    relative_path: str
    line: int
    expression: str


class _RawCanonicalConnectVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.sqlite_modules: set[str] = {"sqlite3"}
        self.sqlite_connects: set[str] = set()
        self.violations: list[tuple[int, str]] = []
        self.audited_bridges: list[tuple[str, int]] = []
        self._canonical_names: list[set[str]] = [set(CANONICAL_GLOBAL_NAMES)]
        self._function_names: list[str | None] = [None]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "sqlite3":
                self.sqlite_modules.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "sqlite3":
            for alias in node.names:
                if alias.name == "connect":
                    self.sqlite_connects.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_sqlite_connect(node.func) and self._targets_canonical_memory(node):
            if self._is_provably_read_only(node):
                pass
            elif self._is_audited_mutation_bridge():
                self.audited_bridges.append((self._function_names[-1] or "", node.lineno))
            else:
                expression = ast.get_source_segment(self.source, node) or "sqlite connect"
                self.violations.append((node.lineno, expression))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._canonical_names.append(set(self._canonical_names[-1]))
        self._function_names.append(node.name)
        self.generic_visit(node)
        self._function_names.pop()
        self._canonical_names.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._canonical_expression(node.value):
            for target in node.targets:
                self._record_canonical_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and self._canonical_expression(node.value):
            self._record_canonical_target(node.target)
        self.generic_visit(node)

    def _is_sqlite_connect(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self.sqlite_connects
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "connect"
            and isinstance(func.value, ast.Name)
            and func.value.id in self.sqlite_modules
        )

    def _targets_canonical_memory(self, call: ast.Call) -> bool:
        target = call.args[0] if call.args else next(
            (keyword.value for keyword in call.keywords if keyword.arg == "database"),
            None,
        )
        return target is not None and self._canonical_expression(target)

    def _canonical_expression(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self._canonical_names[-1]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return "memory.db" in node.value
        if isinstance(node, ast.JoinedStr):
            return any(self._canonical_expression(value) for value in node.values)
        if isinstance(node, ast.FormattedValue):
            return self._canonical_expression(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # ``MEMORY_DIR / 'learning.db'`` is a separate store; only the
            # filename-bearing right-hand side establishes canonical memory.db.
            return self._canonical_expression(node.right)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"state_path", "memory_db_path", "_memory_db_path"}:
                return not node.args or any(self._canonical_expression(arg) for arg in node.args)
            return any(self._canonical_expression(arg) for arg in node.args)
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            return dotted in {"config.db_path", "engine._config.db_path"}
        return False

    def _record_canonical_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._canonical_names[-1].add(target.id)

    def _is_audited_mutation_bridge(self) -> bool:
        # The visitor receives its file-specific bridge list through a dynamic
        # attribute set by the source scanner.  Exact function names make a
        # newly introduced writer fail this gate rather than disappear behind
        # a directory-wide exception.
        return (getattr(self, "relative_path", ""), self._function_names[-1]) in AUDITED_MUTATION_BRIDGES

    @staticmethod
    def _is_provably_read_only(call: ast.Call) -> bool:
        # ``mode=ro`` only has meaning when passed through SQLite's URI mode.
        keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        uri_true = isinstance(keywords.get("uri"), ast.Constant) and keywords["uri"].value is True
        target = call.args[0] if call.args else keywords.get("database")
        return uri_true and _contains_mode_ro(target)


def _dotted_name(node: ast.Attribute) -> str:
    parts = [node.attr]
    value: ast.expr = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _contains_mode_ro(node: ast.AST | None) -> bool:
    """Accept a literal URI or f-string whose static fragments contain mode=ro."""
    if isinstance(node, ast.Constant):
        return "mode=ro" in str(node.value)
    if isinstance(node, ast.JoinedStr):
        fragments = "".join(
            str(value.value)
            for value in node.values
            if isinstance(value, ast.Constant)
        )
        return "mode=ro" in fragments
    return False


def find_writable_canonical_raw_connects() -> list[RawConnectViolation]:
    """Return raw canonical opens in query surfaces that evade read-only mode."""
    violations: list[RawConnectViolation] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if not relative.startswith(CLIENT_QUERY_SURFACE_ROOTS):
            continue
        source = path.read_text(encoding="utf-8")
        visitor = _RawCanonicalConnectVisitor(source)
        visitor.relative_path = relative
        visitor.visit(ast.parse(source, filename=str(path)))
        violations.extend(
            RawConnectViolation(relative, line, expression)
            for line, expression in visitor.violations
        )
    return violations


def find_audited_mutation_bridge_opens() -> dict[tuple[str, str], list[int]]:
    """Return the explicitly transitional raw canonical opens, by owner."""
    bridges = {bridge: [] for bridge in AUDITED_MUTATION_BRIDGES}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if not relative.startswith(CLIENT_QUERY_SURFACE_ROOTS):
            continue
        source = path.read_text(encoding="utf-8")
        visitor = _RawCanonicalConnectVisitor(source)
        visitor.relative_path = relative
        visitor.visit(ast.parse(source, filename=str(path)))
        for function_name, line in visitor.audited_bridges:
            bridges[(relative, function_name)].append(line)
    return bridges


def test_386_no_non_coordinator_writable_raw_canonical_sqlite_connections() -> None:
    """Client and dashboard query paths cannot open canonical memory.db writable."""
    violations = find_writable_canonical_raw_connects()
    rendered = "\n".join(
        f"{item.relative_path}:{item.line}: {item.expression}"
        for item in violations[:25]
    )
    assert violations == [], (
        "Writable or non-provably-read-only sqlite3.connect calls can target "
        "canonical memory.db outside the query-read boundary:\n"
        f"{rendered}\n... {max(0, len(violations) - 25)} additional violation(s)"
    )


def test_386_audited_mutation_bridges_are_exact_and_bounded() -> None:
    """A bridge cannot turn into a directory or function-wide writer escape hatch."""
    bridge_opens = find_audited_mutation_bridge_opens()
    assert {bridge: len(lines) for bridge, lines in bridge_opens.items()} == {
        bridge: 1 for bridge in AUDITED_MUTATION_BRIDGES
    }


def test_386_ast_guard_detects_aliases_and_accepts_only_explicit_read_only() -> None:
    """Lock the guard itself against easy alias/keyword bypasses."""
    source = """
import sqlite3 as sql
from sqlite3 import connect as open_sqlite
sql.connect(str(DB_PATH))
open_sqlite(MEMORY_DB)
sql.connect('file:/tmp/memory.db?mode=ro', uri=True)
"""
    visitor = _RawCanonicalConnectVisitor(source)
    visitor.visit(ast.parse(source))

    assert [line for line, _ in visitor.violations] == [4, 5]


def test_386_ast_guard_classifies_canonical_dataflow_not_generic_database_names() -> None:
    """Separate pending/learning stores do not mask canonical-memory violations."""
    source = """
import sqlite3
pending_db = root / 'pending.db'
learning_db = root / 'learning.db'
canonical = root / 'memory.db'
sqlite3.connect(pending_db)
sqlite3.connect(learning_db)
sqlite3.connect(canonical)
"""
    visitor = _RawCanonicalConnectVisitor(source)
    visitor.relative_path = "server/routes/example.py"
    visitor.visit(ast.parse(source))

    assert [line for line, _ in visitor.violations] == [8]


def test_386_manual_client_lease_remains_physically_read_only(tmp_path: Path) -> None:
    """Legacy manual-close callers retain the strict factory boundary."""
    db_path = tmp_path / "memory.db"
    writable = sqlite3.connect(db_path)
    writable.execute("CREATE TABLE facts (fact_id TEXT)")
    writable.commit()
    writable.close()

    conn = ReadConnectionFactory(db_path).open()
    try:
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO facts VALUES ('not-allowed')")
    finally:
        conn.close()
