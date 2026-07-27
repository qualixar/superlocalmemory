"""3.8.6 dashboard refreshes must remain physically query-only.

This is intentionally a source contract as well as a SQLite authorizer test.
The dashboard's read handlers are spread over small route modules, so a future
raw ``sqlite3.connect(memory.db)`` would otherwise bypass the read factory
without a focused route test noticing.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.memory_write import memory_read

_ROOT = Path(__file__).resolve().parents[2]
_ROUTE_ROOT = _ROOT / "src" / "superlocalmemory" / "server" / "routes"

_READ_HELPERS = {
    "helpers.py": {"get_db_connection", "_get_db_profiles"},
    "abstraction.py": {"_conn"},
    "insights.py": {"_get_conn"},
    "tiers.py": {"_db"},
}

_READ_ENDPOINTS = {
    "agents.py": {"get_agent_memory_activity", "get_trust_stats"},
    "entity.py": {"list_entities", "get_entity"},
    "lifecycle.py": {"lifecycle_status"},
    "timeline.py": {"get_timeline"},
    "v3_api.py": {
        "dashboard", "get_associations", "get_association_stats",
        "get_consolidation_status", "get_core_memory",
        "get_vector_store_status", "forgetting_stats", "quantization_stats",
        "ccq_blocks", "get_soft_prompts", "get_graph_communities",
        "v33_overview",
    },
}

_DML_DDL = {
    "ALTER", "ANALYZE", "ATTACH", "CREATE", "DELETE", "DETACH", "DROP",
    "INSERT", "PRAGMA", "REINDEX", "REPLACE", "UPDATE", "VACUUM",
}


def _function_node(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _raw_sqlite_open_calls(node: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr == "connect" and isinstance(child.func.value, ast.Name):
            if child.func.value.id in {"sqlite3", "_sqlite3"}:
                calls.append(child)
    return calls


def _sql_mutations(node: ast.AST) -> list[str]:
    statements: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            token = child.value.lstrip().upper().split(maxsplit=1)[0] if child.value.strip() else ""
            if token in _DML_DDL:
                statements.append(child.value)
    return statements


@pytest.mark.parametrize(
    ("filename", "names"),
    [*_READ_HELPERS.items(), *_READ_ENDPOINTS.items()],
)
def test_dashboard_read_helpers_and_refreshes_use_no_raw_writable_open_or_dml(
    filename: str, names: set[str],
):
    """Every listed dashboard query reaches the read boundary, never a writer."""
    path = _ROUTE_ROOT / filename
    for name in names:
        node = _function_node(path, name)
        assert not _raw_sqlite_open_calls(node), f"{filename}:{name} opened SQLite directly"
        assert not _sql_mutations(node), f"{filename}:{name} contains DML or DDL"


def test_read_factory_authorizer_observes_only_select_during_dashboard_style_query(tmp_path):
    """A representative dashboard query cannot cause a write authorizer event."""
    db_path = tmp_path / "memory.db"
    writable = sqlite3.connect(db_path)
    writable.execute("CREATE TABLE atomic_facts (profile_id TEXT, fact_id TEXT)")
    writable.execute("INSERT INTO atomic_facts VALUES ('default', 'fact-1')")
    writable.commit()
    writable.close()

    writes: list[int] = []
    write_actions = {
        sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
    }
    with memory_read(db_path) as conn:
        conn.set_authorizer(
            lambda action, _arg1, _arg2, _db, _source: (
                writes.append(action) if action in write_actions else sqlite3.SQLITE_OK
            ),
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE profile_id = ?", ("default",),
        ).fetchone()[0] == 1

    assert writes == []


def test_legacy_dashboard_connection_helper_remains_query_only(tmp_path):
    """Compatibility callers cannot turn ``get_db_connection`` into a writer."""
    from superlocalmemory.server.routes.helpers import get_read_connection

    db_path = tmp_path / "memory.db"
    writable = sqlite3.connect(db_path)
    writable.execute("CREATE TABLE facts (fact_id TEXT)")
    writable.commit()
    writable.close()

    conn = get_read_connection(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO facts VALUES ('not-allowed')")
    finally:
        conn.close()
