"""The retention engine talks to a sqlite3 connection, never to a DatabaseManager.

DatabaseManager.execute returns a list of rows, so ``.fetchone()`` / ``.fetchall()`` on its
result fail. The engine also commits, opens savepoints and reads ``lastrowid``, none of which a
DatabaseManager offers, so handing it one has to be refused at the door with a plain message
instead of failing on the first read.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.compliance.retention import RetentionEngine
from tests.test_sources.conftest import ListDb


def test_a_database_manager_shaped_handle_is_refused_with_a_plain_message():
    conn = sqlite3.connect(":memory:")
    with pytest.raises(TypeError, match="sqlite3 connection"):
        RetentionEngine(ListDb(conn))


def test_a_real_connection_still_works_end_to_end():
    conn = sqlite3.connect(":memory:")
    engine = RetentionEngine(conn)
    engine.add_rule("p1", "GDPR-30d", 30, "thirty days")
    assert [r["rule_name"] for r in engine.get_rules("p1")] == ["GDPR-30d"]
    assert engine.list_rules() and engine.delete_rule("p1", "GDPR-30d") is True
