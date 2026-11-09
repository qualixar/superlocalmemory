# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""``fetch`` is how an agent verifies a write. It must not lie about one.

GitHub #135. ``fetch(fact_ids: str)`` split on commas and returned whatever
matched. A token that matched nothing produced::

    {"success": true, "results": [], "count": 0}

which is byte-identical to a correct answer for a fact that does not exist. The
caller cannot tell "my call was malformed" from "the store lost my data", so
the reasonable next step is to start debugging the store. That is what the
reporter did, and it cost hours.

MEASURED, because the original diagnosis had this backwards. Against the real
SDK (mcp 2.0.0 / pydantic 2.13.4) the server does NOT stringify a list -- it
rejects it::

    fetch(fact_ids=["abc123"])    -> ValidationError: Input should be a valid string
    fetch(fact_ids='["abc123"]')  -> {"success": true, "results": [], "count": 0}

So the ``"['abc123']"`` the reporter saw was produced by their CLIENT, before
the call. Accepting ``list[str]`` therefore does NOT fix the reported symptom
on its own -- that client still sends a string. The honest error is the
load-bearing fix; list acceptance is ergonomics. Both are here, and the parser
also understands the two stringified-array shapes a client can produce, because
that is the shape that actually reached the store.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.mcp.shared import parse_id_list
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"


class _CapturingServer:
    """Collects the functions register_core_tools decorates."""

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def decorate(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorate


class _Engine:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self.profile_id = _PROFILE


@pytest.fixture()
def fetch_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m1', ?, 'source')", (_PROFILE,),
    )
    conn.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
        " scope, created_at) VALUES ('real-1','m1',?,'a real stored fact',"
        " 'global', '2026-08-01T00:00:00+00:00')", (_PROFILE,),
    )
    conn.commit()
    conn.close()

    from superlocalmemory.mcp import tools_core

    engine = _Engine(DatabaseManager(str(path)))

    async def _profile(*args, **kwargs) -> str:
        return _PROFILE

    monkeypatch.setattr(tools_core, "_runtime_profile", _profile)
    server = _CapturingServer()
    tools_core.register_core_tools(server, lambda: engine)
    fetch = server.tools["fetch"]

    def call(fact_ids):
        return asyncio.run(fetch(fact_ids))

    return call


class TestTheParserUnderstandsWhatClientsActuallySend:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("abc123", ["abc123"]),
            ("abc123,def456", ["abc123", "def456"]),
            (" abc123 , def456 ,", ["abc123", "def456"]),
            (["abc123"], ["abc123"]),
            (["abc123", "def456"], ["abc123", "def456"]),
            # The reporter's exact wire value: Python repr of a list.
            ("['abc123']", ["abc123"]),
            ("['abc123', 'def456']", ["abc123", "def456"]),
            # A JSON-serialising client.
            ('["abc123"]', ["abc123"]),
            ('["abc123", "def456"]', ["abc123", "def456"]),
            ("", []),
            (None, []),
            ([], []),
        ],
    )
    def test_shapes(self, raw, expected) -> None:
        assert parse_id_list(raw) == expected

    def test_a_bare_string_is_never_exploded_into_characters(self) -> None:
        """The failure mode a naive list() would introduce."""
        assert parse_id_list("abc") == ["abc"]


class TestFetchNamesWhatItCouldNotFind:
    def test_nothing_matched_is_not_success(self, fetch_tool) -> None:
        """The #135 repro: this used to be success/0 and unexplainable."""
        out = fetch_tool("['abc123']")
        assert out["success"] is False, (
            "fetch reported success for a token that matched nothing"
        )
        assert "abc123" in str(out), "the error does not name the failing token"

    def test_a_genuinely_absent_id_is_also_reported(self, fetch_tool) -> None:
        """"Not found" must be distinguishable, but it is still not success."""
        out = fetch_tool("no-such-fact")
        assert out["success"] is False
        assert out.get("not_found") == ["no-such-fact"]

    def test_a_partial_miss_is_visible(self, fetch_tool) -> None:
        """Two of three found is the case that silently lost data before."""
        out = fetch_tool("real-1,no-such-fact")
        assert out["success"] is True
        assert out["count"] == 1
        assert out.get("not_found") == ["no-such-fact"]

    def test_a_list_argument_works(self, fetch_tool) -> None:
        out = fetch_tool(["real-1"])
        assert out["success"] is True and out["count"] == 1

    def test_the_happy_path_is_unchanged(self, fetch_tool) -> None:
        out = fetch_tool("real-1")
        assert out["success"] is True and out["count"] == 1
        assert out["results"][0]["fact_id"] == "real-1"
