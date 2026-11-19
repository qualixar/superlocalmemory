# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Every session starts knowing the rules and decisions you confirmed - and
nothing a model only suggested, archived, or marked as replaced."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.mcp.tools_active import register_active_tools
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from tests.helpers.agent_rule import agent_stores, firewall_on  # noqa: F401
from tests.mcp.test_session_init_prospective_surface import _patches, _ToolServer
from tests.test_core.test_coverage_backfill import _db as _migrated_store

# Every test here runs with the origin firewall ON: the rules are saved as an agent saves them
# (``tests/helpers/agent_rule.py``): an agent's own words are at the firewall's floor, so a rule it
# saved stands with no promotion step (owner decision, 19 Nov).
pytestmark = pytest.mark.usefixtures("firewall_on")


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    return _migrated_store(tmp_path / "memory.db")  # origins, trust and the decision log exist


def _fact(db, content, kind=None, source=None, *, created="2026-10-01T10:00:00+00:00",
          lifecycle=None) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                      fact_type=FactType.SEMANTIC, created_at=created)
    fact.memory_kind, fact.memory_kind_source = kind, source
    fact_id = agent_stores(db, fact)
    if lifecycle:
        db.execute("UPDATE atomic_facts SET lifecycle=? WHERE fact_id=?", (lifecycle, fact_id))
    return fact_id


def _context(db, config=None) -> str:
    engine = SimpleNamespace(profile_id="default", mode="B", db=db, _db=None, config=config)
    server = _ToolServer()
    register_active_tools(server, lambda: engine)
    p = _patches(engine)
    with p[0], p[1], p[2]:
        result = asyncio.run(server.tools["session_init"](project_path="/test/proj"))
    assert result["success"] is True, result
    return result.get("context", "")


def test_confirmed_rules_and_decisions_reach_the_session(db) -> None:
    _fact(db, "Always run the full suite once before a release.", "rule", "caller")
    _fact(db, "We chose a 3 second recall ceiling.", "decision", "user")
    context = _context(db)
    assert "Always run the full suite once before a release." in context
    assert "We chose a 3 second recall ceiling." in context


def test_suggested_archived_and_replaced_rules_are_never_injected(db) -> None:
    _fact(db, "Suggested: deploy on Fridays.", "rule", "model:laya")
    _fact(db, "Archived: use the old reranker.", "rule", "caller", lifecycle="archived")
    old = _fact(db, "Old decision: 2 second recall ceiling.", "decision", "caller")
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = "
               "'2026-10-02T00:00:00+00:00' WHERE fact_id = ?", (old,))
    assert db.execute("SELECT 1 FROM fact_temporal_validity WHERE fact_id=? "
                      "AND system_expired_at IS NOT NULL", (old,))
    context = _context(db)
    for text in ("deploy on Fridays", "use the old reranker", "2 second recall ceiling"):
        assert text not in context


def test_a_cold_rule_still_counts(db) -> None:
    _fact(db, "Never print API keys in logs.", "rule", "caller", lifecycle="cold")
    assert "Never print API keys in logs." in _context(db)


def test_flag_off(db) -> None:
    _fact(db, "Always run the full suite once before a release.", "rule", "caller")
    config = SimpleNamespace(memory_kinds=SimpleNamespace(standing_rules_in_session=False))
    assert "full suite once" not in _context(db, config)


def test_cap_10_and_no_duplicates_of_pinned(db) -> None:
    ids = [_fact(db, f"Rule number {i:02d} for every session.", "rule", "caller",
                 created=f"2026-10-01T10:{i:02d}:00+00:00") for i in range(12)]
    db.set_pinned(ids[-1], True)
    context = _context(db)
    assert context.count("Rule number 11 for every session.") == 1      # pinned, once
    shown = [i for i in range(12) if f"Rule number {i:02d} " in context]
    assert shown == list(range(1, 12))  # the pin + the 10 newest other rules
