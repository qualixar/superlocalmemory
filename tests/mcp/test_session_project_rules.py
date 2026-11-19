"""BRN-16: unrelated standing rules must not consume the bounded context."""

import pytest

from tests.mcp.test_session_init_standing_rules import _context, _fact
from tests.mcp.test_session_init_standing_rules import db as _fixture_db


@pytest.fixture(autouse=True)
def _origin_firewall_off(monkeypatch):
    """These tests are about what session start selects, not the origin firewall: they save through agent
    doors (``agent_inferred``), which the firewall (on by default since 19 Nov) holds back until a promotion.
    tests/test_trust_firewall covers that rule."""
    monkeypatch.setenv("SLM_FEATURE_TRUST_TAINT", "off")

db = _fixture_db


def test_unrelated_project_rules_are_filtered_before_the_cap(db):
    relevant = _fact(db, "Rule for this project.", "rule", "caller")
    db.execute(
        'UPDATE memories SET metadata_json=\'{"project":"/test/proj"}\' '
        "WHERE memory_id=(SELECT memory_id FROM atomic_facts WHERE fact_id=?)",
        (relevant,),
    )
    _fact(db, "Global rule: never print a secret.", "rule", "user")
    for i in range(20):
        fid = _fact(
            db, f"Foreign project rule {i}.", "rule", "caller", created="2026-10-10T10:00:00+00:00"
        )
        db.execute(
            'UPDATE memories SET metadata_json=\'{"project":"/else/foreign"}\' '
            "WHERE memory_id=(SELECT memory_id FROM atomic_facts WHERE fact_id=?)",
            (fid,),
        )
    context = _context(db)
    assert "Rule for this project." in context
    assert "Global rule:" in context
    assert "Foreign project rule" not in context
