"""BRN-16: unrelated standing rules must not consume the bounded context."""

import pytest

from tests.helpers.agent_rule import firewall_on  # noqa: F401
from tests.mcp.test_session_init_standing_rules import _context, _fact
from tests.mcp.test_session_init_standing_rules import db as _fixture_db

# Runs with the origin firewall ON: the rules are saved as an agent saves them
# (``tests/helpers/agent_rule.py``): an agent's own words are at the firewall's floor, so a rule it
# saved stands with no promotion step (owner decision, 19 Nov).
pytestmark = pytest.mark.usefixtures("firewall_on")


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
