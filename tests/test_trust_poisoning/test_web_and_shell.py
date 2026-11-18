# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Poisoning scenarios 2 and 3 (W4 exit gate).

2. A memory taken from a web page cannot supersede what the user said (package A's lattice).
3. Origin fails closed (D-49): only the dashboard's own door is ``user_direct``; a shell, a
   script, an agent, or a forged header is ``agent_inferred``. The TTY header is gone.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import agent_attribution as aa
from superlocalmemory.trust import fact_trust, supersession
from tests.test_server.test_canonical_remember_route import _client
from tests.test_trust_firewall.conftest import connect
from tests.test_trust_origin.test_origin_supersession import (
    PERSON as A_PERSON,
)
from tests.test_trust_origin.test_origin_supersession import (
    WEB as A_WEB,
)
from tests.test_trust_origin.test_origin_supersession import (
    _judge,
    _open,
    _pair,
    _reason,
)
from tests.test_truth.test_slot_pass import _retired

# -- 2. a web page cannot supersede the user -----------------------------------------------------

def test_a_web_page_memory_cannot_retire_a_user_statement(tmp_path) -> None:
    db, head, new, counts = _pair(tmp_path, A_PERSON, A_WEB, taint="on")
    assert _reason(db) == supersession.ORIGIN_BELOW_HEAD and _open(db) == 1
    counts, tier = _judge(db, tmp_path, taint="on")  # even a judge that says "replaces"
    assert _retired(db, head) == (None, None, None), "the user's statement was retired by a web page"
    assert tier.calls == 0 and not any(k.startswith("resolved") for k in counts)
    assert _open(db) == 1


def test_the_same_pair_retires_the_older_statement_while_the_firewall_is_off(tmp_path) -> None:
    db, head, new, _counts = _pair(tmp_path, A_PERSON, A_WEB, taint=None)
    counts, _tier = _judge(db, tmp_path)
    assert counts["resolved_replaces"] == 1 and _retired(db, head)[2] == new


# -- 3. only the dashboard's own door is the user's statement (D-49: fails closed) ---------------

def _save_as(engine, *, door: str, text: str, key: str):
    """``slm remember`` / an agent (the daemon capability) or the dashboard page (install token)."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    with _client(engine) as client:
        headers = {}
        if door == "dashboard":
            client.headers.pop("X-SLM-Daemon-Capability")
            client.headers.pop("X-SLM-Target-Instance")
            headers = {"X-Install-Token": ensure_install_token(), aa.CHANNEL_HEADER: "dashboard"}
        elif door == "forged":  # a capability holder that names the dashboard's channel
            headers = {aa.CHANNEL_HEADER: "dashboard"}
        elif door == "terminal":  # the old terminal header means nothing now
            headers = {"X-SLM-Client-Shell": "tty"}
        r = client.post("/remember?wait=true", headers=headers,
                        json={"content": text, "idempotency_key": key})
        assert r.status_code == 200, r.text[:300]
        memory = engine._db.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?",
                                    (r.json()["fact_ids"][0],))[0]["memory_id"]
    whole = engine._db.execute(
        "SELECT f.fact_id FROM atomic_facts f JOIN fact_origin o ON o.fact_id = f.fact_id "
        "WHERE f.memory_id = ? AND o.transform IN ('verbatim', 'excerpt')", (memory,))
    return whole[0]["fact_id"]


def _class_of(engine, fact_id):
    with connect(engine._db) as conn:
        return fact_trust.origin_of(conn, fact_id)["class"]


@pytest.mark.parametrize("door,expected", [
    ("dashboard", "user_direct"), ("agent", "agent_inferred"), ("terminal", "agent_inferred"),
    ("forged", "agent_inferred")])
def test_only_the_dashboards_own_door_is_the_users_statement(
        engine_with_mock_deps, door, expected) -> None:
    fact = _save_as(engine_with_mock_deps, door=door, key=f"door-{door}",
                    text="The deploy window for the billing service is Tuesday evening.")
    assert _class_of(engine_with_mock_deps, fact) == expected


def test_no_signal_is_never_the_trusted_case(engine_with_mock_deps) -> None:
    """The A-06 probe: a save that says nothing about who made it is not 'you said this'."""
    fact = _save_as(engine_with_mock_deps, door="agent", key="silent",
                    text="Remember that the staging database moved to the new cluster.")
    assert _class_of(engine_with_mock_deps, fact) != "user_direct"


def test_what_an_agent_saved_cannot_supersede_what_the_dashboard_saved(
        engine_with_mock_deps) -> None:
    from superlocalmemory.trust import origin_class as oc

    engine = engine_with_mock_deps
    person = _save_as(engine, door="dashboard", key="p1",
                      text="The release cadence of SLM is one release every week.")
    agent = _save_as(engine, door="agent", key="a1",
                     text="The release cadence of SLM is one big release.")
    with connect(engine._db) as conn:
        verdict = supersession.decide(conn, winner=agent, loser=person)
    assert not verdict.allowed and verdict.winner_class == oc.AGENT_INFERRED
    assert verdict.loser_class == oc.USER_DIRECT
