# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Poisoning scenarios 2 and 3 (W4 exit gate).

2. A memory taken from a web page cannot supersede what the user said (package A's lattice).
3. A shell-CLI save from a non-terminal is not ``user_direct``: the client says whether it
   sat at a terminal, the daemon records it, and the class follows (package B's door).
"""

from __future__ import annotations

import io

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


# -- 3. a shell without a terminal is not the user -------------------------------------------------

class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _save_as(engine, monkeypatch, *, terminal: bool, text: str, key: str):
    """What ``slm remember`` does: the client's own header decides, then POST /remember."""
    monkeypatch.setattr("sys.stdin", _Stream(terminal))
    monkeypatch.setattr("sys.stdout", _Stream(terminal))
    headers = aa.shell_headers("capability")
    with _client(engine) as client:
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


@pytest.mark.parametrize("terminal,expected", [(False, "agent_inferred"), (True, "user_direct")])
def test_a_save_from_a_shell_is_the_users_only_at_a_terminal(
        engine_with_mock_deps, monkeypatch, terminal, expected) -> None:
    fact = _save_as(engine_with_mock_deps, monkeypatch, terminal=terminal,
                    text="The deploy window for the billing service is Tuesday evening.",
                    key=f"shell-{terminal}")
    assert _class_of(engine_with_mock_deps, fact) == expected


def test_what_an_agent_types_in_a_shell_cannot_supersede_what_the_person_typed(
        engine_with_mock_deps, monkeypatch) -> None:
    """The reason the gap mattered: an agent's shell save must sit below the user's statement."""
    from superlocalmemory.trust import origin_class as oc

    engine = engine_with_mock_deps
    person = _save_as(engine, monkeypatch, terminal=True, key="p1",
                      text="The release cadence of SLM is one release every week.")
    agent = _save_as(engine, monkeypatch, terminal=False, key="a1",
                     text="The release cadence of SLM is one big release.")
    with connect(engine._db) as conn:
        verdict = supersession.decide(conn, winner=agent, loser=person)
    assert not verdict.allowed and verdict.winner_class == oc.AGENT_INFERRED
    assert verdict.loser_class == oc.USER_DIRECT
