# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""GDPR entity erasure through a REAL daemon erases a memory a person edited.

Varun (2026-10-07): "Erasure wins". A person edits a memory naming someone
(PATCH, which proposes a correction case a person made), then that someone is
erased (POST /api/compliance/gdpr/erase-entity). The route must report success
truthfully, the case is closed with a text-free audit row, and the memory's
words are left in no table of any database in the data root.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import time
import uuid

from tests.test_integration.test_erasure_integrity_e2e import (  # noqa: F401
    _count,
    _recall,
    _remember_complete,
    _ro,
    _text_copies,
    daemon,
    stub_embedder,
)


def _person_edit(daemon, fact_id: str, content: str) -> None:
    for _ in range(45):  # 503 = the writer is busy; the route says retry
        code, body = daemon.request("PATCH", f"/api/memories/{fact_id}", {"content": content})
        if code != 503:
            break
        time.sleep(2)
    assert code in (200, 202), body


def test_gdpr_erasure_of_a_person_erases_the_memory_they_edited(daemon) -> None:
    tag = uuid.uuid4().hex[:6]
    word = "".join(chr(97 + int(c, 16)) for c in tag)  # a name is letters only
    person = f"Brannoc{word}"
    facts = _remember_complete(
        daemon, f"{person} keeps the synthetic teal kettle{tag} in the north pantry.", "")
    _person_edit(daemon, facts[0], f"{person} keeps the teal kettle{tag} in the south pantry.")
    ph = ",".join("?" * len(facts))
    case_ids = [r[0] for r in _ro(daemon, "SELECT case_id FROM correction_cases WHERE "
                                  f"predecessor_fact_id IN ({ph})", tuple(facts))]
    assert len(case_ids) == 1, case_ids
    successor = _ro(daemon, "SELECT successor_fact_id FROM correction_cases WHERE case_id = ?",
                    (case_ids[0],))[0][0]
    names = [r[0] for r in _ro(daemon, "SELECT canonical_name FROM canonical_entities "
                               "WHERE instr(canonical_name, ?) > 0", (person,))]
    assert names, ("precondition: the person is a known entity", [r[0] for r in _ro(
        daemon, "SELECT canonical_name FROM canonical_entities")][-10:])

    code, body = daemon.request("POST", "/api/compliance/gdpr/erase-entity",
                                {"entity_name": names[0], "confirm": names[0]})

    assert code == 200 and body.get("success") is True, body
    assert body.get("correction_cases_erased") == 1 and body.get("erasure_complete") == 1, body
    gone = (*facts, successor)
    gph = ",".join("?" * len(gone))
    assert _count(daemon, f"SELECT COUNT(*) FROM atomic_facts WHERE fact_id IN ({gph})", gone) == 0
    assert _count(daemon, "SELECT COUNT(*) FROM correction_cases WHERE case_id = ?",
                  (case_ids[0],)) == 0
    audit = _ro(daemon, "SELECT closed_reason, actor_id, prior_status FROM "
                "correction_cases_erased WHERE case_id = ?", (case_ids[0],))
    assert audit == [("erased_on_request", "gdpr", "proposed")], audit
    assert _text_copies(daemon, f"kettle{tag}") == {}
    for result in _recall(daemon, f"teal kettle{tag} pantry", ""):
        assert result.get("fact_id") not in gone
