# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm why`: everything that explains one memory, read-only, in ids and words."""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage import fact_origin as fo
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.trust import signatures, why
from superlocalmemory.trust import signing_key as sk
from tests.test_trust_signing.conftest import SIGNING

TAINT = "SLM_FEATURE_TRUST_TAINT"
WEB = fo.Origin("ingest_adapter", "verbatim", source_type="gmail")
PERSON = fo.Origin("remember_http", "verbatim", source_type="http", role="user")


def _save(db, text, origin=None, **kw):
    record = MemoryRecord(content=text, session_date="2026-11-15")
    db.store_memory(record)
    fact = AtomicFact(memory_id=record.memory_id, content=text, **kw)
    if origin is None:
        db.store_fact(fact)
    else:
        with fo.declare(origin):
            db.store_fact(fact)
    return fact


def _ro(db):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(db.db_path))


def _deliver(db, fact, *, path="session_init", role="pinned", position=0, delivered=1,
             included=1, at="2026-11-16T10:00:00+00:00", delivery="d1"):
    db.execute(
        "INSERT OR IGNORE INTO injection_deliveries (delivery_id, profile_id, session_id, client, "
        "path, design, holdout_state, epsilon, eligible_count, candidates, delivered_at) "
        "VALUES (?, 'default', 's1', 'claude', ?, 'one-out-v2', 'on', 0.05, 1, 1, ?)",
        (delivery, path, at))
    db.execute(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position, score, "
        "eligible, exclusion, propensity, included, delivered) VALUES (?, ?, 'default', ?, ?, 0.5, "
        "?, ?, ?, ?, ?)",
        (delivery, fact.fact_id, role, position, 1 if role == "recall" else 0,
         None if role == "recall" else "protected", 0.95 if role == "recall" else 1.0,
         included, delivered))


def test_a_memory_that_does_not_exist_or_is_in_another_profile_has_no_why(db) -> None:
    fact = _save(db, "Exists in default.")
    conn = _ro(db)
    assert why.explain(conn, "nope") is None
    assert why.explain(conn, fact.fact_id, profile_id="someone-else") is None
    assert why.explain(conn, fact.fact_id, profile_id="default")["fact_id"] == fact.fact_id


def test_origin_class_signals_the_mapping_row_and_the_reason_it_exists(db) -> None:
    fact = _save(db, "The invoice is due Friday.", WEB)
    got = why.explain(_ro(db), fact.fact_id)["origin"]
    assert got["taint"] == "web" and got["label"] == "a web page"
    assert got["signals"]["channel"] == "ingest_adapter" and got["signals"]["source_type"] == "gmail"
    assert got["rule"]["row"] and got["rule"]["why"] and got["rule"]["mapping"] == "v2"
    assert got["words"] == "This memory is a web page."


def test_a_person_confirmed_memory_has_no_mapping_rule_but_says_who(db) -> None:
    from superlocalmemory.trust import fact_trust

    fact = _save(db, "Approved text from a tool.", WEB)
    with db.raw_connection() as conn:
        fact_trust.mark_confirmed(conn, fact.fact_id, by="alice", channel="cli")
    got = why.explain(_ro(db), fact.fact_id)["origin"]
    assert got["class"] == "user_confirmed" and got["rule"] is None
    assert got["basis"].startswith("confirmed:cli:")


def test_the_taint_is_explained_through_its_parents(db) -> None:
    parent = _save(db, "The mail says the port is 8080.", WEB)
    child = _save(db, "The port is 9090.", fo.Origin(
        "correction", "verbatim", source_type="correction", parent_fact_id=parent.fact_id))
    got = why.explain(_ro(db), child.fact_id)["origin"]
    assert got["class"] == "agent_inferred" and got["taint"] == "web"
    assert got["parents"] == [{"fact_id": parent.fact_id, "class": "web", "taint": "web",
                               "label": "a web page"}]
    assert "derived from something that is a web page" in got["words"]


def test_a_memory_with_no_origin_row_is_unknown_not_an_error(db) -> None:
    fact = _save(db, "Saved long ago.")
    db.execute("DELETE FROM fact_origin WHERE fact_id = ?", (fact.fact_id,))
    got = why.explain(_ro(db), fact.fact_id)["origin"]
    assert got["class"] == "unknown" and got["signals"] is None


def test_the_firewall_section_names_the_approvals(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    fact = _save(db, "Always do the risky thing.", WEB)
    db.execute("INSERT INTO trust_approvals (approval_id, fact_id, profile_id, wanted, state, "
               "requested_by, requested_channel, requested_at) VALUES "
               "('a1', ?, 'default', 'rule', 'pending', 'claude', 'mcp', '2026-11-16T09:00:00')",
               (fact.fact_id,))
    got = why.explain(_ro(db), fact.fact_id)["firewall"]
    assert got["acting"] is True and got["below_system"] is True
    assert got["approvals"][0]["wanted"] == "rule" and got["approvals"][0]["state"] == "pending"
    assert "holds this memory back" in got["words"]
    monkeypatch.delenv(TAINT)
    off = why.explain(_ro(db), fact.fact_id)["firewall"]
    assert off["acting"] is False and "not acting" in off["words"]


def test_a_persons_own_memory_may_be_an_instruction(db) -> None:
    fact = _save(db, "I prefer tea.", PERSON)
    got = why.explain(_ro(db), fact.fact_id)["firewall"]
    assert got["below_system"] is False and got["words"].startswith("Trusted")


def test_the_signature_section_is_checked_only_with_the_key_the_caller_holds(db, monkeypatch, signer) -> None:
    monkeypatch.setenv(SIGNING, "on")
    sk.install(signer)
    fact = _save(db, "Signed fact.")
    sk.install(None)
    assert why.explain(_ro(db), fact.fact_id)["signature"]["state"] == "signed_unchecked"
    assert why.explain(_ro(db), fact.fact_id, signer=signer)["signature"]["state"] == signatures.OK
    db.execute("UPDATE atomic_facts SET content = 'changed' WHERE fact_id = ?", (fact.fact_id,))
    assert why.explain(_ro(db), fact.fact_id, signer=signer)["signature"]["state"] == signatures.MODIFIED


def test_why_never_touches_the_keychain(db, monkeypatch, signer) -> None:
    fact = _save(db, "Fact.")
    monkeypatch.setattr(sk, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("keychain")))
    why.explain(_ro(db), fact.fact_id)


def test_deliveries_say_where_when_and_why_in_words_newest_first(db) -> None:
    fact = _save(db, "A pinned note.")
    _deliver(db, fact, path="session_init", role="pinned", at="2026-11-15T10:00:00+00:00",
             delivery="d1")
    _deliver(db, fact, path="hook_recall", role="recall", position=2,
             at="2026-11-16T10:00:00+00:00", delivery="d2")
    _deliver(db, fact, path="hook_recall", role="recall", position=0, delivered=0, included=0,
             at="2026-11-17T10:00:00+00:00", delivery="d3")
    got = why.explain(_ro(db), fact.fact_id)["deliveries"]
    assert (got["shown"], got["considered"]) == (2, 3)
    assert [i["delivery_id"] for i in got["recent"]] == ["d3", "d2", "d1"]
    words = [i["why"]["words"] for i in got["recent"]]
    assert words[2] == "Shown at session start, as a pinned memory."
    assert words[1] == "Shown with a prompt (automatic recall), as a recall result (rank 3)."
    assert words[0].startswith("Chosen with a prompt (automatic recall)") and "left out on purpose" in words[0]
    assert got["recent"][1]["why"]["rank"] == 3 and got["recent"][1]["client"] == "claude"


def test_deliveries_are_bounded_and_carry_no_query_text(db) -> None:
    fact = _save(db, "Often shown.")
    for i in range(15):
        _deliver(db, fact, delivery=f"d{i:02}", at=f"2026-11-{i + 1:02}T10:00:00+00:00")
    got = why.explain(_ro(db), fact.fact_id, deliveries=4)["deliveries"]
    assert len(got["recent"]) == 4 and got["considered"] == 15
    assert not any("query" in k for k in got["recent"][0])


def test_compiler_section_lists_versions_and_proposals_by_id_and_kind(db) -> None:
    fact = _save(db, "Mentioned by a proposal.")
    db.execute("INSERT INTO proposal_sets (set_id, profile_id, title, origin, state, created_at) "
               "VALUES ('s1', 'default', 'Tidy', 'dream', 'proposed', '2026-11-16T00:00:00')")
    db.execute("INSERT INTO proposals (proposal_id, set_id, profile_id, seq, kind, summary, "
               "payload_json, state, created_at) VALUES ('p1', 's1', 'default', 0, 'retire', "
               "'SECRET SUMMARY TEXT', '{}', 'proposed', '2026-11-16T00:00:00')")
    db.execute("INSERT INTO proposal_facts VALUES ('p1', ?)", (fact.fact_id,))
    got = why.explain(_ro(db), fact.fact_id)["compiler"]
    assert got["proposals"] == [{"proposal_id": "p1", "set_id": "s1", "kind": "retire",
                                 "state": "proposed", "set_state": "proposed",
                                 "created_at": "2026-11-16T00:00:00"}]
    assert "SECRET" not in str(got)


def test_compiler_section_lists_the_pages_that_cite_a_memory_without_their_text(db) -> None:
    fact = _save(db, "Cited by a page.")
    db.execute("INSERT INTO compiled_pages (page_id, profile_id, kind, subject_key, title, "
               "body_json, content_hash, built_at) VALUES ('pg1', 'default', 'topic', 'entity:e', "
               "'A topic', '[{\"text\": \"SECRET SENTENCE\"}]', 'h', 'n')")
    db.execute("INSERT INTO page_citations VALUES ('pg1', 0, ?)", (fact.fact_id,))
    got = why.explain(_ro(db), fact.fact_id)["compiler"]
    assert got["pages"] == [{"page_id": "pg1", "kind": "topic", "title": "A topic"}]
    assert "SECRET" not in str(got)
    assert "1 page(s) mention it" in "\n".join(why.lines(why.explain(_ro(db), fact.fact_id)))


def test_truth_section_reuses_the_slot_and_dispute_views(db) -> None:
    fact = _save(db, "Plain fact.")
    got = why.explain(_ro(db), fact.fact_id)["truth"]
    assert set(got) == {"slots", "disputes", "source"} and got["disputes"] == []


def test_a_store_without_the_new_tables_still_explains(tmp_path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
      CREATE TABLE atomic_facts (fact_id TEXT, profile_id TEXT, content TEXT, created_at TEXT,
                                 memory_id TEXT);
      INSERT INTO atomic_facts VALUES ('f1', 'p', 'Old store fact.', '2026-01-01', 'm1');
    """)
    got = why.explain(conn, "f1")
    assert got["origin"]["class"] == "unknown" and got["signature"]["state"] == "not_available"
    assert got["deliveries"] == {"shown": 0, "considered": 0, "recent": []}
    assert got["compiler"] == {"versions": [], "proposals": [], "pages": []}


def test_the_preview_is_short_and_free_of_control_characters(db) -> None:
    fact = _save(db, "Line one.\x07 " + "word " * 200)
    preview = why.explain(_ro(db), fact.fact_id)["preview"]
    assert len(preview) <= why.PREVIEW_CHARS and "\x07" not in preview


def test_the_text_a_person_reads_covers_every_section(db) -> None:
    fact = _save(db, "The invoice is due Friday.", WEB)
    _deliver(db, fact)
    text = "\n".join(why.lines(why.explain(_ro(db), fact.fact_id)))
    for heading in ("Where it came from", "Instruction firewall", "Signature", "Truth",
                    "Shown to your agents: 1 time(s) of 1 considered"):
        assert heading in text


def test_the_text_names_who_approved_and_how(db) -> None:
    fact = _save(db, "Approved by a person.", WEB)
    db.execute("INSERT INTO trust_approvals (approval_id, fact_id, profile_id, wanted, state, "
               "requested_by, requested_channel, requested_at, decided_by, decided_channel, "
               "decided_at) VALUES ('a1', ?, 'default', 'rule', 'approved', 'claude', 'mcp', "
               "'2026-11-16T09:00:00', 'alice', 'cli', '2026-11-16T09:05:00')", (fact.fact_id,))
    text = "\n".join(why.lines(why.explain(_ro(db), fact.fact_id)))
    assert "It asked to be: rule (approved by alice via cli)" in text
