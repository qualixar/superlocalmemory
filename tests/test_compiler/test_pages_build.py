# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Drafting pages (D-18): spans and templates only, each sentence cites live facts."""

from __future__ import annotations

from superlocalmemory.compiler import page_sources
from superlocalmemory.compiler import page_templates as tpl
from superlocalmemory.compiler.page_text import MAX_SENTENCE_CHARS, MAX_SENTENCES, content_hash, page_id, sentences_of
from tests.test_compiler.conftest import ENTITY, PROFILE, raw
from tests.test_compiler.helpers import add_fact, bind_slot


def _drafts(db, kinds=None):
    conn = raw(db)
    try:
        return page_sources.build_drafts(conn, PROFILE, kinds)
    finally:
        conn.close()


def _by_kind(drafts, kind):
    return [d for d in drafts if d["kind"] == kind]


def test_sentences_are_substrings_in_order() -> None:
    text = "First one. Second one!\nThird? Fourth"
    parts = sentences_of(text)
    assert parts == ["First one.", "Second one!", "Third?", "Fourth"]
    assert all(p in text for p in parts)


def test_current_state_page_has_the_head_and_since_from_slot_values(store) -> None:
    db, _ids = store
    fid = add_fact(db, "The release cadence of SLM is one release every week", entity=None)
    bind_slot(db, fid, "slm", "release_cadence", "weekly", "2026-03-01T00:00:00+00:00")
    page = _by_kind(_drafts(db), "current_state")[0]
    texts = [s["text"] for s in page["sentences"]]
    assert texts == ["slm release cadence: The release cadence of SLM is one release every week",
                     "slm release cadence has had this value since 2026-03-01."]
    assert all(s["type"] == "template" and s["facts"] == [fid] for s in page["sentences"])
    assert page["page_id"] == page_id(PROFILE, "current_state", "name:slm")
    assert page["withheld"] == {} and page["candidates"] == 2


def test_a_disputed_slot_is_shown_as_a_dispute_never_as_a_value(store) -> None:
    db, _ids = store
    old = add_fact(db, "SLM ships every week", "2026-03-01T00:00:00+00:00", entity=None)
    new = add_fact(db, "SLM ships once a month", "2026-06-01T00:00:00+00:00", entity=None)
    bind_slot(db, old, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    bind_slot(db, new, "slm", "cadence", "monthly", "2026-06-01T00:00:00+00:00")
    page = _by_kind(_drafts(db), "current_state")[0]
    assert len(page["sentences"]) == 1
    text = page["sentences"][0]["text"]
    assert text.startswith("Disputed, not settled:") and "SLM ships every week" in text
    assert "SLM ships once a month" in text
    assert sorted(page["sentences"][0]["facts"]) == sorted([old, new])


def test_a_fact_longer_than_a_sentence_is_withheld_and_counted(store) -> None:
    db, _ids = store
    fid = add_fact(db, "x" * (MAX_SENTENCE_CHARS + 10), entity=None)
    bind_slot(db, fid, "slm", "notes", "long", "2026-03-01T00:00:00+00:00")
    short = add_fact(db, "A short value", entity=None)
    bind_slot(db, short, "slm", "other", "short", "2026-03-01T00:00:00+00:00")
    page = _by_kind(_drafts(db), "current_state")[0]
    assert page["withheld"] == {"too_long": 1}
    assert page["candidates"] == 4 and len(page["sentences"]) == 3


def test_story_page_orders_decisions_in_time(store) -> None:
    db, _ids = store
    b = add_fact(db, "We chose SQLite for storage. It is local first.", "2026-04-02T00:00:00+00:00",
                 kind="decision", entity="ent-db")
    a = add_fact(db, "We chose Python for the core.", "2026-03-01T00:00:00+00:00",
                 kind="decision", entity="ent-db")
    add_fact(db, "A different decision on another subject.", kind="decision", entity="ent-other")
    story = _by_kind(_drafts(db), "story")
    assert len(story) == 1
    assert [s["text"] for s in story[0]["sentences"]] == [
        "2026-03-01: We chose Python for the core.", "2026-04-02: We chose SQLite for storage."]
    assert [s["facts"] for s in story[0]["sentences"]] == [[a], [b]]


def test_topic_page_is_spans_with_limits_and_withheld_counts(store) -> None:
    db, ids = store
    add_fact(db, "One. Two. Three. Four.", "2026-08-01T00:00:00+00:00")
    topic = _by_kind(_drafts(db), "topic")[0]
    assert topic["subject_key"] == f"entity:{ENTITY}" and topic["title"].startswith("SuperLocalMemory")
    assert all(s["type"] == "span" and len(s["facts"]) == 1 for s in topic["sentences"])
    assert topic["withheld"] == {"over_limit": 2}  # two sentences of the long fact left out
    assert topic["candidates"] == 7 and len(topic["sentences"]) == 5


def test_a_page_over_the_size_limit_says_so(store) -> None:
    db, _ids = store
    for i in range(MAX_SENTENCES):
        add_fact(db, f"Sentence number {i} says something.", f"2026-09-{i % 28 + 1:02d}T00:00:00+00:00")
    topic = _by_kind(_drafts(db), "topic")[0]
    assert len(topic["sentences"]) == MAX_SENTENCES
    assert topic["withheld"]["over_limit"] == topic["candidates"] - MAX_SENTENCES


def test_too_few_facts_make_no_topic_or_story_page(store) -> None:
    db, _ids = store
    add_fact(db, "A single decision about the team.", kind="decision", entity="ent-team")
    assert _by_kind(_drafts(db, ("story",)), "story") == []
    add_fact(db, "One fact.", entity="ent-two")
    add_fact(db, "Another fact.", entity="ent-two")
    assert not [d for d in _drafts(db, ("topic",)) if d["subject_key"] == "entity:ent-two"]


def test_tags_group_topic_pages(store) -> None:
    db, _ids = store
    for i in range(3):
        fid = add_fact(db, f"Tagged note {i}.", entity=None)
        mem = db.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?", (fid,))[0][0]
        db.execute("UPDATE memories SET metadata_json = ? WHERE memory_id = ?",
                   ('{"tags": "Launch, plan"}', mem))
    keys = {d["subject_key"] for d in _drafts(db, ("topic",))}
    assert {"tag:launch", "tag:plan"} <= keys


def test_a_stale_source_leaves_its_fact_out_and_says_so(store) -> None:
    db, ids = store
    db.execute("INSERT INTO memory_leases (fact_id, profile_id, source_kind, locator, fingerprint, "
               "state, leased_at, checked_at) VALUES (?, ?, 'file', '/x/notes.md', 'f', 'changed', "
               "'2026-01-01', '2026-01-02')", (ids[0], PROFILE))
    topic = _by_kind(_drafts(db), "topic")[0]
    assert all(ids[0] not in s["facts"] for s in topic["sentences"])
    assert topic["withheld"] == {"source_changed": 1}


def test_retired_facts_are_not_cited(store) -> None:
    db, ids = store
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = '2026-10-01' WHERE fact_id = ?",
               (ids[0],))
    topic = _by_kind(_drafts(db), "topic")
    assert not topic  # two live facts left: under the minimum


def test_the_same_store_drafts_the_same_pages(store) -> None:
    db, _ids = store
    assert _drafts(db) == _drafts(db)
    d = _drafts(db)[0]
    assert d["content_hash"] == content_hash(d["title"], d["sentences"])


def test_the_span_check_passes_then_fails_when_the_fact_changes(store) -> None:
    db, ids = store
    conn = raw(db)
    try:
        topic = _by_kind(page_sources.build_drafts(conn, PROFILE), "topic")[0]
        item = topic["sentences"][0]
        assert tpl.verify(conn, PROFILE, item)[0]
        db.execute("UPDATE atomic_facts SET content = 'Something else entirely now.' "
                   "WHERE fact_id = ?", (item["facts"][0],))
        ok, why = tpl.verify(conn, PROFILE, item)
        assert not ok and "no longer found" in why
        kept, dropped = tpl.verify_all(conn, PROFILE, topic["sentences"])
        assert dropped == 1 and len(kept) == len(topic["sentences"]) - 1
    finally:
        conn.close()


def test_a_template_fails_when_its_slot_moves_on(store) -> None:
    db, _ids = store
    fid = add_fact(db, "SLM ships every week", entity=None)
    bind_slot(db, fid, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    conn = raw(db)
    try:
        page = _by_kind(page_sources.build_drafts(conn, PROFILE), "current_state")[0]
        assert all(tpl.verify(conn, PROFILE, s)[0] for s in page["sentences"])
        db.execute("UPDATE memory_slots SET state = 'empty' WHERE subject_key = 'name:slm'")
        assert not any(tpl.verify(conn, PROFILE, s)[0] for s in page["sentences"])
    finally:
        conn.close()


def test_a_span_cannot_cite_two_facts_or_none(store) -> None:
    db, ids = store
    conn = raw(db)
    try:
        assert not tpl.verify(conn, PROFILE, {"type": "span", "text": "x", "facts": ids[:2]})[0]
        assert not tpl.verify(conn, PROFILE, {"type": "span", "text": "x", "facts": []})[0]
        assert not tpl.verify(conn, PROFILE, {"type": "model", "text": "x", "facts": ids[:1]})[0]
        assert not tpl.verify(conn, PROFILE, {"type": "template", "text": "x", "facts": ids[:1],
                                              "template": "nope", "params": {}})[0]
    finally:
        conn.close()


def test_a_template_fails_when_the_value_it_quotes_changes(store) -> None:
    db, _ids = store
    head = add_fact(db, "SLM ships every week", entity=None)
    bind_slot(db, head, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    conn = raw(db)
    try:
        page = _by_kind(page_sources.build_drafts(conn, PROFILE), "current_state")[0]
        db.execute("UPDATE atomic_facts SET content = 'SLM ships monthly' WHERE fact_id = ?", (head,))
        assert not tpl.verify(conn, PROFILE, page["sentences"][0])[0]
        assert tpl.verify_all(conn, PROFILE, page["sentences"])[1] == 1  # the since-line still holds
    finally:
        conn.close()


def test_a_template_fails_when_it_cites_other_facts_than_it_renders_from(store) -> None:
    db, _ids = store
    old = add_fact(db, "SLM ships every week", "2026-03-01T00:00:00+00:00", entity=None)
    new = add_fact(db, "SLM ships once a month", "2026-06-01T00:00:00+00:00", entity=None)
    bind_slot(db, old, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    bind_slot(db, new, "slm", "cadence", "monthly", "2026-06-01T00:00:00+00:00")
    conn = raw(db)
    try:
        item = _by_kind(page_sources.build_drafts(conn, PROFILE), "current_state")[0]["sentences"][0]
        assert tpl.verify(conn, PROFILE, item)[0]
        assert not tpl.verify(conn, PROFILE, {**item, "facts": [old]})[0]
    finally:
        conn.close()


def test_a_span_cites_exactly_one_memory_even_when_its_words_are_in_both(store) -> None:
    db, ids = store
    conn = raw(db)
    try:
        both = {"type": "span", "text": "SuperLocalMemory", "facts": ids[:2]}
        ok, why = tpl.verify(conn, PROFILE, both)
        assert not ok and "exactly one" in why
    finally:
        conn.close()


def test_quarantined_and_unapplied_successor_facts_are_never_cited(store) -> None:
    db, ids = store
    db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (ids[0],))
    conn = raw(db)
    try:
        assert set(tpl.live_facts(conn, PROFILE, ids)) == set(ids[1:])
        db.execute("UPDATE atomic_facts SET quarantined = 0 WHERE fact_id = ?", (ids[0],))
        db.execute(
            "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
            "successor_fact_id, reason_code, status, version, idempotency_key, "
            "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, created_at, "
            "updated_at) VALUES ('c1', ?, 'personal', ?, ?, 'direct_content_correction', "
            "'proposed', 0, 'k', 'a', 'host_authenticated', 'trusted', 'n', 'n')",
            (PROFILE, ids[0], ids[1]))
        assert set(tpl.live_facts(conn, PROFILE, ids)) == {ids[0], ids[2]}
        db.execute("UPDATE correction_cases SET status = 'applied'")
        assert set(tpl.live_facts(conn, PROFILE, ids)) == set(ids)
    finally:
        conn.close()


def test_a_sentence_too_long_for_a_topic_page_is_withheld_not_cut(store) -> None:
    db, _ids = store
    add_fact(db, "A short one. " + "x" * (MAX_SENTENCE_CHARS + 5) + ".", "2026-08-01T00:00:00+00:00")
    topic = _by_kind(_drafts(db), "topic")[0]
    assert topic["withheld"] == {"too_long": 1}
    assert all(len(s["text"]) <= MAX_SENTENCE_CHARS for s in topic["sentences"])


def test_a_story_leaves_out_a_decision_whose_source_changed(store) -> None:
    db, _ids = store
    add_fact(db, "We chose Python for the core.", "2026-03-01T00:00:00+00:00", kind="decision", entity="ent-d")
    stale = add_fact(db, "We chose SQLite.", "2026-04-01T00:00:00+00:00", kind="decision", entity="ent-d")
    add_fact(db, "We chose Rust for the CLI.", "2026-05-01T00:00:00+00:00", kind="decision", entity="ent-d")
    db.execute("INSERT INTO memory_leases (fact_id, profile_id, source_kind, locator, fingerprint, "
               "state, leased_at, checked_at) VALUES (?, ?, 'file', '/x/a.md', 'f', 'changed', "
               "'2026-01-01', '2026-01-02')", (stale, PROFILE))
    story = _by_kind(_drafts(db), "story")[0]
    assert len(story["sentences"]) == 2 and story["withheld"] == {"source_changed": 1}
    assert all(stale not in s["facts"] for s in story["sentences"])
