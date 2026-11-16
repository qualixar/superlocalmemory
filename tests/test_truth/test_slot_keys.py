# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Tier 0 slot keys: confident on value statements, silent on everything else."""

from __future__ import annotations

import pytest

from superlocalmemory.truth.slot_keys import extract, normalise_term, value_hash


@pytest.mark.parametrize("text,subject,relation,qualifier,cardinality", [
    ("The release cadence of SLM is one release per month.", "name:slm", "release_cadence",
     "", "single"),
    ("The release cadence for SLM is now one big release", "name:slm", "release_cadence", "",
     "single"),
    ("Alex's preferred voice model is the studio replica", "name:alex",
     "preferred_voice_model", "", "single"),
    ("Alex prefers the studio TTS for voiceovers", "name:alex", "prefers",
     "voiceovers", "single"),
    ("Alex switched to the cloud renderer for video clips", "name:alex", "prefers", "video_clips",
     "single"),
    ("Acme uses SQLite for local storage", "name:acme", "uses", "local_storage",
     "plural"),
])
def test_value_statements_get_a_key(text, subject, relation, qualifier, cardinality) -> None:
    key = extract(text)
    assert key is not None
    assert (key.subject_key, key.relation_key, key.qualifier, key.cardinality) == (
        subject, relation, qualifier, cardinality)
    assert key.key_source == "pattern" and key.confidence == 0.6


@pytest.mark.parametrize("text", [
    "", "Is the release cadence of SLM monthly?",
    "The release cadence of SLM is not monthly.",
    "Alex's voice model is no longer the old one.",
    "The owner of the repo is maybe Alex.",
    "We shipped 1.2.0. The owner of the repo is Alex.",
    "Line one\nline two is here",
    "A long narrative " * 20 + "is here.",
    "Alex prefers the studio voice",  # prefers what for? ambiguous, no key
    "Session ended after the release.",
    # Shapes a shadow reading of a real store showed the first rules got wrong:
    "The modification of build_index.py was done in the past",
    "The modification of test_api.py is done using an unknown tool",
    "I pulled the patched image for the first one and asked it again which server was answering",
    "A lot of these failures are because of the rushed code only.",
    "The demo's first post was live on 11 sites for 62 minutes.",
    "The fix for Bug 1 is in the main branch",
    "The total estimated lines of code for the master plan of the whole project is 6,900",
])
def test_non_statements_get_no_key(text) -> None:
    assert extract(text) is None


def test_explicit_key_wins_and_validates() -> None:
    key = extract("We ship one big release", explicit="SLM/release cadence",
                  explicit_value="one big release")
    assert key is not None and (key.subject_key, key.relation_key, key.qualifier) == (
        "name:slm", "release_cadence", "")
    assert key.key_source == "explicit" and key.confidence == 1.0
    assert key.value_hash == "stated:" + value_hash("One big release.")  # R-34
    bare = extract("We ship one big release", explicit="SLM/release cadence")
    assert bare is not None and bare.key_source == "explicit_unvalued"
    assert bare.value_hash == value_hash("We ship one big release")
    assert extract("x", explicit="only-one-part") is None
    assert extract("x", explicit="a//c") is None
    assert extract("  ", explicit="a/b") is None


def test_a_resolved_subject_uses_the_entity_id() -> None:
    key = extract("The release cadence of SuperLocalMemory is monthly",
                  resolve_subject=lambda name: "ent_42" if name == "SuperLocalMemory" else None)
    assert key is not None and key.subject_key == "entity:ent_42"


def test_value_hash_ignores_case_space_and_trailing_punctuation() -> None:
    assert value_hash("One  Release per month.") == value_hash("one release per month")
    assert value_hash("monthly") != value_hash("weekly")
    assert normalise_term("  The Release   Cadence ") == "release_cadence"


@pytest.mark.parametrize("text,relation", [
    ("Alex's laptop is managed by the IT team", "laptop__managed_by"),
    ("Alex's machine is running Docker 29", "machine__running"),
    ("Alex's session files are stored in /tmp/files", "session_files__stored_in"),
    ("The project's code is now fully owned by Acme", "code__owned_by"),
    ("Alex's laptop is an Apple laptop", "laptop"),
])
def test_a_leading_participle_is_part_of_the_relation(text, relation) -> None:
    key = extract(text)
    assert key is not None and key.relation_key == relation


def test_no_key_supersedes_on_its_own() -> None:
    import sqlite3

    from superlocalmemory.storage.migrations import M069_memory_slots as M069
    from superlocalmemory.truth import slot_store as ss

    conn = sqlite3.connect(":memory:")
    M069.apply(conn)
    pattern = [extract("The owner of SLM is Alex"), extract("The owner of SLM is Sam")]
    explicit = [extract("Alex runs it", explicit="slm/maintainer", explicit_value="Alex"),
                extract("Sam runs it", explicit="slm/maintainer", explicit_value="Sam")]
    for key in pattern + explicit:
        assert key is not None
    ss.bind(conn, profile_id="p", scope="s", fact_id="f1", key=pattern[0], valid_from="2026-01-01")
    assert ss.bind(conn, profile_id="p", scope="s", fact_id="f2", key=pattern[1],
                   valid_from="2026-02-01").outcome == ss.DISPUTE
    ss.bind(conn, profile_id="p", scope="s", fact_id="f3", key=explicit[0], valid_from="2026-01-01")
    assert ss.bind(conn, profile_id="p", scope="s", fact_id="f4", key=explicit[1],
                   valid_from="2026-02-01").outcome == ss.DISPUTE  # D-36: for explicit-v1
    reasons = [r[0] for r in conn.execute("SELECT reason_code FROM slot_disputes "
                                          "ORDER BY opened_at, rowid")]
    assert reasons == ["pattern_key", ss.EXPLICIT_VALUE_CHANGED]
