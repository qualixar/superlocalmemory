# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Ids, dates, paths, status words, everyday words and fragments never become
entities, from any source.

Every name below was a top entity on a clone of the author's store after the
first entity repair relinked facts from the names the old extractor had saved
("Users" 529 facts from /Users/... paths, "LIVE" 327, "VERIFIED" 282, 16-hex
fact ids 211 each, "2026-04-29 16:57" with 437 aliases). "Claude Code" was
typed as a person.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.encoding.entity_mentions import extract_mentions
from superlocalmemory.encoding.entity_resolver import EntityResolver, _guess_entity_type
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager

JUNK = ["Users", "User", "LIVE", "VERIFIED", "PASS", "FAIL", "DONE", "Final", "plan", "IST",
        "b2f4b74fbbad434c", "314a6feed02e400b", "c51df5026863416b", "cf43af4b67a347fd",
        "3f2b1c4e-9a7d-4e2b-8c1d-0a9e8f7d6c5b", "Configure", "Tested", "Comma",
        "2026-09-14 00:55", "2026-04-29 16:57", "/api/v3/consolidation/trigger",
        "/.well-known/oauth-protected-resource", "Xenobot Section Still Has",
        "Check Acme", "spill over", "gate.py", "notes.md", "SCREEN RECORDING",
        "/api/*", "[Agantic_official]", "honest/critical beat", "Url", "AUTO-CREATED"]

NAMES = ["SuperLocalMemory", "Claude Code", "SLM 4.1.25", "Alex", "Acme", "pgbouncer",
         "vitamin D", "billing service", "llama3.2", "Ollama", "NASA",
         "The Mirror and the Light", "Bank of America", "Xenobot Section", "TCP/IP"]

NOTE = ("Final: PASS. Status LIVE and VERIFIED at 14:55 IST on 2026-09-14 00:55.\n"
        "Configure the gateway first. Tested on fact b2f4b74fbbad434c and "
        "cf43af4b67a347fd via /api/v3/consolidation/trigger and "
        "/.well-known/oauth-protected-resource.\n"
        "Users: see /Users/yourusername/Documents/notes.md. Comma placement fixed.\n"
        "Alex asked Acme about pgbouncer; the Xenobot Section Still Has gaps.")


#: Capitalised everyday words: junk where a sentence or line starts, a name
#: elsewhere ("the Final Fantasy save"), so they are judged against the text.
EVERYDAY = {"Users", "User", "Final", "Tested", "Comma"}


@pytest.mark.parametrize("name", sorted(set(JUNK) - EVERYDAY))
def test_a_junk_name_never_reaches_an_entity(tmp_path: Path, name: str) -> None:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(real_schema)
    assert EntityResolver(db, llm=None).resolve([name], "default") == {}


@pytest.mark.parametrize("name", sorted(EVERYDAY))
def test_an_everyday_word_from_a_line_start_is_not_a_name(name: str) -> None:
    from superlocalmemory.encoding.entity_validity import valid_names

    assert valid_names([name], f"{name}: done.\n{name} again later.") == []
    assert valid_names([name], f"We met the {name} team.") == [name]


@pytest.mark.parametrize("name", NAMES)
def test_a_real_name_still_becomes_an_entity(tmp_path: Path, name: str) -> None:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(real_schema)
    assert name in EntityResolver(db, llm=None).resolve([name], "default")


def test_the_extractor_finds_the_names_in_a_noisy_note_and_none_of_the_junk() -> None:
    got = set(extract_mentions(NOTE))
    assert {"Alex", "Acme", "pgbouncer", "Xenobot Section"} <= got
    assert not got & set(JUNK), got & set(JUNK)
    assert not any(n in got for n in ("Documents", "varun", "notes.md", "Status"))


def test_an_everyday_word_counts_only_where_no_sentence_starts() -> None:
    assert "Atlas" in extract_mentions("Check the Atlas report before the review.")
    assert "Final" not in extract_mentions("Final verdict: the build is green.")
    assert "Atlas" not in extract_mentions("Atlas of the world is on the shelf.")


def test_a_models_names_are_judged_against_the_text_on_save(tmp_path: Path) -> None:
    from superlocalmemory.core.store_pipeline import enrich_fact
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(real_schema)
    text = "Final run: PASS. Varun shipped the Atlas build at 14:55 IST."
    fact = AtomicFact(fact_id="f1", content=text, entities=[
        "Final", "PASS", "IST", "Varun", "Atlas", "14:55", "b2f4b74fbbad434c", "the user"])
    out = enrich_fact(fact, MemoryRecord(memory_id="m1", content=text), "default",
                      embedder=None, entity_resolver=EntityResolver(db, llm=None),
                      temporal_parser=None)
    assert sorted(out.entities) == ["Atlas", "Varun"]
    names = {db.execute("SELECT canonical_name FROM canonical_entities WHERE entity_id = ?",
                        (e,))[0]["canonical_name"] for e in out.canonical_entities}
    assert names == {"Atlas", "Varun"}


@pytest.mark.parametrize(("name", "kind"), [
    ("Claude Code", "concept"), ("Claude Opus", "concept"), ("Gemini Flash", "concept"),
    ("Xenobot Section", "concept"), ("Nimbus Analytics", "concept"), ("Final Report", "concept"),
    ("Alice Smith", "person"), ("Varun Pratap Bhardwaj", "person"), ("Priya Sharma", "person"),
])
def test_a_product_is_not_typed_as_a_person(name: str, kind: str) -> None:
    assert _guess_entity_type(name) == kind
