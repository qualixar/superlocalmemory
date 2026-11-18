# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-9: a round trip per importer through the real writer; origin; never overwrite; idempotent."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.feature_flags import env_name
from superlocalmemory.portability import service
from superlocalmemory.portability.model import ImportFormatError
from superlocalmemory.portability.service import ImportOff
from tests.test_portability import fixtures as fx
from tests.test_portability.conftest import WHO


def _memories(w, run_id):
    return [dict(r) for r in w.db.execute(
        "SELECT m.memory_id, m.content, m.session_id, m.speaker, m.role, m.session_date, "
        "m.scope, m.metadata_json FROM memories m JOIN import_run_items i "
        "ON i.memory_id = m.memory_id WHERE i.run_id = ? ORDER BY m.content", (run_id,))]


def test_chatgpt_round_trip_keeps_session_speaker_role_and_date(world) -> None:
    run = world.run("conversations.json", fx.dumps(fx.chatgpt_list()))
    assert run["state"] == "done" and run["saved_items"] == 4 and run["format"] == "chatgpt"
    rows = _memories(world, run["run_id"])
    assert sorted(r["content"] for r in rows) == sorted([
        "I prefer short answers with one example.", "Understood, short answers with an example.",
        "Our deploy day is Thursday.", "Noted: deploys happen on Thursday."])
    assert {r["session_id"] for r in rows} == {"import:chatgpt:conv-1"}
    assert {(r["role"], r["speaker"]) for r in rows} == {("user", "user"), ("assistant", "assistant")}
    assert {r["session_date"][:10] for r in rows} == {"2023-11-14"}  # the conversation's own time
    assert {r["scope"] for r in rows} == {"personal"}
    assert json.loads(rows[0]["metadata_json"])["_slm_import"]["run"] == run["run_id"]
    assert run["skipped"]["other_branch"] == 1 and run["skipped"]["empty_conversation"] == 1


def test_imported_facts_have_the_import_origin_the_bottom_of_the_lattice(world) -> None:
    from superlocalmemory.trust import origin_class as oc

    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    rows = world.db.execute(
        "SELECT o.channel, o.transform, o.source_type, o.actor_id FROM fact_origin o JOIN "
        "atomic_facts f ON f.fact_id = o.fact_id JOIN import_run_items i ON i.memory_id = "
        "f.memory_id WHERE i.run_id = ?", (run["run_id"],))
    assert rows and {r["channel"] for r in rows} == {"import"}
    assert {r["actor_id"] for r in rows} == {f"import:{run['run_id']}"}
    signals = oc.Signals(channel="import", transform="verbatim", source_type="http-import")
    assert oc.classify(signals).origin_class == oc.IMPORTED_FILE
    assert oc.rank_of(oc.IMPORTED_FILE) > oc.rank_of(oc.AGENT_INFERRED)


@pytest.mark.parametrize("name,payload,fmt,expected", [
    ("claude.json", fx.claude_list(), "claude", 2),
    ("mem0-list.json", fx.MEM0_ITEMS, "mem0", 2),
    ("mem0-results.json", {"results": fx.MEM0_ITEMS}, "mem0", 2),
    ("mem0-page.json", {"count": 5, "next": None, "previous": None, "results": fx.MEM0_ITEMS},
     "mem0", 2),
    ("agent.af", fx.letta_file(), "letta", 4),
])
def test_each_json_importer_round_trips(world, name, payload, fmt, expected) -> None:
    run = world.run(name, fx.dumps(payload))
    assert run["state"] == "done" and run["format"] == fmt and run["saved_items"] == expected
    assert world.count("import_run_items", "run_id = ?", (run["run_id"],)) == expected


def test_markdown_okf_and_claude_memory_text_round_trip(world) -> None:
    md = world.run("AGENTS.md", fx.AGENTS_MD.encode())
    assert md["format"] == "markdown" and md["saved_items"] == 3
    okf = world.run("release.md", fx.OKF_CONCEPT.encode())
    assert okf["format"] == "okf" and okf["saved_items"] == 2
    text = world.run("memory.txt", fx.CLAUDE_MEMORY_TXT.encode())
    assert text["format"] == "claude-memory" and text["saved_items"] == 3
    dated = [r for r in _memories(world, text["run_id"]) if r["content"].startswith("Runs a small")]
    assert dated[0]["session_date"].startswith("2026-01-10")


def test_a_second_import_of_the_same_file_saves_nothing_and_changes_nothing(world) -> None:
    first = world.run("conversations.json", fx.dumps(fx.chatgpt_list()))
    before = world.count("memories"), world.count("atomic_facts")
    again = world.run("conversations.json", fx.dumps(fx.chatgpt_list()))
    assert again["saved_items"] == 0 and again["skipped"]["already_there"] == 4
    assert (world.count("memories"), world.count("atomic_facts")) == before
    assert first["saved_items"] == 4


def test_import_never_overwrites_a_memory_that_is_already_there(world) -> None:
    from tests.test_truth.test_slot_pass import _fact

    existing = _fact(world.db, "Prefers short answers.", "2026-03-01T00:00:00+00:00")
    world.db.execute("UPDATE atomic_facts SET evidence_count = 7 WHERE fact_id = ?", (existing,))
    memory_id = dict(world.db.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?",
                                      (existing,))[0])["memory_id"]
    before = [dict(r) for r in world.db.execute("SELECT * FROM memories WHERE memory_id = ?",
                                                (memory_id,))]
    facts_before = [dict(r) for r in world.db.execute("SELECT * FROM atomic_facts "
                                                      "WHERE fact_id = ?", (existing,))]
    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))  # also holds "Prefers short answers."
    assert run["saved_items"] == 1 and run["skipped"]["already_there"] == 1
    assert [dict(r) for r in world.db.execute("SELECT * FROM memories WHERE memory_id = ?",
                                              (memory_id,))] == before
    assert [dict(r) for r in world.db.execute("SELECT * FROM atomic_facts WHERE fact_id = ?",
                                              (existing,))] == facts_before


def test_preview_writes_nothing_and_says_what_it_found_in_plain_words(world) -> None:
    from superlocalmemory.portability import readers

    before = world.count("memories"), world.count("import_runs")
    parsed, view = service.preview(world.engine, world.profile,
                                   readers.from_bytes("c.json", fx.dumps(fx.chatgpt_list())), None)
    assert view["summary"].startswith("We found 2 conversations from ChatGPT, November 2023.")
    assert "1 empty one will be skipped." in view["summary"]
    assert view["would_save"] == 4 and view["items"] == 4 and view["label"] == "ChatGPT"
    assert (world.count("memories"), world.count("import_runs")) == before
    assert not any(k in json.dumps(view) for k in ("Thursday", "short answers"))  # counts, no text


def test_preview_counts_what_is_already_in_memory(world) -> None:
    from superlocalmemory.portability import readers

    world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    _p, view = service.preview(world.engine, world.profile,
                               readers.from_bytes("m.json", fx.dumps(fx.MEM0_ITEMS)), None)
    assert view["would_save"] == 0 and view["already_there"] == 2
    assert "2 are already in your memory and will be left exactly as they are." in view["summary"]


def test_the_letta_preview_says_archival_memory_is_not_in_the_file(world) -> None:
    from superlocalmemory.portability import readers

    _p, view = service.preview(world.engine, world.profile,
                               readers.from_bytes("a.af", fx.dumps(fx.letta_file())), None)
    assert "archival memory" in view["summary"] and view["notes"]


def test_a_file_in_no_known_format_is_refused_naming_the_formats(world) -> None:
    from superlocalmemory.portability import readers

    with pytest.raises(ImportFormatError, match="ChatGPT, Claude, Letta, Mem0"):
        service.preview(world.engine, world.profile,
                        readers.from_bytes("x.json", b'{"hello": 1}'), None)


def test_off_is_refused_with_the_switch_named_and_the_kill_switch_stops_it(world, monkeypatch) -> None:
    from superlocalmemory.portability import readers

    docs = readers.from_bytes("m.json", fx.dumps(fx.MEM0_ITEMS))
    monkeypatch.setenv(env_name("portability.import"), "off")
    for call in (lambda: service.preview(world.engine, world.profile, docs, None),
                 lambda: service.start_run(world.engine, world.runtime, profile_id=world.profile,
                                           docs=docs, fmt=None, label="m.json", decider=WHO)):
        with pytest.raises(ImportOff, match="portability.import"):
            call()
    monkeypatch.setenv(env_name("portability.import"), "on")
    monkeypatch.setenv("SLM_4230_OFF", "1")
    with pytest.raises(ImportOff, match="kill switch"):
        service.preview(world.engine, world.profile, docs, None)
    assert world.count("import_runs") == 0


def test_one_import_runs_at_a_time_in_a_profile(world) -> None:
    from superlocalmemory.portability import runs

    with world.db.raw_connection() as conn:
        runs.create(conn, profile_id=world.profile, fmt="mem0", source_label="x", actor="a",
                    channel="cli", conversations=0, total_items=1, skipped={}, first_at="",
                    last_at="")
    with pytest.raises(service.ImportRefused, match="already running"):
        world.run("m.json", fx.dumps(fx.MEM0_ITEMS), wait=False)


def test_the_import_is_recorded_as_a_decision_by_whoever_started_it(world) -> None:
    from superlocalmemory.core import decisions

    run = world.run("c.json", fx.dumps(fx.chatgpt_list()))
    [row] = [d for d in decisions.recent(_ro(world), world.profile) if d["kind"] == "import"]
    assert (row["actor"], row["channel"], row["action"]) == (WHO.actor, WHO.channel, "run")
    assert row["subject_id"] == run["run_id"] and row["can_undo"]
    assert row["summary"] == "Imported 2 conversations from ChatGPT"


def _ro(world):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(world.db.db_path))


def test_text_that_differs_only_in_case_or_spacing_counts_as_already_there(world) -> None:
    from tests.test_truth.test_slot_pass import _fact

    _fact(world.db, "prefers  SHORT answers.", "2026-03-01T00:00:00+00:00")
    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    assert run["saved_items"] == 1 and run["skipped"]["already_there"] == 1
