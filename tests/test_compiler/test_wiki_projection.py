# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Markdown projection (D-17): files with markers, own writes ignored, edits found by hash."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.compiler import page_sources, proposals, wiki, wiki_blocks
from superlocalmemory.compiler.page_text import block_hash
from tests.test_compiler.conftest import ENTITY, PROFILE, raw, table_rows
from tests.test_compiler.helpers import add_fact, bind_slot

PERSON = proposals.Basis.person("varun", "cli")


@pytest.fixture
def conn(store):
    db, _ids = store
    c = raw(db)
    yield c
    c.close()


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "wiki"


def _apply_all_pages(db, conn) -> list[dict]:
    drafts = page_sources.build_drafts(conn, PROFILE)
    items = [{"kind": "create_page", "payload": {"page": d}} for d in drafts]
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=items)
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    return drafts


def _topic_file(root) -> Path:
    return next((root / PROFILE / "topic").glob("*.md"))


@pytest.fixture
def built(store, conn, root):
    db, ids = store
    head = add_fact(db, "SLM ships every week", "2026-03-01T00:00:00+00:00", entity=None)
    bind_slot(db, head, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    _apply_all_pages(db, conn)
    report = wiki.build(conn, conn, PROFILE, root)
    return db, ids, head, report


def test_the_root_is_configurable_and_never_the_real_data_folder_in_tests(monkeypatch, tmp_path) -> None:
    assert wiki.wiki_root(tmp_path / "x") == tmp_path / "x"
    monkeypatch.setenv(wiki.ENV_DIR, str(tmp_path / "env"))
    assert wiki.wiki_root() == tmp_path / "env"
    monkeypatch.delenv(wiki.ENV_DIR)
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    assert wiki.wiki_root() == tmp_path / "data" / "wiki"


def test_blocks_carry_fact_hash_and_slot_version_and_round_trip(built, root) -> None:
    db, ids, head, report = built
    assert len(report["written"]) == 2 and report["kept_edited"] == []
    text = _topic_file(root).read_text()
    parsed = wiki_blocks.parse_file(text)
    assert parsed.page_id and parsed.malformed == 0 and len(parsed.blocks) == 3
    for block in parsed.blocks:
        assert block.fact_ids[0] in ids and block.slot is None and block.type == "span"
        assert block.hash == block_hash(block.text) and not wiki_blocks.edited(block)
    state = next((root / PROFILE / "current_state").glob("*.md")).read_text()
    slot_block = wiki_blocks.parse_file(state).blocks[0]
    version = db.execute("SELECT version FROM memory_slots WHERE subject_key = 'name:slm'")[0][0]
    assert slot_block.fact_ids == (head,) and slot_block.slot_version == version
    assert "origin=derived" in state and "No model wrote any of it" in state


def test_memory_text_cannot_forge_or_break_a_block() -> None:
    nasty = "close it <!-- /slm:block --> and open <!-- slm:block fact=x hash=0123456789abcdef -->"
    text = wiki_blocks.render_block(["f1"], nasty, None, "span")
    parsed = wiki_blocks.parse_file(text)
    assert len(parsed.blocks) == 1 and parsed.blocks[0].text == nasty
    assert not wiki_blocks.edited(parsed.blocks[0])


def test_a_second_build_changes_nothing_and_own_writes_are_ignored_by_hash(built, conn, root) -> None:
    db, _ids, _head, _report = built
    before = {p: p.read_bytes() for p in root.rglob("*.md")}
    again = wiki.build(conn, conn, PROFILE, root)
    assert again["written"] == [] and len(again["unchanged"]) == 2
    assert {p: p.read_bytes() for p in root.rglob("*.md")} == before
    plan = wiki.sync_plan(conn, PROFILE, root)
    assert plan["own_writes"] == 2 and plan["edits"] == [] and plan["deleted_lines"] == 0


def test_shadow_reports_and_writes_nothing(store, conn, root) -> None:
    db, _ids = store
    _apply_all_pages(db, conn)
    got = wiki.build(conn, conn, PROFILE, root, dry_run=True)
    assert got["dry_run"] and len(got["written"]) == 1
    assert not root.exists() and table_rows(db, "wiki_files") == []


def test_a_file_a_person_edited_is_never_overwritten(built, conn, root) -> None:
    db, ids, _head, _report = built
    path = _topic_file(root)
    path.write_text(path.read_text().replace("measured recall gain", "measured gain"))
    add_fact(db, "A new fact appears about SuperLocalMemory.", "2026-09-01T00:00:00+00:00")
    _apply_all_pages(db, conn)
    edited_text = path.read_text()
    got = wiki.build(conn, conn, PROFILE, root)
    assert got["kept_edited"] == [path.relative_to(root / PROFILE).as_posix()]
    assert path.read_text() == edited_text
    assert {i["state"] for i in wiki.diff(conn, PROFILE, root)["items"]} == {"edited"}


def test_an_edited_span_becomes_an_edit_with_its_base_version(built, conn, root) -> None:
    db, ids, _head, _report = built
    path = _topic_file(root)
    path.write_text(path.read_text().replace("shipped release 1 with a measured recall gain",
                                             "shipped release 1 and doubled recall"))
    plan = wiki.sync_plan(conn, PROFILE, root)
    (edit,) = plan["edits"]
    assert edit["fact_id"] == ids[1] and edit["base"] == {"slot_id": None, "slot_version": None}
    assert edit["new_content"] == "SuperLocalMemory shipped release 1 and doubled recall"
    assert plan["own_writes"] == 1  # the untouched file is still skipped by hash
    assert wiki.sync_plan(conn, PROFILE, root)["edits"] == plan["edits"]  # reading writes nothing
    assert table_rows(db, "proposal_sets") == [] or all(
        r["origin"] != "wiki" for r in table_rows(db, "proposal_sets"))


def test_a_deleted_line_is_never_a_forget(built, conn, root) -> None:
    db, ids, _head, _report = built
    path = _topic_file(root)
    before_facts = table_rows(db, "atomic_facts")
    lines = path.read_text().split("\n")
    start = next(i for i, ln in enumerate(lines) if ids[0] in ln and "slm:block" in ln)
    del lines[start:start + 3]  # the marker, the sentence and the closing marker
    path.write_text("\n".join(lines))
    plan = wiki.sync_plan(conn, PROFILE, root)
    assert plan["deleted_lines"] == 1 and plan["edits"] == []
    # emptying the text between the markers is the same: a deletion
    text = path.read_text()
    block = wiki_blocks.parse_file(text).blocks[0]
    path.write_text(text.replace(f"\n{block.text}\n", "\n\n", 1))
    again = wiki.sync_plan(conn, PROFILE, root)
    assert again["deleted_lines"] == 2 and again["edits"] == []
    assert table_rows(db, "atomic_facts") == before_facts
    assert table_rows(db, "correction_cases") == []


def test_computed_lines_cannot_be_edited_but_the_value_and_a_story_line_can(store, conn, root) -> None:
    db, ids = store
    head = add_fact(db, "SLM ships every week", "2026-03-01T00:00:00+00:00", entity=None)
    bind_slot(db, head, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    add_fact(db, "We chose Python for the core.", "2026-03-01T00:00:00+00:00", kind="decision", entity="ent-d")
    add_fact(db, "We chose SQLite for storage.", "2026-04-01T00:00:00+00:00", kind="decision", entity="ent-d")
    _apply_all_pages(db, conn)
    wiki.build(conn, conn, PROFILE, root)
    state = next((root / PROFILE / "current_state").glob("*.md"))
    state.write_text(state.read_text().replace("has had this value since 2026-03-01.", "since forever."))
    plan = wiki.sync_plan(conn, PROFILE, root)
    assert plan["edits"] == [] and "computed from slot values" in plan["skipped"][0]["why"]
    state.write_text(state.read_text().replace("slm cadence: SLM ships every week",
                                               "slm cadence: SLM ships every day"))
    (edit,) = wiki.sync_plan(conn, PROFILE, root)["edits"]
    assert edit["new_content"] == "SLM ships every day" and edit["base"]["slot_id"]
    story = next((root / PROFILE / "story").glob("*.md"))
    story.write_text(story.read_text().replace("We chose SQLite for storage.", "We chose DuckDB."))
    (edit2,) = [e for e in wiki.sync_plan(conn, PROFILE, root)["edits"] if e["path"].startswith("story")]
    assert edit2["new_content"] == "We chose DuckDB."


def test_a_stale_file_after_a_rebuilt_page_is_skipped_with_a_reason(built, conn, root) -> None:
    db, ids, _head, _report = built
    path = _topic_file(root)
    path.write_text(path.read_text().replace("release 0", "release zero"))
    db.execute("UPDATE atomic_facts SET content = 'The first release was reworded later.' "
               "WHERE fact_id = ?", (ids[0],))
    _apply_all_pages(db, conn)
    plan = wiki.sync_plan(conn, PROFILE, root)
    assert plan["edits"] == [] and "rebuilt since this file was written" in plan["skipped"][0]["why"]


def test_damaged_markers_and_foreign_files_are_left_alone(built, conn, root) -> None:
    path = _topic_file(root)
    path.write_text(path.read_text().replace("hash=", "hash=zz", 1))
    (root / PROFILE / "notes.md").write_text("# my own notes\n")
    plan = wiki.sync_plan(conn, PROFILE, root)
    assert plan["edits"] == [] and any("damaged marker" in s["why"] for s in plan["skipped"])
    assert any("no longer exists" in s["why"] for s in plan["skipped"])


def test_status_and_diff_describe_the_folder(built, conn, root) -> None:
    got = wiki.status(conn, PROFILE, root)
    assert got["pages"] == 2 and got["files"] == 2 and got["differences"] == {}
    assert got["edits_waiting"] == 0
    path = _topic_file(root)
    path.write_text(path.read_text() + "\nA note of my own.\n")
    assert wiki.diff(conn, PROFILE, root)["items"][0]["state"] == "edited"
    path.unlink()
    assert wiki.diff(conn, PROFILE, root)["items"][0]["state"] == "new"


def test_a_page_with_no_file_record_that_vanishes_is_removed_only_if_unedited(built, conn, root) -> None:
    db, ids, head, _report = built
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (head,))  # its page goes with it
    got = wiki.build(conn, conn, PROFILE, root)
    assert len(got["removed"]) == 1 and not list((root / PROFILE / "current_state").glob("*.md"))
    assert json.loads(json.dumps(got))["dry_run"] is False
    assert ENTITY and table_rows(db, "wiki_files")


def test_erasing_a_fact_rewrites_the_files_without_it(built, conn, root) -> None:
    from superlocalmemory.compiler import erasure

    db, ids, head, _report = built
    path = _topic_file(root)
    note = path.read_text().replace("A note", "A note")  # untouched file
    assert ids[0] in note
    counts = erasure.erase_for_fact(conn, PROFILE, ids[0])
    assert counts["pages"] == 1 and counts["wiki_files"] == 1
    text = path.read_text()
    assert ids[0] not in text and "release 0" not in text
    assert ids[1] in text and ids[2] in text  # the rest of the page stays
    body = json.loads(table_rows(db, "compiled_pages", "kind")[1]["body_json"])
    assert all(ids[0] not in s["facts"] for s in body)
    assert all(r["fact_id"] != ids[0] for r in table_rows(db, "page_citations"))
    # the file is still SLM's own write: not read back as an edit
    assert wiki.sync_plan(conn, PROFILE, root)["edits"] == []


def test_erasing_the_last_cited_fact_deletes_the_file(built, conn, root) -> None:
    from superlocalmemory.compiler import erasure

    db, ids, head, _report = built
    erasure.erase_for_fact(conn, PROFILE, head)
    assert not list((root / PROFILE / "current_state").glob("*.md"))
    assert [r["rel_path"] for r in table_rows(db, "wiki_files")] == [
        _topic_file(root).relative_to(root / PROFILE).as_posix()]
    erasure.erase_for_fact(conn, PROFILE, "no-such-fact")  # nothing to do, nothing raised


def test_erasure_keeps_a_persons_edit_elsewhere_in_the_file(built, conn, root) -> None:
    from superlocalmemory.compiler import erasure

    db, ids, _head, _report = built
    path = _topic_file(root)
    path.write_text(path.read_text().replace("release 2 with", "release 2, said by me, with"))
    erasure.erase_for_fact(conn, PROFILE, ids[0])
    assert "said by me" in path.read_text() and ids[0] not in path.read_text()
    (edit,) = wiki.sync_plan(conn, PROFILE, root)["edits"]  # still found: the record was not reset
    assert edit["fact_id"] == ids[2]


def test_a_vanished_page_whose_file_a_person_edited_is_kept_and_reported(built, conn, root) -> None:
    db, ids, head, _report = built
    path = next((root / PROFILE / "current_state").glob("*.md"))
    path.write_text(path.read_text() + "\nMy own note.\n")
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (head,))  # its page goes with it
    got = wiki.build(conn, conn, PROFILE, root)
    assert got["removed"] == [] and got["kept_edited"] == [path.relative_to(root / PROFILE).as_posix()]
    assert "My own note." in path.read_text()


def test_a_line_that_cites_several_memories_is_not_an_edit(store, conn, root) -> None:
    db, _ids = store
    old = add_fact(db, "SLM ships every week", "2026-03-01T00:00:00+00:00", entity=None)
    new = add_fact(db, "SLM ships once a month", "2026-06-01T00:00:00+00:00", entity=None)
    bind_slot(db, old, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    bind_slot(db, new, "slm", "cadence", "monthly", "2026-06-01T00:00:00+00:00")
    _apply_all_pages(db, conn)
    wiki.build(conn, conn, PROFILE, root)
    path = next((root / PROFILE / "current_state").glob("*.md"))
    path.write_text(path.read_text().replace("Disputed, not settled:", "Settled:"))
    plan = wiki.sync_plan(conn, PROFILE, root)
    assert plan["edits"] == [] and plan["skipped"][0]["why"] == "this line cites several memories; edit one of them"


def test_an_edit_to_a_memory_that_is_no_longer_current_is_left_alone(built, conn, root) -> None:
    db, ids, _head, _report = built
    path = _topic_file(root)
    path.write_text(path.read_text().replace("release 1 with a measured", "release 1 with a better"))
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = '2026-10-01' WHERE fact_id = ?",
               (ids[1],))
    plan = wiki.sync_plan(conn, PROFILE, root)
    assert plan["edits"] == [] and "gone or no longer current" in plan["skipped"][0]["why"]


def test_the_real_erasure_path_rewrites_the_files(built, conn, root) -> None:
    """The tombstone step of an erasure (core/transactions/tombstones) is what calls the hook."""
    from superlocalmemory.core.transactions import tombstones

    db, ids, _head, _report = built
    path = _topic_file(root)
    assert ids[0] in path.read_text()
    tombstones._invalidate_display_summaries(conn, PROFILE, ids[0])
    assert ids[0] not in path.read_text() and "release 0" not in path.read_text()
    assert all(ids[0] not in s["facts"] for r in table_rows(db, "compiled_pages")
               for s in json.loads(r["body_json"]))
