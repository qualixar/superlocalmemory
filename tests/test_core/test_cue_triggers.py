# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-8: cue triggers. A cue fires on the matching step, only then, once per
conversation; a lease warning follows an edit; erasing a fact erases its cues."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from superlocalmemory.core import cue_binding as cb
from superlocalmemory.core import cue_triggers as ct
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M078_memory_cues as M078
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.truth.invariants import open_read_only

TODAY = date(2026, 11, 18)


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _memory(db, content="Run the schema migration check before editing the schema.", *,
            cues=(), kind=None, sentences=("It is a two step check.",)):
    """A memory with its whole-text fact first, then one fact per sentence."""
    record = MemoryRecord(content=content, session_date="2026-11-16",
                          metadata={cb.METADATA_KEY: list(cues)} if cues else {})
    db.store_memory(record)
    first = AtomicFact(memory_id=record.memory_id, content=content)
    db.store_fact(first)
    for sentence in sentences:
        db.store_fact(AtomicFact(memory_id=record.memory_id, content=sentence))
    if kind:
        db.execute("UPDATE atomic_facts SET memory_kind = ? WHERE memory_id = ?",
                   (kind, record.memory_id))
    return first.fact_id


def _bind(db) -> int:
    with db.raw_connection() as conn:
        return cb.bind_explicit(conn)[0] + cb.bind_extracted(conn)


def _ev(name="PreToolUse", tool="Edit", cwd="/work/proj", response="", **tool_input):
    return ct.Event(name, tool, tool_input, response, cwd, "sess-1")


def _find(db, event, **kw):
    conn = open_read_only(str(db.db_path))
    try:
        return ct.find(conn, "default", event, today=TODAY, **kw)
    finally:
        conn.close()


def _rows(db, sql):
    conn = sqlite3.connect(str(db.db_path))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


# -- the cue format, refused at the door ------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    "path:src/**/schema.py", "tool:Bash:git push", "tool:Edit", "error:database is locked",
    "repo:superlocalmemory", "date:2026-11-20", "PATH:src/a.py", " repo : x ",
])
def test_a_well_formed_cue_is_accepted(spec) -> None:
    assert ct.parse_cue(spec).spec.split(":")[0] in ct.KINDS


@pytest.mark.parametrize("spec", [
    "", "path", "nope:x", "path:", "path:   ", "date:tomorrow", "date:2026-13-40", "date:20261120",
    "repo:a/b", "tool:Bash:", "tool:bad name:x", "path:" + "a" * 201, "error:two\nlines",
    "path:" + "*/" * 9, None, 5, ["path:a"],
])
def test_a_malformed_cue_is_refused(spec) -> None:
    with pytest.raises(ct.CueError):
        ct.parse_cue(spec)


def test_a_list_of_cues_is_checked_whole_and_deduplicated() -> None:
    cues = ct.parse_cues(["path:a/b.py", "path:a/b.py", "repo:x"])
    assert [c.spec for c in cues] == ["path:a/b.py", "repo:x"]
    assert ct.parse_cues(None) == ()
    with pytest.raises(ct.CueError):
        ct.parse_cues(["path:a/b.py", "bad"])
    with pytest.raises(ct.CueError):
        ct.parse_cues("path:a/b.py")  # one string is not a list
    with pytest.raises(ct.CueError):
        ct.parse_cues([f"repo:r{i}" for i in range(ct.MAX_CUES + 1)])


# -- matching ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(("pattern", "path", "expected"), [
    ("src/**/schema.py", "/work/proj/src/storage/schema.py", True),
    ("src/**/schema.py", "/work/proj/src/schema.py", True),     # ** may match no folder
    ("src/**/schema.py", "/work/proj/src/storage/models.py", False),
    ("/work/proj/src/*.py", "/work/proj/src/a.py", True),
    ("/work/other/*.py", "/work/proj/src/a.py", False),
    ("*.md", "/work/proj/docs/guide.md", True),
    ("schema.py", "/work/proj/src/xschema.py", False),          # a tail must start at a folder
])
def test_path_globs(pattern, path, expected) -> None:
    assert ct.path_matches(pattern, path, "/work/proj") is expected


def test_a_path_cue_matches_relative_to_the_working_directory() -> None:
    assert ct.path_matches("src/a.py", "/work/proj/src/a.py", "/work/proj")
    assert not ct.path_matches("", "", "/work/proj")


def test_an_event_names_the_file_of_an_edit_or_read_only() -> None:
    assert _ev(file_path="/a/b.py").path == "/a/b.py"
    assert _ev(file_path="rel/b.py", cwd="/work").path == "/work/rel/b.py"
    assert _ev(tool="NotebookEdit", notebook_path="/a/n.ipynb").path == "/a/n.ipynb"
    assert _ev(tool="Bash", command="cat /a/b.py").path == ""
    assert _ev(file_path="/a/../a/b.py").path == "/a/b.py"


def test_tool_cues_match_the_name_and_a_part_of_the_command() -> None:
    push = ct.Cue("tool", "Bash:git push")
    assert ct.matches(push, _ev(tool="bash", command="cd x && GIT PUSH origin main"), TODAY)
    assert not ct.matches(push, _ev(tool="Bash", command="git status"), TODAY)
    assert not ct.matches(push, _ev(tool="Edit", command="git push"), TODAY)
    assert ct.matches(ct.Cue("tool", "Edit"), _ev(file_path="/a.py"), TODAY)


def test_a_tool_cue_part_is_a_substring_never_a_pattern() -> None:
    cue = ct.Cue("tool", "Bash:(a+)+$")
    assert not ct.matches(cue, _ev(tool="Bash", command="a" * 5000 + "!"), TODAY)
    assert ct.matches(cue, _ev(tool="Bash", command="echo (a+)+$ done"), TODAY)


def test_error_cues_read_only_what_a_tool_returned() -> None:
    cue = ct.Cue("error", "Database is locked")
    post = _ev("PostToolUse", "Bash", response="sqlite3.OperationalError: database is locked")
    assert ct.matches(cue, post, TODAY)
    assert not ct.matches(cue, _ev("PreToolUse", "Bash", response=post.tool_response), TODAY)
    late = _ev("PostToolUse", "Bash", response="x" * ct.ERROR_WINDOW + "database is locked")
    assert not ct.matches(cue, late, TODAY)  # only the first 8 KB is read


def test_a_payload_becomes_an_event_with_the_text_of_the_response() -> None:
    event = ct.event_from_payload({
        "hook_event_name": "PostToolUse", "tool_name": "Bash", "session_id": " s1 ", "cwd": "/w",
        "tool_input": {"command": "ls"},
        "tool_response": {"stdout": "ok", "stderr": "database is locked", "n": 3}})
    assert event.session_id == "s1" and event.cwd == "/w" and event.post
    assert "database is locked" in event.tool_response
    pre = ct.event_from_payload({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                 "tool_response": "ignored"})
    assert pre.tool_response == ""
    for bad in (None, [], {}, {"hook_event_name": "Stop", "tool_name": "Bash"},
                {"hook_event_name": "PreToolUse"}, {"hook_event_name": "PreToolUse", "tool_name": ""}):
        assert ct.event_from_payload(bad) is None


def test_repo_and_date_cues() -> None:
    repo = ct.Cue("repo", "superlocalmemory")
    assert ct.matches(repo, _ev(cwd="/Users/x/work/superlocalmemory/src"), TODAY)
    assert not ct.matches(repo, _ev(cwd="/Users/x/work/superlocalmemory-fork"), TODAY)
    assert not ct.matches(repo, _ev(cwd=""), TODAY)
    due = ct.Cue("date", "2026-11-18")
    assert ct.matches(due, _ev(), date(2026, 11, 18)) and ct.matches(due, _ev(), date(2027, 1, 1))
    assert not ct.matches(due, _ev(), date(2026, 11, 17))
    assert not ct.matches(ct.Cue("date", "garbage"), _ev(), TODAY)


# -- cues in a store: fires on the matching step, not on another -------------------------------------

def test_a_cue_fires_on_the_matching_edit_and_not_on_another(store) -> None:
    fact = _memory(store, cues=["path:src/**/schema.py"])
    assert _bind(store) == 1
    (hit,) = _find(store, _ev(file_path="/work/proj/src/storage/schema.py"))
    assert (hit.fact_id, hit.via, hit.reason) == (fact, "cue", "path:src/**/schema.py")
    assert _find(store, _ev(file_path="/work/proj/src/storage/models.py")) == []
    assert _find(store, _ev(tool="Bash", command="cat src/storage/schema.py")) == []


def test_each_kind_fires_from_the_store(store) -> None:
    by_cue = {spec: _memory(store, content=f"plan for {spec}", cues=[spec], sentences=())
              for spec in ("tool:Bash:git push", "error:database is locked", "repo:proj",
                           "date:2026-11-01", "path:a/b.py")}
    _bind(store)

    def fired(event):
        return {h.reason for h in _find(store, event)}

    push = _ev(tool="Bash", command="git push", cwd="/elsewhere")
    assert fired(push) == {"tool:Bash:git push", "date:2026-11-01"}
    assert fired(_ev(tool="Bash", command="ls", cwd="/work/proj")) == {"repo:proj", "date:2026-11-01"}
    locked = _ev("PostToolUse", "Bash", cwd="/elsewhere", command="make",
                 response="database is locked")
    assert fired(locked) == {"error:database is locked", "date:2026-11-01"}
    assert fired(_ev(file_path="/work/proj/a/b.py", cwd="/elsewhere")) == {
        "path:a/b.py", "date:2026-11-01"}
    assert {h.fact_id for h in _find(store, push)} == {
        by_cue["tool:Bash:git push"], by_cue["date:2026-11-01"]}


def test_a_memory_whose_several_cues_match_is_shown_once(store) -> None:
    fact = _memory(store, cues=["tool:Bash:git push", "repo:proj"])
    _bind(store)
    hits = _find(store, _ev(tool="Bash", command="git push"))
    assert [h.fact_id for h in hits] == [fact]


def test_a_cue_fires_once_per_conversation(store) -> None:
    fact = _memory(store, cues=["repo:proj"])
    _bind(store)
    assert [h.fact_id for h in _find(store, _ev(file_path="/a.py"))] == [fact]
    assert _find(store, _ev(file_path="/a.py"), shown=frozenset({fact})) == []


def test_a_retired_or_archived_memory_never_fires(store) -> None:
    gone, archived, live = (_memory(store, cues=["repo:proj"]) for _ in range(3))
    _bind(store)
    store.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (archived,))
    store.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (gone,))
    assert [h.fact_id for h in _find(store, _ev(file_path="/a.py"))] == [live]


def test_at_most_the_cap_is_shown_per_step_the_rest_wait(store) -> None:
    facts = [_memory(store, content=f"plan {i}", cues=["repo:proj"], sentences=())
             for i in range(ct.MAX_FIRED + 2)]
    _bind(store)
    first = [h.fact_id for h in _find(store, _ev(file_path="/a.py"))]
    assert len(first) == ct.MAX_FIRED
    rest = [h.fact_id for h in _find(store, _ev(file_path="/a.py"), shown=frozenset(first))]
    assert sorted(first + rest) == sorted(facts)


def test_cues_of_another_profile_do_not_fire(store) -> None:
    _memory(store, cues=["repo:proj"])
    _bind(store)
    conn = open_read_only(str(store.db_path))
    try:
        assert ct.find(conn, "someone-else", _ev(file_path="/a.py"), today=TODAY) == []
    finally:
        conn.close()


def test_a_store_without_the_cue_tables_answers_nothing() -> None:
    conn = sqlite3.connect(":memory:")
    assert ct.find(conn, "default", _ev(file_path="/a.py")) == []


# -- the lease re-check -----------------------------------------------------------------------------

def _lease(db, fact_id, path, kind="file"):
    db.execute("INSERT INTO memory_leases (fact_id, profile_id, source_kind, locator, fingerprint,"
               " state, leased_at, checked_at) VALUES (?, 'default', ?, ?, 'abc', 'fresh', 'x', 'x')",
               (fact_id, kind, path))


def test_editing_a_leased_file_warns_about_memories_saved_from_it(store, tmp_path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    fact = _memory(store, content="The port is 8765.")
    _lease(store, fact, str(src.resolve()))
    done = _ev("PostToolUse", "Edit", file_path=str(src))
    (hit,) = _find(store, done)
    assert (hit.fact_id, hit.via, hit.reason) == (fact, "lease", str(src))
    assert "You just edited" in ct.lease_note(hit.reason)


def test_the_lease_warning_needs_a_finished_edit_of_that_file(store, tmp_path) -> None:
    src, other = tmp_path / "a.yaml", tmp_path / "b.yaml"
    for p in (src, other):
        p.write_text("x")
    fact = _memory(store)
    _lease(store, fact, str(src.resolve()))
    assert _find(store, _ev("PreToolUse", "Edit", file_path=str(src))) == []     # not yet edited
    assert _find(store, _ev("PostToolUse", "Read", file_path=str(src))) == []    # only read
    assert _find(store, _ev("PostToolUse", "Edit", file_path=str(other))) == []   # another file
    assert _find(store, _ev("PostToolUse", "Bash", command="cat a.yaml")) == []
    assert _find(store, _ev("PostToolUse", "Write", file_path=str(src)))[0].fact_id == fact


def test_a_lease_warning_is_not_repeated_in_a_conversation(store, tmp_path) -> None:
    src = tmp_path / "a.yaml"
    src.write_text("x")
    fact = _memory(store)
    _lease(store, fact, str(src.resolve()))
    done = _ev("PostToolUse", "Edit", file_path=str(src))
    assert _find(store, done, shown={fact}) == []


def test_only_file_leases_warn(store, tmp_path) -> None:
    src = tmp_path / "a.yaml"
    src.write_text("x")
    _lease(store, _memory(store), str(src.resolve()), kind="url")
    assert _find(store, _ev("PostToolUse", "Edit", file_path=str(src))) == []


# -- binding ----------------------------------------------------------------------------------------

def test_cues_bind_to_the_first_fact_only_and_binding_is_idempotent(store) -> None:
    first = _memory(store, cues=["repo:proj", "path:a/b.py"], sentences=("one.", "two."))
    assert _bind(store) == 2
    assert _bind(store) == 0
    rows = _rows(store, "SELECT fact_id, origin, profile_id FROM memory_cues")
    assert {r[0] for r in rows} == {first} and {r[1] for r in rows} == {"explicit"}
    assert {r[2] for r in rows} == {"default"}


def test_the_watermark_moves_so_a_pass_never_rescans(store, tmp_path) -> None:
    _memory(store, cues=["repo:one"])
    with store.raw_connection() as conn:
        made, mark = cb.bind_explicit(conn, 0)
    assert made == 1 and mark > 0
    _memory(store, cues=["repo:two"])
    with store.raw_connection() as conn:
        assert cb.bind_explicit(conn, mark)[0] == 1
        assert cb.bind_explicit(conn, mark + 10_000)[0] == 0
    assert cb.run(store, tmp_path / "state") == 0
    assert (tmp_path / "state" / cb.STATE).is_file()
    assert cb.run(store, tmp_path / "state") == 0


def test_a_stored_cue_that_no_door_accepts_is_never_bound(store) -> None:
    _memory(store, cues=["nonsense", "repo:ok"])
    assert _bind(store) == 1
    assert _rows(store, "SELECT pattern FROM memory_cues") == [("ok",)]


def test_a_memory_without_cues_binds_nothing(store) -> None:
    _memory(store)
    assert _bind(store) == 0


# -- extraction from plans ---------------------------------------------------------------------------

def test_a_plan_that_names_a_file_gets_an_extracted_path_cue(store) -> None:
    fact = _memory(store, content="TODO: migrate src/storage/schema.py to the new layout, then "
                                  "update docs/guide.md.", kind="prospective")
    assert _bind(store) == 2
    rows = _rows(store, "SELECT fact_id, kind, pattern, origin FROM memory_cues ORDER BY pattern")
    assert rows == [(fact, "path", "docs/guide.md", "extracted"),
                    (fact, "path", "src/storage/schema.py", "extracted")]
    assert _find(store, _ev(file_path="/work/proj/src/storage/schema.py"))[0].fact_id == fact
    assert _bind(store) == 0


def test_extraction_is_conservative(store) -> None:
    _memory(store, content="Remember src/app/main.py needs a fix.", kind="rule")     # not a plan
    _memory(store, content="Plan: read https://example.com/docs/index.html later.", kind="prospective")
    _memory(store, content="Plan: finish the report by Friday.", kind="prospective")  # no path
    _memory(store, content="Plan: bump version 1.2/3.4 and ping a/b", kind="prospective")
    assert _bind(store) == 0


def test_extraction_binds_the_first_fact_and_caps_the_count() -> None:
    paths = cb.extract_paths("see a/b.py, c/d.py, e/f.py, g/h.py and a/b.py again")
    assert paths == ["a/b.py", "c/d.py", "e/f.py"]
    assert cb.extract_paths("edit ~/notes/todo.md and /etc/app/conf.yaml.") == [
        "~/notes/todo.md", "/etc/app/conf.yaml"]
    assert cb.extract_paths("") == [] and cb.extract_paths("../../x/y.py") == []


def test_extraction_binds_only_the_whole_text_fact(store) -> None:
    fact = _memory(store, content="Plan: tidy docs/guide.md.", kind="prospective",
                   sentences=("Plan: tidy docs/guide.md.",))
    _bind(store)
    assert {r[0] for r in _rows(store, "SELECT fact_id FROM memory_cues")} == {fact}


# -- erasing -----------------------------------------------------------------------------------------

def test_erasing_a_fact_erases_its_cues(store) -> None:
    keep = _memory(store, cues=["repo:keep"])
    gone = _memory(store, cues=["repo:gone", "path:x/y.py"])
    _bind(store)
    assert len(_rows(store, "SELECT * FROM memory_cues")) == 3
    store.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (gone,))
    assert {r[0] for r in _rows(store, "SELECT fact_id FROM memory_cues")} == {keep}


# -- the migration -----------------------------------------------------------------------------------

def test_m078_is_deferred_additive_and_repairable() -> None:
    assert _MODULES[M078.NAME] is M078
    assert M078.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M078.DOWNGRADE_FLOOR == 51 and M078.DB_TARGET == "memory"
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M078.verify(conn)
    M078.apply(conn)
    M078.apply(conn)
    assert M078.verify(conn)
    conn.execute("DROP TRIGGER trg_memory_cues_fact_erased")
    assert not M078.verify(conn)
    M078.repair(conn)
    assert M078.verify(conn)


def test_m078_builds_without_facts_and_checks_only_invariants() -> None:
    conn = sqlite3.connect(":memory:")
    M078.apply(conn)
    assert M078.verify(conn)
    conn.execute("INSERT INTO memory_cues VALUES ('c', 'p', 'f', 'anything', 'x', 'now', 'other')")
    for bad in ("('d', 'p', 'f', '', 'x', 'now', 'explicit')",
                "('e', 'p', 'f', 'path', '', 'now', 'explicit')"):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"INSERT INTO memory_cues VALUES {bad}")


def test_the_pass_runs_in_the_maintenance_scheduler(store, tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    _memory(store, cues=["repo:proj"])
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "home"))
    steps = []
    fake = SimpleNamespace(_running=True, _db=store,
                           _record_step=lambda *a: steps.append(a))
    MaintenanceScheduler._cue_pass(fake)
    assert steps == [("cue pass", True)]
    assert _rows(store, "SELECT pattern FROM memory_cues") == [("proj",)]
    fake._running = False
    MaintenanceScheduler._cue_pass(fake)
    assert len(steps) == 1


def test_the_cue_listing_shows_each_cue_with_its_memory(store) -> None:
    from superlocalmemory.cli.cues_cmd import list_cues

    _memory(store, cues=["repo:proj"])
    _bind(store)
    (row,) = list_cues(Path(store.db_path), "default")
    assert (row["kind"], row["pattern"], row["origin"]) == ("repo", "proj", "explicit")
    assert row["memory"].startswith("Run the schema migration check")
    assert list_cues(Path(store.db_path), "other") == []


# -- receipts ---------------------------------------------------------------------------------------

def test_a_cue_delivery_is_a_pull_path_that_is_never_randomised_or_withheld() -> None:
    from superlocalmemory.outcomes import randomiser

    assert "cue" in randomiser.PULL_PATHS <= randomiser.PATHS
    assert "cue" not in randomiser.RANDOMISED_PATHS
    cands = [randomiser.Candidate(f"f{i}", "recall", 1.0) for i in range(6)]
    for n in range(200):
        a = randomiser.choose(b"k" * 32, f"d{n}", "cue", cands, holdout_on=True)
        assert all(c.included and c.propensity == 1.0 for c in a.choices)


def test_a_cue_is_not_counted_as_the_agent_asking_again() -> None:
    from superlocalmemory.outcomes import effects, randomiser

    assert "cue" not in randomiser.ASKED_PATHS
    assert randomiser.ASKED_PATHS == randomiser.PULL_PATHS - {"cue"}
    _sql, params = effects._outcome_sql(effects.REASK)
    assert "cue" not in params and "mcp_recall" in params
