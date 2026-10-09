# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Fast tests: every check must fail when its evidence is missing or bad."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _slm_env as env  # noqa: E402
import _verdicts as v  # noqa: E402
import build_fixture as bf  # noqa: E402
import corpus  # noqa: E402

IDS = [[m.id for m in corpus.MEMORIES][i:i + 5] for i in range(12)]


def exp_runs():
    """Per-query top-5 that contain the expected id for every query."""
    return [list(q.expected) + ["QX-9999"] for q in corpus.QUERIES]


def base(runs=None, errors=None):
    r = exp_runs()
    return {"runs": runs if runs is not None else [r, r], "errors": errors or [], "status": "s"}


# --- recall -------------------------------------------------------------------

def test_recall_passes_when_clean_and_equal():
    r = v.recall_check(base(), exp_runs(), [], "s")
    assert r["passed"] is True and r["order_identical"] is True


def test_recall_fails_on_reference_errors():
    assert v.recall_check(base(errors=["x"]), exp_runs(), [], "s")["passed"] is False


def test_recall_fails_on_new_errors():
    assert v.recall_check(base(), exp_runs(), ["q: boom"], "s")["passed"] is False


def test_recall_fails_when_nothing_was_possible():
    empty = [[] for _ in corpus.QUERIES]
    r = v.recall_check(base(runs=[empty, empty]), empty, [], "s")
    assert r["passed"] is False and r["errors"]


def test_recall_fails_without_new_results_or_runs():
    assert v.recall_check(base(), [], [], "s")["passed"] is False
    assert v.recall_check(base(runs=[exp_runs()]), exp_runs(), [], "s")["passed"] is False


def test_recall_fails_when_expected_hits_drop_below_baseline():
    wrong = [["QX-9990", "QX-9991"] for _ in corpus.QUERIES]
    r = v.recall_check(base(), wrong, [], "s")
    assert r["passed"] is False


def test_order_identical_notices_reordering():
    swapped = [list(reversed(x)) for x in exp_runs()]
    r = v.recall_check(base(), swapped, [], "s")
    assert r["order_identical"] is False
    assert r["verdict"] == "identical"  # same sets


def test_recall_json_without_results_key_is_an_error():
    rows, _, err = v.parse_recall("{\"data\": {\"channel_status\": {}}}")
    assert rows == [] and "results" in err
    rows, _, err = v.parse_recall("noise {\"results\": []}")
    assert err == "" and rows == []
    assert v.parse_recall("not json")[2]
    assert v.parse_recall("{\"results\": null}")[2]


# --- upgrade ------------------------------------------------------------------

def good_upgrade():
    return {"daemon_up": True, "schema_version": 54, "contents_unchanged": True, "corpus_after": 40,
            "readiness": {"migrations": True, "migration_failures": []}, "timed_out": False,
            "memory_counts_unchanged": True, "errors": []}


def test_upgrade_ok_baseline():
    assert v.upgrade_ok(good_upgrade()) is True


@pytest.mark.parametrize("change", [
    {"daemon_up": False}, {"schema_version": 53}, {"schema_version": None}, {"contents_unchanged": False},
    {"corpus_after": 39}, {"readiness": {}}, {"readiness": {"migrations": False}},
    {"readiness": {"migrations": True, "migration_failures": ["m1"]}}, {"timed_out": True},
    {"memory_counts_unchanged": False}, {"errors": ["Traceback"]},
])
def test_upgrade_fails_closed(change):
    assert v.upgrade_ok({**good_upgrade(), **change}) is False


def test_upgrade_missing_evidence_fails():
    d = good_upgrade()
    del d["readiness"]
    assert v.upgrade_ok(d) is False


# --- downgrade ----------------------------------------------------------------

def good_down():
    return {"prepare_rc": 0, "daemon_up": True, "schema_after_prepare": 51, "schema_version": 51,
            "queries_answered": 12, "corpus_before": 40, "corpus_after": 40, "errors": []}


def test_downgrade_ok_baseline():
    assert v.downgrade_ok(good_down(), {"verdict": "identical"}, 54) is True


@pytest.mark.parametrize("change", [
    {"prepare_rc": 1}, {"daemon_up": False}, {"schema_after_prepare": 54}, {"schema_after_prepare": None},
    {"schema_version": 50}, {"schema_version": 55}, {"schema_version": None},
    {"queries_answered": 11}, {"corpus_after": 39}, {"errors": ["SchemaVersionError"]},
])
def test_downgrade_fails_closed(change):
    assert v.downgrade_ok({**good_down(), **change}, {"verdict": "identical"}, 54) is False


def test_older_version_may_remigrate_up_to_its_own_ceiling_only():
    d = {**good_down(), "schema_version": 53}
    assert v.downgrade_ok(d, {"verdict": "identical"}, 53) is True
    assert v.downgrade_ok({**d, "schema_version": 54}, {"verdict": "identical"}, 53) is False


def test_downgrade_needs_a_recall_verdict_that_is_not_worse():
    assert v.downgrade_ok(good_down(), None, 54) is False
    assert v.downgrade_ok(good_down(), {"verdict": "worse"}, 54) is False
    assert v.downgrade_ok(good_down(), {"verdict": "within_noise"}, 54) is True


def test_core_counts_ignore_derived_tables_but_not_user_data():
    before = {"memories": 40, "atomic_facts": 40, "projection_outbox": 40}
    assert v.core_counts_unchanged(before, {"memories": 40, "atomic_facts": 40, "projection_outbox": 0, "new": 1})
    assert not v.core_counts_unchanged(before, {"memories": 39, "atomic_facts": 40})
    assert not v.core_counts_unchanged(before, {"atomic_facts": 40})
    assert not v.core_counts_unchanged({"projection_outbox": 1}, {"projection_outbox": 1})  # no core table at all


def test_mapping_diff_reports_added_removed_changed():
    d = v.mapping_diff({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert d == {"added": ["c"], "removed": ["a"], "changed": ["b"]}


# --- restore ------------------------------------------------------------------

def test_restore_ok_requires_schema_columns_and_rows():
    want = {"t": ["a", "b"]}
    ok = dict(want_cols=want, got_cols=want, want_counts={"t": 2}, got_counts={"t": 2},
              want_schema=51, got_schema=51, contents_ok=True)
    assert v.restore_ok(**ok) is True
    assert v.restore_ok(**{**ok, "got_schema": 54}) is False
    assert v.restore_ok(**{**ok, "want_schema": None, "got_schema": None}) is True
    assert v.restore_ok(**{**ok, "got_cols": {"t": ["a"]}}) is False
    assert v.restore_ok(**{**ok, "got_cols": {"t": ["a", "b"], "extra": ["x"]}}) is False
    assert v.restore_ok(**{**ok, "got_counts": {"t": 3}}) is False
    assert v.restore_ok(**{**ok, "contents_ok": False}) is False


def test_snapshot_status_needs_restore_points_rc_zero():
    restores = [{"ok": True}]
    assert v.snapshot_passed(restores, restore_points_rc=0) is True
    assert v.snapshot_passed(restores, restore_points_rc=1) is False
    assert v.snapshot_passed([{"ok": True}, {"ok": False}], restore_points_rc=0) is False
    assert v.snapshot_passed([], restore_points_rc=0) is False


# --- overall ------------------------------------------------------------------

def test_check_ok_treats_na_as_acceptable_and_missing_as_failure():
    assert v.check_ok({"passed": True}) is True
    assert v.check_ok({"passed": None, "status": "n/a"}) is True
    assert v.check_ok({"passed": None}) is False
    assert v.check_ok({}) is False
    assert v.check_ok({"passed": False, "status": "n/a"}) is False
    assert v.failing({"a": {"passed": True}, "b": {"passed": None, "status": "n/a"}, "c": {}}) == ["c"]


# --- sqlite helpers -----------------------------------------------------------

def make_db(path: Path, ddl: list[str]):
    conn = sqlite3.connect(path)
    for s in ddl:
        conn.execute(s)
    conn.commit()
    conn.close()


def test_schema_signature_and_columns_detect_changes(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    make_db(a, ["CREATE TABLE t (x INTEGER)"])
    make_db(b, ["CREATE TABLE t (x INTEGER)"])
    assert env.schema_signature(a) == env.schema_signature(b)
    assert env.table_columns(a) == {"t": ["x"]}
    make_db(b, ["ALTER TABLE t ADD COLUMN y TEXT"])
    assert env.schema_signature(a) != env.schema_signature(b)
    assert env.table_columns(b) == {"t": ["x", "y"]}


def test_base_table_counts_ignore_fts_shadow_tables(tmp_path):
    db = tmp_path / "m.db"
    make_db(db, ["CREATE TABLE facts (x)", "INSERT INTO facts VALUES (1)",
                 "CREATE VIRTUAL TABLE facts_fts USING fts5(x)"])
    assert env.base_table_counts(db) == {"facts": 1}


# --- snapshot n/a -------------------------------------------------------------

def test_snapshot_na_only_when_both_schemas_unchanged(tmp_path):
    for side in ("f", "u"):
        (tmp_path / side).mkdir()
        for name in ("memory.db", "learning.db"):
            make_db(tmp_path / side / name, ["CREATE TABLE t (x)"])
    assert v.schema_unchanged(tmp_path / "f", tmp_path / "u") is True
    make_db(tmp_path / "u" / "learning.db", ["CREATE TABLE extra (y)"])
    assert v.schema_unchanged(tmp_path / "f", tmp_path / "u") is False


# --- build --------------------------------------------------------------------

def test_short_fixture_is_a_build_error():
    assert bf.fixture_problems(settled=40, errors=[]) == []
    assert bf.fixture_problems(settled=39, errors=[])
    assert bf.fixture_problems(settled=40, errors=["x"]) == ["x"]


# --- reference interpreter ----------------------------------------------------

def test_run_checks_refuses_a_reference_that_is_not_4_1_24(monkeypatch, tmp_path):
    import upgrade_check as uc

    monkeypatch.setattr(env, "package_version", lambda py, package="superlocalmemory": "4.1.23")
    with pytest.raises(RuntimeError, match="expected superlocalmemory 4.1.24"):
        uc.run_checks(tmp_path, Path("/x/bin/python"), Path("/y/bin/python"), tmp_path)


def test_run_checks_refuses_a_down_interpreter_that_is_not_4_1_20(monkeypatch, tmp_path):
    import upgrade_check as uc

    versions = {"/x/bin/python": "4.1.24", "/z/bin/python": "4.1.21"}
    monkeypatch.setattr(env, "package_version", lambda py, package="superlocalmemory": versions[str(py)])
    with pytest.raises(RuntimeError, match="expected superlocalmemory 4.1.20"):
        uc.run_checks(tmp_path, Path("/x/bin/python"), Path("/y/bin/python"), tmp_path, Path("/z/bin/python"))


# --- snapshot classification, reasons, virtual tables -------------------------

def sig(*rows):
    return sorted(rows)


T = ("table", "t", "CREATE TABLE t (x)")
IDX = ("index", "idx_t", "CREATE INDEX idx_t ON t(x)")
TRG = ("trigger", "trg_t", "CREATE TRIGGER trg_t AFTER INSERT ON t BEGIN SELECT 1; END")
CORE = {"memories": 5, "atomic_facts": 9}


def test_unchanged_schema_is_not_applicable():
    r = v.snapshotless_verdict({"memory.db": sig(T, IDX)}, {"memory.db": sig(T, IDX)}, CORE, CORE)
    assert r["status"] == "n/a" and r["passed"] is None and r["errors"] == []


def test_additive_only_diff_is_n_a_additive_with_evidence():
    after = sig(T, IDX, ("table", "new", "CREATE TABLE new (y)"), ("index", "idx_new", "CREATE INDEX idx_new ON new(y)"),
                ("trigger", "trg_t", TRG[2] + " "))
    r = v.snapshotless_verdict({"memory.db": sig(T, IDX, TRG)}, {"memory.db": after}, CORE, CORE)
    assert r["status"] == "n/a-additive" and r["passed"] is None and r["errors"] == []
    assert r["diff"]["memory.db"]["added"] == ["idx_new", "new"]
    assert r["diff"]["memory.db"]["changed"] == ["trg_t"]


def test_changed_or_removed_table_fails():
    for after in (sig(("table", "t", "CREATE TABLE t (x, y)")), sig()):
        r = v.snapshotless_verdict({"memory.db": sig(T)}, {"memory.db": after}, CORE, CORE)
        assert r["passed"] is False and r["status"] == "fail" and r["errors"]


def test_user_data_count_change_fails_even_when_additive():
    after = sig(T, ("table", "new", "CREATE TABLE new (y)"))
    r = v.snapshotless_verdict({"memory.db": sig(T)}, {"memory.db": after}, CORE, {**CORE, "memories": 4})
    assert r["passed"] is False and any("memories" in e for e in r["errors"])


def test_check_ok_accepts_n_a_additive_only_with_null_passed():
    assert v.check_ok({"passed": None, "status": "n/a-additive"}) is True
    assert v.check_ok({"passed": False, "status": "n/a-additive"}) is False


def test_failing_snapshot_always_has_a_reason():
    assert v.snapshot_reasons([], 0)
    assert v.snapshot_reasons([{"ok": False}], 0)
    assert v.snapshot_reasons([{"ok": False, "reasons": ["columns differ"]}], 0) == ["columns differ"]
    assert v.snapshot_reasons([{"ok": True}], 1)
    assert v.snapshot_reasons([{"ok": True}], 0) == []


def test_restore_problems_name_the_failed_condition_and_allow_extra_empty_tables():
    want = {"t": ["a"]}
    good = dict(want_cols=want, got_cols={**want, "memory_events": ["id"]}, want_counts={"t": 2},
                got_counts={"t": 2, "memory_events": 0}, want_schema=None, got_schema=None, contents_ok=True)
    assert v.restore_problems(**good) == [] and v.restore_ok(**good) is True
    bad = v.restore_problems(**{**good, "got_counts": {"t": 2, "memory_events": 3}})
    assert bad and "memory_events" in bad[0]
    assert v.restore_problems(**{**good, "contents_ok": False})
    assert v.restore_problems(**{**good, "got_cols": {"t": ["a", "b"]}})


def test_virtual_table_does_not_crash_the_reader(tmp_path):
    db = tmp_path / "v.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE plain (a, b)")
    # a module sqlite cannot load here: the row exists in sqlite_master but any read of it fails
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute("INSERT INTO sqlite_master (type,name,tbl_name,rootpage,sql) VALUES "
                 "('table','vec_x','vec_x',0,'CREATE VIRTUAL TABLE vec_x USING vec0(e float[3])')")
    conn.commit()
    conn.close()
    cols = env.table_columns(db)
    assert cols["plain"] == ["a", "b"] and "vec_x" in cols and cols["vec_x"]
    assert env.table_counts(db)["vec_x"] == -1


def test_unreadable_table_becomes_an_error_entry(tmp_path, monkeypatch):
    db = tmp_path / "u.db"
    make_db(db, ["CREATE TABLE ok (a)", "CREATE TABLE bad (b)"])
    real = env._pragma_columns

    def flaky(conn, name):
        if name == "bad":
            raise sqlite3.OperationalError("no such module: vec0")
        return real(conn, name)
    monkeypatch.setattr(env, "_pragma_columns", flaky)
    cols = env.table_columns(db)
    assert cols["ok"] == ["a"] and cols["bad"][0].startswith("<unreadable")
