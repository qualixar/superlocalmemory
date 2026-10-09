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
    return {"prepare_rc": 0, "daemon_up": True, "schema_version": 51, "queries_answered": 12,
            "corpus_before": 40, "corpus_after": 40, "errors": []}


def test_downgrade_ok_baseline():
    assert v.downgrade_ok(good_down(), {"verdict": "identical"}) is True


@pytest.mark.parametrize("change", [
    {"prepare_rc": 1}, {"daemon_up": False}, {"schema_version": 54}, {"schema_version": None},
    {"queries_answered": 11}, {"corpus_after": 39}, {"errors": ["SchemaVersionError"]},
])
def test_downgrade_fails_closed(change):
    assert v.downgrade_ok({**good_down(), **change}, {"verdict": "identical"}) is False


def test_downgrade_needs_a_recall_verdict_that_is_not_worse():
    assert v.downgrade_ok(good_down(), None) is False
    assert v.downgrade_ok(good_down(), {"verdict": "worse"}) is False
    assert v.downgrade_ok(good_down(), {"verdict": "within_noise"}) is True


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
