# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com | https://varunpratap.com

"""A feature gated on schema that is absent must be reported as never having run.

Falling back when a column is missing is correct behaviour -- it is what keeps an
old store openable. The problem is that it is indistinguishable from working:
nothing raises, static analysis passes, a call-graph trace reaches the code, and
coverage counts the line. Only asking the store answers the question.

The verdict that matters most is ``SATISFIED_ELSEWHERE``. The common case is not
that the data is missing; it is that the guard asks the wrong table. Reporting
where the data actually lives turns a dead feature into a one-line fix with no
migration, and that is the difference between a diagnostic and a bug report.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.reliability import check_schema_guards
from superlocalmemory.reliability.join_liveness import Guard, Requirement

_GUARD = (
    Guard(
        name="example_feature",
        describes="a feature gated on an author column",
        requires=(Requirement("trust_scores"), Requirement("facts", "created_by")),
        fallback_behaviour="every row takes a neutral weight, so the feature is inert",
    ),
)


def _store(path, *, with_column: bool, elsewhere: int | None = None, facts: int = 0):
    """``elsewhere`` provenance rows over ``facts`` rows, so coverage is
    expressible as a fraction rather than only as a presence flag."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trust_scores (source_id TEXT, alpha REAL, beta REAL)")
    cols = "fact_id TEXT PRIMARY KEY, content TEXT"
    if with_column:
        cols += ", created_by TEXT"
    conn.execute(f"CREATE TABLE facts ({cols})")
    if facts:
        conn.executemany(
            "INSERT INTO facts (fact_id, content) VALUES (?, ?)",
            [(f"f{i}", "x") for i in range(facts)],
        )
    if elsewhere is not None:
        conn.execute("CREATE TABLE provenance (fact_id TEXT, created_by TEXT)")
        conn.executemany(
            "INSERT INTO provenance VALUES (?, ?)",
            [(f"f{i}", f"client-{i}") for i in range(elsewhere)],
        )
    conn.commit()
    conn.close()
    return path


class TestALiveGuardIsReportedLive:
    def test_all_requirements_present(self, tmp_path) -> None:
        db = _store(tmp_path / "m.db", with_column=True)

        (v,) = check_schema_guards(db, guards=_GUARD)

        assert v.verdict == "LIVE"
        assert v.is_live
        assert v.missing == ()


class TestAGuardWhoseDataLivesElsewhereSaysSo:
    """The actionable case, and the one the live store actually exhibits."""

    def test_it_finds_the_column_on_another_table(self, tmp_path) -> None:
        db = _store(tmp_path / "m.db", with_column=False, elsewhere=4, facts=4)

        (v,) = check_schema_guards(db, guards=_GUARD)

        assert v.verdict == "SATISFIED_ELSEWHERE"
        assert v.missing == ("facts.created_by",)
        table, column, populated, coverage = v.found_elsewhere[0]
        assert (table, column, populated) == ("provenance", "created_by", 4)
        assert coverage == 100.0

    def test_it_names_the_table_the_row_count_and_the_remedy(self, tmp_path) -> None:
        db = _store(tmp_path / "m.db", with_column=False, elsewhere=12, facts=12)

        (v,) = check_schema_guards(db, guards=_GUARD)

        assert "provenance.created_by" in v.detail
        assert "12 populated rows" in v.detail
        assert "without a backfill" in v.detail
        # The fallback must be stated, because "off" and "computing the same
        # thing as being off" are different things to an operator.
        assert "inert" in v.detail

    def test_partial_coverage_is_not_offered_as_a_free_fix(self, tmp_path) -> None:
        """The correction that matters. A column fully populated on its OWN table
        can still describe only part of the set the join needs, and reporting the
        populated count alone overstates the remedy. This is the live store's
        actual shape: every provenance row carries an author, and 30% of facts
        have no provenance row at all."""
        db = _store(tmp_path / "m.db", with_column=False, elsewhere=7, facts=10)

        (v,) = check_schema_guards(db, guards=_GUARD)

        _, _, populated, coverage = v.found_elsewhere[0]
        assert populated == 7, "every row on its own table is populated"
        assert coverage == 70.0, "but it covers only 70% of what the guard needs"
        assert "only 70.0%" in v.detail
        assert "partial remedy" in v.detail
        assert "backfill decision" in v.detail
        assert "no migration" not in v.detail, (
            "must not promise a free fix at partial coverage"
        )

    def test_an_empty_column_elsewhere_is_not_offered_as_a_fix(
        self, tmp_path,
    ) -> None:
        """A column that exists but is unpopulated would need a backfill, so it
        must not be reported as a drop-in remedy."""
        db = _store(tmp_path / "m.db", with_column=False, elsewhere=0)

        (v,) = check_schema_guards(db, guards=_GUARD)

        assert v.verdict == "DEAD"
        assert "never executed" in v.detail


class TestADeadGuardIsReportedDead:
    def test_nothing_carries_the_column(self, tmp_path) -> None:
        db = _store(tmp_path / "m.db", with_column=False)

        (v,) = check_schema_guards(db, guards=_GUARD)

        assert v.verdict == "DEAD"
        assert not v.is_live
        assert v.found_elsewhere == ()

    def test_a_missing_table_is_named(self, tmp_path) -> None:
        path = tmp_path / "m.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE facts (fact_id TEXT, created_by TEXT)")
        conn.commit()
        conn.close()

        (v,) = check_schema_guards(path, guards=_GUARD)

        assert "trust_scores" in v.missing


class TestItNeverBreaksTheCaller:
    def test_an_unreadable_store_yields_no_verdicts(self, tmp_path) -> None:
        assert check_schema_guards(tmp_path / "nope.db", guards=_GUARD) == []

    def test_the_shipped_guard_registry_is_evaluable(self, tmp_path) -> None:
        """The default registry must not throw on a store missing everything."""
        path = tmp_path / "empty.db"
        sqlite3.connect(path).close()

        for v in check_schema_guards(path):
            assert v.verdict in {"LIVE", "SATISFIED_ELSEWHERE", "DEAD"}
