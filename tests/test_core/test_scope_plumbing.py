# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""v3.6.15 multi-scope PLUMBING proof — the #44 scope-drop fixes.

These tests close the gap the equivalence proof (DB layer) can't see: that the
``scope`` actually SURVIVES the engine write paths (store_fast + the async
pending materializer), and that recall is SHARED-OFF BY DEFAULT (opt-in) while
still honouring an explicit per-call opt-in.

Backward-compat guarantee under test:
  * a default recall returns ONLY this profile's own facts (3.6.14 behaviour);
  * personal writes/recall are entirely unaffected by the scope machinery;
  * an explicit scope (``--scope global`` / ``include_global=True``) is never
    silently downgraded to personal anywhere on the write or read path.
"""

from __future__ import annotations

import pytest

from superlocalmemory.cli import pending_store


def _scope_of(engine, fact_id: str) -> str:
    row = engine._db.execute(
        "SELECT scope FROM atomic_facts WHERE fact_id = ?", (fact_id,)
    )
    return row[0][0]


# ---------------------------------------------------------------------------
# W5 — store_fast must honour an explicit scope (write-through path)
# ---------------------------------------------------------------------------

class TestStoreFastScope:
    def test_store_fast_defaults_personal(self, engine_with_mock_deps):
        ids = engine_with_mock_deps.store_fast("a plain personal note")
        assert ids
        assert _scope_of(engine_with_mock_deps, ids[0]) == "personal"

    def test_store_fast_honours_global_scope(self, engine_with_mock_deps):
        # The exact bug: the write-through verbatim insert dropped scope and
        # stored everything personal. It must now land as global.
        ids = engine_with_mock_deps.store_fast(
            "a team-wide global decision", scope="global",
        )
        assert ids
        assert _scope_of(engine_with_mock_deps, ids[0]) == "global"

    def test_store_fast_shared_with_persisted(self, engine_with_mock_deps):
        ids = engine_with_mock_deps.store_fast(
            "shared with alice", scope="shared", shared_with=["alice"],
        )
        assert ids
        row = engine_with_mock_deps._db.execute(
            "SELECT scope, shared_with FROM atomic_facts WHERE fact_id = ?",
            (ids[0],),
        )[0]
        assert row[0] == "shared"
        assert "alice" in (row[1] or "")


# ---------------------------------------------------------------------------
# Recall is SHARED-OFF BY DEFAULT, opt-in per call (policy + read plumbing)
#
# We prove threading by SPYING on the scope-filtered DB method every retrieval
# channel funnels through (get_all_facts). This isolates the policy+plumbing
# under test from mock-embedder ranking noise; the DB-layer scope FILTERING
# itself is proven separately in test_scope_recall_equivalence.
# ---------------------------------------------------------------------------

class TestRecallSharedOffByDefault:
    @staticmethod
    def _spy_get_all_facts(eng, monkeypatch):
        calls: list[tuple] = []
        orig = eng._db.get_all_facts

        def spy(profile, include_global=True, include_shared=True, **kw):
            calls.append((include_global, include_shared))
            return orig(profile, include_global=include_global,
                        include_shared=include_shared, **kw)

        monkeypatch.setattr(eng._db, "get_all_facts", spy)
        return calls

    def test_default_recall_threads_shared_off(self, engine_with_mock_deps, monkeypatch):
        eng = engine_with_mock_deps
        eng.store_fast("zebra primary personal fact")
        calls = self._spy_get_all_facts(eng, monkeypatch)

        eng.recall("zebra")  # no flags → must resolve to config default (False)
        assert calls, "recall did not query the scope-filtered DB method"
        assert all(ig is False and ish is False for ig, ish in calls), calls

    def test_opt_in_threads_true_to_db(self, engine_with_mock_deps, monkeypatch):
        eng = engine_with_mock_deps
        eng.store_fast("zebra primary personal fact")
        calls = self._spy_get_all_facts(eng, monkeypatch)

        eng.recall("zebra", include_global=True, include_shared=True)
        assert calls
        assert all(ig is True and ish is True for ig, ish in calls), calls

    def test_explicit_false_overrides(self, engine_with_mock_deps, monkeypatch):
        eng = engine_with_mock_deps
        eng.store_fast("zebra primary personal fact")
        calls = self._spy_get_all_facts(eng, monkeypatch)

        eng.recall("zebra", include_global=False, include_shared=False)
        assert calls
        assert all(ig is False and ish is False for ig, ish in calls), calls

    def test_personal_recall_identical_regardless_of_flags(self, engine_with_mock_deps):
        # Pure-personal data: the flag must make ZERO difference (3.6.14 parity).
        eng = engine_with_mock_deps
        eng.store_fast("octopus personal one")
        eng.store_fast("octopus personal two")
        default = {r.fact.fact_id for r in eng.recall("octopus").results}
        forced = {r.fact.fact_id for r in eng.recall(
            "octopus", include_global=True, include_shared=True).results}
        assert default == forced


# ---------------------------------------------------------------------------
# W6 — the async pending materializer must REPLAY scope from metadata
# ---------------------------------------------------------------------------

class TestPendingMaterializerScope:
    def test_process_pending_replays_global_scope(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        base_dir = eng._config.base_dir
        # Mirror the async /remember path: scope rides inside the pending
        # metadata blob (store_pending already has the column).
        pending_store.store_pending(
            "a queued global memory",
            metadata={"scope": "global"},
            base_dir=base_dir,
        )
        eng._process_pending_memories()

        rows = eng._db.execute(
            "SELECT scope FROM atomic_facts WHERE content LIKE '%queued global%'"
        )
        assert rows, "materializer did not store the pending memory"
        assert all(r[0] == "global" for r in rows)

    def test_process_pending_personal_stays_personal(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        base_dir = eng._config.base_dir
        pending_store.store_pending("a queued plain memory", base_dir=base_dir)
        eng._process_pending_memories()

        rows = eng._db.execute(
            "SELECT scope FROM atomic_facts WHERE content LIKE '%queued plain%'"
        )
        assert rows
        assert all(r[0] == "personal" for r in rows)
