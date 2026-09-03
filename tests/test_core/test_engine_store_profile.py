# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Per-request ``profile_id`` threading on the three engine write paths.

Spec §4: a caller may target any profile on a single write without
switching the engine's active profile. Contract under test:

  * ``profile_id="b"`` — the durable rows (memory, fact, ingestion
    operation journal) land under profile ``b``;
  * ``profile_id=None`` (and ``""``) — the engine's active profile is
    used, byte-for-byte the pre-feature behaviour;
  * ``engine._profile_id`` is NEVER mutated by a write call.

Fixture convention: ``engine_with_mock_deps`` from ``tests/conftest.py``
(real SQLite DB on tmp_path, real schema, mocked embedder, no LLM) —
the same convention ``test_engine_store_path.py`` uses. Assertions are
made against the real database rows (the actual downstream write), not
against internal mocks.
"""

from __future__ import annotations

import inspect

import pytest

from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import AtomicFact, FactType

TARGET = "profile-target"


def _seed_profile(engine: MemoryEngine, name: str = TARGET) -> None:
    """FK on memories/atomic_facts → profiles; seed the target row.

    Uses the same INSERT OR IGNORE statement schema.create_all_tables
    uses for the 'default' profile.
    """
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (name, name),
    )


def _fact_profile(engine: MemoryEngine, fact_id: str) -> str | None:
    row = engine._db.execute(
        "SELECT profile_id FROM atomic_facts WHERE fact_id = ?", (fact_id,),
    )
    return None if not row else row[0][0]


def _memory_profile(engine: MemoryEngine, fact_id: str) -> str | None:
    row = engine._db.execute(
        "SELECT m.profile_id FROM memories m "
        "JOIN atomic_facts f ON f.memory_id = m.memory_id "
        "WHERE f.fact_id = ?",
        (fact_id,),
    )
    return None if not row else row[0][0]


def _operation_profile(engine: MemoryEngine, idempotency_key: str) -> str | None:
    """The durable M018 journal row carries profile_id as data (spec §4.3)."""
    row = engine._db.execute(
        "SELECT profile_id FROM ingestion_operations WHERE idempotency_key = ?",
        (idempotency_key,),
    )
    return None if not row else row[0][0]


# ---------------------------------------------------------------------------
# Signature contract — Task 3/5 consume this exact shape
# ---------------------------------------------------------------------------

class TestSignatureContract:
    @pytest.mark.parametrize(
        "method_name", ["store", "store_fact_direct", "store_fast"],
    )
    def test_profile_id_is_keyword_only_last_and_defaults_none(
        self, method_name: str,
    ) -> None:
        params = inspect.signature(
            getattr(MemoryEngine, method_name),
        ).parameters
        assert "profile_id" in params, f"{method_name} lost profile_id"
        param = params["profile_id"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None
        assert param is list(params.values())[-1], (
            "profile_id must be appended last so existing positional "
            "callers cannot shift"
        )


# ---------------------------------------------------------------------------
# store() — canonical journal + queryable receipt path
# ---------------------------------------------------------------------------

class TestStoreProfile:
    def test_explicit_profile_lands_in_target(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        ids = eng.store(
            "Zed keeps a per-request profile routing test notebook",
            session_id="s-profile", profile_id=TARGET,
        )
        assert ids, "store() produced no facts"
        assert _fact_profile(eng, ids[0]) == TARGET
        assert _memory_profile(eng, ids[0]) == TARGET

    def test_explicit_profile_on_the_operation_journal(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        ids = eng.store(
            "Yara archives the journal profile of this write",
            profile_id=TARGET,
        )
        assert ids
        # The queryable-receipt fact id is the journal's first fact id.
        row = eng._db.execute(
            "SELECT profile_id FROM ingestion_operations "
            "WHERE queryable_fact_ids_json LIKE ?",
            (f"%{ids[0]}%",),
        )
        assert row, "no ingestion_operations row for this store"
        assert row[0][0] == TARGET

    def test_none_falls_back_to_active(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        ids = eng.store(
            "Uma writes without naming a profile", profile_id=None,
        )
        assert ids
        assert _fact_profile(eng, ids[0]) == eng._profile_id

    def test_empty_string_falls_back_to_active(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        ids = eng.store(
            "Tomas passes an explicitly empty profile id", profile_id="",
        )
        assert ids
        assert _fact_profile(eng, ids[0]) == eng._profile_id

    def test_store_does_not_mutate_active_profile(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        before = eng._profile_id
        eng.store("Sara mutates nothing by naming a profile", profile_id=TARGET)
        assert eng._profile_id == before


# ---------------------------------------------------------------------------
# store_fact_direct() — prebuilt-fact canonical path
# ---------------------------------------------------------------------------

class TestStoreFactDirectProfile:
    @staticmethod
    def _fact(fact_id: str, content: str) -> AtomicFact:
        fact = AtomicFact(
            fact_id=fact_id, memory_id="", content=content,
            fact_type=FactType.SEMANTIC, entities=["Rita"], confidence=0.9,
        )
        return fact

    def test_explicit_profile_lands_in_target(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        fact = self._fact("direct-pid-1", "Rita routes a prebuilt fact by profile")
        returned = eng.store_fact_direct(fact, profile_id=TARGET)
        assert returned == "direct-pid-1"
        assert _fact_profile(eng, "direct-pid-1") == TARGET
        assert _memory_profile(eng, "direct-pid-1") == TARGET
        assert _operation_profile(eng, "prebuilt:direct-pid-1") == TARGET

    def test_none_falls_back_to_active(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        fact = self._fact("direct-pid-2", "Quinn omits the profile argument")
        returned = eng.store_fact_direct(fact, profile_id=None)
        assert returned == "direct-pid-2"
        assert _fact_profile(eng, "direct-pid-2") == eng._profile_id

    def test_does_not_mutate_active_profile(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        before = eng._profile_id
        fact = self._fact("direct-pid-3", "Petra checks the active pointer")
        eng.store_fact_direct(fact, profile_id=TARGET)
        assert eng._profile_id == before


# ---------------------------------------------------------------------------
# store_fast() — synchronous write-through path
# ---------------------------------------------------------------------------

class TestStoreFastProfile:
    def test_explicit_profile_lands_in_target(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        ids = eng.store_fast(
            "Omar writes through fast into a named profile", profile_id=TARGET,
        )
        assert ids
        assert _fact_profile(eng, ids[0]) == TARGET
        assert _memory_profile(eng, ids[0]) == TARGET

    def test_none_falls_back_to_active(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        ids = eng.store_fast(
            "Nadia writes through fast with no profile", profile_id=None,
        )
        assert ids
        assert _fact_profile(eng, ids[0]) == eng._profile_id

    def test_does_not_mutate_active_profile(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        before = eng._profile_id
        eng.store_fast("Mona mutates nothing on the fast path", profile_id=TARGET)
        assert eng._profile_id == before


# ---------------------------------------------------------------------------
# Isolation — a targeted write must not leak into the active profile
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    def test_targeted_write_leaves_active_profile_empty(
        self, engine_with_mock_deps,
    ):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        ids = eng.store(
            "Lena is visible only to the profile she names", profile_id=TARGET,
        )
        assert ids
        active_facts = eng._db.get_all_facts(eng._profile_id)
        assert all(f.fact_id not in ids for f in active_facts), (
            "per-request write leaked into the active profile"
        )


# ---------------------------------------------------------------------------
# enrich_new_facts_now / _projection_has — the inline-enrichment seam
# (final-review I-1: routed writes must enrich against THEIR profile)
# ---------------------------------------------------------------------------

class TestEnrichmentProfileSignatureContract:
    """profile_id is keyword-only, defaults None, appended last — the same
    convention the three store paths above established for Tasks 3/5."""

    @pytest.mark.parametrize(
        "method_name", ["enrich_new_facts_now", "_projection_has"],
    )
    def test_profile_id_is_keyword_only_last_and_defaults_none(
        self, method_name: str,
    ) -> None:
        params = inspect.signature(
            getattr(MemoryEngine, method_name),
        ).parameters
        assert "profile_id" in params, f"{method_name} lost profile_id"
        param = params["profile_id"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None
        assert param is list(params.values())[-1], (
            "profile_id must be appended last so existing positional "
            "callers cannot shift"
        )


def _warm_mock_embedder(eng, monkeypatch):
    """Make inline enrichment actually embed under mocked deps.

    The fixture's mock embedder reports itself cold-and-remote (MagicMock
    attributes), so the warm guard declines by design. Patching the guard is
    the sanctioned seam: the embedding itself is orthogonal to the profile
    routing under test, and this makes an enriched>0 outcome possible so the
    assertions are not vacuously true.
    """
    monkeypatch.setattr(
        eng, "_warm_guard_embed",
        lambda text, *, timeout_s=None: ([0.01] * 768, [0.0] * 768, [1.0] * 768),
    )


class TestEnrichNewFactsNowProfile:
    def _spy_get_fact(self, eng, monkeypatch):
        """Record every (fact_id, profile_id) lookup, then serve the real row."""
        real = eng._db.get_fact
        seen: list[tuple[str, object]] = []

        def _spy(fact_id, profile_id=None):
            seen.append((fact_id, profile_id))
            return real(fact_id, profile_id)

        monkeypatch.setattr(eng._db, "get_fact", _spy)
        return seen

    def test_routed_facts_resolve_against_the_routed_profile(
        self, engine_with_mock_deps, monkeypatch,
    ):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        ids = eng.store(
            "Wren logs the enrichment routing probe for the relay desk",
            profile_id=TARGET,
        )
        assert ids
        _warm_mock_embedder(eng, monkeypatch)
        seen = self._spy_get_fact(eng, monkeypatch)

        enriched = eng.enrich_new_facts_now(
            ids, profile_id=TARGET, timeout_s=5.0,
        )

        lookups = [p for fid, p in seen if fid in ids]
        assert lookups, "enrichment never looked up the routed facts"
        assert set(lookups) == {TARGET}, (
            f"routed enrichment must resolve facts against {TARGET!r}, "
            f"got {set(lookups)!r} (active is {eng._profile_id!r})"
        )
        # The routed rows resolved, so enrichment ran to its honest outcome
        # instead of skipping every fact as unfindable.
        assert enriched == len(ids), (
            f"expected all {len(ids)} routed fact(s) searchable by meaning, "
            f"got {enriched}"
        )

    def test_without_the_anchor_routed_facts_are_invisible(
        self, engine_with_mock_deps,
    ):
        """The lookups are tenant-scoped: this documents why the anchor must
        be threaded. WITHOUT profile_id the routed rows resolve to None and
        inline enrichment silently skips them — the exact pre-fix daemon
        behaviour for every routed write."""
        eng = engine_with_mock_deps
        _seed_profile(eng)
        ids = eng.store(
            "Sable files the anchorless enrichment control note",
            profile_id=TARGET,
        )
        assert ids
        assert eng._profile_id != TARGET

        assert eng.enrich_new_facts_now(ids, timeout_s=5.0) == 0

    def test_legacy_facts_enrich_against_the_active_profile(
        self, engine_with_mock_deps, monkeypatch,
    ):
        eng = engine_with_mock_deps
        ids = eng.store("Orla enriches on the legacy active path")
        assert ids
        _warm_mock_embedder(eng, monkeypatch)
        seen = self._spy_get_fact(eng, monkeypatch)

        enriched = eng.enrich_new_facts_now(ids, timeout_s=5.0)

        lookups = [p for fid, p in seen if fid in ids]
        assert lookups
        assert set(lookups) == {eng._profile_id}
        assert enriched == len(ids)


class TestAuditUnknownProfile:
    """4.1.14 audit: the engine fails closed on unknown profiles itself."""

    def test_store_fast_rejects_unknown_profile(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        with pytest.raises(ValueError, match="unknown profile"):
            eng.store_fast(
                "Ghost content must never land anywhere.", profile_id="ghost",
            )

    def test_store_rejects_unknown_profile(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        with pytest.raises(ValueError, match="unknown profile"):
            eng.store(
                "Ghost content must never land anywhere.", profile_id="ghost",
            )

    def test_whitespace_profile_id_is_legacy(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        ids = eng.store_fast(
            "Whitespace anchor resolves to the active profile.",
            profile_id="   ",
        )
        assert ids
        assert _fact_profile(eng, ids[0]) == eng._profile_id

    def test_padded_profile_id_routes_stripped(self, engine_with_mock_deps):
        eng = engine_with_mock_deps
        _seed_profile(eng)
        ids = eng.store_fast(
            "Padded anchor routes to the stripped profile.",
            profile_id=f"  {TARGET}  ",
        )
        assert ids
        assert _fact_profile(eng, ids[0]) == TARGET
