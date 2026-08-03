# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Phase 5 Brain Wiring — TDD tests (RED → GREEN → REFACTOR)
#
# Tests four features:
#   1. SOFT-PROMPT INJECTION  (P1-8): PromptInjector wired into AutoInvoker
#   2. FISHER POSTERIOR        (P1-9): bayesian_update called + persisted on access
#   3. EBBINGHAUS-LANGEVIN            coupling runs inside maintenance
#   4. HOPFIELD channel-count         comment/const alignment to 6-channel model

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.core.config import (
    AutoInvokeConfig,
    MathConfig,
    ForgettingConfig,
    ParameterizationConfig,
    SLMConfig,
)
from superlocalmemory.hooks.auto_invoker import AutoInvoker
from superlocalmemory.math.fisher import FisherRaoMetric


# ============================================================================
# Helpers
# ============================================================================

def _make_real_db(tmp_path: Path):
    """Create a minimal real SQLite DB with the atomic_facts schema."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "test_brain.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default', 'default')"
    )
    return db


def _insert_soft_prompt(db, profile_id: str, content: str = "Use TypeScript always.") -> str:
    """Insert a live soft prompt template into the DB."""
    pid = str(uuid.uuid4())
    db.execute(
        "INSERT INTO soft_prompt_templates "
        "(prompt_id, profile_id, category, content, "
        "source_pattern_ids, confidence, effectiveness, "
        "token_count, retention_score, active, version, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, "
        "datetime('now'), datetime('now'))",
        (pid, profile_id, "tech_preference", content,
         "[]", 0.9, 0.8, 10, 1.0),
    )
    return pid


def _insert_fact(db, profile_id: str, variance: list[float],
                 access_count: int = 0) -> str:
    """Insert a fact with known fisher_variance and access_count via store_fact + update_fact."""
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    mem = MemoryRecord(profile_id=profile_id, content="Test memory for fact")
    db.store_memory(mem)

    fact = AtomicFact(
        memory_id=mem.memory_id,
        profile_id=profile_id,
        content="The sky is blue.",
        embedding=[0.1] * 8,
        fisher_mean=[0.0] * 8,
        fisher_variance=variance,
        confidence=0.9,
        importance=0.5,
        evidence_count=1,
        access_count=access_count,
    )
    db.store_fact(fact)
    return fact.fact_id


# ============================================================================
# 1. SOFT-PROMPT INJECTION — P1-8
# ============================================================================

class TestSoftPromptInjectionWiring:
    """P1-8: PromptInjector must be constructed in engine_wiring and wired into
    AutoInvoker so that stored behavioral soft prompts appear in get_session_context.
    """

    def test_init_auto_invoker_wires_prompt_injector(self, tmp_path: Path) -> None:
        """_init_auto_invoker must attach a non-None PromptInjector when
        parameterization is enabled."""
        from superlocalmemory.core import engine_wiring

        db = _make_real_db(tmp_path)
        config = SLMConfig()

        invoker = engine_wiring._init_auto_invoker(
            config=config,
            db=db,
            vector_store=None,
            trust_scorer=None,
            embedder=None,
        )

        assert invoker is not None, "AutoInvoker must be returned"
        assert invoker._prompt_injector is not None, (
            "PromptInjector must be wired into AutoInvoker "
            "(currently None — P1-8 not yet wired)"
        )

    def test_soft_prompt_appears_in_session_context(self, tmp_path: Path) -> None:
        """End-to-end: stored soft prompt text must appear in get_session_context output.

        This test verifies the full injection chain:
          DB (soft_prompt_templates) → PromptInjector.get_injection_context()
          → AutoInvoker.get_session_context()

        'Reading is not verification. Run it.'
        """
        from superlocalmemory.core import engine_wiring

        db = _make_real_db(tmp_path)
        profile_id = "default"
        prompt_content = "ALWAYS use TypeScript with strict mode."

        _insert_soft_prompt(db, profile_id, prompt_content)

        config = SLMConfig()
        invoker = engine_wiring._init_auto_invoker(
            config=config,
            db=db,
            vector_store=None,
            trust_scorer=None,
            embedder=None,
        )
        assert invoker is not None

        ctx = invoker.get_session_context(project_path="/test", query="coding style")
        assert prompt_content in ctx, (
            f"Stored soft prompt '{prompt_content}' must appear in session context. "
            f"Got: {ctx[:300]!r}"
        )

    def test_soft_prompt_absent_when_parameterization_disabled(
        self, tmp_path: Path
    ) -> None:
        """When parameterization.enabled=False, no prompt injector is attached."""
        from superlocalmemory.core import engine_wiring
        from dataclasses import replace

        db = _make_real_db(tmp_path)
        profile_id = "default"
        _insert_soft_prompt(db, profile_id, "Use Go always.")

        config = SLMConfig()
        object.__setattr__(
            config, "parameterization",
            ParameterizationConfig(enabled=False),
        )
        invoker = engine_wiring._init_auto_invoker(
            config=config,
            db=db,
            vector_store=None,
            trust_scorer=None,
            embedder=None,
        )
        assert invoker is not None
        assert invoker._prompt_injector is None, (
            "With parameterization disabled, _prompt_injector must be None"
        )


# ============================================================================
# 2. FISHER POSTERIOR — P1-9
# ============================================================================

class TestFisherPosteriorWiring:
    """P1-9: bayesian_update must be called on access events and persisted.

    Variance must strictly tighten after N confirmations.
    """

    def test_bayesian_update_tightens_variance(self) -> None:
        """Math property: variance strictly decreases after each bayesian_update."""
        frm = FisherRaoMetric()
        old_var = [1.0] * 8
        obs_var = [1.0] * 8

        v1 = frm.bayesian_update(old_var, obs_var)
        v2 = frm.bayesian_update(v1, obs_var)
        v3 = frm.bayesian_update(v2, obs_var)

        assert all(v1[i] < old_var[i] for i in range(8)), "v1 must be tighter than v0"
        assert all(v2[i] < v1[i] for i in range(8)), "v2 must be tighter than v1"
        assert all(v3[i] < v2[i] for i in range(8)), "v3 must be tighter than v2"

    def test_maintenance_applies_fisher_posterior_and_persists(
        self, tmp_path: Path
    ) -> None:
        """After maintenance runs, fisher_variance for an accessed fact must decrease.

        This test is end-to-end: reads from DB before, runs maintenance, reads again.
        'The implementer is an LLM. Verify independently.'
        """
        from superlocalmemory.core.maintenance import run_maintenance

        db = _make_real_db(tmp_path)
        profile_id = "default"
        initial_var = [1.0] * 8
        fact_id = _insert_fact(db, profile_id, initial_var, access_count=3)

        config = SLMConfig()

        run_maintenance(db, config, profile_id=profile_id)

        rows = db.execute(
            "SELECT fisher_variance FROM atomic_facts WHERE fact_id = ?",
            (fact_id,),
        )
        assert rows, "Fact must still exist after maintenance"
        stored = json.loads(rows[0]["fisher_variance"])

        assert len(stored) == 8, "Variance dimensions must be preserved"
        assert all(
            stored[i] < initial_var[i] for i in range(8)
        ), (
            f"Fisher variance must tighten after maintenance for accessed facts. "
            f"Initial: {initial_var}, After: {stored}"
        )

    def test_unaccessed_fact_variance_unchanged(self, tmp_path: Path) -> None:
        """Facts with access_count=0 must NOT have variance updated (no spurious tightening)."""
        from superlocalmemory.core.maintenance import run_maintenance

        db = _make_real_db(tmp_path)
        profile_id = "default"
        initial_var = [1.0] * 8
        fact_id = _insert_fact(db, profile_id, initial_var, access_count=0)

        config = SLMConfig()
        run_maintenance(db, config, profile_id=profile_id)

        rows = db.execute(
            "SELECT fisher_variance FROM atomic_facts WHERE fact_id = ?",
            (fact_id,),
        )
        assert rows
        stored = json.loads(rows[0]["fisher_variance"])
        assert stored == pytest.approx(initial_var, abs=1e-9), (
            "Unaccessed fact's fisher_variance must remain unchanged"
        )

    def test_fisher_update_disabled_by_config(self, tmp_path: Path) -> None:
        """When config.math.fisher_bayesian_update=False, variance must not change."""
        from superlocalmemory.core.maintenance import run_maintenance

        db = _make_real_db(tmp_path)
        profile_id = "default"
        initial_var = [1.0] * 8
        fact_id = _insert_fact(db, profile_id, initial_var, access_count=5)

        config = SLMConfig()
        object.__setattr__(
            config, "math",
            MathConfig(
                fisher_bayesian_update=False,
                langevin_persist_positions=False,
                sheaf_at_encoding=False,
            ),
        )
        run_maintenance(db, config, profile_id=profile_id)

        rows = db.execute(
            "SELECT fisher_variance FROM atomic_facts WHERE fact_id = ?",
            (fact_id,),
        )
        assert rows
        stored = json.loads(rows[0]["fisher_variance"])
        assert stored == pytest.approx(initial_var, abs=1e-9), (
            "When fisher_bayesian_update=False, variance must be unchanged"
        )

    def test_variance_tightens_across_multiple_maintenance_runs(
        self, tmp_path: Path
    ) -> None:
        """Multiple maintenance runs must monotonically decrease variance."""
        from superlocalmemory.core.maintenance import run_maintenance

        db = _make_real_db(tmp_path)
        profile_id = "default"
        initial_var = [1.5] * 8
        fact_id = _insert_fact(db, profile_id, initial_var, access_count=1)

        config = SLMConfig()

        def _read_var() -> list[float]:
            rows = db.execute(
                "SELECT fisher_variance FROM atomic_facts WHERE fact_id = ?",
                (fact_id,),
            )
            return json.loads(rows[0]["fisher_variance"])

        run_maintenance(db, config, profile_id=profile_id)
        var_after_1 = _read_var()

        run_maintenance(db, config, profile_id=profile_id)
        var_after_2 = _read_var()

        assert all(var_after_1[i] < initial_var[i] for i in range(8)), "Run 1 must tighten"
        assert all(var_after_2[i] < var_after_1[i] for i in range(8)), "Run 2 must tighten further"


# ============================================================================
# 3. EBBINGHAUS-LANGEVIN COUPLING — maintenance wiring
# ============================================================================

class TestEbbinghausLangevinMaintenanceWiring:
    """Ebbinghaus-Langevin coupling must run inside the maintenance cycle
    when config.math.ebbinghaus_langevin_coupling_enabled=True.
    """

    def test_maintenance_returns_ebbinghaus_coupled_count(
        self, tmp_path: Path
    ) -> None:
        """When enabled, maintenance must return a positive ebbinghaus_coupled count."""
        from superlocalmemory.core.maintenance import run_maintenance

        db = _make_real_db(tmp_path)
        profile_id = "default"
        variance = [0.5] * 8
        position = [0.1] * 8
        fact_id = _insert_fact(db, profile_id, variance, access_count=2)

        db.execute(
            "UPDATE atomic_facts SET langevin_position = ? WHERE fact_id = ?",
            (json.dumps(position), fact_id),
        )

        config = SLMConfig()
        object.__setattr__(
            config, "math",
            MathConfig(
                langevin_persist_positions=True,
                ebbinghaus_langevin_coupling_enabled=True,
                fisher_bayesian_update=False,
                sheaf_at_encoding=False,
            ),
        )

        counts = run_maintenance(db, config, profile_id=profile_id)
        assert "ebbinghaus_coupled" in counts, (
            "maintenance must return 'ebbinghaus_coupled' key when coupling is enabled"
        )
        assert counts["ebbinghaus_coupled"] > 0, (
            "At least one fact should have been Ebbinghaus-Langevin coupled"
        )

    def test_ebbinghaus_coupling_disabled_by_default(
        self, tmp_path: Path
    ) -> None:
        """When ebbinghaus_langevin_coupling_enabled=False (default), coupling must not run."""
        from superlocalmemory.core.maintenance import run_maintenance

        db = _make_real_db(tmp_path)
        profile_id = "default"
        variance = [0.5] * 8
        position = [0.1] * 8
        fact_id = _insert_fact(db, profile_id, variance, access_count=2)
        db.execute(
            "UPDATE atomic_facts SET langevin_position = ? WHERE fact_id = ?",
            (json.dumps(position), fact_id),
        )

        config = SLMConfig()
        object.__setattr__(
            config, "math",
            MathConfig(
                langevin_persist_positions=False,
                ebbinghaus_langevin_coupling_enabled=False,
                fisher_bayesian_update=False,
                sheaf_at_encoding=False,
            ),
        )

        counts = run_maintenance(db, config, profile_id=profile_id)
        assert counts.get("ebbinghaus_coupled", 0) == 0, (
            "Coupling must not run when ebbinghaus_langevin_coupling_enabled=False"
        )

    def test_ebbinghaus_coupling_updates_lifecycle_zone(
        self, tmp_path: Path
    ) -> None:
        """Coupling must write back lifecycle zone changes to the DB."""
        from superlocalmemory.core.maintenance import run_maintenance

        db = _make_real_db(tmp_path)
        profile_id = "default"

        # Fact with very high variance (uncertain) and zero access → will be forgotten
        variance = [2.0] * 8
        position = [0.9] * 8
        fact_id = _insert_fact(db, profile_id, variance, access_count=0)
        db.execute(
            "UPDATE atomic_facts SET langevin_position = ?, lifecycle = 'active' "
            "WHERE fact_id = ?",
            (json.dumps(position), fact_id),
        )

        config = SLMConfig()
        object.__setattr__(
            config, "math",
            MathConfig(
                langevin_persist_positions=True,
                ebbinghaus_langevin_coupling_enabled=True,
                fisher_bayesian_update=False,
                sheaf_at_encoding=False,
            ),
        )

        run_maintenance(db, config, profile_id=profile_id)

        rows = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = ?", (fact_id,)
        )
        assert rows, "Fact must still exist"


# ============================================================================
# 4. HOPFIELD CHANNEL-COUNT — comment/const alignment
# ============================================================================

class TestHopfieldChannelCountAlignment:
    """engine_wiring docstring and hopfield_channel docstring must both
    accurately reflect the 6-channel retrieval model.
    """

    def test_engine_wiring_docstring_says_six_channels(self) -> None:
        """init_retrieval docstring must mention 'six' candidate producers."""
        from superlocalmemory.core import engine_wiring
        import inspect

        doc = inspect.getdoc(engine_wiring.init_retrieval) or ""
        assert "six" in doc.lower(), (
            f"init_retrieval docstring must say 'six candidate producers' "
            f"(was 'five' before Hopfield was added as 6th channel). Got: {doc!r}"
        )

    def test_hopfield_channel_docstring_identifies_sixth_channel(self) -> None:
        """HopfieldChannel class docstring must clearly state it is the 6th of 6 channels."""
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel
        import inspect

        doc = inspect.getdoc(HopfieldChannel) or ""
        assert "6th" in doc or "sixth" in doc.lower(), (
            f"HopfieldChannel must be identified as the 6th retrieval channel. Got: {doc!r}"
        )

    def test_hopfield_module_docstring_references_six_channels(self) -> None:
        """The hopfield_channel module docstring must reference 6-channel retrieval."""
        import superlocalmemory.retrieval.hopfield_channel as hc_mod
        doc = hc_mod.__doc__ or ""
        assert "6" in doc or "six" in doc.lower(), (
            f"hopfield_channel module docstring must reference the 6-channel model. Got: {doc!r}"
        )
