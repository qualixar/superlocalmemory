# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Integration tests for v2.8 MCP tools -- end-to-end workflows.

Exercises all 6 new v2.8 tool handlers through realistic multi-step
scenarios:

    1. Behavioral feedback loop: report outcomes -> extract patterns -> query
    2. Lifecycle flow: create memory -> compact (dry + execute) -> verify
    3. Audit flow: log events -> query -> verify chain integrity
    4. Retention policy creation (GDPR + HIPAA)
    5. Cross-engine workflow: behavioral -> lifecycle -> audit
    6. All tools return the mandatory 'success' key

All tests use temporary databases -- NEVER touches production ~/.claude-memory/.

Run with:
    python3 -m pytest tests/test_mcp_integration_v28.py -v
"""
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure src/ is importable
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))


class TestMCPIntegrationV28:
    """End-to-end integration tests for all v2.8 MCP tools."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.memory_db = os.path.join(self.tmp_dir, "memory.db")
        self.learning_db = os.path.join(self.tmp_dir, "learning.db")
        self.audit_db = os.path.join(self.tmp_dir, "audit.db")

        # Create memory.db with lifecycle-enabled schema and test data
        conn = sqlite3.connect(self.memory_db)
        conn.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                lifecycle_state TEXT DEFAULT 'active',
                lifecycle_updated_at TIMESTAMP,
                lifecycle_history TEXT DEFAULT '[]',
                access_level TEXT DEFAULT 'public',
                profile TEXT DEFAULT 'default',
                tags TEXT DEFAULT '[]',
                project_name TEXT
            )
        """)
        now = datetime.now()

        # Memory 1: active, fresh, high importance -- should NOT be compacted
        conn.execute(
            "INSERT INTO memories "
            "(content, importance, last_accessed, created_at, tags) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "Python best practices",
                8,
                now.isoformat(),
                (now - timedelta(days=5)).isoformat(),
                '["python"]',
            ),
        )

        # Memory 2: active but stale (45 days no access, importance 3)
        # Evaluator default: active->warm requires no_access>=30d AND importance<=6
        # 45 >= 30 AND 3 <= 6 => should be recommended for active -> warm
        conn.execute(
            "INSERT INTO memories "
            "(content, importance, last_accessed, created_at, lifecycle_state) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "old stale memory",
                3,
                (now - timedelta(days=45)).isoformat(),
                (now - timedelta(days=120)).isoformat(),
                "active",
            ),
        )

        conn.commit()
        conn.close()

        # Override tool DB paths so they hit temp databases
        import mcp_tools_v28 as tools

        tools.DEFAULT_MEMORY_DB = self.memory_db
        tools.DEFAULT_LEARNING_DB = self.learning_db
        tools.DEFAULT_AUDIT_DB = self.audit_db

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(coro):
        """Run an async coroutine synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Test 1: Behavioral feedback loop
    # ------------------------------------------------------------------

    def test_report_outcome_and_get_patterns_loop(self):
        """Behavioral feedback loop: report outcomes -> extract patterns -> query patterns.

        Flow:
        1. Report 6 successes + 2 failures for action_type='code_written'
        2. Extract patterns using BehavioralPatternExtractor directly
        3. Query patterns via the get_behavioral_patterns MCP tool
        4. Verify at least one pattern was discovered
        """
        import mcp_tools_v28 as tools

        # Step 1: Report multiple outcomes to build history
        for _ in range(6):
            result = self._run(
                tools.report_outcome(
                    [1], "success", action_type="code_written", project="proj_a"
                )
            )
            assert result["success"] is True
            assert isinstance(result["outcome_id"], int)

        for _ in range(2):
            result = self._run(
                tools.report_outcome(
                    [1], "failure", action_type="code_written", project="proj_a"
                )
            )
            assert result["success"] is True

        # Step 2: Extract patterns from the recorded outcomes
        # (The MCP tool reads stored patterns, so we need to extract + save first)
        from behavioral.behavioral_patterns import BehavioralPatternExtractor

        extractor = BehavioralPatternExtractor(self.learning_db)
        raw_patterns = extractor.extract_patterns()
        assert len(raw_patterns) >= 1, "Expected at least 1 pattern from 8 outcomes"
        saved_count = extractor.save_patterns()
        assert saved_count >= 1

        # Step 3: Query patterns via the MCP tool handler
        result = self._run(tools.get_behavioral_patterns())
        assert result["success"] is True
        assert result["count"] >= 1

        # Verify the pattern has sensible values
        patterns = result["patterns"]
        assert len(patterns) >= 1
        found_action_type_pattern = any(
            p["pattern_type"] == "action_type_success"
            and p["pattern_key"] == "code_written"
            for p in patterns
        )
        assert found_action_type_pattern, (
            "Expected an action_type_success pattern for 'code_written'"
        )

    # ------------------------------------------------------------------
    # Test 2: Lifecycle compact flow
    # ------------------------------------------------------------------

    def test_lifecycle_compact_flow(self):
        """Lifecycle flow: check status -> compact (dry) -> compact (execute) -> verify.

        Flow:
        1. Verify initial state: 2 active memories
        2. Dry-run compact: should recommend at least 1 transition
        3. Execute compact: stale memory transitions active -> warm
        4. Verify final distribution reflects the transition
        """
        import mcp_tools_v28 as tools

        # Step 1: Check initial status -- both memories in 'active'
        status = self._run(tools.get_lifecycle_status())
        assert status["success"] is True
        assert status["distribution"]["active"] == 2
        assert status["total_memories"] == 2

        # Step 2: Dry-run compact -- should flag the stale memory
        dry = self._run(tools.compact_memories(dry_run=True))
        assert dry["success"] is True
        assert dry["dry_run"] is True
        assert dry["recommendations"] >= 1, "Stale memory should have a recommendation"

        # Check the recommendation details
        details = dry["details"]
        stale_rec = [d for d in details if d["memory_id"] == 2]
        assert len(stale_rec) == 1
        assert stale_rec[0]["from"] == "active"
        assert stale_rec[0]["to"] == "warm"

        # Step 3: Execute compact (not dry run)
        result = self._run(tools.compact_memories(dry_run=False))
        assert result["success"] is True
        assert result["dry_run"] is False
        assert result["transitioned"] >= 1

        # Step 4: Verify the stale memory transitioned to 'warm'
        status2 = self._run(tools.get_lifecycle_status())
        assert status2["success"] is True
        assert status2["distribution"]["active"] == 1, "Fresh memory stays active"
        assert status2["distribution"]["warm"] == 1, "Stale memory should be warm"

        # Also verify via single-memory lookup
        single = self._run(tools.get_lifecycle_status(memory_id=2))
        assert single["success"] is True
        assert single["lifecycle_state"] == "warm"

    # ------------------------------------------------------------------
    # Test 3: Audit trail flow
    # ------------------------------------------------------------------

    def test_audit_trail_flow(self):
        """Audit flow: log events -> query -> verify hash chain integrity.

        Flow:
        1. Log 3 audit events directly via AuditDB
        2. Query all events via the audit_trail MCP tool
        3. Verify event count matches
        4. Verify the hash chain is intact
        5. Filter by event_type and by actor
        """
        import mcp_tools_v28 as tools
        from compliance.audit_db import AuditDB

        # Step 1: Log events directly (simulating EventBus emissions)
        db = AuditDB(self.audit_db)
        db.log_event("memory.created", actor="user", resource_id=1)
        db.log_event("memory.recalled", actor="agent_a", resource_id=1)
        db.log_event("memory.deleted", actor="user", resource_id=2)

        # Step 2: Query all events via MCP tool
        result = self._run(tools.audit_trail(limit=10))
        assert result["success"] is True
        assert result["count"] == 3

        # Step 3: Verify the hash chain is intact
        verified = self._run(tools.audit_trail(verify_chain=True))
        assert verified["success"] is True
        assert verified["chain_valid"] is True
        assert verified["chain_entries"] == 3

        # Step 4: Filter by event_type
        created_only = self._run(tools.audit_trail(event_type="memory.created"))
        assert created_only["success"] is True
        assert created_only["count"] == 1
        assert created_only["events"][0]["event_type"] == "memory.created"

        # Step 5: Filter by actor
        agent_only = self._run(tools.audit_trail(actor="agent_a"))
        assert agent_only["success"] is True
        assert agent_only["count"] == 1
        assert agent_only["events"][0]["actor"] == "agent_a"

    # ------------------------------------------------------------------
    # Test 4: Retention policy creation
    # ------------------------------------------------------------------

    def test_retention_policy_creation(self):
        """Retention: create GDPR + HIPAA policies and verify they persist.

        Flow:
        1. Create a GDPR erasure policy (0-day retention, tombstone action)
        2. Create a HIPAA retention policy (2555-day retention, retain action)
        3. Verify both returned success and policy IDs
        """
        import mcp_tools_v28 as tools

        # GDPR: immediate tombstone for memories tagged 'gdpr'
        r1 = self._run(
            tools.set_retention_policy(
                "GDPR Erasure", "gdpr", 0, "tombstone", ["gdpr"]
            )
        )
        assert r1["success"] is True
        assert isinstance(r1["policy_id"], int)
        assert r1["framework"] == "gdpr"

        # HIPAA: retain for 7 years (2555 days) for memories tagged 'hipaa'
        r2 = self._run(
            tools.set_retention_policy(
                "HIPAA", "hipaa", 2555, "retain", ["hipaa"]
            )
        )
        assert r2["success"] is True
        assert isinstance(r2["policy_id"], int)
        assert r2["framework"] == "hipaa"

        # Policy IDs should be distinct
        assert r1["policy_id"] != r2["policy_id"]

        # Verify policies exist in the DB
        from lifecycle.retention_policy import RetentionPolicyManager

        mgr = RetentionPolicyManager(self.memory_db)
        policies = mgr.list_policies()
        assert len(policies) == 2
        names = {p["name"] for p in policies}
        assert "GDPR Erasure" in names
        assert "HIPAA" in names

    # ------------------------------------------------------------------
    # Test 5: All tools return 'success' key
    # ------------------------------------------------------------------

    def test_all_tools_return_success_key(self):
        """Every v2.8 tool should return a dict with a 'success' key.

        This is the contract: callers always check result['success'] first.
        """
        import mcp_tools_v28 as tools

        results = [
            self._run(tools.report_outcome([1], "success")),
            self._run(tools.get_lifecycle_status()),
            self._run(
                tools.set_retention_policy("test", "internal", 30, "archive")
            ),
            self._run(tools.compact_memories(dry_run=True)),
            self._run(tools.get_behavioral_patterns()),
            self._run(tools.audit_trail()),
        ]

        for i, r in enumerate(results):
            assert isinstance(r, dict), f"Tool {i} did not return a dict: {type(r)}"
            assert "success" in r, f"Tool {i} missing 'success' key: {r}"

    # ------------------------------------------------------------------
    # Test 6: Cross-engine workflow
    # ------------------------------------------------------------------

    def test_cross_engine_workflow(self):
        """Cross-engine: behavioral -> lifecycle -> audit in sequence.

        Exercises all three v2.8 engines in a single scenario to verify
        they don't interfere with each other.

        Flow:
        1. Report a behavioral outcome (writes to learning.db)
        2. Query lifecycle status for a specific memory (reads memory.db)
        3. Query audit trail (reads audit.db -- empty since no EventBus wired)
        """
        import mcp_tools_v28 as tools

        # Behavioral: report an outcome
        r1 = self._run(
            tools.report_outcome([1], "success", project="cross_test")
        )
        assert r1["success"] is True
        assert r1["outcome"] == "success"

        # Lifecycle: check memory 1 state (should still be 'active')
        r2 = self._run(tools.get_lifecycle_status(memory_id=1))
        assert r2["success"] is True
        assert r2["lifecycle_state"] == "active"

        # Audit: query trail (empty -- no events logged via EventBus in this test)
        r3 = self._run(tools.audit_trail())
        assert r3["success"] is True
        assert r3["count"] == 0

        # Now manually log an audit event and re-query
        from compliance.audit_db import AuditDB

        db = AuditDB(self.audit_db)
        db.log_event("behavioral.outcome_recorded", actor="user", resource_id=1)

        r4 = self._run(tools.audit_trail())
        assert r4["success"] is True
        assert r4["count"] == 1
