# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Tests that start from "an agent saved a rule" run with the origin firewall ON.

An agent's save is ``agent_inferred``, which is at the firewall's floor (owner decision, 19 Nov): an
agent's own rule or decision is trusted for instruction slots with no promotion step. :func:`agent_stores`
stores a fact the way an agent's save does, so the tests that follow prove what session start, standing
rules, cues and the core block do WITH the firewall on. What the firewall still holds (web, tool, peer,
import, model text) is in ``tests/test_trust_firewall`` and ``tests/test_minimal_firewall``.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, env_name
from superlocalmemory.storage import fact_origin as fo

#: What an agent door records for a save: the agent's own words, not a person's.
AGENT_ORIGIN = fo.Origin("remember_http", "verbatim", source_type="http", role="assistant",
                         agent_id="claude-code")
#: Kind sources that mean a caller declared or a person confirmed the kind (a classifier's never).
CONFIRMED_SOURCES = ("caller", "user")


@pytest.fixture
def firewall_on(monkeypatch, tmp_path):
    """The firewall acting, as shipped, on a data folder of its own."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    monkeypatch.delenv(KILL_ENV, raising=False)
    monkeypatch.setenv(env_name("trust.taint"), "on")


def agent_stores(db, fact, *, by: str = "agent:claude-code", channel: str = "mcp") -> str:
    """``db.store_fact(fact)`` as an agent's save (``by`` and ``channel`` are kept for the callers)."""
    with fo.declare(AGENT_ORIGIN):
        return db.store_fact(fact)
