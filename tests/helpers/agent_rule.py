# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Tests that start from "an agent saved a rule" run with the origin firewall ON.

An agent's save is ``agent_inferred``, below the firewall's floor. Saving (or confirming) it as a
rule or decision is its explicit promotion (``trust/declared_promotion``, owner decision 19 Nov).
:func:`agent_stores` stores a fact the way an agent's save does and, for a kind a caller or a
person confirmed, runs that same promotion step, so the tests that follow prove what session
start, standing rules, cues and the core block do WITH the firewall on. The route-level proof
(HTTP ``/remember``, kind confirmation, undo) is ``tests/test_trust_firewall/test_declared_promotion.py``.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, env_name
from superlocalmemory.storage import fact_origin as fo
from superlocalmemory.trust import declared_promotion

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
    """``db.store_fact(fact)`` as an agent's save; a confirmed rule or decision is promoted."""
    with fo.declare(AGENT_ORIGIN):
        fact_id = db.store_fact(fact)
    kind, source = getattr(fact, "memory_kind", None), getattr(fact, "memory_kind_source", None)
    if kind and source in CONFIRMED_SOURCES:
        with db.raw_connection() as conn:
            declared_promotion.promote(conn, fact.profile_id, [(fact_id, kind)], by=by,
                                       channel=channel, how=declared_promotion.SAVED)
    return fact_id
