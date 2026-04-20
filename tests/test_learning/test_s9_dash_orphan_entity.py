# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 9 dashboard fix (S9-DASH-01)

"""Regression tests for the orphan-entity leak in Auto-Detected Patterns.

The dashboard ``Patterns`` panel surfaced raw hex entity_ids as
"entity_preferences" (e.g. ``ea701bf01f1ff4df8``) when
``canonical_entities_json`` in an atomic_fact referenced an entity_id
with no row in ``canonical_entities``. Two layers of defence:

1. Producer — ``pattern_miner._mine_entity_preferences`` now skips
   unresolvable entity_ids entirely (no pattern emitted).
2. Consumer — ``BehavioralPatternStore.get_patterns`` filters out any
   row whose ``pattern_key`` / ``metadata.value`` is a pure hex id
   (16-20 chars), so historic bad rows do not need a destructive
   DB-cleanup migration on live installs.

This test covers both layers.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Consumer-side filter — BehavioralPatternStore
# ---------------------------------------------------------------------------

def test_hex_id_pattern_keys_are_filtered_on_read(tmp_path: Path) -> None:
    """Historic ``_store_patterns`` rows whose pattern_key is a bare
    hex id are invisible to the dashboard."""
    from superlocalmemory.learning.behavioral import BehavioralPatternStore

    db_path = tmp_path / "learning.db"
    store = BehavioralPatternStore(str(db_path))

    # Seed two rows: one legitimate, one orphan-hex.
    store.record_pattern(
        profile_id="p", pattern_type="entity_preferences",
        data={"topic": "Qualixar", "pattern_key": "entity:Qualixar",
              "value": "Qualixar", "evidence": 10, "source": "t"},
        success_rate=0.9, confidence=0.9,
    )
    store.record_pattern(
        profile_id="p", pattern_type="entity_preferences",
        data={"topic": "ea701bf01f1ff4df8",
              "pattern_key": "entity:ea701bf01f1ff4df8",
              "value": "ea701bf01f1ff4df8",
              "evidence": 191, "source": "t"},
        success_rate=1.0, confidence=1.0,
    )

    out = store.get_patterns(profile_id="p")
    keys = [p.get("pattern_key") for p in out]
    values = [(p.get("metadata") or {}).get("value") for p in out]
    assert "Qualixar" in keys, \
        f"legitimate pattern missing: {keys!r}"
    assert "ea701bf01f1ff4df8" not in keys, \
        f"orphan hex id leaked into pattern_key: {keys!r}"
    assert "ea701bf01f1ff4df8" not in values, \
        f"orphan hex id leaked into metadata.value: {values!r}"


def test_short_hex_under_15_chars_not_filtered(tmp_path: Path) -> None:
    """Real words that happen to look hex but are under 15 chars
    (e.g. ``cafe``, ``ace12``) must NOT be filtered — the pattern is
    entity_id-specific (16-20 hex)."""
    from superlocalmemory.learning.behavioral import BehavioralPatternStore

    store = BehavioralPatternStore(str(tmp_path / "l.db"))
    store.record_pattern(
        profile_id="p", pattern_type="entity_preferences",
        data={"topic": "cafe", "pattern_key": "entity:cafe",
              "value": "cafe", "evidence": 5, "source": "t"},
        success_rate=0.8, confidence=0.8,
    )
    out = store.get_patterns(profile_id="p")
    keys = [p.get("pattern_key") for p in out]
    assert "cafe" in keys, f"short word was wrongly filtered: {keys!r}"


# ---------------------------------------------------------------------------
# Producer-side — _mine_entity_preferences skips orphans
# ---------------------------------------------------------------------------

def test_mine_entity_preferences_skips_orphan(tmp_path: Path) -> None:
    """If an entity_id in ``canonical_entities_json`` has no row in
    ``canonical_entities``, the miner produces no pattern for it."""
    from superlocalmemory.learning import pattern_miner
    from superlocalmemory.learning.behavioral import BehavioralPatternStore

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE canonical_entities (
            entity_id TEXT PRIMARY KEY,
            canonical_name TEXT,
            entity_type TEXT
        )
    """)
    # Legitimate row + NO row for orphan_id.
    conn.execute(
        "INSERT INTO canonical_entities VALUES (?, ?, ?)",
        ("aaaa0000bbbb1111", "Qualixar", "company"),
    )
    conn.commit()

    # Build 5 facts referencing BOTH a real entity and an orphan id.
    orphan_id = "ea701bf01f1ff4df8"
    real_id = "aaaa0000bbbb1111"
    facts = []
    for i in range(5):
        facts.append({
            "canonical_entities_json": f'["{real_id}","{orphan_id}"]',
        })

    store = BehavioralPatternStore(str(tmp_path / "learning.db"))
    gen = pattern_miner._mine_entity_preferences(
        store, conn, facts, profile_id="p", dry_run=False,
    )

    out = store.get_patterns(profile_id="p")
    values = [(p.get("metadata") or {}).get("value") for p in out]
    assert "Qualixar" in values, \
        f"real entity missing from output: {values!r}"
    assert orphan_id not in values, \
        f"orphan id leaked into patterns: {values!r}"
    conn.close()
