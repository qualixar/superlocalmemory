# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for superlocalmemory.storage.memory_kinds — the kind vocabulary.

WP-2 (memory-kinds-4.1.19). Covers parse_kind, is_confirmed, the AUTHORITY
order, COARSE totality, legacy spelling recovery, and kind_fields' display
states (confirmed / suggested / legacy / untyped).
"""

from __future__ import annotations

import pytest

from superlocalmemory.storage.memory_kinds import (
    ALIASES,
    AUTHORITY,
    COARSE,
    CONFIRMED_SOURCES,
    LEGACY_TO_KIND,
    KindAssignment,
    KindSource,
    MemoryKind,
    is_confirmed,
    kind_fields,
    parse_kind,
)
from superlocalmemory.storage.models import FactType


class TestParseKind:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("semantic", MemoryKind.SEMANTIC),
            ("RULE", MemoryKind.RULE),
            ("  RULE  ", MemoryKind.RULE),
            ("project_state", MemoryKind.STATUS),
            ("project-state", MemoryKind.STATUS),
            ("to-do", MemoryKind.PROSPECTIVE),
            ("todo", MemoryKind.PROSPECTIVE),
            ("fact", MemoryKind.SEMANTIC),
            ("knowledge", MemoryKind.SEMANTIC),
            ("event", MemoryKind.EPISODIC),
            ("experience", MemoryKind.EPISODIC),
            ("preference", MemoryKind.OPINION),
            ("view", MemoryKind.OPINION),
            ("instruction", MemoryKind.RULE),
            ("constraint", MemoryKind.RULE),
            ("policy", MemoryKind.RULE),
            ("convention", MemoryKind.RULE),
            ("choice", MemoryKind.DECISION),
            ("how-to", MemoryKind.PROCEDURE),
            ("howto", MemoryKind.PROCEDURE),
            ("steps", MemoryKind.PROCEDURE),
            ("recipe", MemoryKind.PROCEDURE),
            ("plan", MemoryKind.PROSPECTIVE),
            ("commitment", MemoryKind.PROSPECTIVE),
            ("task", MemoryKind.PROSPECTIVE),
            ("erratum", MemoryKind.CORRECTION),
            ("fix", MemoryKind.CORRECTION),
        ],
    )
    def test_parse_kind_accepts_values_and_aliases(self, raw: str, expected: MemoryKind) -> None:
        assert parse_kind(raw) is expected

    @pytest.mark.parametrize(
        "raw",
        [None, 5, "", "   ", "not-a-kind", "DROP TABLE x", object(), [], {}],
    )
    def test_parse_kind_never_raises(self, raw: object) -> None:
        # Must not raise for any of these; a bad kind is never fatal (I1).
        result = parse_kind(raw)
        assert result is None or isinstance(result, MemoryKind)

    def test_parse_kind_recovers_whitespace_and_case(self) -> None:
        assert parse_kind("  RULE ") is MemoryKind.RULE
        assert parse_kind("Decision") is MemoryKind.DECISION


class TestCoarseAndLegacy:
    def test_coarse_is_total_and_maps_to_legacy_values(self) -> None:
        legacy_values = {ft.value for ft in FactType}
        for kind in MemoryKind:
            assert kind in COARSE, f"COARSE missing {kind}"
            assert COARSE[kind] in legacy_values, f"COARSE[{kind}] not a FactType value"

    def test_legacy_spellings_map(self) -> None:
        assert LEGACY_TO_KIND["temporal"] is MemoryKind.PROSPECTIVE
        assert LEGACY_TO_KIND["world"] is MemoryKind.SEMANTIC
        assert LEGACY_TO_KIND["experience"] is MemoryKind.EPISODIC
        assert LEGACY_TO_KIND["semantic"] is MemoryKind.SEMANTIC
        assert LEGACY_TO_KIND["episodic"] is MemoryKind.EPISODIC
        assert LEGACY_TO_KIND["opinion"] is MemoryKind.OPINION
        assert LEGACY_TO_KIND["prospective"] is MemoryKind.PROSPECTIVE


class TestAuthorityOrder:
    def test_authority_order(self) -> None:
        assert AUTHORITY[KindSource.USER] > AUTHORITY[KindSource.CALLER]
        assert AUTHORITY[KindSource.CALLER] > AUTHORITY[KindSource.MODEL_LLM]
        assert AUTHORITY[KindSource.CALLER] > AUTHORITY[KindSource.MODEL_LAYA]
        assert AUTHORITY[KindSource.CALLER] > AUTHORITY[KindSource.MODEL_JEV]
        assert AUTHORITY[KindSource.MODEL_LLM] == AUTHORITY[KindSource.MODEL_JEV]
        assert AUTHORITY[KindSource.MODEL_LLM] == AUTHORITY[KindSource.MODEL_LAYA]
        assert AUTHORITY[KindSource.MODEL_LLM] > AUTHORITY[KindSource.RULES]
        assert AUTHORITY[KindSource.RULES] > AUTHORITY[KindSource.LEGACY]
        # Every member of KindSource is covered — a silently-missing entry
        # would make a comparison raise deep inside an upsert.
        assert set(AUTHORITY) == set(KindSource)

    def test_is_confirmed(self) -> None:
        assert is_confirmed("user") is True
        assert is_confirmed("caller") is True
        assert is_confirmed("model:llm") is False
        assert is_confirmed("rules") is False
        assert is_confirmed("legacy") is False
        assert is_confirmed(None) is False
        assert is_confirmed(123) is False  # type: ignore[arg-type]
        assert is_confirmed("") is False

    def test_confirmed_sources_matches_authority_top(self) -> None:
        assert CONFIRMED_SOURCES == {KindSource.USER, KindSource.CALLER}


class TestKindAssignment:
    def test_as_columns(self) -> None:
        assignment = KindAssignment(
            kind=MemoryKind.DECISION, source=KindSource.CALLER,
            confidence=None, recipe="caller",
        )
        cols = assignment.as_columns("2026-01-01T00:00:00+00:00")
        assert cols == {
            "memory_kind": "decision",
            "memory_kind_source": "caller",
            "memory_kind_confidence": None,
            "memory_kind_recipe": "caller",
            "memory_kind_at": "2026-01-01T00:00:00+00:00",
        }


class TestKindFields:
    def test_confirmed_state(self) -> None:
        fields = kind_fields({
            "memory_kind": "rule", "memory_kind_source": "caller",
            "memory_kind_confidence": None, "fact_type": "semantic",
        })
        assert fields["memory_kind"] == "rule"
        assert fields["memory_kind_state"] == "confirmed"
        assert fields["memory_kind_label"] == "Standing rule"
        assert fields["memory_kind_source"] == "caller"

    def test_suggested_state_above_threshold(self) -> None:
        fields = kind_fields({
            "memory_kind": "decision", "memory_kind_source": "model:llm",
            "memory_kind_confidence": 0.75, "fact_type": "episodic",
        }, display_min_confidence=0.20)
        assert fields["memory_kind_state"] == "suggested"
        assert fields["memory_kind"] == "decision"

    def test_below_threshold_falls_back_to_legacy(self) -> None:
        fields = kind_fields({
            "memory_kind": "decision", "memory_kind_source": "model:llm",
            "memory_kind_confidence": 0.05, "fact_type": "episodic",
        }, display_min_confidence=0.20)
        assert fields["memory_kind_state"] == "legacy"
        assert fields["memory_kind"] == "episodic"
        assert fields["memory_kind_source"] is None

    def test_rules_suggestion_is_shown_without_a_confidence(self) -> None:
        # The rules never give a confidence; their kind must still be shown,
        # not hidden behind the old fact type.
        fields = kind_fields({
            "memory_kind": "decision", "memory_kind_source": "rules",
            "memory_kind_confidence": None, "fact_type": "semantic",
        }, display_min_confidence=0.20)
        assert fields["memory_kind_state"] == "suggested"
        assert fields["memory_kind"] == "decision"
        assert fields["memory_kind_source"] == "rules"
        assert fields["memory_kind_confidence"] is None

    def test_model_suggestion_without_a_confidence_is_not_shown(self) -> None:
        # A model answer always carries a confidence; one without it is not
        # trusted for display.
        fields = kind_fields({
            "memory_kind": "decision", "memory_kind_source": "model:laya",
            "memory_kind_confidence": None, "fact_type": "semantic",
        })
        assert fields["memory_kind_state"] == "legacy"
        assert fields["memory_kind"] == "semantic"

    def test_untyped_row_falls_back_to_legacy_fact_type(self) -> None:
        fields = kind_fields({
            "memory_kind": None, "memory_kind_source": None,
            "memory_kind_confidence": None, "fact_type": "prospective",
        })
        assert fields["memory_kind_state"] == "legacy"
        assert fields["memory_kind"] == "prospective"

    def test_unmappable_row_is_untyped(self) -> None:
        fields = kind_fields({
            "memory_kind": None, "memory_kind_source": None,
            "memory_kind_confidence": None, "fact_type": "not-a-real-type",
        })
        assert fields["memory_kind_state"] == "untyped"
        assert fields["memory_kind"] is None

    def test_reads_attribute_access_too(self) -> None:
        class Row:
            memory_kind = "opinion"
            memory_kind_source = "user"
            memory_kind_confidence = None
            fact_type = "opinion"

        fields = kind_fields(Row())
        assert fields["memory_kind_state"] == "confirmed"
        assert fields["memory_kind"] == "opinion"

    def test_never_raises_on_garbage_input(self) -> None:
        for obj in (None, 5, "x", object(), [], {}):
            fields = kind_fields(obj)
            assert fields["memory_kind_state"] in {"confirmed", "suggested", "legacy", "untyped"}

    def test_aliases_cover_every_kind(self) -> None:
        assert set(ALIASES.values()) <= set(MemoryKind)
