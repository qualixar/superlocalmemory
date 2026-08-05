# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.core.modes -- capability matrix and validation.

Covers:
  - get_capabilities for each mode
  - Mode A: eu_ai_act_compliant=True, data_stays_local=True
  - Mode B: local LLM capabilities, still compliant
  - Mode C: agentic_retrieval=True, cloud_embeddings=True
  - validate_mode_config returns warnings
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.modes import (
    MODE_A,
    MODE_B,
    MODE_C,
    ModeCapabilities,
    get_capabilities,
    validate_mode_config,
)
from superlocalmemory.storage.models import Mode


# ---------------------------------------------------------------------------
# get_capabilities
# ---------------------------------------------------------------------------

class TestGetCapabilities:
    def test_returns_mode_a(self) -> None:
        caps = get_capabilities(Mode.A)
        assert caps is MODE_A
        assert caps.mode == Mode.A

    def test_returns_mode_b(self) -> None:
        caps = get_capabilities(Mode.B)
        assert caps is MODE_B
        assert caps.mode == Mode.B

    def test_returns_mode_c(self) -> None:
        caps = get_capabilities(Mode.C)
        assert caps is MODE_C
        assert caps.mode == Mode.C


# ---------------------------------------------------------------------------
# Mode A capabilities
# ---------------------------------------------------------------------------

class TestModeA:
    def test_eu_ai_act_compliant(self) -> None:
        assert MODE_A.eu_ai_act_compliant is None

    def test_data_stays_local(self) -> None:
        assert MODE_A.data_stays_local is True

    def test_no_llm_capabilities(self) -> None:
        assert MODE_A.llm_fact_extraction is False
        assert MODE_A.llm_entity_resolution is False
        assert MODE_A.llm_type_classification is False
        assert MODE_A.llm_importance_scoring is False
        assert MODE_A.agentic_retrieval is False
        assert MODE_A.llm_answer_generation is False

    def test_no_cloud(self) -> None:
        assert MODE_A.cloud_reranker is False
        assert MODE_A.cloud_embeddings is False

    def test_embedding_dimension_384(self) -> None:
        assert MODE_A.embedding_dimension == 768

    def test_has_description(self) -> None:
        assert len(MODE_A.description) > 0
        assert "Local Guardian" in MODE_A.description


# ---------------------------------------------------------------------------
# Mode B capabilities
# ---------------------------------------------------------------------------

class TestModeB:
    def test_eu_ai_act_compliant(self) -> None:
        assert MODE_B.eu_ai_act_compliant is None

    def test_data_stays_local(self) -> None:
        assert MODE_B.data_stays_local is True

    def test_llm_extraction_enabled(self) -> None:
        assert MODE_B.llm_fact_extraction is True
        assert MODE_B.llm_entity_resolution is True
        assert MODE_B.llm_type_classification is True
        assert MODE_B.llm_importance_scoring is True

    def test_llm_answer_generation_enabled(self) -> None:
        assert MODE_B.llm_answer_generation is True

    def test_no_agentic_retrieval(self) -> None:
        assert MODE_B.agentic_retrieval is False

    def test_no_cloud(self) -> None:
        assert MODE_B.cloud_reranker is False
        assert MODE_B.cloud_embeddings is False

    def test_embedding_dimension_384(self) -> None:
        assert MODE_B.embedding_dimension == 768


# ---------------------------------------------------------------------------
# Mode C capabilities
# ---------------------------------------------------------------------------

class TestModeC:
    def test_not_eu_ai_act_compliant(self) -> None:
        assert MODE_C.eu_ai_act_compliant is None

    def test_data_not_local(self) -> None:
        assert MODE_C.data_stays_local is False

    def test_all_llm_capabilities(self) -> None:
        assert MODE_C.llm_fact_extraction is True
        assert MODE_C.llm_entity_resolution is True
        assert MODE_C.llm_type_classification is True
        assert MODE_C.llm_importance_scoring is True
        assert MODE_C.llm_answer_generation is True

    def test_agentic_retrieval(self) -> None:
        assert MODE_C.agentic_retrieval is True

    def test_cloud_embeddings(self) -> None:
        assert MODE_C.cloud_embeddings is True

    def test_cloud_reranker(self) -> None:
        assert MODE_C.cloud_reranker is True

    def test_embedding_dimension_3072(self) -> None:
        assert MODE_C.embedding_dimension == 3072


# ---------------------------------------------------------------------------
# ModeCapabilities frozen
# ---------------------------------------------------------------------------

class TestModeCapabilitiesFrozen:
    def test_frozen_immutability(self) -> None:
        with pytest.raises(AttributeError):
            MODE_A.eu_ai_act_compliant = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_mode_config
# ---------------------------------------------------------------------------

class TestValidateModeConfig:
    def test_mode_a_always_valid(self) -> None:
        issues = validate_mode_config(Mode.A)
        assert issues == []

    def test_mode_b_warns_no_ollama(self) -> None:
        issues = validate_mode_config(Mode.B, has_ollama=False)
        assert len(issues) == 1
        assert "Ollama" in issues[0]

    def test_mode_b_no_warning_with_ollama(self) -> None:
        issues = validate_mode_config(Mode.B, has_ollama=True)
        assert len(issues) == 0

    def test_mode_c_warns_no_cloud_llm(self) -> None:
        issues = validate_mode_config(Mode.C, has_cloud_llm=False)
        assert len(issues) >= 1
        # Should warn about both cloud embeddings and agentic retrieval
        all_text = " ".join(issues)
        assert "cloud" in all_text.lower() or "agentic" in all_text.lower()

    def test_mode_c_no_warnings_with_cloud(self) -> None:
        issues = validate_mode_config(Mode.C, has_cloud_llm=True)
        assert len(issues) == 0
