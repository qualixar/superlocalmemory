# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V4 mode capability descriptors.

Three operating modes with clear capability boundaries.
Mode A: deterministic local extraction and local inference.
Mode B: local LLM enrichment and local inference.
Mode C: configured provider-assisted inference.

An operating mode does not determine EU AI Act compliance.  That assessment
depends on the deployment, intended use, operator role, and applicable duties.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from dataclasses import dataclass

from superlocalmemory.storage.models import Mode


@dataclass(frozen=True)
class ModeCapabilities:
    """What each mode can and cannot do."""

    mode: Mode

    # Encoding capabilities
    llm_fact_extraction: bool      # Can use LLM for fact extraction?
    llm_entity_resolution: bool    # Can use LLM for entity disambiguation?
    llm_type_classification: bool  # Can use LLM for fact type routing?
    llm_importance_scoring: bool   # Can use LLM for importance assessment?

    # Retrieval capabilities
    agentic_retrieval: bool        # Can do multi-round LLM-guided retrieval?
    llm_answer_generation: bool    # Can LLM generate answers from context?
    cloud_reranker: bool           # Can use Cohere / cloud reranker?

    # Embedding capabilities
    cloud_embeddings: bool         # Can use cloud embedding API?
    embedding_dimension: int       # Expected embedding dimension

    # Compliance
    eu_ai_act_compliant: bool | None  # None: requires deployment assessment
    data_stays_local: bool         # Does ALL data stay on device?

    # Description
    description: str = ""


# ---------------------------------------------------------------------------
# Mode Definitions
# ---------------------------------------------------------------------------

MODE_A = ModeCapabilities(
    mode=Mode.A,
    llm_fact_extraction=False,
    llm_entity_resolution=False,
    llm_type_classification=False,
    llm_importance_scoring=False,
    agentic_retrieval=False,
    llm_answer_generation=False,
    cloud_reranker=False,
    cloud_embeddings=False,
    embedding_dimension=768,
    eu_ai_act_compliant=None,
    data_stays_local=True,
    description=(
        "Local Guardian — Zero LLM, zero cloud. "
        "Uses nomic-embed-text-v1.5 encoder (768d, 8K context) for embeddings. "
        "Deterministic rules for extraction and a local PyTorch cross-encoder. "
        "EU AI Act classification requires deployment assessment."
    ),
)

MODE_B = ModeCapabilities(
    mode=Mode.B,
    llm_fact_extraction=True,
    llm_entity_resolution=True,
    llm_type_classification=True,
    llm_importance_scoring=True,
    agentic_retrieval=False,
    llm_answer_generation=True,
    cloud_reranker=False,
    cloud_embeddings=False,
    embedding_dimension=768,
    eu_ai_act_compliant=None,
    data_stays_local=True,
    description=(
        "Smart Local — Local Ollama LLM (Phi-3, Llama 3.2). "
        "LLM-quality extraction and classification, fully local. "
        "Local PyTorch cross-encoder reranking. No configured cloud inference. "
        "EU AI Act classification requires deployment assessment."
    ),
)

MODE_C = ModeCapabilities(
    mode=Mode.C,
    llm_fact_extraction=True,
    llm_entity_resolution=True,
    llm_type_classification=True,
    llm_importance_scoring=True,
    agentic_retrieval=True,
    llm_answer_generation=True,
    cloud_reranker=True,
    cloud_embeddings=True,
    embedding_dimension=3072,
    eu_ai_act_compliant=None,
    data_stays_local=False,
    description=(
        "FULL POWER — UNRESTRICTED. Best embeddings (text-embedding-3-large, 3072-dim). "
        "Best configured cloud LLMs (e.g. GPT-5, Claude Opus 4). Agentic multi-round retrieval. "
        "Cohere reranker option. Cloud processing requires deployment-specific "
        "privacy, contractual, and EU AI Act assessment."
    ),
)


def get_capabilities(mode: Mode) -> ModeCapabilities:
    """Get capability matrix for a mode."""
    _map = {Mode.A: MODE_A, Mode.B: MODE_B, Mode.C: MODE_C}
    return _map[mode]


def validate_mode_config(mode: Mode, *, has_ollama: bool = False, has_cloud_llm: bool = False) -> list[str]:
    """Validate that required services are available for the chosen mode.

    Returns list of warnings/errors. Empty list = all good.
    """
    issues: list[str] = []
    caps = get_capabilities(mode)

    if caps.llm_fact_extraction and mode == Mode.B and not has_ollama:
        issues.append("Mode B requires Ollama but it is not available. Falling back to Mode A extraction.")

    if caps.cloud_embeddings and not has_cloud_llm:
        issues.append("Mode C cloud embeddings configured but no API endpoint provided.")

    if caps.agentic_retrieval and not has_cloud_llm:
        issues.append("Mode C agentic retrieval requires cloud LLM but none configured.")

    return issues
