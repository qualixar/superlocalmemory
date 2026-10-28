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

    @property
    def data_locality_label(self) -> str:
        """UI/API locality label derived from the mode record (single source).

        Never invent a parallel mode→label map in the dashboard — consume this.
        """
        return "local-only" if self.data_stays_local else "provider-assisted"


# ---------------------------------------------------------------------------
# Mode copy — single source of truth for user-facing mode text (#112)
# ---------------------------------------------------------------------------
#
# Modes are named by what the user GETS, never by a vendor:
#   A = no language model
#   B = a model on this machine (Ollama by default, any local server works)
#   C = your own endpoint or a cloud provider
#
# Every surface that describes a mode to a user (CLI help, setup wizard, MCP
# tool descriptions, the dashboard) must read from here, or use wording that
# matches it. A vendor name may appear as an EXAMPLE default, never as the
# definition of the mode — Mode B's "requirement" is a model on this machine,
# not Ollama specifically; Mode C's is an endpoint the user points at, not
# "the cloud".

MODE_SHORT_NAME: dict[Mode, str] = {
    Mode.A: "Local Guardian",
    Mode.B: "Smart Local",
    Mode.C: "Full Power",
}

#: One-line, experience-defined description of each mode's model boundary.
#: Mode A and B deliberately contain the exact substring "nothing leaves this
#: device" — tools_v3._mode_description() swaps it for "nothing else leaves
#: this device" once the optional online answer check is on. Do not reword
#: that phrase without updating the swap.
MODE_TAGLINE: dict[Mode, str] = {
    Mode.A: (
        "Memories are saved and recalled without a language model — nothing leaves "
        "this device. Fastest and most private."
    ),
    Mode.B: (
        "A model on this machine — Ollama by default, any local server works — "
        "and nothing leaves this device."
    ),
    Mode.C: (
        "Your own endpoint, or a cloud provider, for the best recall quality. "
        "Your queries leave this device."
    ),
}


def mode_short_name(mode: Mode | str) -> str:
    """The 2-word product name for a mode (e.g. "Smart Local")."""
    if isinstance(mode, str):
        mode = Mode(mode.strip().lower())
    return MODE_SHORT_NAME[mode]


def mode_tagline(mode: Mode | str) -> str:
    """The one-line, vendor-neutral capability blurb for a mode."""
    if isinstance(mode, str):
        mode = Mode(mode.strip().lower())
    return MODE_TAGLINE[mode]


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
        "Local Guardian — No language model, zero cloud. "
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
        "Smart Local — a model on this machine (Ollama by default, any local "
        "OpenAI-compatible server works). "
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
        "Full Power — your own endpoint, or a cloud provider, for the best "
        "embeddings and LLMs you configure (3072-dim embeddings available). "
        "Agentic multi-round retrieval. Optional cloud reranker. "
        "Processing through a provider requires deployment-specific "
        "privacy, contractual, and EU AI Act assessment."
    ),
)


def get_capabilities(mode: Mode) -> ModeCapabilities:
    """Get capability matrix for a mode."""
    _map = {Mode.A: MODE_A, Mode.B: MODE_B, Mode.C: MODE_C}
    return _map[mode]


def dashboard_mode_fields(
    mode: Mode | str, embedding: object = None, llm: object = None,
) -> dict[str, object]:
    """Fields the dashboard API must surface about where data goes.

    Keeps UI locality claims bound to :class:`ModeCapabilities` so Mode C
    cannot be labeled local-only by a divergent hardcode. With the embedding
    and language-model configuration given (#303), the answer is the effective
    one: a remote endpoint turns a local-only mode into provider-assisted. The
    parts are named, never their hosts, since this payload reaches viewers.
    """
    from superlocalmemory.core.data_locality import effective_data_locality

    if isinstance(mode, str):
        mode = Mode(mode.strip().lower())
    locality = effective_data_locality(mode, embedding, llm)
    return {
        "data_locality_label": locality.label,
        "data_stays_local": locality.stays_local,
        "data_remote_parts": list(locality.remote_parts),
    }


def validate_mode_config(
    mode: Mode,
    *,
    has_llm: bool = False,
    has_cloud_llm: bool = False,
    embedding: object = None,
    llm: object = None,
) -> list[str]:
    """Validate that required services are available for the chosen mode.

    ``has_llm`` reports whether Mode B's model-on-this-machine is reachable —
    Ollama by default, but any local OpenAI-compatible server qualifies
    (#112). Naming it ``has_ollama`` baked one vendor into the check itself;
    the capability Mode B needs is "a local model answers", not "Ollama
    answers".

    ``embedding`` and ``llm`` are the configured blocks (#303). Given them, a
    local mode whose embedding endpoint or language model is not on this
    computer gets a warning saying so, since the mode's local-only default no
    longer holds.

    Returns list of warnings/errors. Empty list = all good.
    """
    issues: list[str] = []
    caps = get_capabilities(mode)

    if caps.llm_fact_extraction and mode == Mode.B and not has_llm:
        issues.append(
            "Mode B has no local model available. Falling back to Mode A extraction."
        )

    if caps.cloud_embeddings and not has_cloud_llm:
        issues.append("Mode C cloud embeddings configured but no API endpoint provided.")

    if caps.agentic_retrieval and not has_cloud_llm:
        issues.append("Mode C agentic retrieval requires cloud LLM but none configured.")

    if caps.data_stays_local and (embedding is not None or llm is not None):
        from superlocalmemory.core.data_locality import (
            effective_data_locality,
            remote_parts_sentence,
        )

        locality = effective_data_locality(mode, embedding, llm)
        if not locality.stays_local:
            issues.append(
                f"Mode {mode.value.upper()} is local by default. "
                f"{remote_parts_sentence(locality)}"
            )

    return issues
