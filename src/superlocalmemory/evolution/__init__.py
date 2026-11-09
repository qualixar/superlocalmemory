# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Skill Evolution Engine — track, analyze, and evolve AI agent skills.

3-trigger system (post-session + degradation + health check) with
LLM confirmation gate and blind verification.

Inspired by: HKUDS/OpenSpace (arXiv:2604.01687), ECC continuous learning.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from superlocalmemory.evolution.types import (
    EvolutionCandidate,
    EvolutionRecord,
    EvolutionStatus,
    EvolutionType,
    TriggerType,
)

__all__ = [
    "EvolutionCandidate",
    "EvolutionRecord",
    "EvolutionType",
    "TriggerType",
    "EvolutionStatus",
]
