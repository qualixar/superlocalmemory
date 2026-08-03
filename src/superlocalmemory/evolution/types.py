# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Skill Evolution type definitions.

Immutable data classes for evolution candidates, records, and lineage.
All types are frozen dataclasses — no mutation after creation.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvolutionType(str, Enum):
    """How the skill is being evolved."""
    FIX = "fix"           # Repair broken skill in-place
    DERIVED = "derived"   # Create specialized variant
    CAPTURED = "captured" # Extract new skill from patterns


class TriggerType(str, Enum):
    """What triggered the evolution."""
    POST_SESSION = "post_session"   # Session Stop hook analysis
    DEGRADATION = "degradation"     # Behavioral assertion confidence drop
    HEALTH_CHECK = "health_check"   # Periodic consolidation scan


class EvolutionStatus(str, Enum):
    """Pipeline status.

    State machine (Phase 2):
      CANDIDATE → VERIFIED_QUARANTINED → APPROVED → ACTIVE
                              ↓                ↓
                           REJECTED        ROLLED_BACK

    Legacy values (CONFIRMED, MUTATED, VERIFIED, PROMOTED) are retained for
    DB-row compatibility with pre-Phase-2 records.  New code uses the states
    above.  PROMOTED is an alias for what is now VERIFIED_QUARANTINED in the
    live-in-quarantine sense; new records use VERIFIED_QUARANTINED explicitly.
    """
    CANDIDATE            = "candidate"             # Detected, not yet processed
    CONFIRMED            = "confirmed"             # Legacy: LLM gate passed
    MUTATED              = "mutated"               # Legacy: new SKILL.md generated
    VERIFIED             = "verified"              # Legacy: blind verify passed
    PROMOTED             = "promoted"              # Legacy alias for quarantined
    REJECTED             = "rejected"              # Failed gate or verification
    FAILED               = "failed"               # Error during evolution
    # --- Phase 2 states ---
    VERIFIED_QUARANTINED = "verified_quarantined"  # Passed blind verify, in quarantine
    APPROVED             = "approved"              # Human/policy approved; ready to activate
    ACTIVE               = "active"               # Live in skill directory
    ROLLED_BACK          = "rolled_back"           # Activation reverted; prior artifact restored


@dataclass(frozen=True)
class EvolutionCandidate:
    """A skill flagged for potential evolution."""
    skill_name: str
    evolution_type: EvolutionType
    trigger: TriggerType
    evidence: tuple[str, ...] = ()
    effective_score: float = 0.0
    invocation_count: int = 0
    session_id: str = ""
    project_path: str = ""


@dataclass(frozen=True)
class EvolutionRecord:
    """Persisted record of an evolution attempt."""
    id: str
    skill_name: str
    parent_skill_id: Optional[str]
    evolution_type: EvolutionType
    trigger: TriggerType
    generation: int = 0
    status: EvolutionStatus = EvolutionStatus.CANDIDATE
    mutation_summary: str = ""
    evidence: tuple[str, ...] = ()
    original_content: str = ""
    evolved_content: str = ""
    content_diff: str = ""
    blind_verified: bool = False
    rejection_reason: str = ""
    created_at: str = ""
    completed_at: str = ""
    # Phase 2 (CRIT-2): the sanitized directory name inside the quarantine root,
    # e.g. "brainstorming-vabc12".  Distinct from skill_name ("brainstorming").
    # Stored in-memory; not persisted to skill_evolution_log (no schema column).
    # The activator reads this to locate the artifact; it is also stored in the
    # VERIFIED_QUARANTINED transition metadata for auditability.
    quarantine_dir_name: str = ""


@dataclass(frozen=True)
class SkillLineage:
    """Lineage metadata for an evolved skill."""
    skill_id: str
    parent_skill_id: Optional[str]
    evolution_type: EvolutionType
    generation: int
    trigger: TriggerType
    mutation_summary: str = ""
    created_at: str = ""

    @property
    def is_root(self) -> bool:
        return self.parent_skill_id is None
