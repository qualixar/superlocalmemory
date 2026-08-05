# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""EU AI Act technical deployment-posture reporting.

An operating mode can establish technical facts such as locality and use of
generative AI.  It cannot determine the Act's risk classification or certify
legal compliance without the system's intended purpose and deployment context.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from superlocalmemory.core.modes import get_capabilities
from superlocalmemory.storage.models import Mode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComplianceReport:
    """EU AI Act compliance assessment for a specific mode."""

    mode: Mode
    compliant: bool | None
    risk_category: str
    data_stays_local: bool
    uses_generative_ai: bool
    transparency_met: bool | None
    human_oversight: bool | None
    deployment_context_required: bool
    findings: list[str]
    timestamp: str


class EUAIActChecker:
    """Report mode-level technical posture without giving legal certification."""

    def check_compliance(self, mode: Mode) -> ComplianceReport:
        """Generate compliance report for a mode."""
        caps = get_capabilities(mode)
        findings: list[str] = []

        # Data locality
        data_local = caps.data_stays_local
        if not data_local:
            findings.append(
                "Data leaves device for cloud LLM processing. "
                "Requires Data Processing Agreement (DPA) with provider."
            )

        # Generative AI usage
        uses_gen_ai = caps.llm_fact_extraction or caps.llm_answer_generation
        local_gen_ai = uses_gen_ai and data_local

        if uses_gen_ai and not data_local:
            findings.append(
                "Uses cloud generative AI. EU AI Act Art. 52 requires "
                "transparency: users must be informed AI generates content."
            )

        if local_gen_ai:
            findings.append("Generative AI processing is configured to remain local.")
        findings.append(
            "Legal classification is undetermined: intended purpose, affected persons, "
            "sector, deployment context, and operator controls must be assessed."
        )
        findings.append(
            "Transparency and human-oversight obligations require deployment evidence; "
            "the operating mode alone cannot mark them as met."
        )

        return ComplianceReport(
            mode=mode,
            compliant=None,
            risk_category="undetermined",
            data_stays_local=data_local,
            uses_generative_ai=uses_gen_ai,
            transparency_met=None,
            human_oversight=None,
            deployment_context_required=True,
            findings=findings,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def verify_all_modes(self) -> dict[str, ComplianceReport]:
        """Generate compliance reports for all three modes."""
        return {
            mode.value: self.check_compliance(mode)
            for mode in Mode
        }

    def get_compliant_modes(self) -> list[Mode]:
        """Return no certified modes; mode alone is insufficient evidence."""
        return []
