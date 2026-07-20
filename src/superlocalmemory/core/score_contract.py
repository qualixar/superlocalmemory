# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Truthful public retrieval score contract.

Retrieval produces an ordering, not an answer probability.  This module keeps
query relevance, stored assertion confidence, trust, and internal ranking
utility distinct while preserving the V3.6 aliases for one compatibility
release.
"""

from __future__ import annotations

import math

from superlocalmemory.storage.models import RecallResponse


def _bounded(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return min(1.0, max(0.0, number))


def finalize_score_contract(response: RecallResponse) -> RecallResponse:
    """Finalize aliases, rank positions, and response abstention metadata."""
    for position, result in enumerate(response.results or (), start=1):
        relevance = _bounded(
            result.relevance_score
            if result.relevance_score is not None
            else result.score
        )
        memory_confidence = _bounded(
            result.memory_confidence
            if result.memory_confidence is not None
            else getattr(result.fact, "confidence", 0.0)
        )
        result.relevance_score = relevance
        result.score = relevance
        result.memory_confidence = memory_confidence
        result.confidence = memory_confidence
        result.rank_position = position

    response.score_contract_version = "2"
    response.calibration_status = "uncalibrated"
    response.calibration_id = None
    response.answer_confidence = None
    response.abstained = not bool(response.results)
    if response.abstained:
        response.abstention_reason = (
            "evidence_floor" if response.no_confident_match else "no_candidates"
        )
    else:
        response.abstention_reason = None
    return response


__all__ = ["finalize_score_contract"]
