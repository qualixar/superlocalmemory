# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The correction ledger's errors (split out of ``correction_cases``, 4.2.30 R-39).

``correction_cases`` re-exports every name, so callers keep importing from it.
"""

from __future__ import annotations


class CorrectionCaseError(RuntimeError):
    """Base exception for correction-case lifecycle failures."""


class CorrectionAuthorizationError(CorrectionCaseError):
    """The host/server did not attest the acting identity as trusted."""


class CorrectionCompareAndSetError(CorrectionCaseError):
    """The caller attempted a state transition against a stale case version."""


class CorrectionFactErasingError(CorrectionCompareAndSetError):
    """A fact the case names is being deleted (its erasure tombstone exists)."""


class CorrectionIdempotencyError(CorrectionCaseError):
    """A replay key names a different correction proposal."""


class CorrectionDismissedError(CorrectionCaseError):
    """The user dismissed this machine proposal in bulk; it is not filed again."""


class CorrectionNotFoundError(CorrectionCaseError):
    """No correction case exists for the requested identifier."""


__all__ = ["CorrectionAuthorizationError", "CorrectionCaseError", "CorrectionCompareAndSetError",
           "CorrectionDismissedError", "CorrectionFactErasingError", "CorrectionIdempotencyError",
           "CorrectionNotFoundError"]
