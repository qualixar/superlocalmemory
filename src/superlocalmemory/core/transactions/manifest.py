# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from superlocalmemory.core.transactions.obligations import Obligation
from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    ObligationState,
    is_terminal_success,
)


class ManifestState(StrEnum):
    COMPLETE = "COMPLETE"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class OwnerEvidence:
    owner: str
    kind: ObligationKind
    state: ObligationState
    revision: int
    checksum: str

    def as_dict(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "kind": str(self.kind),
            "state": str(self.state),
            "revision": self.revision,
            "checksum": self.checksum,
        }


@dataclass(frozen=True, slots=True)
class CompletionManifest:
    operation_id: str
    profile_id: str
    state: ManifestState
    all_met: bool
    obligation_count: int
    owner_evidence: tuple[OwnerEvidence, ...]
    manifest_hash: str
    created_at: float
    updated_at: float


def build_evidence(obligations: Iterable[Obligation]) -> tuple[OwnerEvidence, ...]:
    evidence = [
        OwnerEvidence(
            owner=o.owner,
            kind=o.kind,
            state=o.state,
            revision=o.revision,
            checksum=o.checksum or "",
        )
        for o in obligations
    ]
    evidence.sort(key=lambda e: (e.owner, str(e.kind), e.revision))
    return tuple(evidence)


def compute_manifest_hash(evidence: Iterable[OwnerEvidence]) -> str:
    ordered = sorted(
        (e.as_dict() for e in evidence),
        key=lambda d: (d["owner"], d["kind"], d["revision"]),
    )
    canonical = json.dumps(
        ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_json(evidence: Iterable[OwnerEvidence]) -> str:
    ordered = sorted(
        (e.as_dict() for e in evidence),
        key=lambda d: (d["owner"], d["kind"], d["revision"]),
    )
    return json.dumps(
        ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def derive_state(
    obligations: tuple[Obligation, ...], *, canonical_committed: bool,
) -> tuple[ManifestState, bool]:
    if not canonical_committed:
        return ManifestState.FAILED, False
    if not obligations:
        return ManifestState.COMPLETE, True
    all_met = all(is_terminal_success(o.state) for o in obligations)
    if all_met:
        return ManifestState.COMPLETE, True
    return ManifestState.DEGRADED, False


__all__ = [
    "CompletionManifest",
    "ManifestState",
    "OwnerEvidence",
    "build_evidence",
    "compute_manifest_hash",
    "derive_state",
    "evidence_json",
]
