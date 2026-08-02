# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


class ObligationKind(StrEnum):
    APPLY = "apply"
    ERASE = "erase"


class ObligationState(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    COMPENSATED = "compensated"
    ERASED = "erased"


_TERMINAL_SUCCESS = frozenset({ObligationState.VERIFIED, ObligationState.ERASED})


def is_terminal_success(state: ObligationState) -> bool:
    return state in _TERMINAL_SUCCESS


def _freeze(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True, slots=True)
class OperationContext:
    operation_id: str
    profile_id: str
    subject_id: str
    revision: int = 0
    generation: int = 0
    fact_ids: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id is required")
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if not self.subject_id:
            raise ValueError("subject_id is required")
        object.__setattr__(self, "fact_ids", tuple(self.fact_ids))
        object.__setattr__(self, "payload", _freeze(self.payload))


@dataclass(frozen=True, slots=True)
class OwnerResult:
    owner: str
    ok: bool
    checksum: str | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", _freeze(self.detail))


@dataclass(frozen=True, slots=True)
class OwnerErasureProof:
    owner: str
    erased: bool
    checksum: str | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", _freeze(self.detail))


@dataclass(frozen=True, slots=True)
class OwnerHealth:
    owner: str
    healthy: bool
    detail: str = ""


@runtime_checkable
class ProjectionOwner(Protocol):
    @property
    def name(self) -> str: ...

    def apply(self, context: OperationContext) -> OwnerResult: ...

    def verify(self, context: OperationContext) -> OwnerResult: ...

    def compensate(self, context: OperationContext) -> OwnerResult: ...

    def erase(self, context: OperationContext) -> OwnerErasureProof: ...

    def prove_erased(self, context: OperationContext) -> OwnerErasureProof: ...

    def health(self) -> OwnerHealth: ...


__all__ = [
    "ObligationKind",
    "ObligationState",
    "OperationContext",
    "OwnerErasureProof",
    "OwnerHealth",
    "OwnerResult",
    "ProjectionOwner",
    "is_terminal_success",
]
