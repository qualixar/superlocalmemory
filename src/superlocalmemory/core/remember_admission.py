# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Journal-first service boundary for a durable, immediately-queryable remember."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    AdmissionJournalUnavailable,
    PreparedAdmission,
    RememberRequest,
    TerminalAdmissionError,
)


class AdmissionRejected(RuntimeError):
    """Canonical admission was not committed; the journal records its outcome."""

    def __init__(self, error_code: str, *, retryable: bool = False) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RememberReceipt:
    """Canonical receipt returned only after admission is queryable."""

    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, receipt: Mapping[str, Any]) -> RememberReceipt:
        return cls(payload=dict(receipt))


@dataclass(frozen=True, slots=True)
class RememberAdmissionCommand:
    """Typed foreground input for a coordinator's bounded canonical transaction."""

    journal_id: str
    request_hash: str
    request: RememberRequest
    profile_id: str
    idempotency_key: str

    @classmethod
    def from_prepared(
        cls, prepared: PreparedAdmission, request: RememberRequest
    ) -> RememberAdmissionCommand:
        return cls(
            journal_id=prepared.journal_id,
            request_hash=prepared.request_hash,
            request=request,
            profile_id=prepared.profile_id,
            idempotency_key=prepared.idempotency_key,
        )


class RememberCoordinator(Protocol):
    def submit(self, command: RememberAdmissionCommand, *, wait_ms: int) -> Any: ...


class RememberService:
    """Apply the journaling pattern without doing enrichment inside admission."""

    def __init__(self, journal: AdmissionJournal, coordinator: RememberCoordinator) -> None:
        self._journal = journal
        self._coordinator = coordinator

    def remember(
        self, request: RememberRequest, actor: Actor, *, deadline_ms: int
    ) -> RememberReceipt:
        if deadline_ms <= 0:
            raise ValueError("deadline_ms must be greater than zero")
        deadline = time.monotonic() + deadline_ms / 1_000
        prepared = self._journal.prepare(
            request,
            actor,
            deadline=deadline,
        )
        if prepared.original_receipt is not None:
            return RememberReceipt.from_mapping(prepared.original_receipt)
        if prepared.state == "rejected":
            raise AdmissionRejected(prepared.error_code or "COMMAND_REJECTED")

        command_request = self._journal.request_for(
            prepared,
            deadline=deadline,
        )
        dispatched = self._journal.mark_dispatched(
            prepared.journal_id,
            deadline=deadline,
        )
        if dispatched.original_receipt is not None:
            return RememberReceipt.from_mapping(dispatched.original_receipt)
        try:
            result = self._coordinator.submit(
                RememberAdmissionCommand.from_prepared(prepared, command_request),
                wait_ms=_remaining_milliseconds(deadline),
            )
        except TerminalAdmissionError as exc:
            self._journal.mark_rejected(
                prepared.journal_id,
                exc.error_code,
                deadline=deadline,
            )
            raise AdmissionRejected(exc.error_code) from exc
        state = _result_value(result, "state")
        receipt = _result_value(result, "receipt") or {}
        if state in {"committed", "duplicate"}:
            if not isinstance(receipt, Mapping):
                raise AdmissionRejected("COMMAND_REJECTED: canonical result had no receipt")
            committed = self._journal.mark_committed(
                prepared.journal_id,
                receipt,
                deadline=deadline,
            )
            return RememberReceipt.from_mapping(committed.original_receipt or receipt)

        error_code = str(_result_value(result, "error_code") or "COMMAND_REJECTED")
        if state == "rejected":
            self._journal.mark_rejected(
                prepared.journal_id,
                error_code,
                deadline=deadline,
            )
        raise AdmissionRejected(error_code, retryable=state != "rejected")


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(key)
    return getattr(result, key, None)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AdmissionJournalUnavailable("remember admission deadline expired")
    return remaining


def _remaining_milliseconds(deadline: float) -> int:
    return max(1, math.ceil(_remaining_seconds(deadline) * 1_000))
