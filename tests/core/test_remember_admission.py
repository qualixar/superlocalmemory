# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Focused service tests: journal first, canonical dispatch second."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from superlocalmemory.core.remember_admission import (
    AdmissionRejected,
    RememberAdmissionCommand,
    RememberService,
)
from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    AdmissionJournalUnavailable,
    RememberRequest,
    TerminalAdmissionError,
)

_FUNCTIONAL_DEADLINE_MS = 1_500
_BUDGET_TEST_DEADLINE_MS = 250


@dataclass(frozen=True)
class _TestCodec:
    def encrypt(self, plaintext: bytes) -> bytes:
        return b"policy:" + plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        return ciphertext[len(b"policy:") :][::-1]


class _Coordinator:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.commands: list[RememberAdmissionCommand] = []
        self.wait_ms: list[int] = []

    def submit(self, command: RememberAdmissionCommand, *, wait_ms: int) -> dict[str, object]:
        assert 0 < wait_ms <= _FUNCTIONAL_DEADLINE_MS
        self.commands.append(command)
        self.wait_ms.append(wait_ms)
        return self.result


class _TerminalCoordinator:
    def submit(self, _command, *, wait_ms: int):
        assert 0 < wait_ms <= _FUNCTIONAL_DEADLINE_MS
        raise TerminalAdmissionError("DETERMINISTIC_POLICY_REJECTED")


def _request() -> RememberRequest:
    return RememberRequest(
        content="Acknowledge only after canonical facts are queryable.",
        profile_id="default",
        source_type="http",
        idempotency_key="remember-service:one",
    )


def _actor() -> Actor:
    return Actor(
        principal_id="daemon-capability:service-test",
        allowed_profiles=frozenset({"default"}),
        allowed_scopes=frozenset({"personal"}),
    )


def test_remember_prepares_before_dispatch_and_duplicate_returns_original_receipt(tmp_path) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    coordinator = _Coordinator(
        {
            "state": "committed",
            "receipt": {
                "operation_id": "op-1",
                "fact_ids": ["fact-1"],
                "state": "queryable",
                "commit_sequence": 1,
            },
        }
    )
    service = RememberService(journal, coordinator)

    first = service.remember(
        _request(), _actor(), deadline_ms=_FUNCTIONAL_DEADLINE_MS
    )
    duplicate = service.remember(
        _request(), _actor(), deadline_ms=_FUNCTIONAL_DEADLINE_MS
    )

    assert first == duplicate
    assert len(coordinator.commands) == 1
    assert coordinator.commands[0].request.content == _request().content
    assert journal.get(coordinator.commands[0].journal_id).state == "committed"


def test_coordinator_receives_only_the_remaining_end_to_end_budget(
    tmp_path,
    monkeypatch,
) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    coordinator = _Coordinator(
        {
            "state": "committed",
            "receipt": {
                "operation_id": "op-budget",
                "fact_ids": ["fact-budget"],
                "state": "queryable",
                "commit_sequence": 1,
            },
        }
    )
    original_prepare = journal.prepare

    def delayed_prepare(*args, **kwargs):
        time.sleep(0.05)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(journal, "prepare", delayed_prepare)
    RememberService(journal, coordinator).remember(
        _request(),
        _actor(),
        deadline_ms=_BUDGET_TEST_DEADLINE_MS,
    )

    assert len(coordinator.wait_ms) == 1
    assert 0 < coordinator.wait_ms[0] < 225


def test_slow_codec_cannot_rebase_deadline_and_persist_after_expiry(tmp_path) -> None:
    class SlowCodec(_TestCodec):
        def encrypt(self, plaintext: bytes) -> bytes:
            time.sleep(0.06)
            return super().encrypt(plaintext)

    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=SlowCodec())
    coordinator = _Coordinator({"state": "committed", "receipt": {}})

    with pytest.raises(AdmissionJournalUnavailable, match="deadline"):
        RememberService(journal, coordinator).remember(
            _request(),
            _actor(),
            deadline_ms=50,
        )

    assert journal.count() == 0
    assert coordinator.commands == []


def test_rejected_dispatch_leaves_durable_rejected_record_for_diagnosis(tmp_path) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    service = RememberService(
        journal,
        _Coordinator({"state": "rejected", "receipt": {}, "error_code": "COMMAND_REJECTED"}),
    )

    with pytest.raises(AdmissionRejected, match="COMMAND_REJECTED"):
        service.remember(
            _request(), _actor(), deadline_ms=_FUNCTIONAL_DEADLINE_MS
        )

    entry = journal.get_by_idempotency_key("default", _request().idempotency_key)
    assert entry is not None
    assert entry.state == "rejected"
    assert entry.error_code == "COMMAND_REJECTED"


def test_nonpositive_deadline_and_retryable_result_do_not_acknowledge(tmp_path) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    service = RememberService(
        journal,
        _Coordinator({"state": "retryable", "receipt": {}, "error_code": "WRITE_OVERLOADED"}),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        service.remember(_request(), _actor(), deadline_ms=0)
    with pytest.raises(AdmissionRejected, match="WRITE_OVERLOADED"):
        service.remember(
            _request(), _actor(), deadline_ms=_FUNCTIONAL_DEADLINE_MS
        )

    entry = journal.get_by_idempotency_key("default", _request().idempotency_key)
    assert entry is not None
    assert entry.state == "dispatched"


def test_terminal_dispatch_exception_is_rejected_and_remains_terminal(tmp_path) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    service = RememberService(journal, _TerminalCoordinator())

    for _attempt in range(2):
        with pytest.raises(AdmissionRejected, match="DETERMINISTIC_POLICY_REJECTED") as caught:
            service.remember(
                _request(), _actor(), deadline_ms=_FUNCTIONAL_DEADLINE_MS
            )
        assert caught.value.retryable is False

    entry = journal.get_by_idempotency_key("default", _request().idempotency_key)
    assert entry is not None
    assert entry.state == "rejected"
    assert entry.error_code == "DETERMINISTIC_POLICY_REJECTED"
