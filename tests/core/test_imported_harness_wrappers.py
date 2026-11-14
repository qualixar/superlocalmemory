# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Issue #319: importing a saved conversation that quotes agent-harness
notifications.

Three things were wrong. The prompt-template firewall, meant for automatic
capture, refused those parts of a file the user chose to import. The refusal
lost its reason on the way to the journal and the folder report. And because
a refusal is cached under its idempotency key, while a folder retry re-sends
the same keys, a refused part stayed refused forever even after a fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from superlocalmemory.core.ingest_gate import apply_ingest_gate
from superlocalmemory.core.injection import is_prompt_template, prompt_template_kind
from superlocalmemory.core.remember_admission import AdmissionRejected, RememberService
from superlocalmemory.storage.admission_journal import (
    UNEXPLAINED_REJECTION,
    Actor,
    AdmissionJournal,
    RememberRequest,
    TerminalAdmissionError,
)

QUOTED = (
    "### User\n\n<task-notification>\n<task-id>b1</task-id>\n"
    "<status>completed</status>\n<summary>Build finished</summary>\n"
    "</task-notification>\n\n### Assistant\n\nThe build passed."
)
TEMPLATE = "You are summarizing a Claude Code session. The first line must be exactly..."


# --- the gate -------------------------------------------------------------

@pytest.mark.parametrize("source_type", [None, "", "mcp", "hook", "http-remember"])
def test_captured_harness_wrappers_are_still_refused(source_type):
    gate = apply_ingest_gate(QUOTED, source_type=source_type)
    assert gate.rejected and gate.rejection_code == "harness_wrapper"
    assert is_prompt_template(QUOTED, source_type=source_type)


@pytest.mark.parametrize("source_type", ["folder", "document"])
def test_imported_files_keep_quoted_wrappers_as_source_text(source_type):
    gate = apply_ingest_gate(QUOTED, source_type=source_type)
    assert not gate.rejected
    assert gate.fact_content == QUOTED  # kept exactly, nothing stripped


@pytest.mark.parametrize("source_type", [None, "folder", "document"])
def test_prompt_templates_are_refused_from_every_source(source_type):
    gate = apply_ingest_gate(TEMPLATE, source_type=source_type)
    assert gate.rejected and gate.rejection_code == "prompt_template"
    assert prompt_template_kind(TEMPLATE, source_type=source_type) == "prompt_template"


def test_oversize_names_its_code():
    gate = apply_ingest_gate("x" * 20, max_ingest_bytes=10)
    assert gate.rejected and gate.rejection_code == "oversize"


# --- the real write path ----------------------------------------------------

def _writer(tmp_path):
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    db.execute("INSERT INTO profiles(profile_id, name, description) VALUES (?, ?, ?)",
               ("p1", "p1", "test profile"))
    return db, build_immediate_admission_handler(db, profile_id="p1")


def _ingestion(source_type, key):
    from superlocalmemory.core.ingestion_command import IngestionRequest

    return IngestionRequest(content=QUOTED, profile_id="p1", source_type=source_type,
                            idempotency_key=key, trusted_actor_id="local-capability:test")


def test_folder_part_with_a_quoted_notification_is_saved(tmp_path):
    db, writer = _writer(tmp_path)
    fact_ids = writer(_ingestion("folder", "src:a:1:61"), "op-61")
    assert fact_ids
    rows = db.execute("SELECT content FROM memories WHERE profile_id = ?", ("p1",))
    assert str(rows[0]["content"]) == QUOTED


def test_a_refusal_carries_its_reason(tmp_path):
    from superlocalmemory.core.ingestion_command import IngestionRejectedError

    _db, writer = _writer(tmp_path)
    with pytest.raises(IngestionRejectedError) as refused:
        writer(_ingestion("mcp", "k-mcp"), "op-mcp")
    assert refused.value.code == "ingest_gate:harness_wrapper"


# --- the journal: a refusal from an older policy is judged once more --------

@dataclass(frozen=True)
class _Codec:
    def encrypt(self, plaintext: bytes) -> bytes:
        return b"c:" + plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        return ciphertext[2:][::-1]


class _Coordinator:
    """Commits unless told to refuse; counts what reached it."""

    def __init__(self, refuse_with: str | None = None) -> None:
        self.refuse_with = refuse_with
        self.calls = 0

    def submit(self, command, *, wait_ms):
        self.calls += 1
        if self.refuse_with:
            raise TerminalAdmissionError(self.refuse_with)
        return {"state": "committed", "receipt": {
            "operation_id": "op-61", "fact_ids": ["fact-61"], "state": "queryable",
            "commit_sequence": 1}}


def _request():
    return RememberRequest(content=QUOTED, profile_id="default", source_type="folder",
                           idempotency_key="src:a:1:61")


def _actor():
    return Actor(principal_id="daemon-capability:test",
                 allowed_profiles=frozenset({"default"}),
                 allowed_scopes=frozenset({"personal"}))


def _refused_entry(tmp_path, code):
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_Codec())
    entry = journal.prepare(_request(), _actor())
    journal.mark_rejected(entry.journal_id, code)
    return journal, entry


def test_a_pre_fix_refusal_is_judged_again_and_saves_once(tmp_path):
    journal, entry = _refused_entry(tmp_path, UNEXPLAINED_REJECTION)
    coordinator = _Coordinator()
    service = RememberService(journal, coordinator)

    receipt = service.remember(_request(), _actor(), deadline_ms=1_500)
    assert receipt.payload["fact_ids"] == ["fact-61"]
    assert journal.get(entry.journal_id).state == "committed"
    # The same key again returns the stored receipt: nothing is saved twice.
    service.remember(_request(), _actor(), deadline_ms=1_500)
    assert coordinator.calls == 1


def test_a_refusal_with_a_recorded_reason_stays_final(tmp_path):
    journal, _entry = _refused_entry(tmp_path, "COMMAND_REJECTED:ingest_gate:prompt_template")
    coordinator = _Coordinator()
    with pytest.raises(AdmissionRejected) as refused:
        RememberService(journal, coordinator).remember(_request(), _actor(), deadline_ms=1_500)
    assert refused.value.error_code == "COMMAND_REJECTED:ingest_gate:prompt_template"
    assert coordinator.calls == 0


def test_a_reopened_entry_refused_again_is_not_reopened_a_third_time(tmp_path):
    journal, entry = _refused_entry(tmp_path, UNEXPLAINED_REJECTION)
    coordinator = _Coordinator(refuse_with="COMMAND_REJECTED:ingest_gate:prompt_template")
    service = RememberService(journal, coordinator)
    for _ in range(3):
        with pytest.raises(AdmissionRejected):
            service.remember(_request(), _actor(), deadline_ms=1_500)
    assert coordinator.calls == 1
    assert journal.get(entry.journal_id).error_code.endswith("prompt_template")


def test_reopen_touches_only_the_bare_legacy_code(tmp_path):
    journal, entry = _refused_entry(tmp_path, "UNKNOWN_PROFILE")
    assert journal.reopen_unexplained_rejection(entry.journal_id) is None
    assert journal.get(entry.journal_id).state == "rejected"


# --- the folder report ------------------------------------------------------

def test_folder_report_names_the_code_and_the_part():
    from superlocalmemory.core.remember_admission import AdmissionRejected as Refused
    from superlocalmemory.sources.ingest import PartialSave
    from superlocalmemory.sources.reconcile import failure_reason

    cause = Refused("COMMAND_REJECTED:ingest_gate:harness_wrapper")
    exc = PartialSave([], part=61, total=821)
    assert failure_reason(exc, cause) == (
        "AdmissionRejected:COMMAND_REJECTED:ingest_gate:harness_wrapper at part 61/821")
    assert failure_reason(ValueError("x"), ValueError("x")) == "ValueError"
