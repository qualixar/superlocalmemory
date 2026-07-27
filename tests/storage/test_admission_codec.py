from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from superlocalmemory.storage.admission_codec import (
    AdmissionKeyError,
    MachineKeyCommandCodec,
)
from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    AdmissionPayloadError,
    RememberRequest,
)


def test_machine_codec_round_trip_and_ciphertext_is_random(tmp_path: Path) -> None:
    codec = MachineKeyCommandCodec(tmp_path / "admission-key.bin")
    plaintext = b'{"content":"private memory"}'

    first = codec.encrypt(plaintext)
    second = codec.encrypt(plaintext)

    assert first != second
    assert plaintext not in first
    assert codec.decrypt(first) == plaintext
    assert codec.decrypt(second) == plaintext


def test_machine_codec_survives_process_restart(tmp_path: Path) -> None:
    key_path = tmp_path / "admission-key.bin"
    first = MachineKeyCommandCodec(key_path)
    ciphertext = first.encrypt(b"durable")

    second = MachineKeyCommandCodec(key_path)

    assert second.decrypt(ciphertext) == b"durable"
    if os.name != "nt":
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600


def test_machine_codec_rejects_tampering(tmp_path: Path) -> None:
    codec = MachineKeyCommandCodec(tmp_path / "admission-key.bin")
    ciphertext = bytearray(codec.encrypt(b"durable"))
    ciphertext[-1] ^= 1

    with pytest.raises(InvalidTag):
        codec.decrypt(bytes(ciphertext))


def test_machine_codec_rejects_invalid_or_non_regular_key(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid-key.bin"
    invalid.write_bytes(b"short")
    with pytest.raises(AdmissionKeyError, match="invalid length"):
        MachineKeyCommandCodec(invalid)

    directory = tmp_path / "directory-key"
    directory.mkdir()
    with pytest.raises(AdmissionKeyError, match="regular file"):
        MachineKeyCommandCodec(directory)


def test_journal_reports_wrong_machine_key_as_bounded_payload_error(tmp_path: Path) -> None:
    journal_path = tmp_path / "admission.db"
    first = AdmissionJournal(
        journal_path,
        codec=MachineKeyCommandCodec(tmp_path / "first-key.bin"),
    )
    actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
    entry = first.prepare(
        RememberRequest(
            content="encrypted recovery witness",
            profile_id="default",
            source_type="test",
            idempotency_key="wrong-key-witness",
            trusted_actor_id="actor",
        ),
        actor,
    )
    restarted_with_wrong_key = AdmissionJournal(
        journal_path,
        codec=MachineKeyCommandCodec(tmp_path / "second-key.bin"),
    )

    with pytest.raises(AdmissionPayloadError, match="cannot be decrypted"):
        restarted_with_wrong_key.request_for(entry)
