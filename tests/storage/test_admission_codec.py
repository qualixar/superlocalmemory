from __future__ import annotations

import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
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


def test_machine_codec_writes_random_newlines_without_text_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows must open the binary key with O_BINARY, not text translation."""
    key_path = tmp_path / "admission-key.bin"
    monkeypatch.setattr(
        "superlocalmemory.storage.admission_codec.os.urandom",
        lambda size: b"\n" * size,
    )

    MachineKeyCommandCodec(key_path)

    assert key_path.read_bytes() == b"\n" * 32


def test_concurrent_codec_start_waits_for_complete_exclusive_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing O_EXCL readers must not reject the winning process's partial write."""
    key_path = tmp_path / "admission-key.bin"
    real_write = os.write

    def slow_write(fd: int, data) -> int:
        time.sleep(0.003)
        return real_write(fd, bytes(data[:1]))

    monkeypatch.setattr(
        "superlocalmemory.storage.admission_codec.os.write",
        slow_write,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        codecs = list(pool.map(lambda _: MachineKeyCommandCodec(key_path), range(8)))

    ciphertext = codecs[0].encrypt(b"concurrent-start")
    assert all(codec.decrypt(ciphertext) == b"concurrent-start" for codec in codecs)
    assert len(key_path.read_bytes()) == 32


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
