# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-08 (19 Nov 2026): the secret screen must not redact public material.

The screen runs by default on every save and cannot be undone, so it may only replace what is
secret. A host's SSH public key, a public certificate, a PEM public key and a lockfile's integrity
hash are public; storing a placeholder for them destroys what the person asked to keep. Private
keys and tokens keep being screened, and a string that only dresses as public material (a key
label in front of random text) is still screened.
"""

from __future__ import annotations

import base64
import hashlib
import struct

import pytest

from superlocalmemory.core import security_primitives as sp


def _wire(*parts: bytes) -> bytes:
    return b"".join(struct.pack(">I", len(p)) + p for p in parts)


def _rand(seed: str, n: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}{counter}".encode()).digest()
        counter += 1
    return out[:n]


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


_EXPONENT, _ZERO, _POINT = b"\x01\x00\x01", b"\x00", b"\x04"
ED25519 = "ssh-ed25519 " + _b64(_wire(b"ssh-ed25519", _rand("ed", 32)))
RSA = "ssh-rsa " + _b64(_wire(b"ssh-rsa", _EXPONENT, _ZERO + _rand("rsa", 256)))
ECDSA = "ecdsa-sha2-nistp256 " + _b64(_wire(b"ecdsa-sha2-nistp256", b"nistp256", _POINT + _rand("ec", 64)))
SRI = {
    "sha256": "sha256-" + _b64(hashlib.sha256(b"a").digest()),
    "sha384": "sha384-" + _b64(hashlib.sha384(b"a").digest()),
    "sha512": "sha512-" + _b64(hashlib.sha512(b"a").digest()),
}


def _pem(label: str, size: int = 300) -> str:
    body = _b64(_rand(label, size))
    lines = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    return f"-----BEGIN {label}-----\n{lines}\n-----END {label}-----"


@pytest.mark.parametrize("key", [ED25519, RSA, ECDSA], ids=["ed25519", "rsa", "ecdsa"])
@pytest.mark.parametrize("aggression", ["normal", "high"])
def test_an_ssh_public_key_is_kept(key, aggression):
    text = f"The bastion host key is {key} varun@mac and it is in known_hosts."
    assert sp.redact_secrets(text, aggression=aggression) == text


@pytest.mark.parametrize("label", ["PUBLIC KEY", "CERTIFICATE", "CERTIFICATE REQUEST", "RSA PUBLIC KEY"])
@pytest.mark.parametrize("aggression", ["normal", "high"])
def test_a_pem_public_block_is_kept_whole(label, aggression):
    block = _pem(label)
    text = f"Pin this one:\n{block}\nthen restart."
    assert sp.redact_secrets(text, aggression=aggression) == text


@pytest.mark.parametrize("algo", ["sha256", "sha384", "sha512"])
def test_a_subresource_integrity_hash_is_kept(algo):
    text = f'<script src="/app.js" integrity="{SRI[algo]}" crossorigin="anonymous"></script>'
    assert sp.redact_secrets(text) == text
    lock = f'"node_modules/left-pad": {{"version": "1.3.0", "integrity": "{SRI[algo]}"}}'
    assert sp.redact_secrets(lock) == lock


# -- what is still screened -------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY",
                                   "OPENSSH PRIVATE KEY", "ENCRYPTED PRIVATE KEY",
                                   "PGP PRIVATE KEY BLOCK"])
def test_a_private_key_is_still_screened(label):
    block = _pem(label)
    out = sp.redact_secrets(f"mine:\n{block}\n")
    assert "[REDACTED:PRIVATE_KEY:" in out
    assert _b64(_rand(label, 300))[:40] not in out


def test_the_marker_for_a_private_key_still_comes_from_the_begin_line_alone():
    assert sp.redact_secrets("-----BEGIN RSA PRIVATE KEY-----") .startswith("[REDACTED:PRIVATE_KEY:")


def test_a_key_label_in_front_of_random_text_is_not_a_free_pass():
    smuggled = "ssh-ed25519 " + _b64(_rand("smuggle", 48))  # not an SSH wire blob
    assert "[REDACTED:" in sp.redact_secrets(f"note: {smuggled}")
    wrong_type = "ssh-rsa " + _b64(_wire(b"ssh-ed25519", _rand("ed", 32)))  # blob says another type
    assert "[REDACTED:" in sp.redact_secrets(wrong_type)
    trailing = ED25519 + _b64(_rand("tail", 48))  # extra random text glued to a real key
    assert sp.redact_secrets(trailing) != trailing


def test_a_certificate_label_around_a_secret_is_not_a_free_pass():
    token = "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    text = f"-----BEGIN CERTIFICATE-----\n{token}\n-----END CERTIFICATE-----"
    assert token not in sp.redact_secrets(text)
    unterminated = f"-----BEGIN CERTIFICATE-----\n{_b64(_rand('x', 200))}\n"
    assert "[REDACTED:ENTROPY" in sp.redact_secrets(unterminated)  # no END: not a certificate block


def test_a_hash_shaped_string_of_the_wrong_size_is_not_an_integrity_hash():
    for algo, size in (("sha512", 31), ("sha512", 33), ("sha384", 45), ("sha384", 51), ("sha256", 33), ("sha256", 35)):
        odd = f"{algo}-" + _b64(_rand(f"odd{size}", size))  # not the digest length of its algorithm
        assert "[REDACTED:" in sp.redact_secrets(f"x {odd} y"), (algo, size)


def test_a_real_token_beside_public_material_is_screened_and_the_public_part_kept():
    token = "ghp_" + "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4zAb3d"
    text = f"host key {ED25519}\nintegrity {SRI['sha512']}\ntoken {token}\n{_pem('CERTIFICATE')}\n"
    out = sp.redact_secrets(text)
    assert ED25519 in out and SRI["sha512"] in out and _pem("CERTIFICATE") in out
    assert token not in out and "[REDACTED:GITHUB:" in out


def test_text_that_already_holds_the_shield_characters_is_still_screened():
    token = "ghp_" + "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4zAb3d"
    own = "\ue000" + "0" + "\ue001"
    out = sp.redact_secrets(f"{own} {ED25519} {token}")
    assert token not in out
    assert own in out  # the text's own characters are not mistaken for a placeholder


def test_detect_secrets_does_not_flag_a_public_certificate():
    assert sp.detect_secrets(_pem("CERTIFICATE")) == []
    assert any(h.kind == "PRIVATE_KEY" for h in sp.detect_secrets("-----BEGIN PRIVATE KEY-----"))


def test_a_crafted_pile_of_begin_lines_does_not_stall_the_screen():
    import time

    text = "-----BEGIN CERTIFICATE-----\n" * 9000
    started = time.monotonic()
    sp.redact_secrets(text)
    assert time.monotonic() - started < 5
