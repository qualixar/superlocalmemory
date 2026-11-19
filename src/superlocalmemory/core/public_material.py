# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Public material the secret screen must leave alone (4.2.30, F-08).

``redact_secrets`` runs on every save by default and its work cannot be undone, so it may
only replace what is secret. Three kinds of text look random and are not:

* an SSH **public** key line (``ssh-ed25519 AAAA... comment``);
* a PEM **public** block (``PUBLIC KEY``, ``CERTIFICATE``, ``CERTIFICATE REQUEST`` ...);
* a subresource-integrity hash (``sha256-`` / ``sha384-`` / ``sha512-`` and a base64 digest of the
  exact length), as in a lockfile or a ``<script integrity=...>`` tag.

:func:`shield` swaps each one for a placeholder the later passes cannot match, and
:func:`unshield` puts them back. Each kind is checked by shape and not by label alone, so a key
label in front of random text, or a certificate label around a token, is still screened: the SSH
blob must parse as the wire format and name the same key type, the PEM body may hold only base64
and whitespace, and the hash must be exactly as long as its algorithm's digest.
"""

from __future__ import annotations

import base64
import binascii
import re

#: Private-use characters that bracket a placeholder. Text that already holds them is not shielded.
_OPEN, _CLOSE = "", ""

_SSH_TYPES = ("ssh-rsa", "ssh-ed25519", "ssh-dss", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384",
              "ecdsa-sha2-nistp521", "sk-ssh-ed25519@openssh.com", "sk-ecdsa-sha2-nistp256@openssh.com")
_SSH = re.compile(
    r"(?<![A-Za-z0-9+/=@.-])(" + "|".join(re.escape(t) for t in _SSH_TYPES) + r")"
    r"[ \t]+([A-Za-z0-9+/]{20,4000}={0,3})(?![A-Za-z0-9+/=])")

_PEM_TYPES = ("PUBLIC KEY", "RSA PUBLIC KEY", "CERTIFICATE", "TRUSTED CERTIFICATE", "X509 CERTIFICATE",
              "CERTIFICATE REQUEST", "NEW CERTIFICATE REQUEST", "X509 CRL")
_PEM_BEGIN = re.compile(r"-----BEGIN (" + "|".join(_PEM_TYPES) + r")-----")
_PEM_BODY = re.compile(r"[A-Za-z0-9+/=\s]*")
#: A certificate or key block is a few KB; a longer span is not scanned (bounds the work).
_PEM_MAX = 12000

#: The digest, in base64, exactly as long as the algorithm's output.
_SRI = re.compile(
    r"(?<![A-Za-z0-9+/=-])sha(?:256-[A-Za-z0-9+/]{43}=|384-[A-Za-z0-9+/]{64}|512-[A-Za-z0-9+/]{86}==)"
    r"(?![A-Za-z0-9+/=])")


def _is_ssh_blob(kind: str, blob: str) -> bool:
    """The base64 is an SSH public-key blob: length-prefixed strings, the first one naming ``kind``."""
    try:
        data = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
    except (binascii.Error, ValueError):
        return False
    fields, at = [], 0
    while at < len(data):
        if at + 4 > len(data):
            return False
        size = int.from_bytes(data[at:at + 4], "big")
        at += 4
        if at + size > len(data):
            return False
        fields.append(data[at:at + size])
        at += size
    return len(fields) >= 2 and fields[0] == kind.encode("ascii")


def _spans(text: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for m in _SSH.finditer(text):
        if _is_ssh_blob(m.group(1), m.group(2)):
            found.append((m.start(), m.end()))
    for m in _PEM_BEGIN.finditer(text):
        end_marker = f"-----END {m.group(1)}-----"
        stop = text.find(end_marker, m.end(), m.end() + _PEM_MAX)
        if stop != -1 and _PEM_BODY.fullmatch(text, m.end(), stop):
            found.append((m.start(), stop + len(end_marker)))
    found.extend((m.start(), m.end()) for m in _SRI.finditer(text))
    return found


def shield(text: str) -> tuple[str, list[str]]:
    """``text`` with each public span replaced by a placeholder, and the spans to put back."""
    if _OPEN in text or _CLOSE in text:
        return text, []
    pieces: list[str] = []
    saved: list[str] = []
    at = 0
    for start, end in sorted(set(_spans(text))):
        if start < at:
            continue  # inside an earlier span
        pieces.append(text[at:start])
        pieces.append(f"{_OPEN}{len(saved)}{_CLOSE}")
        saved.append(text[start:end])
        at = end
    if not saved:
        return text, []
    pieces.append(text[at:])
    return "".join(pieces), saved


def unshield(text: str, saved: list[str]) -> str:
    """Put the shielded spans back; a placeholder a later pass consumed stays redacted."""
    if not saved:
        return text
    return re.sub(f"{_OPEN}(\\d+){_CLOSE}",
                  lambda m: saved[int(m.group(1))] if int(m.group(1)) < len(saved) else m.group(0), text)


__all__ = ["shield", "unshield"]
