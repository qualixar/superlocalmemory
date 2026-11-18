# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Never overwrite what is already remembered: find what the store already holds.

The normal save path, given text identical to a memory that is already stored (same scope, role
and speaker), reuses that memory and rewrites its session date and metadata, and an identical
fact inside one memory only bumps a counter. An import must never touch an existing memory, so
before anything is saved each item is compared with the text the profile already holds, and an
item that is already there is left out and counted ("already in your memory"). The comparison is
on the text as it would be stored (credentials stripped, personal-data redaction as configured),
case and spacing ignored; only short fixed-size digests are kept in memory, never the text.

Read-only: it opens no write transaction.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterable

from superlocalmemory.memory_core.save_path import (
    ContentOrigin,
    effective_pii_redaction,
    prepare_for_save,
)
from superlocalmemory.portability.model import Item

_SPACE = re.compile(r"\s+")
CHUNK = 2_000


def key_of(text: str) -> str:
    """A short digest of the text, ignoring case and spacing."""
    flat = _SPACE.sub(" ", text).strip().casefold()
    return hashlib.blake2b(flat.encode("utf-8"), digest_size=12).hexdigest()


def stored_form(item: Item, config: object | None) -> str:
    """The text of ``item`` as the save path would store it."""
    return prepare_for_save(item.text, origin=ContentOrigin.DERIVED_TEXT,
                            pii_redaction=effective_pii_redaction(config)).text


def present_keys(conn: sqlite3.Connection, profile_id: str, wanted: Iterable[str]) -> set[str]:
    """Which of the ``wanted`` keys name a memory that profile already holds.

    Streams the profile's stored memories once and keeps only the digests that are asked for.
    A fact cut from a longer memory is not compared: facts only ever merge inside their own
    memory, so a sentence an import shares with an older memory overwrites nothing.
    """
    want = set(wanted)
    found: set[str] = set()
    if not want:
        return found
    try:
        cursor = conn.execute("SELECT content FROM memories WHERE profile_id = ?", (profile_id,))
    except sqlite3.Error:
        return found
    while found != want:
        rows = cursor.fetchmany(CHUNK)
        if not rows:
            break
        for (content,) in rows:
            if isinstance(content, str):
                k = key_of(content)
                if k in want:
                    found.add(k)
    return found


__all__ = ["key_of", "present_keys", "stored_form"]
