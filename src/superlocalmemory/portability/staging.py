# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Keep an uploaded file between "preview" and "Import", then delete it.

A preview reads a file and writes nothing to memory. Pressing Import should not need the file
sent again, so the bytes wait in a private folder inside SuperLocalMemory's data folder, owner
only, under a random id. They are deleted when the run starts reading them has finished, when
the person starts over, and after ``STAGING_TTL_S`` whatever happens. The id is 32 hex
characters; nothing else is ever turned into a path.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path

from superlocalmemory.portability import constants as C
from superlocalmemory.portability.readers import Doc, IntakeRefused

_ID = re.compile(r"[0-9a-f]{32}")
MAX_STAGED = 8


def _root(base: Path | None = None) -> Path:
    from superlocalmemory.infra.data_root import canonical_data_root

    return Path(base if base is not None else canonical_data_root()) / "import-staging"


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def sweep(now: float | None = None) -> int:
    """Delete staged uploads older than the TTL, and the oldest beyond ``MAX_STAGED - 1``."""
    root = _root()
    if not root.is_dir():
        return 0
    stamp = time.time() if now is None else now
    entries = sorted((p for p in root.iterdir() if p.is_dir() and _ID.fullmatch(p.name)),
                     key=lambda p: p.stat().st_mtime)
    stale = [p for p in entries if stamp - p.stat().st_mtime > C.STAGING_TTL_S]
    live = [p for p in entries if p not in stale]
    stale += live[:max(0, len(live) - (MAX_STAGED - 1))]  # room for the one about to be put
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
    return len(stale)


def put(docs: list[Doc]) -> str:
    """Stage the docs; returns the upload id."""
    sweep()
    upload_id = secrets.token_hex(16)
    folder = _root() / upload_id
    _private_dir(_root())
    _private_dir(folder)
    names = []
    for n, doc in enumerate(docs):
        fd = os.open(folder / f"{n:05d}.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(doc.data)
        names.append(doc.name)
    (folder / "names.json").write_text(json.dumps(names), encoding="utf-8")
    return upload_id


def get(upload_id: str) -> list[Doc]:
    """The staged docs; ``IntakeRefused`` in plain words when they are gone."""
    gone = IntakeRefused("That upload is no longer here. Choose the file again.")
    if not _ID.fullmatch(str(upload_id or "")):
        raise gone
    folder = _root() / upload_id
    try:
        names = json.loads((folder / "names.json").read_text(encoding="utf-8"))
        return [Doc(name, (folder / f"{n:05d}.bin").read_bytes()) for n, name in enumerate(names)]
    except (OSError, ValueError):
        raise gone from None


def drop_all(base: Path | None = None) -> int:
    """Delete every staged upload (a profile was erased; they name no profile). Returns how many."""
    root = _root(base)
    if not root.is_dir():
        return 0
    found = [p for p in root.iterdir() if p.is_dir()]
    for path in found:
        shutil.rmtree(path, ignore_errors=True)
    return len(found)


def drop(upload_id: str) -> None:
    if _ID.fullmatch(str(upload_id or "")):
        shutil.rmtree(_root() / upload_id, ignore_errors=True)


__all__ = ["MAX_STAGED", "drop", "drop_all", "get", "put", "sweep"]
