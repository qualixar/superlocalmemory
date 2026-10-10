# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Turn one folder file into memories through the normal write paths. Never writes to the folder."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from superlocalmemory.core.security_primitives import SecretHit, detect_secrets
from superlocalmemory.documents.chunking import chunk_text
from superlocalmemory.memory_core import ContentOrigin
from superlocalmemory.memory_core.submit import SaveRequest, submit_memory
from superlocalmemory.sources.host import SourceHost
from superlocalmemory.sources.ignore import size_cap
from superlocalmemory.sources.safe_read import open_regular

logger = logging.getLogger(__name__)

SCREEN_BYTES = 256 * 1024
SECTION_LIMIT = 24_000
_HEADING = re.compile(r"^#{1,2} \S")
_FENCE = re.compile(r"^\s*(```|~~~)")
_MARKDOWN = (".md", ".markdown")


@dataclass
class Ingested:
    """What a file became: memory entries (``m`` memory id, ``f`` fact ids) and any owner ids."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    document_id: str | None = None
    media_id: str | None = None
    shared: bool = False
    skip_reason: str = ""
    retry: bool = False
    links: list | None = None  # Obsidian notes only: the links to record (None leaves them alone)


def screen(data: bytes) -> list[SecretHit]:
    """Credential shapes in the first 256 KB of a text file."""
    return detect_secrets(data[:SCREEN_BYTES].decode("utf-8", errors="replace"))


def split_markdown(text: str, limit: int = SECTION_LIMIT) -> list[str]:
    """The note as one part, or (when longer than ``limit``) one part per H1/H2 section."""
    if len(text) <= limit:
        return [text] if text.strip() else []
    sections: list[list[str]] = [[]]
    fenced = False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        if not fenced and _HEADING.match(line) and any(l.strip() for l in sections[-1]):
            sections.append([])
        sections[-1].append(line)
    parts: list[str] = []
    for lines in sections:
        parts += chunk_text("".join(lines), limit)
    return parts


def split_text(relpath: str, text: str) -> list[str]:
    if relpath.lower().endswith(_MARKDOWN):
        return split_markdown(text)
    return chunk_text(text, SECTION_LIMIT)


def provenance(source_id: str, relpath: str, version: str) -> dict[str, str]:
    return {"type": "folder", "source_id": source_id, "relpath": relpath, "version": version}


def _key(source_id: str, relpath: str, n: int, part: int) -> str:
    """Per source, path and save number ``n``: every save of a path is its own save, never a repeat."""
    path = hashlib.sha256(relpath.encode()).hexdigest()[:12]
    return f"src:{source_id[:12]}:{path}:{n}:{part}"


def ingest_text(host: SourceHost, runtime: Any, source: dict, relpath: str, data: bytes,
                version: str, n: int) -> Ingested:
    """Save a text or markdown file as memories (credentials stripped as derived text)."""
    parts = split_text(relpath, data.decode("utf-8", errors="replace"))
    if not parts:
        return Ingested(skip_reason="empty")
    return save_parts(host, runtime, source, relpath, parts, version, n)


def save_parts(host: SourceHost, runtime: Any, source: dict, relpath: str, parts: list[str], version: str,
               n: int, *, tags: str = "", session_date: str = "",
               extra: dict[str, Any] | None = None) -> Ingested:
    """One memory per part; ``extra`` is server-prepared metadata added beside the provenance."""
    out = Ingested()
    for number, part in enumerate(parts, 1):
        request = SaveRequest(
            segments=((part, ContentOrigin.DERIVED_TEXT),), profile_id=source["profile_id"],
            source_type="folder", trusted_actor_id=host.actor_id(), tags=tags, session_date=session_date,
            trusted_metadata={**(extra or {}), "_slm_source": provenance(source["source_id"], relpath, version)},
            idempotency_key=_key(source["source_id"], relpath, n, number))
        saved = submit_memory(runtime, request, config=host.config())
        out.entries.append({"m": saved.memory_id, "f": list(saved.fact_ids), "v": version})
    return out


def folder_tag(source_id: str, relpath: str, version: str) -> dict[str, str]:
    """Added beside a picture's or page's own ``type``, which stays ``media`` or ``document``."""
    return {"origin": "folder", "source_id": source_id, "relpath": relpath, "version": version}


def load_verified(path: Path, file_id: str | None, sha: str, kind: str) -> bytes | None:
    """The file's bytes, read without following a link; None when they are no longer the bytes that were hashed.

    The read stops one byte past the kind's size cap, so a file that grew cannot fill memory.
    OSError is raised for a link, a pipe or another file.
    """
    cap = size_cap(kind)
    with open_regular(path, file_id) as fh:
        data = fh.read(cap + 1)
    if len(data) > cap or hashlib.sha256(data).hexdigest() != sha:
        return None
    return data


def ingest_pdf(host: SourceHost, source: dict, relpath: str, data: bytes, version: str, n: int) -> Ingested:
    from superlocalmemory.documents import submit_document
    from superlocalmemory.media.ingest import MediaInput

    receipt = submit_document(
        MediaInput(data=data, file_name=Path(relpath).name), profile_id=source["profile_id"],
        actor_id=host.actor_id(), config=host.config(),
        idempotency_key=_key(source["source_id"], relpath, n, 0),
        folder=folder_tag(source["source_id"], relpath, version))
    if receipt.status == "refused":
        return Ingested(skip_reason=receipt.reason[:120] or "refused")
    if receipt.status == "duplicate":  # someone else's document: borrowed, never owned
        entries = [{"shared_doc": receipt.document_id}] if receipt.document_id else []
        return Ingested(entries=entries, shared=True)
    return Ingested(document_id=receipt.document_id)


def ingest_image(host: SourceHost, runtime: Any, source: dict, relpath: str, data: bytes,
                 version: str, n: int) -> Ingested:
    from superlocalmemory.media.ingest import MediaInput, remember_media

    receipt = remember_media(
        MediaInput(data=data, file_name=Path(relpath).name), profile_id=source["profile_id"],
        actor_id=host.actor_id(), runtime=runtime, config=host.config(),
        idempotency_key=_key(source["source_id"], relpath, n, 0),
        folder=folder_tag(source["source_id"], relpath, version))
    if receipt.status == "warming":
        return Ingested(retry=True)
    if receipt.status == "refused":
        return Ingested(skip_reason=receipt.reason[:120] or "refused")
    if receipt.status == "duplicate":  # someone else's picture: borrowed, never owned
        entries = [{"shared_m": receipt.memory_id}] if receipt.memory_id else []
        return Ingested(entries=entries, shared=True)
    entry = {"m": receipt.memory_id, "f": [], "v": version} if receipt.memory_id else None
    return Ingested(entries=[entry] if entry else [], media_id=receipt.media_id)


def facts_of(runtime: Any, memory_ids: list[str]) -> list[str]:
    """Fact ids of memories whose receipt did not list them (pictures)."""
    db = getattr(runtime, "_db", None)
    if db is None or not memory_ids:
        return []
    found: list[str] = []
    try:
        for i in range(0, len(memory_ids), 400):
            part = memory_ids[i:i + 400]
            rows = db.execute("SELECT fact_id FROM atomic_facts WHERE memory_id IN ("
                              + ",".join("?" * len(part)) + ")", tuple(part))
            found += [r["fact_id"] for r in rows]
    except Exception as exc:  # noqa: BLE001 - what cannot be found cannot be hidden here
        logger.warning("could not look up facts of folder memories (%s)", type(exc).__name__)
    return found


def any_archived(runtime: Any, memory_ids: list[str]) -> bool:
    """True when a returned save points at memories whose facts are archived (hidden from recall)."""
    db = getattr(runtime, "_db", None)
    if db is None or not memory_ids:
        return False
    try:
        row = db.execute("SELECT 1 FROM atomic_facts WHERE lifecycle = 'archived' AND memory_id IN ("
                         + ",".join("?" * len(memory_ids)) + ") LIMIT 1", tuple(memory_ids)).fetchone()
    except Exception as exc:  # noqa: BLE001 - not knowing is treated as "not archived"
        logger.warning("could not look up folder memories (%s)", type(exc).__name__)
        return False
    return row is not None


__all__ = ["Ingested", "SCREEN_BYTES", "any_archived", "facts_of", "ingest_image", "ingest_pdf", "ingest_text",
           "load_verified", "provenance", "save_parts", "screen", "split_markdown", "split_text"]
