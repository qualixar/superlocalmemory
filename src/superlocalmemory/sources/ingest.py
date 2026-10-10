# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Turn one folder file into memories through the normal write paths. Never writes to the folder."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from superlocalmemory.core.ingestion_command import is_terminal_failure
from superlocalmemory.core.security_primitives import SecretHit, detect_secrets
from superlocalmemory.documents.chunking import chunk_text
from superlocalmemory.memory_core import ContentOrigin
from superlocalmemory.memory_core import submit as _submit
from superlocalmemory.memory_core.submit import SavePending, SaveRequest, submit_memory_settled
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


class PartialSave(RuntimeError):
    """A file failed part-way: ``entries`` are the parts already saved, kept so they stay owned."""

    def __init__(self, entries: list[dict[str, Any]]):
        super().__init__("a folder file was saved only in part")
        self.entries = entries


class SaveBudgetSpent(PartialSave):
    """The file's settle budget ran out while a part was still queued.

    The parts sent so far (the queued one by its key) stay owned by the row; the rest of the file
    is sent on the next pass under the same keys. A deferral, not a failure."""


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
    """Per source, path and save number ``n``. A new save of a path has a new ``n``; a retry of the
    same file version after a failure keeps its ``n``, so the writer recognises the parts it already has."""
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
               extra: dict[str, Any] | None = None, settle_until: float | None = None) -> Ingested:
    """One memory per part; ``extra`` is server-prepared metadata added beside the provenance.

    Waiting for queued saves to commit has one budget per file (``settle_until``, a
    ``time.monotonic`` deadline; by default the settle wait from now). When it runs out on a part
    that is still queued, the rest of the file is deferred (``SaveBudgetSpent``).
    """
    out = Ingested()
    until = settle_until if settle_until is not None else time.monotonic() + _submit.SETTLE_WAIT_S
    for number, part in enumerate(parts, 1):
        key = _key(source["source_id"], relpath, n, number)
        request = SaveRequest(
            segments=((part, ContentOrigin.DERIVED_TEXT),), profile_id=source["profile_id"],
            source_type="folder", trusted_actor_id=host.actor_id(), tags=tags, session_date=session_date,
            trusted_metadata={**(extra or {}), "_slm_source": provenance(source["source_id"], relpath, version)},
            idempotency_key=key)
        try:
            saved = submit_memory_settled(runtime, request, config=host.config(),
                                          wait_s=max(0.0, until - time.monotonic()))
        except SavePending:
            # Durable and queued: owned by its key until it commits (see ``facts_of_keys``).
            out.entries.append({"m": None, "f": [], "v": version, "k": key})
            if number < len(parts):
                raise SaveBudgetSpent(out.entries) from None
            continue
        except Exception as exc:
            raise PartialSave(out.entries) from exc
        # ``k`` is kept beside the ids: a retry of this version re-sends the same key, and the
        # entry it gets back is recognised as the same save (see ``reconcile._supersede``).
        out.entries.append({"m": saved.memory_id, "f": list(saved.fact_ids), "v": version, "k": key})
    return out


@dataclass(frozen=True)
class KeyFacts:
    """What queued saves (known only by key) resolved to: ``facts`` of the committed ones, and the
    ``pending`` keys that have not committed (or could not be looked up) and so own nothing findable yet."""

    facts: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)


def resolve_keys(runtime: Any, profile_id: str, keys: list[str]) -> KeyFacts:
    """Resolve folder saves recorded only by their key.

    A key is *settled* when it can no longer gain facts: its operation holds fact ids, finished,
    or failed for good; or, with no operation row, the admission journal says the request was
    rejected, committed (the receipt names the facts) or never admitted. Only a key that may still
    commit is *pending*: an operation that is raw, queued, enriching or retryable, or a journal
    entry still prepared or dispatched, or a journal that cannot be read. Callers treat hiding or
    erasing a pending key as not done, never as "nothing to hide".

    Text parts are written as ``folder`` saves and pictures as ``media`` saves; folder keys
    (``src:...``) are unique to the folder either way.
    """
    if not keys:
        return KeyFacts()
    db = getattr(runtime, "_db", None)
    if db is None:
        return KeyFacts(pending=list(keys))
    found: list[str] = []
    settled: set[str] = set()
    live: set[str] = set()  # an operation exists and may still gain facts: the journal has no say
    try:
        for i in range(0, len(keys), 400):
            part = keys[i:i + 400]
            rows = db.execute(
                "SELECT idempotency_key, state, attempt_count, next_retry_at, queryable_fact_ids_json,"
                " final_fact_ids_json FROM ingestion_operations WHERE profile_id = ?"
                " AND source_type IN ('folder', 'media') AND idempotency_key IN ("
                + ",".join("?" * len(part)) + ")", (profile_id, *part))
            for row in rows:
                ids = [str(f) for column in (row[4], row[5]) for f in json.loads(column or "[]")]
                done = bool(ids) or row[1] == "complete" or is_terminal_failure(row[1], row[2], row[3])
                (settled if done else live).add(str(row[0]))
                found += ids
    except Exception as exc:  # noqa: BLE001 - what cannot be looked up is not known to be hidden
        logger.warning("could not look up queued folder saves (%s)", type(exc).__name__)
        return KeyFacts(facts=found, pending=list(keys))
    pending: list[str] = []
    for key in keys:
        if key in settled:
            continue
        if key in live:
            pending.append(key)
            continue
        verdict = _journal_verdict(runtime, profile_id, key)
        if verdict is None:
            pending.append(key)
        else:
            found += verdict
    return KeyFacts(facts=found, pending=pending)


def _journal_verdict(runtime: Any, profile_id: str, key: str) -> list[str] | None:
    """For a key with no operation row: the fact ids it saved (possibly none) once the admission
    journal says it can no longer commit, or None while it may still commit or is unknown."""
    journal = getattr(runtime, "journal", None)
    if journal is None:
        return None
    try:
        entry = journal.get_by_idempotency_key(profile_id, key)
    except Exception as exc:  # noqa: BLE001 - unknown is not settled
        logger.warning("could not look up the admission journal (%s)", type(exc).__name__)
        return None
    if entry is None:  # never durably admitted: nothing was saved
        return []
    if entry.state == "rejected":
        return []
    if entry.state == "committed":
        receipt = entry.original_receipt or {}
        return [str(f) for f in receipt.get("fact_ids") or []]
    return None  # prepared / dispatched: the request is still queued


def facts_of_keys(runtime: Any, profile_id: str, keys: list[str]) -> list[str]:
    """Fact ids of the committed folder saves among ``keys`` (see ``resolve_keys`` for the rest)."""
    return resolve_keys(runtime, profile_id, keys).facts


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


#: Skip reason for a PDF or picture skipped only because images & documents were off or not set up.
#: Unlike other skips it is temporary: the file is read again once the feature is ready.
MEDIA_NOT_READY = "media_not_ready"


def media_ready() -> bool:
    """Whether images & documents are on and their set-up has finished."""
    try:
        from superlocalmemory.runtimes.features import media_enabled
        from superlocalmemory.runtimes.media_env import media_env

        return bool(media_enabled()) and media_env().status().state == "ready"
    except Exception as exc:  # noqa: BLE001 - unknown counts as not ready
        logger.warning("could not read the images & documents state (%s)", type(exc).__name__)
        return False


def ingest_pdf(host: SourceHost, source: dict, relpath: str, data: bytes, version: str, n: int) -> Ingested:
    from superlocalmemory.documents import submit_document
    from superlocalmemory.media.ingest import MediaInput

    if not media_ready():
        return Ingested(skip_reason=MEDIA_NOT_READY)
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

    if not media_ready():
        return Ingested(skip_reason=MEDIA_NOT_READY)
    key = _key(source["source_id"], relpath, n, 0)
    receipt = remember_media(
        MediaInput(data=data, file_name=Path(relpath).name), profile_id=source["profile_id"],
        actor_id=host.actor_id(), runtime=runtime, config=host.config(),
        idempotency_key=key,
        folder=folder_tag(source["source_id"], relpath, version))
    if receipt.status == "warming":
        return Ingested(retry=True)
    if receipt.status == "refused":
        return Ingested(skip_reason=receipt.reason[:120] or "refused")
    if receipt.status == "duplicate":  # someone else's picture: borrowed, never owned
        entries = [{"shared_m": receipt.memory_id}] if receipt.memory_id else []
        return Ingested(entries=entries, shared=True)
    # A picture whose memory was still queued is owned by its key until it commits.
    entry = ({"m": receipt.memory_id, "f": [], "v": version} if receipt.memory_id
             else {"m": None, "f": [], "v": version, "k": key})
    return Ingested(entries=[entry], media_id=receipt.media_id)


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
        # ``execute`` returns a list of rows (storage/database.py), never a cursor.
        rows = db.execute("SELECT 1 FROM atomic_facts WHERE lifecycle = 'archived' AND memory_id IN ("
                          + ",".join("?" * len(memory_ids)) + ") LIMIT 1", tuple(memory_ids))
    except Exception as exc:  # noqa: BLE001 - not knowing is treated as "not archived"
        logger.warning("could not look up folder memories (%s)", type(exc).__name__)
        return False
    return len(rows) > 0


__all__ = ["Ingested", "KeyFacts", "SCREEN_BYTES", "SaveBudgetSpent", "any_archived", "facts_of", "ingest_image", "ingest_pdf", "ingest_text",
           "load_verified", "provenance", "resolve_keys", "save_parts", "screen", "split_markdown", "split_text"]
