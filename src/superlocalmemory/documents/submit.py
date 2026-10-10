# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Hand a PDF over: check it, keep the original, and queue the page-by-page job.

Nothing is read page by page here; that is the job's work. The person's own words
are stored already prepared and only ever go into the document's own memory.
Nothing here logs content, text or paths.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from superlocalmemory.media import files
from superlocalmemory.media.ingest import MediaInput
from superlocalmemory.memory_core import prepare_user_text

logger = logging.getLogger(__name__)

MAX_BASE64_BYTES = 25 * 1024 * 1024
QUOTA_BYTES = 2 * 1024 ** 3
DEFAULT_MAX_MB = 100.0
_OFF = "Images and documents are turned off. Turn them on in settings to save documents."
_HOLD_STATES = ("processing", "ready")


@dataclass(frozen=True)
class DocumentReceipt:
    status: Literal["processing", "duplicate", "refused"]
    document_id: str | None = None
    job_id: str | None = None
    reason: str = ""


class _Stop(Exception):
    def __init__(self, receipt: DocumentReceipt) -> None:
        super().__init__(receipt.status)
        self.receipt = receipt


def _refuse(reason: str) -> _Stop:
    return _Stop(DocumentReceipt("refused", reason=reason))


def _refuse_own_data(path: Path) -> None:
    from superlocalmemory.infra.data_root import DATA_ROOT_REFUSAL, overlaps_data_root

    if overlaps_data_root(path):
        raise _refuse(DATA_ROOT_REFUSAL)


def _max_bytes() -> int:
    try:
        return int(float(os.environ.get("SLM_DOC_MAX_MB", DEFAULT_MAX_MB)) * 1024 * 1024)
    except ValueError:
        return int(DEFAULT_MAX_MB * 1024 * 1024)


def _resolve(store: Any) -> tuple[Any, bool]:
    if store is not None:
        return store, False
    from superlocalmemory.media import open_media_store
    from superlocalmemory.runtimes.features import media_enabled

    if not media_enabled():
        raise _refuse(_OFF)
    from superlocalmemory.runtimes.media_env import media_env

    state = media_env().status().state
    if state != "ready":
        from superlocalmemory.media.readiness import setup_message

        raise _refuse(setup_message(state))
    opened = open_media_store()
    if opened is None:
        raise _refuse(_OFF)
    return opened, True


# -- staging the bytes ---------------------------------------------------------

def _check_head(head: bytes) -> None:
    if not head:
        raise _refuse("That document is empty.")
    if head.startswith(b"%PDF-"):
        return
    if head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF")):
        raise _refuse("That is an image. Use the image option to save pictures.")
    raise _refuse("That file type is not supported (PDF only).")


def _new_tmp(tmp_dir: Path) -> tuple[Path, Any]:
    path = tmp_dir / ("doc-" + secrets.token_hex(12))
    return path, os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _stage_base64(inp: MediaInput, out: Any) -> tuple[str, int]:
    if len(inp.base64 or "") * 3 // 4 > MAX_BASE64_BYTES + 3:
        raise _refuse("That document is too large for pasted data. Give the file path instead.")
    try:
        data = base64.b64decode(inp.base64 or "", validate=True)
    except (binascii.Error, ValueError):
        raise _refuse("That document data could not be read.") from None
    if len(data) > min(MAX_BASE64_BYTES, _max_bytes()):
        raise _refuse("That document is too large.")
    _check_head(data[:8])
    out.write(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _stage_path(inp: MediaInput, out: Any) -> tuple[str, int]:
    limit, digest, size = _max_bytes(), hashlib.sha256(), 0
    try:
        path = Path(inp.path)
        _refuse_own_data(path)
        if not path.is_file():
            raise _refuse("That document could not be found.")
        with open(path, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                raise _refuse("That document could not be found.")
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                if size == 0:
                    _check_head(chunk[:8])
                size += len(chunk)
                if size > limit:
                    raise _refuse("That document is too large.")
                digest.update(chunk)
                out.write(chunk)
    except OSError:
        raise _refuse("That document could not be read.") from None
    if size == 0:
        raise _refuse("That document is empty.")
    return digest.hexdigest(), size


def _stage_data(inp: MediaInput, out: Any) -> tuple[str, int]:
    data = inp.data or b""
    if len(data) > _max_bytes():
        raise _refuse("That document is too large.")
    _check_head(data[:8])
    out.write(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _stage(inp: MediaInput, root: Path) -> tuple[Path, str, int]:
    if inp.data is None and (inp.base64 is None) == (inp.path is None):
        raise _refuse("Give a document file or document data.")
    tmp, out = _new_tmp(files.tmp_dir(root))
    try:
        with out:
            if inp.data is not None:
                sha, size = _stage_data(inp, out)
            else:
                sha, size = _stage_base64(inp, out) if inp.base64 is not None else _stage_path(inp, out)
        return tmp, sha, size
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# -- the submit ------------------------------------------------------------------

def _title(inp: MediaInput) -> str:
    name = inp.file_name or (Path(inp.path).name if inp.path else "")
    return Path(name).stem.strip()[:200]


def _document_id(profile_id: str, key: str) -> str:
    if key:
        return hashlib.sha256(f"{profile_id}\0{key}".encode()).hexdigest()[:32]
    return secrets.token_hex(16)


def _existing(store: Any, profile_id: str, doc_id: str, sha: str, keyed: bool,
              folder: bool = False) -> dict | None:
    """A document that makes this submit a repeat, or None; raises when a key names other content."""
    if keyed:
        row = store.get_document(doc_id)
        if row and row["profile_id"] == profile_id and row["state"] != "tombstoned":
            if row["sha256"] != sha:
                raise _refuse("That key was already used for a different document.")
            return row
    return store.find_document_by_sha(profile_id, sha, exclude_origin=None if folder else "folder")


def _repeat(store: Any, row: dict) -> DocumentReceipt | None:
    """A receipt when the repeat needs no new work (still running or done)."""
    if row["state"] not in _HOLD_STATES:
        return None
    job = store.job_for_document(row["document_id"])
    return DocumentReceipt("duplicate", document_id=row["document_id"], job_id=job["job_id"] if job else None)


def _queue(store: Any, doc_id: str, profile_id: str, payload: dict[str, Any]) -> str:
    return store.enqueue_job(profile_id, "document", 0, {**payload, "document_id": doc_id})


def _place(root: Path, tmp: Path, profile_id: str) -> tuple[str, bool]:
    try:
        new = not files.planned_path(root, profile_id, tmp, "pdf").exists()
        return files.place_original(root, tmp, profile_id, "pdf"), new
    except (OSError, ValueError):
        raise _refuse("The document could not be saved.") from None


def _create(store: Any, root: Path, tmp: Path, sha: str, size: int, doc_id: str, profile_id: str,
            title: str, payload: dict[str, Any]) -> DocumentReceipt:
    relpath, placed_new = _place(root, tmp, profile_id)
    try:
        store.insert_document(document_id=doc_id, profile_id=profile_id, sha256=sha, title=title,
                              mime="application/pdf", bytes=size, source_relpath=relpath,
                              origin="folder" if "folder" in payload else "user")
        job_id = _queue(store, doc_id, profile_id, payload)
    except Exception as exc:  # noqa: BLE001 - nothing usable was stored; undo the file
        logger.warning("document was not queued (%s)", type(exc).__name__)
        if placed_new:
            files.remove_original(root, relpath)
        raise _refuse("The document could not be saved right now. Try again.") from None
    return DocumentReceipt("processing", document_id=doc_id, job_id=job_id)


def _retry(store: Any, row: dict, payload: dict[str, Any]) -> DocumentReceipt:
    store.update_document(row["document_id"], state="processing")
    return DocumentReceipt("processing", document_id=row["document_id"],
                           job_id=_queue(store, row["document_id"], row["profile_id"], payload))


def _submit(store: Any, inp: MediaInput, root: Path, profile_id: str, payload: dict, key: str) -> DocumentReceipt:
    tmp, sha, size = _stage(inp, root)
    try:
        doc_id = _document_id(profile_id, key)
        row = _existing(store, profile_id, doc_id, sha, bool(key), "folder" in payload)
        if row:
            tmp.unlink(missing_ok=True)
            return _repeat(store, row) or _retry(store, row, payload)
        _, used = store.count_and_bytes(profile_id)
        if used + store.document_bytes(profile_id) + size > QUOTA_BYTES:
            raise _refuse("The library is full (2 GB limit). Remove some items first.")
        return _create(store, root, tmp, sha, size, doc_id, profile_id, _title(inp), payload)
    finally:
        tmp.unlink(missing_ok=True)


def submit_document(
    inp: MediaInput, *, content: str = "", profile_id: str, actor_id: str, config: Any,
    tags: str = "", session_date: str = "", idempotency_key: str = "", store: Any = None,
    folder: dict[str, Any] | None = None, scope: str | None = None, shared_with: tuple[str, ...] = (),
) -> DocumentReceipt:
    """Queue a PDF for page-by-page saving; see ``DocumentReceipt`` for the outcomes.

    ``folder`` (set only by folder sources) is added to every memory's provenance.
    """
    opened, store_ref = False, store
    try:
        store_ref, opened = _resolve(store)
        words = prepare_user_text(config, content).text if content.strip() else ""
        payload = {"user_words": words, "tags": tags, "session_date": session_date, "actor_id": actor_id}
        if scope:
            payload["scope"] = scope
            payload["shared_with"] = list(shared_with)
        if folder:
            payload["folder"] = dict(folder)
        receipt = _submit(store_ref, inp, Path(store_ref.path).parent, profile_id, payload, idempotency_key)
    except _Stop as stop:
        return stop.receipt
    finally:
        if opened and store_ref is not None:
            store_ref.close()
    if receipt.status == "processing":
        from superlocalmemory.documents.runner import wake_active

        wake_active()
    return receipt
