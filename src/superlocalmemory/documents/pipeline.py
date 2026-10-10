# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Run one document job: read the pages, keep each as a memory with its source.

Page text is text derived by SLM, so credentials in it are always stripped. The
person's own words go only into the one memory that belongs to the document. A
page is recorded last, after its memories, so a run that stops anywhere resumes
at the first unrecorded page and every memory it saves again has the same key.
Nothing here logs document text, OCR text or paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from contextlib import ExitStack
from typing import Any, Callable, ContextManager

from superlocalmemory.core.recall_gate import background_work, yield_to_recalls
from superlocalmemory.documents.chunking import chunk_text
from superlocalmemory.documents.heavy import parse_reservation
from superlocalmemory.documents.parse_proc import ParseFailed, ParseLimit, ParseSession, ParseStopped
from superlocalmemory.media import files
from superlocalmemory.media.labels import DOCUMENT
from superlocalmemory.memory_core import (
    ContentOrigin, effective_pii_redaction, prepare_for_save, scan_sensitive,
)
from superlocalmemory.memory_core.submit import SavePending, SaveRequest, submit_memory_settled
from superlocalmemory.runtimes.worker_client import MediaWorkerError

logger = logging.getLogger(__name__)

PART_LIMIT = 24_000
MAX_PAGE_TEXT = 1_000_000
MAX_THUMB_BYTES = 262_144
RECALL_YIELD_S = 5.0
_CODE = re.compile(r"[a-z_]{1,40}")


class _End(Exception):
    """Ends the run; the subclasses say how."""


class Failed(_End):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Stopped(_End):
    pass


class LeaseLost(_End):
    pass


class Cancelled(_End):
    pass


class Deferred(_End):
    """Not now (memory is short, or a model swap is under way): the job goes back to the queue."""


@dataclass(frozen=True)
class JobContext:
    store: Any
    client: Any
    runtime: Any
    config: Any
    python: Path
    script: Path
    limits: Any
    owner: str
    should_stop: Callable[[], bool]
    #: A context manager held around each parse step (the shared RAM reservation).
    heavy: Callable[[], ContextManager[None]] = parse_reservation
    #: True while a model swap asks background work to stand aside.
    paused: Callable[[], bool] = lambda: False


@dataclass
class _Text:
    origin: str = "none"
    text: str = ""
    pii: int = 0
    secrets: int = 0
    #: True only when the whole page text was scanned, whatever the redaction
    #: setting. Unscanned is never clean.
    scanned: bool = False


def _derived(text: str, redact: bool) -> _Text:
    prepared = prepare_for_save(text[:MAX_PAGE_TEXT], origin=ContentOrigin.DERIVED_TEXT, pii_redaction=redact)
    origin = "text_layer" if prepared.text.strip() else "none"
    try:  # vetting reads ALL of the text; what is stored stays the cut, redacted text
        found = scan_sensitive(text)
    except Exception:  # noqa: BLE001 - an unscanned page is simply not cleared for remote use
        logger.warning("page text could not be scanned; it stays local-only")
        return _Text(origin, prepared.text, prepared.pii_count, prepared.secret_count)
    return _Text(origin, prepared.text, found.pii, found.secrets, scanned=True)


class JobRunner:
    def __init__(self, ctx: JobContext, job: dict[str, Any]) -> None:
        self.ctx, self.job = ctx, job
        self.store = ctx.store
        self.payload = json.loads(job.get("payload_json") or "{}")
        self.doc_id = str(self.payload.get("document_id") or "")
        self.profile_id = job["profile_id"]
        self.redact = effective_pii_redaction(ctx.config)
        self.work = Path()
        self.deadline = time.monotonic() + ctx.limits.job_timeout_s
        self.doc: dict[str, Any] = {}
        self._last_renew = time.monotonic()

    # -- the run ----------------------------------------------------------------
    def run(self) -> bool:
        """Run the job; False when it was put back in the queue to wait."""
        self.doc = self.store.get_document(self.doc_id) or {}
        if not self.doc or self.doc["state"] == "tombstoned" or self.doc["profile_id"] != self.profile_id:
            self._finish("cancelled")
            return True
        root = Path(self.store.path).parent
        self.work = Path(tempfile.mkdtemp(dir=files.tmp_dir(root)))
        try:
            with background_work():
                self._process(root)
        except Failed as exc:
            self._fail(exc.reason)
        except Stopped:
            self.store.release_job(self.job["job_id"], self.ctx.owner)
        except Deferred:
            self.store.release_job(self.job["job_id"], self.ctx.owner)
            return False
        except LeaseLost:
            logger.info("document job lost its lease; leaving it to its new owner")
        except Cancelled:
            self._hide_saved()
            self._finish("cancelled")
        except MediaWorkerError:
            self._fail("image_tools")
        except Exception as exc:  # noqa: BLE001 - one job must not stop the service
            logger.warning("document job failed (%s)", type(exc).__name__)
            self._fail("failed")
        finally:
            shutil.rmtree(self.work, ignore_errors=True)
        return True

    def _process(self, root: Path) -> None:
        total = self._read_pages(root)
        self._document_memory()
        self.store.refresh_document_counts(self.doc_id)
        if not self.store.mark_document_ready(self.doc_id, total):
            raise Cancelled()  # removed while this job finished; never brought back
        if not self.store.progress_job(self.job["job_id"], self.ctx.owner, len(self.store.page_numbers(self.doc_id)),
                                       total):
            raise LeaseLost()
        self._finish("done")

    def _hide_saved(self) -> None:
        """Hide every memory the removed document owns, including any this job saved after the
        removal read its list. Same keys as the removal, so hiding twice is one hide."""
        from superlocalmemory.documents.status import archive_document_facts

        current = self.store.get_document(self.doc_id)
        if current and current["state"] == "tombstoned":
            archive_document_facts(self.store, self.ctx.runtime, current)

    def _finish(self, state: str, error: str | None = None) -> None:
        self.store.finish_job(self.job["job_id"], self.ctx.owner, state, error)

    def _fail(self, reason: str) -> None:
        reason = reason if _CODE.fullmatch(reason) else "failed"
        try:
            self.store.refresh_document_counts(self.doc_id)
            self.store.update_document(self.doc_id, state="failed")
        except Exception as exc:  # noqa: BLE001 - the job is still marked failed below
            logger.warning("document state was not updated (%s)", type(exc).__name__)
        self._finish("failed", reason)

    # -- reading pages ------------------------------------------------------------
    def _read_pages(self, root: Path) -> int:
        limits = self.ctx.limits
        source = files.media_root(root) / str(self.doc["source_relpath"])
        request = {"path": str(source), "out_dir": str(self.work), "max_pages": limits.max_pages,
                   "render_long_edge": 1280, "min_text_chars": 20,
                   "skip": sorted(self.store.page_numbers(self.doc_id))}
        total = 0
        session = ParseSession(
            self.ctx.python, self.ctx.script, request, cwd=self.work, page_timeout_s=limits.page_timeout_s,
            job_deadline=self.deadline, rss_limit_mb=limits.rss_limit_mb, poll_s=limits.poll_s,
            should_stop=self.ctx.should_stop, on_tick=self._tick)
        try:
            with session:
                while True:
                    event = self._next_event(session)
                    if event.get("done"):
                        self._apply_meta(event)
                        return total
                    if event.get("opened"):
                        total = int(event.get("page_count") or 0)
                        self.store.update_document(self.doc_id, page_count=total)
                        self._progress(total)
                    else:
                        self._page(event)
                        self._progress(total)
                        self._checkpoint()
        except ParseLimit as exc:
            raise Failed(exc.reason) from None
        except ParseFailed as exc:
            raise Failed(exc.reason) from None
        except ParseStopped:
            raise Stopped() from None

    def _next_event(self, session: ParseSession) -> dict[str, Any]:
        """One parse step, under the shared RAM reservation; a refused reservation defers the job."""
        with ExitStack() as stack:
            try:
                stack.enter_context(self.ctx.heavy())
            except RuntimeError:
                raise Deferred() from None
            return session.next_event()

    def _apply_meta(self, event: dict[str, Any]) -> None:
        title = str(event.get("title") or "").strip()
        if title:
            self.doc["title"] = title[:300]
            self.store.update_document(self.doc_id, title=self.doc["title"])

    def _progress(self, total: int) -> None:
        done = len(self.store.page_numbers(self.doc_id))
        if not self.store.progress_job(self.job["job_id"], self.ctx.owner, done, total or None):
            raise LeaseLost()

    def _tick(self) -> None:
        """Called while waiting on the parse: keep the lease alive."""
        if time.monotonic() - self._last_renew >= max(1.0, self.ctx.limits.lease_s / 4):
            self._renew()

    def _renew(self) -> None:
        self._last_renew = time.monotonic()
        if not self.store.renew_lease(self.job["job_id"], self.ctx.owner, self.ctx.limits.lease_s):
            raise LeaseLost()

    def _checkpoint(self) -> None:
        self._renew()
        if self.ctx.should_stop():
            raise Stopped()
        if self.ctx.paused():
            raise Deferred()
        current = self.store.get_document(self.doc_id)
        if not current or current["state"] == "tombstoned":
            raise Cancelled()
        if time.monotonic() > self.deadline:
            raise Failed("time_limit")
        yield_to_recalls(max_seconds=RECALL_YIELD_S)

    # -- one page -------------------------------------------------------------------
    def _inside(self, name: Any, directory: Path) -> Path:
        path = Path(str(name)).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise Failed("page_failed")
        return path

    def _page(self, event: dict[str, Any]) -> None:
        page_no = int(event["page_no"])
        png = self._inside(event.get("png_path"), self.work)
        text = self._text(event, png)
        thumb, phash = self._thumb(png)
        vector = self.ctx.client.embed_images([png], wait_cold=True)[0]
        memory_ids, fact_ids = self._page_memories(page_no, text)
        media_id = self._page_item(event, page_no, text, memory_ids, thumb, phash, vector)
        self.store.put_page(self.doc_id, page_no, media_id=media_id, memory_ids=memory_ids, fact_ids=fact_ids,
                            text_origin=text.origin, char_count=len(text.text))
        png.unlink(missing_ok=True)

    def _text(self, event: dict[str, Any], png: Path) -> _Text:
        if event.get("has_text_layer"):
            return _derived(str(event.get("text") or ""), self.redact)
        try:
            reply = self.ctx.client.ocr_image(png, wait_cold=True)
        except MediaWorkerError:
            return _Text()
        found = _derived(str(reply.get("text") or ""), self.redact)
        found.origin = "ocr" if found.origin != "none" and reply.get("engine") not in (None, "none") else "none"
        return found if found.origin == "ocr" else _Text()

    def _thumb(self, png: Path) -> tuple[bytes | None, str | None]:
        out = Path(tempfile.mkdtemp(dir=self.work))
        try:
            info = dict(self.ctx.client.prepare_image(png, out, wait_cold=True))
            thumb = info.get("thumb_path")
            data = None
            if thumb:
                path = self._inside(thumb, out)
                data = path.read_bytes() if path.stat().st_size <= MAX_THUMB_BYTES else None
            return data, info.get("phash")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def _save(self, segments: tuple, key: str, source: dict[str, Any]) -> Any:
        request = SaveRequest(
            segments=segments, profile_id=self.profile_id, source_type="document",
            trusted_actor_id=str(self.payload.get("actor_id") or ""), tags=str(self.payload.get("tags") or ""),
            session_date=str(self.payload.get("session_date") or ""),
            trusted_metadata={"_slm_source": {**source, **(self.payload.get("folder") or {})}},
            idempotency_key=key, scope=self.payload.get("scope") or None,
            shared_with=tuple(self.payload.get("shared_with") or ()))
        try:
            return submit_memory_settled(self.ctx.runtime, request, config=self.ctx.config)
        except SavePending:
            # Durable but not committed yet: record nothing for this page now; the retry re-sends
            # the same key and gets the committed ids, so removal can always find them.
            raise Deferred() from None
        except Exception as exc:  # noqa: BLE001 - nothing for this key was stored
            logger.warning("document memory was not saved (%s)", type(exc).__name__)
            raise Failed("save_failed") from None

    def _page_memories(self, page_no: int, text: _Text) -> tuple[list[str], list[str]]:
        header = f"[Page {page_no}]\n"
        parts = chunk_text(text.text, PART_LIMIT - len(header)) if text.origin != "none" else []
        memory_ids: list[str] = []
        fact_ids: list[str] = []
        for number, part in enumerate(parts, 1):
            source: dict[str, Any] = {"type": "document", "document_id": self.doc_id, "page": page_no}
            if len(parts) > 1:
                source["part"] = number
            saved = self._save(((header + part, ContentOrigin.DERIVED_TEXT),),
                               f"doc:{self.doc_id}:{page_no}:{number}", source)
            if saved.memory_id:
                memory_ids.append(saved.memory_id)
            fact_ids += list(saved.fact_ids)
        return memory_ids, fact_ids

    def _page_item(self, event: dict[str, Any], page_no: int, text: _Text, memory_ids: list[str],
                   thumb: bytes | None, phash: str | None, vector: list[float]) -> str:
        media_id = hashlib.sha256(f"{self.doc_id}:{page_no}".encode()).hexdigest()[:32]
        if self.store.get_item(media_id):
            return media_id
        self.store.insert_item(
            media_id=media_id, profile_id=self.profile_id, kind="page", source_sha256=self.doc["sha256"],
            phash=phash, mime="image/png", bytes=0, width=event.get("width"), height=event.get("height"),
            anchor_memory_id=memory_ids[0] if memory_ids else None, document_id=self.doc_id, page_no=page_no,
            origin="document", thumb_webp=thumb,
            remote_ok=int(text.origin != "none" and text.scanned and text.secrets == 0 and text.pii == 0))
        try:
            client = self.ctx.client
            space = self.store.ensure_active_space(client.model_id, client.revision, len(vector))
            self.store.put_vector(media_id, space, self.profile_id, vector)
        except Exception as exc:  # noqa: BLE001 - the page is kept; only searching by picture is lost
            logger.warning("page %s saved without its vector (%s)", page_no, type(exc).__name__)
        return media_id

    # -- the document's own memory ------------------------------------------------
    def _document_memory(self) -> None:
        current = self.store.get_document(self.doc_id) or {}
        if current.get("memory_id") or current.get("fact_ids_json") not in (None, "[]"):
            return
        title, words = str(current.get("title") or "").strip(), str(self.payload.get("user_words") or "")
        segments: list[tuple[str, ContentOrigin]] = []
        if title:
            segments.append((title, ContentOrigin.DERIVED_TEXT))
        if words.strip():
            segments.append((("\n\n" if title else "") + words, ContentOrigin.USER_TEXT))
        if not segments:
            return
        if title:
            segments.insert(1, ("\n" + DOCUMENT, ContentOrigin.DERIVED_TEXT))
        else:
            segments.insert(0, (DOCUMENT + "\n\n", ContentOrigin.DERIVED_TEXT))
        saved = self._save(tuple(segments), f"doc:{self.doc_id}:doc", {"type": "document", "document_id": self.doc_id})
        self.store.update_document(self.doc_id, memory_id=saved.memory_id, fact_ids_json=json.dumps(list(saved.fact_ids)))
