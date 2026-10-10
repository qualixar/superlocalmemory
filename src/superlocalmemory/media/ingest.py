# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Save one image: strip its metadata, keep a copy, read its text, remember it.

The picture is processed in the media worker (the daemon never decodes pixels).
What is kept: the stripped re-encode (content addressed), a small thumbnail, a
perceptual hash, a vector, and one memory whose text is the person's own words
plus the text found in the image. Nothing here logs content, text or paths.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from superlocalmemory.media import files
from superlocalmemory.media.labels import NO_TEXT, TEXT_MARKER
from superlocalmemory.memory_core import ContentOrigin, effective_pii_redaction, prepare_for_save
from superlocalmemory.memory_core.submit import SaveRequest, submit_memory
from superlocalmemory.runtimes.space_plan import compatible, current_space_plan
from superlocalmemory.runtimes.worker_client import MediaWorkerError, MediaWorkerWarming

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_BASE64_BYTES = 8 * 1024 * 1024
QUOTA_BYTES = 2 * 1024 ** 3
MAX_OCR_CHARS = 8_000
PREVIEW_CHARS = 200
NEAR_DUPLICATE_BITS = 4
MAX_THUMB_BYTES = 262_144
MARKER = TEXT_MARKER
_MIMES = frozenset({"image/png", "image/jpeg", "image/webp"})
_EXT = re.compile(r"[a-z0-9]{1,8}")
_EXIF_DATE = re.compile(r"(\d{4}):(\d{2}):(\d{2}) (\d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class MediaInput:
    path: Path | None = None
    base64: str | None = None
    file_name: str = ""
    #: Internal only: bytes the caller already read safely (folder sources). Routes never set it.
    data: bytes | None = None
    download_url: str | None = None
    remote: bool = False


@dataclass(frozen=True)
class MediaReceipt:
    status: Literal["stored", "duplicate", "warming", "refused"]
    media_id: str | None = None
    memory_id: str | None = None
    reason: str = ""
    duplicate_of: str | None = None
    near_duplicate_of: str | None = None
    extracted_text_preview: str = ""


class _Stop(Exception):
    """Ends the run early with this receipt."""

    def __init__(self, receipt: MediaReceipt) -> None:
        super().__init__(receipt.status)
        self.receipt = receipt


def _refuse(reason: str) -> _Stop:
    return _Stop(MediaReceipt("refused", reason=reason))


@dataclass
class _Ocr:
    engine: str = "none"
    text: str = ""
    pii: int = 0
    secrets: int = 0


@dataclass
class _Job:
    client: Any
    store: Any
    cache: Any
    config: Any
    redact: bool
    root: Path
    plan: Any = None
    work: Path = field(default_factory=Path)
    placed: str = ""
    placed_new: bool = False


def _cold_wait_s() -> float:
    try:
        return min(60.0, max(1.0, float(os.environ.get("SLM_MEDIA_COLD_WAIT_S", "20"))))
    except ValueError:
        return 20.0


# -- reading the input ---------------------------------------------------------

def _download(inp: MediaInput) -> bytes:
    from superlocalmemory.core.media_fetch import MediaFetchRefused, fetch_media

    try:
        return fetch_media(inp.download_url or "", remote=inp.remote, max_bytes=MAX_FILE_BYTES).data
    except MediaFetchRefused as refused:
        raise _refuse(refused.reason) from None


def _refuse_own_data(path: Path) -> None:
    """A file or folder in SuperLocalMemory's own data folder is not the person's to hand over."""
    from superlocalmemory.infra.data_root import DATA_ROOT_REFUSAL, overlaps_data_root

    if overlaps_data_root(path):
        raise _refuse(DATA_ROOT_REFUSAL)


def _read_input(inp: MediaInput) -> bytes:
    if inp.data is not None:
        if len(inp.data) > MAX_FILE_BYTES:
            raise _refuse("That image is too large (25 MB limit).")
        return inp.data
    if inp.download_url:
        return _download(inp)
    if inp.base64 is not None:
        if len(inp.base64) * 3 // 4 > MAX_BASE64_BYTES + 3:
            raise _refuse("That image is too large (8 MB limit for pasted images).")
        try:
            data = base64.b64decode(inp.base64, validate=True)
        except (binascii.Error, ValueError):
            raise _refuse("That image data could not be read.") from None
        if len(data) > MAX_BASE64_BYTES:
            raise _refuse("That image is too large (8 MB limit for pasted images).")
        return data
    if inp.path is None:
        raise _refuse("Give an image file or image data.")
    try:
        path = Path(inp.path)
        _refuse_own_data(path)
        if not path.is_file():
            raise _refuse("That image file could not be found.")
        with open(path, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                raise _refuse("That image file could not be found.")
            data = fh.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise _refuse("That image is too large (25 MB limit).")
        return data
    except OSError:
        raise _refuse("That image file could not be read.") from None


def _check_kind(data: bytes) -> None:
    if not data:
        raise _refuse("That image is empty.")
    ok = (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")
          or data[:6] in (b"GIF87a", b"GIF89a") or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"))
    if not ok:
        raise _refuse("That file type is not supported (PNG, JPEG, GIF and WEBP only).")


_OFF = "Images are turned off. Turn them on in settings to save images."


def _unavailable() -> _Stop:
    from superlocalmemory.media.readiness import media_refusal

    return _refuse(media_refusal() or _OFF)


def _resolve(client: Any, store: Any) -> tuple[Any, Any, bool]:
    if client is None:
        from superlocalmemory.runtimes.worker_client import media_embedder

        client = media_embedder()
    if client is None:
        raise _unavailable()
    opened = store is None
    if store is None:
        from superlocalmemory.media import open_media_store

        store = open_media_store()
    if store is None:
        raise _unavailable()
    return client, store, opened


# -- the worker steps ----------------------------------------------------------

def _wait_for_warm(client: Any) -> None:
    if client.is_warm():
        return
    client.warm_up()
    deadline = time.monotonic() + _cold_wait_s()
    while time.monotonic() < deadline:
        if client.is_warm():
            return
        time.sleep(0.1)
    raise _Stop(MediaReceipt("warming", reason="images are starting up; try again in a minute"))


def _inside(work: Path, name: Any) -> Path:
    path = Path(str(name)).resolve()
    if not path.is_relative_to(work.resolve()) or not path.is_file():
        raise _refuse("The image could not be processed.")
    return path


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare(job: _Job, data: bytes) -> dict[str, Any]:
    _wait_for_warm(job.client)
    fd = os.open(job.work / "source.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    info = dict(job.client.prepare_image(job.work / "source.bin", job.work, wait_cold=False))
    ext = str(info.get("stored_ext") or "").lower()
    if info.get("mime") not in _MIMES or not _EXT.fullmatch(ext):
        raise _refuse("The image could not be processed.")
    stored = _inside(job.work, info.get("stored_path"))
    info.update(stored_path=stored, stored_ext=ext, stored_sha=_file_sha(stored), stored_size=stored.stat().st_size)
    thumb = info.get("thumb_path")
    info["thumb"] = None
    if thumb:
        tpath = _inside(job.work, thumb)
        info["thumb"] = tpath.read_bytes() if tpath.stat().st_size <= MAX_THUMB_BYTES else None
    return info


def _place(job: _Job, info: dict[str, Any]) -> str:
    target = files.original_path(job.root, info["stored_sha"], info["stored_ext"])
    job.placed_new = not target.exists()
    try:
        job.placed = files.place_original(job.root, info["stored_path"], info["stored_sha"], info["stored_ext"])
    except (OSError, ValueError):
        raise _refuse("The image could not be saved.") from None
    return job.placed


def _ocr_key(stored_sha: str, redact: bool):
    from superlocalmemory.cache.keys import CacheKey, params_hash

    return CacheKey(stored_sha, "ocr.auto", "1", params_hash=params_hash({"redaction": redact, "lang": "auto"}))


def _ocr(job: _Job, info: dict[str, Any]) -> _Ocr:
    key = _ocr_key(info["stored_sha"], job.redact)
    try:
        hit = job.cache.get(key) if job.cache is not None else None
        if hit:
            return _Ocr(**json.loads(bytes(hit).decode("utf-8")))
    except Exception:  # noqa: BLE001 - a cache fault only costs a recompute
        logger.debug("ocr cache read skipped")
    reply = job.client.ocr_image(info["stored_path"], wait_cold=False)
    engine, text = str(reply.get("engine") or "none"), str(reply.get("text") or "")[:MAX_OCR_CHARS]
    prepared = prepare_for_save(text, origin=ContentOrigin.DERIVED_TEXT, pii_redaction=job.redact)
    out = _Ocr(engine, prepared.text, prepared.pii_count, prepared.secret_count)
    if engine != "none" and job.cache is not None:
        try:
            job.cache.put(key, json.dumps(out.__dict__).encode("utf-8"), kind="json")
        except Exception:  # noqa: BLE001
            logger.debug("ocr cache write skipped")
    return out


def _near_duplicate(store: Any, profile_id: str, phash: str | None) -> str | None:
    try:
        mine = int(phash or "", 16)
    except ValueError:
        return None
    best: tuple[int, str] | None = None
    for media_id, other in store.phash_candidates(profile_id):
        try:
            dist = (mine ^ int(other, 16)).bit_count()
        except ValueError:
            continue
        if dist <= NEAR_DUPLICATE_BITS and (best is None or dist < best[0]):
            best = (dist, media_id)
    return best[1] if best else None


# -- saving --------------------------------------------------------------------

def _segments(content: str, ocr_text: str) -> tuple[tuple[str, ContentOrigin], ...]:
    parts: list[tuple[str, ContentOrigin]] = []
    if content.strip():
        parts.append((content, ContentOrigin.USER_TEXT))
    if ocr_text.strip():
        lead = "\n\n" if parts else ""
        parts.append((f"{lead}{MARKER}{ocr_text}", ContentOrigin.DERIVED_TEXT))
    if not parts:
        parts.append((NO_TEXT, ContentOrigin.DERIVED_TEXT))
    return tuple(parts)


def _captured_at(exif: dict[str, Any]) -> str | None:
    found = _EXIF_DATE.fullmatch(str(exif.get("DateTimeOriginal") or ""))
    return f"{found[1]}-{found[2]}-{found[3]}T{found[4]}" if found else None


def _check_space(job: _Job, dim: int) -> dict[str, Any]:
    """The signature this picture's space must carry; refuses before anything is saved when the index differs."""
    plan = replace(job.plan, image_model=str(job.client.model_id), image_revision=str(job.client.revision), dim=dim)
    if not compatible(plan, job.store.active_signature()):
        raise _refuse("The picture index was built with a different model; rebuild it from the dashboard.")
    return plan.signature()


def _write_row(job: _Job, fields: dict[str, Any], vector: list[float], profile_id: str,
               signature: dict[str, Any]) -> str:
    media_id = job.store.insert_item(**fields)
    try:
        space = job.store.ensure_active_space(job.client.model_id, job.client.revision, len(vector), signature)
        job.store.put_vector(media_id, space, profile_id, vector)
    except Exception as exc:  # noqa: BLE001 - the row is kept; only searching by picture is lost
        logger.warning("image %s saved without its vector (%s)", media_id, type(exc).__name__)
    return media_id


def _cleanup(job: _Job) -> None:
    if job.placed and job.placed_new:
        files.remove_original(job.root, job.placed)
    job.placed = ""


def _store_it(job: _Job, data: bytes, src_sha: str, args: dict[str, Any]) -> MediaReceipt:
    profile_id = args["profile_id"]
    info = _prepare(job, data)
    relpath = _place(job, info)
    ocr = _ocr(job, info)
    vector = job.client.embed_images([info["stored_path"]], wait_cold=False)[0]
    signature = _check_space(job, len(vector))
    near = _near_duplicate(job.store, profile_id, info.get("phash"))
    media_id = uuid.uuid4().hex
    request = SaveRequest(
        segments=_segments(args["content"], ocr.text), profile_id=profile_id, source_type="media",
        trusted_actor_id=args["actor_id"], tags=args["tags"], session_date=args["session_date"],
        trusted_metadata={"_slm_source": {"type": "media", "media_id": media_id, "origin": "tool",
                                          **(args.get("folder") or {})}},
        idempotency_key=args["idempotency_key"], scope=args.get("scope"),
        shared_with=tuple(args.get("shared_with") or ()))
    try:
        saved = submit_memory(args["runtime"], request, config=job.config)
    except Exception as exc:  # noqa: BLE001 - nothing was stored; undo the file
        logger.warning("image memory was not saved (%s)", type(exc).__name__)
        _cleanup(job)
        raise _refuse("The image could not be saved right now. Try again.") from None
    fields = dict(
        media_id=media_id, profile_id=profile_id, kind="image", source_sha256=src_sha,
        stored_sha256=info["stored_sha"], phash=info.get("phash"), mime=info["mime"],
        bytes=info["stored_size"], width=info.get("width"), height=info.get("height"),
        original_relpath=relpath, exif_json=info.get("exif") or {}, captured_at=_captured_at(info.get("exif") or {}),
        anchor_memory_id=saved.memory_id,
        origin="folder" if args.get("folder") else "tool", thumb_webp=info["thumb"],
        remote_ok=int(ocr.engine != "none" and ocr.secrets == 0 and ocr.pii == 0))
    preview = ocr.text[:PREVIEW_CHARS]
    try:
        _write_row(job, fields, vector, profile_id, signature)
    except Exception as exc:  # noqa: BLE001 - the memory exists; the file stays for later reconciliation
        logger.warning("memory %s saved but its image row was not (%s)", saved.memory_id, type(exc).__name__)
        return MediaReceipt("stored", memory_id=saved.memory_id, extracted_text_preview=preview,
                            reason="The text was saved; the picture could not be indexed.")
    return MediaReceipt("stored", media_id=media_id, memory_id=saved.memory_id,
                        near_duplicate_of=near, extracted_text_preview=preview)


def remember_media(
    inp: MediaInput, *, content: str = "", profile_id: str, actor_id: str, runtime: Any, config: Any,
    tags: str = "", session_date: str = "", idempotency_key: str = "",
    client: Any = None, store: Any = None, cache: Any = None, folder: dict[str, Any] | None = None,
    scope: str | None = None, shared_with: tuple[str, ...] = (),
) -> MediaReceipt:
    """Save an image and the words about it as one memory; see ``MediaReceipt`` for the outcomes."""
    opened = False
    store_ref = store
    try:
        data = _read_input(inp)
        _check_kind(data)
        client, store_ref, opened = _resolve(client, store)
        src_sha = hashlib.sha256(data).hexdigest()
        known = store_ref.find_by_sha(profile_id, src_sha, exclude_origin=None if folder else "folder")
        if known:
            return MediaReceipt("duplicate", media_id=known["media_id"], memory_id=known["anchor_memory_id"],
                                duplicate_of=known["media_id"])
        _, used = store_ref.count_and_bytes(profile_id)
        if used + len(data) > QUOTA_BYTES:
            raise _refuse("The image library is full (2 GB limit). Remove some images first.")
        return _run(client, store_ref, cache, config, data, src_sha, dict(
            content=content, profile_id=profile_id, actor_id=actor_id, runtime=runtime, tags=tags,
            session_date=session_date, idempotency_key=idempotency_key, folder=folder,
            scope=scope, shared_with=tuple(shared_with)))
    except _Stop as stop:
        return stop.receipt
    finally:
        if opened and store_ref is not None:
            store_ref.close()


def _run(client: Any, store: Any, cache: Any, config: Any, data: bytes, src_sha: str,
         args: dict[str, Any]) -> MediaReceipt:
    if cache is None:
        try:
            from superlocalmemory.cache.factory import default_cache

            cache = default_cache()
        except Exception:  # noqa: BLE001 - no cache just means no reuse
            cache = None
    root = Path(store.path).parent
    try:
        plan = current_space_plan(root)
    except ValueError:
        raise _refuse("That picture mode is not available in this build.") from None
    job = _Job(client, store, cache, config, effective_pii_redaction(config), root, plan)
    job.work = Path(tempfile.mkdtemp(dir=files.tmp_dir(root)))
    try:
        return _store_it(job, data, src_sha, args)
    except MediaWorkerWarming:
        _cleanup(job)
        return MediaReceipt("warming", reason="images are starting up; try again in a minute")
    except MediaWorkerError:
        _cleanup(job)
        return MediaReceipt("refused", reason="The image tools could not process that image.")
    except _Stop:
        _cleanup(job)
        raise
    finally:
        shutil.rmtree(job.work, ignore_errors=True)
