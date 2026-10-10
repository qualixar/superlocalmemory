# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm media repair``: give every saved picture a vector in the current picture index.

A picture can be saved without a vector (the write failed), and after the picture
model or the space mode changes, no picture has a vector in the new index. Both
leave the memory's words searchable but the picture itself not findable by what it
shows. This reads each kept original again, embeds it and stores the vector. When
the model changed it first creates the index for the current model, only after the
first picture has embedded, so a failed run never retires a working index.

The run covers the whole picture library, because an index belongs to the library
and not to one profile; the caller needs MANAGE. It stops after a time budget and
reports how many are left, so running it again carries on.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from superlocalmemory.media import files, ingest, media_db_exists, open_media_store
from superlocalmemory.runtimes.space_plan import current_space_plan
from superlocalmemory.runtimes.worker_client import MediaWorkerWarming

logger = logging.getLogger(__name__)

BUDGET_S = 240.0
BATCH = 8


@dataclass
class RepairReport:
    dry_run: bool = False
    missing: int = 0
    repaired: int = 0
    failed: int = 0
    skipped_no_file: int = 0
    remaining: int = 0
    index_needs_rebuild: bool = False
    rebuilt_index: bool = False
    reason: str = ""


def _differs(plan: Any, client: Any, signature: dict[str, Any] | None) -> bool:
    """Whether the stored index was built for another model or mode (the size is judged later)."""
    if signature is None:
        return False
    want = {**plan.signature(), "image_model": str(client.model_id),
            "image_revision": str(client.revision)}
    return any(str(signature.get(k)) != str(v) for k, v in want.items() if k != "dim")


def _file_for(root: Path, relpath: str) -> Path | None:
    base = files.media_root(root).resolve()
    path = (base / relpath).resolve()
    if not path.is_relative_to(base) or not path.is_file() or (base / relpath).is_symlink():
        return None
    return path


def _open(client: Any, store: Any, root: Path) -> tuple[Any, Any, bool]:
    opened = store is None
    if client is None:
        from superlocalmemory.runtimes.worker_client import media_embedder

        client = media_embedder()
    if store is None and media_db_exists(root):
        store = open_media_store(data_root=root)
    if client is None or store is None:
        if opened and store is not None:
            store.close()
        from superlocalmemory.media.readiness import media_refusal

        raise ingest._Stop(ingest.MediaReceipt("refused", reason=media_refusal() or ingest._OFF))
    return client, store, opened


def _plan(root: Path) -> Any:
    try:
        return current_space_plan(root)
    except ValueError:
        raise ingest._refuse("That picture mode is not available in this build.") from None


def _put(store: Any, plan: Any, client: Any, item: dict[str, Any], vector: list[float]) -> None:
    signature = replace(plan, image_model=str(client.model_id), image_revision=str(client.revision),
                        dim=len(vector)).signature()
    space = store.ensure_active_space(client.model_id, client.revision, len(vector), signature)
    store.put_vector(item["media_id"], space, item["profile_id"], vector)


def _embed_batch(client: Any, paths: list[Path]) -> list[list[float]]:
    return client.embed_images(paths, wait_cold=False)


def _refresh_signature(store: Any, client: Any, plan: Any, report: RepairReport) -> None:
    """Same image model, other mode or text model: its vectors stand, only the record changes."""
    active = store.active_space()
    if (report.index_needs_rebuild and not report.rebuilt_index and report.failed == 0
            and _same_vectors(active, client)):
        store.record_signature(active["space_id"], replace(
            plan, image_model=str(client.model_id), image_revision=str(client.revision),
            dim=int(active["dim"])).signature())
        report.rebuilt_index = True


def _run_batches(store: Any, client: Any, plan: Any, todo: list[tuple[dict, Path]],
                 report: RepairReport, deadline: float) -> None:
    for start in range(0, len(todo), BATCH):
        if time.monotonic() > deadline:
            report.remaining = len(todo) - start
            report.reason = "Stopped after the time limit; run it again to carry on."
            return
        batch = todo[start:start + BATCH]
        try:
            vectors = _embed_batch(client, [p for _, p in batch])
        except MediaWorkerWarming:
            report.remaining = len(todo) - start
            report.reason = "The image model is starting up; try again in a minute."
            return
        except Exception as exc:  # noqa: BLE001 - count it and carry on with the next batch
            logger.warning("repair: embedding %d pictures failed (%s)", len(batch), type(exc).__name__)
            report.failed += len(batch)
            continue
        for (item, _), vector in zip(batch, vectors):
            try:
                _put(store, plan, client, item, vector)
                report.repaired += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("repair: vector for %s not stored (%s)", item["media_id"], type(exc).__name__)
                report.failed += 1


def _same_vectors(active: dict[str, Any] | None, client: Any) -> bool:
    """The active space was made by this very image model, so its vectors are still right."""
    return active is not None and (active["model_id"], active["model_revision"]) == (
        str(client.model_id), str(client.revision))


def _plan_work(store: Any, client: Any, plan: Any, root: Path,
               report: RepairReport) -> list[tuple[dict, Path]]:
    active = store.active_space()
    report.index_needs_rebuild = active is not None and _differs(plan, client, store.active_signature())
    needs_all = active is None or (report.index_needs_rebuild and not _same_vectors(active, client))
    items = store.images_without_vector(None if needs_all else active["space_id"])
    report.missing = len(items)
    todo: list[tuple[dict, Path]] = []
    for item in items:
        path = _file_for(root, item["original_relpath"])
        if path is None:
            report.skipped_no_file += 1
        else:
            todo.append((item, path))
    return todo


def repair(profile_id: str, dry_run: bool = False, *, data_root: str | Path | None = None,
           client: Any = None, store: Any = None, budget_s: float = BUDGET_S) -> RepairReport:
    """Embed the pictures that have no vector in the current index; see the module note.

    ``profile_id`` is who asked (the route checks MANAGE for it); the run covers the library.
    """
    report = RepairReport(dry_run=dry_run)
    if data_root is None:
        from superlocalmemory.infra.data_root import canonical_data_root

        data_root = canonical_data_root()
    root = Path(data_root)
    opened = False
    try:
        client, store, opened = _open(client, store, root)
        plan = _plan(root)
        todo = _plan_work(store, client, plan, root, report)
        if dry_run:
            return report
        if not todo:
            _refresh_signature(store, client, plan, report)
            return report
        before = store.active_space()
        ingest._wait_for_warm(client)
        _run_batches(store, client, plan, todo, report, time.monotonic() + budget_s)
        after = store.active_space()
        report.rebuilt_index = bool(before and after and before["space_id"] != after["space_id"])
        _refresh_signature(store, client, plan, report)
        return report
    except ingest._Stop as stop:
        report.reason = stop.receipt.reason
        return report
    finally:
        if opened and store is not None:
            store.close()
