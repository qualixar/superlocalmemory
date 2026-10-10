# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Guarded background re-embed: switch the embedding model without stopping.

One daemon thread runs at most one job. While it runs, recall and remember use
the OLD model and the OLD vectors untouched; the job embeds every memory into a
staging space of the NEW dimension in short batches, catches up with memories
written or edited meanwhile, and then swaps the spaces in one transaction while
the daemon holds requests for that moment only (core/embedding_reindex_activate).
The previous space is kept until the next switch, so ``rollback`` can return to
it; ``forget-previous`` frees it. A daemon restart resumes the job at its
cursor; a failure at any step leaves the old space live.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from superlocalmemory.core import embedding_reindex_secrets as secrets
from superlocalmemory.core import embedding_reindex_steps as steps
from superlocalmemory.storage import embedding_canonical_slots as slots
from superlocalmemory.storage import embedding_change_log as change_log
from superlocalmemory.storage import embedding_spaces as sp
from superlocalmemory.storage.embedding_reindex_jobs import (
    JobConflict,
    active_job,
    create_job,
    get_job,
    latest_job,
    public_view,
    update_job,
)
from superlocalmemory.storage.embedding_space_swap import count_facts, drop_gone, facts_after, staged_diff

logger = logging.getLogger(__name__)

_RUNNER: "ReindexRunner | None" = None
_IDLE_POLL_S = 2.0
_PURGE_EVERY_S = 30.0
_CATCH_UP_ROUNDS = 50
_QUIET_WAIT_S = 600.0  # background is paused meanwhile; requests are served
_MAX_BACKOFF_S = 60.0
#: How long a model load waits for another heavy job (a picture-model load, a document parse).
LOAD_WAIT_S = 600.0


class NoChange(ValueError):
    """The requested model is the one the store already holds."""


class Refused(RuntimeError):
    """A request that cannot be honoured; the message says why, in plain words."""


def space_changed(live: Any, target: Any) -> bool:
    """True when ``target`` needs its own vectors (not the live space or an alias)."""
    return not sp.same_space(sp.signature_of(live), sp.signature_of(target))


def pending_switch_message(live: Any, target: Any) -> str:
    """What happens to a model change saved without starting a job: never "on next recall"."""
    return (f"Saved. Your memories are re-indexed with {target.model_name} in the background "
            "once the SLM daemon loads this configuration; recall keeps using "
            f"{live.model_name} until the switch completes. Progress: slm embedder status")


def get_runner() -> "ReindexRunner | None":
    return _RUNNER


def notify_pending(config: Any) -> None:
    """An engine was built on the live space while config named another model."""
    runner = _RUNNER
    if runner is not None and Path(runner.db_path) == Path(config.db_path):
        runner.adopt_pending(config)


class ReindexRunner:
    def __init__(self, *, db_path: Any, data_root: Any, app_state: Any = None) -> None:
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self.app_state = app_state
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._api_lock = threading.Lock()
        self._prebuilt: dict[int, Any] = {}
        self._thread: threading.Thread | None = None
        self._last_purge = 0.0
        self.notice: str | None = None
        self.quiet: dict | None = None
        #: job id -> change-log position of the last full diff that found nothing
        self.clean_marks: dict[int, int] = {}

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "ReindexRunner":
        self._thread = threading.Thread(target=self._loop, name="slm-embedding-reindex",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    def _conn(self) -> Any:
        return sp.connect(self.db_path)

    # -- requests (HTTP routes and CLI land here) ---------------------------

    def _live(self, conn: Any) -> tuple[str, dict]:
        row = sp.read_space(conn)
        if row is None:
            raise Refused("the embedding space of this store is not recorded yet; "
                          "restart SLM once and try again")
        return str(row["live_signature"]), json.loads(row["live_config"])

    def request_switch(self, target: Any, *, kind: str = "switch",
                       prebuilt: Any = None) -> dict:
        target_sig = sp.signature_of(target)
        with self._api_lock:
            conn = self._conn()
            try:
                live_sig, live_cfg = self._live(conn)
                if sp.same_space(live_sig, target_sig):
                    raise NoChange(f"{target.model_name} is already the embedding model")
                with steps.write_txn(conn, self.db_path):
                    job = create_job(conn, kind=kind, from_sig=live_sig, to_sig=target_sig,
                                     from_cfg=live_cfg, to_cfg=sp.public_config(target),
                                     total=count_facts(conn))
                    if prebuilt is not None:  # before COMMIT: the runner cannot see it yet
                        self._prebuilt[int(job["job_id"])] = prebuilt
            finally:
                conn.close()
        if kind == "switch":  # a rollback's key already sits under PREVIOUS
            secrets.put(self.data_root, secrets.TARGET, target.api_key)
        self._cancel.clear()
        self.wake()
        logger.info("embedding re-index job %s queued: %s -> %s", job["job_id"],
                    live_sig, target_sig)
        return public_view(job)

    def request_rollback(self) -> dict:
        conn = self._conn()
        try:
            row = sp.read_space(conn)
            if active_job(conn) is not None:
                raise JobConflict(active_job(conn))
            if row is None or not row.get("prev_signature"):
                raise Refused("there is no previous embedding model to go back to")
            prev = sp.config_from_public(row["prev_config"],
                                         secrets.get(self.data_root, secrets.PREVIOUS))
        finally:
            conn.close()
        embedder = steps.build_embedder(prev)
        try:
            if embedder is None:
                raise steps.StepFailed("it could not be started")
            steps.probe(embedder, prev.dimension)
        except steps.StepFailed as exc:
            if embedder is not None:
                steps.close_embedder(embedder)
            raise Refused(f"the previous model {prev.model_name} is not available ({exc}); "
                          "make it available again, then retry the rollback") from exc
        try:
            return self.request_switch(prev, kind="rollback", prebuilt=embedder)
        except BaseException:
            for key in [k for k, v in self._prebuilt.items() if v is embedder]:
                self._prebuilt.pop(key, None)
            steps.close_embedder(embedder)
            raise

    def cancel(self) -> dict:
        conn = self._conn()
        try:
            job = active_job(conn)
            if job is None:
                raise Refused("no re-index is running")
            if job["state"] == "queued":
                with steps.write_txn(conn, self.db_path):
                    update_job(conn, job["job_id"], state="cancelled", finished_at=time.time(),
                               error="cancelled before it started")
                return public_view(get_job(conn, job["job_id"]))
        finally:
            conn.close()
        self._cancel.set()
        self.wake()
        return public_view(job)

    def forget_previous(self) -> dict:
        with self._api_lock:
            conn = self._conn()
            try:
                job = active_job(conn)
                if job is not None and job["kind"] == "rollback":
                    raise JobConflict(job)
                row = sp.read_space(conn)
                had = sp.has_previous(conn) or bool(row and row.get("prev_signature"))
                if not had:
                    raise Refused("there is no previous embedding space to free")
                with steps.write_txn(conn, self.db_path):
                    freed = int(conn.execute(f"SELECT COUNT(*) FROM {sp.PREV_MAP}").fetchone()[0]) \
                        if sp.table_exists(conn, sp.PREV_MAP) else 0
                    sp.drop_previous(conn)
                    if row is not None:
                        sp.write_space(conn, row["live_signature"], json.loads(row["live_config"]))
            finally:
                conn.close()
        secrets.drop(self.data_root, secrets.PREVIOUS)
        return {"freed_vectors": freed, "previous": None}

    def status(self) -> dict:
        conn = self._conn()
        try:
            job = active_job(conn) or latest_job(conn)
            row = sp.read_space(conn)
            previous = row.get("prev_signature") if row else None
            return {"job": public_view(job), "notice": self.notice,
                    "live": row["live_signature"] if row else None,
                    "previous": previous if previous and sp.has_previous(conn) else previous,
                    "previous_vectors_kept": sp.has_previous(conn)}
        finally:
            conn.close()

    # -- adopting a model named outside the job API -------------------------

    def adopt_pending(self, config: Any) -> dict | None:
        from superlocalmemory.core.embedding_live import PENDING_ATTR

        target = getattr(config, PENDING_ATTR, None)
        if target is None:
            return None
        try:
            delattr(config, PENDING_ATTR)
        except AttributeError:
            pass
        try:
            return self.request_switch(target)
        except NoChange:
            return None
        except JobConflict as exc:
            self.notice = (f"{target.model_name} was named in the configuration while {exc}; "
                           "it was not queued. Switch to it when this one is done: "
                           f"slm embedder switch {target.model_name} --dimension {target.dimension}")
            logger.warning("%s", self.notice)
        except Exception as exc:
            logger.error("could not queue a re-index to %s: %s", target.model_name, exc)
        return None

    # -- the thread ---------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn = self._conn()
                try:
                    job = active_job(conn)
                    busy = job is None and self._housekeeping(conn)
                finally:
                    conn.close()
                if job is not None:
                    self._run(job)
                    continue
                if busy:
                    time.sleep(steps.PAUSE_S or 0.02)
                    continue
            except Exception as exc:  # the thread must outlive any one bad pass
                logger.error("embedding re-index runner pass failed: %s", exc)
            self._wake.wait(_IDLE_POLL_S)
            self._wake.clear()

    def _housekeeping(self, conn: Any) -> bool:
        """Between jobs: drop a replaced space, clear spent column twins, purge."""
        if sp.table_exists(conn, sp.TRASH_VEC):
            with steps.write_txn(conn, self.db_path):
                sp.drop_vec(conn, sp.TRASH_VEC)
            return True
        latest = latest_job(conn)
        stats = json.loads(latest["stats"]) if latest and latest.get("stats") else {}
        clear = stats.get("clear")
        if isinstance(clear, dict):
            with steps.write_txn(conn, self.db_path):
                cursor = slots.clear_next_chunk(conn, int(clear.get("cursor") or 0))
                stats["clear"] = "done" if cursor is None else {"cursor": cursor}
                update_job(conn, latest["job_id"], stats=json.dumps(stats))
            return True
        if time.monotonic() - self._last_purge > _PURGE_EVERY_S:
            self._last_purge = time.monotonic()
            if sp.table_exists(conn, sp.PURGE):
                with steps.write_txn(conn, self.db_path):
                    sp.purge_pending(conn)
        return False

    def _key_for(self, job: dict) -> str:
        role = secrets.TARGET if job["kind"] == "switch" else secrets.PREVIOUS
        return secrets.get(self.data_root, role)

    def _fail(self, job: dict, message: str, state: str = "failed") -> None:
        conn = self._conn()
        try:
            current = get_job(conn, job["job_id"]) or job
            stats = json.loads(current["stats"]) if current.get("stats") else {}
            stats["clear"] = {"cursor": 0}  # the column twins hold part of the new space
            with steps.write_txn(conn, self.db_path):
                sp.drop_staging(conn)
                update_job(conn, job["job_id"], state=state, error=message,
                           finished_at=time.time(), stats=json.dumps(stats))
        finally:
            conn.close()
        if job["kind"] == "switch":
            secrets.drop(self.data_root, secrets.TARGET)
        logger.error("embedding re-index job %s %s: %s; the previous embedding space "
                     "stays active", job["job_id"], state, message)

    def _run(self, job: dict) -> None:
        from superlocalmemory.core.embedding_reindex_activate import ActivationFailed, activate_job

        target = sp.config_from_public(job["to_config"], self._key_for(job))
        embedder = self._prebuilt.pop(job["job_id"], None)
        handed_over = False
        try:
            if embedder is None:
                embedder = self._load_embedder(target)
            if embedder is None:
                raise steps.StepFailed(f"the model {target.model_name} could not be started")
            job = self._prepare(job, embedder, target.dimension)
            if job["state"] == "running":
                job = self._bulk(job, embedder, target.dimension)
            if job["state"] in ("catching_up", "ready"):
                job, handed_over = self._activate(job, embedder, target)
        except _Cancelled:
            self._fail(job, "cancelled", state="cancelled")
        except (steps.StepFailed, ActivationFailed) as exc:
            if not isinstance(exc, ActivationFailed):
                self._fail(job, str(exc))
        except Exception as exc:
            logger.exception("embedding re-index job %s stopped", job["job_id"])
            if self._stop.is_set():
                return  # shutting down: the job resumes on the next start
            self._fail(job, f"unexpected error: {exc}")
        finally:
            if embedder is not None and not handed_over:
                steps.close_embedder(embedder)

    def _load_embedder(self, target: Any) -> Any | None:
        """Start the model, and ask it one question so it is really in memory, under the RAM reservation.

        Only one heavy job runs at a time across the daemon, the picture worker and the
        command line: a second one waits here. The reservation covers the load only, so a
        long re-index does not shut other model loads out for hours.
        """
        from superlocalmemory.core.ram_lock import ram_reservation
        from superlocalmemory.runtimes.media_models import load_mb_for

        try:
            with ram_reservation("embedding-reindex-load", required_mb=load_mb_for(target.model_name),
                                 timeout_s=LOAD_WAIT_S):
                embedder = steps.build_embedder(target)
                if embedder is not None:
                    steps.probe(embedder, target.dimension)
                return embedder
        except steps.StepFailed:
            raise
        except RuntimeError as exc:
            raise steps.StepFailed(
                "there was not enough free memory to load the model right now "
                f"(another large job may be running); try again later ({exc})") from exc

    def _activate(self, job: dict, embedder: Any, target: Any) -> tuple[dict, bool]:
        from superlocalmemory.core.embedding_reindex_activate import activate_job

        # The pause is held across every try: a background unit admitted during a
        # backoff would be in flight again at the next try (measured: one
        # materialization outlasts a whole try on a 22k-fact store).
        backoff = 1.0
        with self._background_paused():
            for _attempt in range(20):
                if job["state"] == "ready" and int(job["job_id"]) not in self.clean_marks:
                    job = {**job, "state": "catching_up"}  # resumed: no clean diff yet
                if job["state"] == "catching_up":
                    job = self._catch_up(job, embedder, target.dimension)
                if job["state"] != "ready":
                    break
                self._wait_for_quiet()
                self._catch_up_logged(job, embedder, target.dimension)
                job = activate_job(self, job, embedder, target)
                if job["state"] == "activated":
                    return job, True
                if job["state"] == "ready":  # requests did not drain: every try holds them
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, _MAX_BACKOFF_S)
        return job, False

    def _background_paused(self):
        from contextlib import nullcontext

        if self.app_state is None:
            return nullcontext()
        from superlocalmemory.server.profile_runtime import get_profile_runtime

        return get_profile_runtime(self.app_state).pausing_background()

    def _wait_for_quiet(self) -> None:
        """Enter the swap when nothing is in flight, so its drain holds no one.

        The swap waits for admitted requests and holds new ones meanwhile; under
        steady load a request is always in flight, and each try that times out
        stalls everyone for the drain timeout. A quiet moment makes the drain
        immediate. Bounded: after _QUIET_WAIT_S it tries anyway.
        """
        if self.app_state is None:
            return
        from superlocalmemory.core.recall_gate import in_flight
        from superlocalmemory.server.profile_runtime import get_profile_runtime

        runtime = get_profile_runtime(self.app_state)
        started = time.monotonic()
        polls = ops_zero = recalls_zero = 0
        while time.monotonic() - started < _QUIET_WAIT_S and not self._stop.is_set():
            ops, recalls = runtime.active_operations, in_flight()
            polls += 1
            ops_zero += ops == 0
            recalls_zero += recalls == 0
            if ops == 0 and recalls == 0:
                self.quiet = {"waited_s": round(time.monotonic() - started, 2), "found": True}
                return
            time.sleep(0.02)
        self.quiet = {"waited_s": round(time.monotonic() - started, 2), "found": False,
                      "polls": polls, "no_requests": ops_zero, "no_recalls": recalls_zero}
        logger.info("re-index: no quiet moment in %.0f s (%s); swapping anyway",
                    _QUIET_WAIT_S, self.quiet)

    def _check(self) -> None:
        if self._cancel.is_set():
            self._cancel.clear()
            raise _Cancelled()
        if self._stop.is_set():
            raise RuntimeError("daemon stopping")

    def _prepare(self, job: dict, embedder: Any, dimension: int) -> dict:
        conn = self._conn()
        try:
            if job["state"] == "queued":
                steps.probe(embedder, dimension)
                with steps.write_txn(conn, self.db_path):
                    sp.drop_staging(conn)
                    sp.ensure_side_tables(conn)
                    sp.create_vec(conn, sp.NEXT_VEC, dimension)
                    slots.ensure_next_columns(conn)
                    change_log.start(conn)
                    update_job(conn, job["job_id"], state="running", started_at=time.time(),
                               total=count_facts(conn), cursor=0, done=0, next_rowid=0,
                               attempts=int(job.get("attempts") or 0) + 1)
            elif not sp.table_exists(conn, sp.NEXT_VEC):
                with steps.write_txn(conn, self.db_path):  # staging lost: start over
                    sp.ensure_side_tables(conn)
                    sp.create_vec(conn, sp.NEXT_VEC, dimension)
                    slots.ensure_next_columns(conn)
                    change_log.start(conn)
                    conn.execute(f"DELETE FROM {sp.NEXT_MAP}")
                    update_job(conn, job["job_id"], state="running", cursor=0, done=0)
            else:
                stats = json.loads(job["stats"]) if job.get("stats") else {}
                stats["resumed_at_done"] = int(job.get("done") or 0)
                with steps.write_txn(conn, self.db_path):
                    change_log.start(conn)  # a job begun before 4.1.22 GA had none
                    update_job(conn, job["job_id"], started_at=time.time(),
                               attempts=int(job.get("attempts") or 0) + 1,
                               stats=json.dumps(stats))
            return get_job(conn, job["job_id"])
        finally:
            conn.close()

    def _stats(self, job: dict) -> steps.LockStats:
        return steps.LockStats(json.loads(job["stats"]) if job.get("stats") else None)

    def _save_stats(self, conn: Any, job: dict, stats: steps.LockStats, **extra: Any) -> None:
        current = json.loads(job["stats"]) if job.get("stats") else {}
        current.update(stats.summary())
        current.update(extra)
        with steps.write_txn(conn, self.db_path):
            update_job(conn, job["job_id"], stats=json.dumps(current))

    def _bulk(self, job: dict, embedder: Any, dimension: int) -> dict:
        conn = self._conn()
        stats = self._stats(job)
        rollback = job["kind"] == "rollback"
        try:
            cursor = int(job["cursor"])
            while True:
                self._check()
                facts = facts_after(conn, cursor, steps.BATCH)
                if not facts:
                    break
                steps.stage(conn, self.db_path, job, embedder, facts, dimension,
                            rollback=rollback, stats=stats, cursor=facts[-1][0],
                            catching_up=False)
                cursor = facts[-1][0]
                if len(stats.samples) % 20 == 0:
                    self._save_stats(conn, get_job(conn, job["job_id"]), stats)
            with steps.write_txn(conn, self.db_path):
                update_job(conn, job["job_id"], state="catching_up")
            self._save_stats(conn, get_job(conn, job["job_id"]), stats)
            return get_job(conn, job["job_id"])
        finally:
            conn.close()

    def _catch_up_logged(self, job: dict, embedder: Any, dimension: int) -> None:
        """Re-stage only what the change log recorded since the last clean mark."""
        job_id = int(job["job_id"])
        conn = self._conn()
        stats = self._stats(job)
        try:
            position = change_log.mark(conn)
            ids = [str(r[0]) for r in conn.execute(
                f"SELECT DISTINCT fact_id FROM {change_log.TABLE} WHERE seq > ?",
                (self.clean_marks.get(job_id, position),))]
            facts = []
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                marks = ",".join("?" for _ in chunk)
                facts += [(int(r[0]), str(r[1]), str(r[2]), r[3] or "") for r in conn.execute(
                    f"SELECT rowid, fact_id, profile_id, content FROM atomic_facts "
                    f"WHERE fact_id IN ({marks})", chunk)]
            for start in range(0, len(facts), steps.BATCH):
                steps.stage(conn, self.db_path, job, embedder, facts[start:start + steps.BATCH],
                            dimension, rollback=job["kind"] == "rollback", stats=stats,
                            cursor=None, catching_up=True)
            self.clean_marks[job_id] = position
        finally:
            conn.close()

    def _catch_up(self, job: dict, embedder: Any, dimension: int) -> dict:
        conn = self._conn()
        stats = self._stats(job)
        try:
            for _round in range(_CATCH_UP_ROUNDS):
                self._check()
                position = change_log.mark(conn)  # read BEFORE the diff it vouches for
                need, gone = staged_diff(conn)
                if gone:
                    with steps.write_txn(conn, self.db_path, stats):
                        drop_gone(conn, gone)
                if not need:
                    self.clean_marks[int(job["job_id"])] = position
                    break
                for start in range(0, len(need), steps.BATCH):
                    self._check()
                    steps.stage(conn, self.db_path, job, embedder, need[start:start + steps.BATCH],
                                dimension, rollback=job["kind"] == "rollback", stats=stats,
                                cursor=None, catching_up=True)
            with steps.write_txn(conn, self.db_path):
                update_job(conn, job["job_id"], state="ready")
            self._save_stats(conn, get_job(conn, job["job_id"]), stats)
            return get_job(conn, job["job_id"])
        finally:
            conn.close()


class _Cancelled(Exception):
    pass


__all__ = ["NoChange", "Refused", "ReindexRunner", "get_runner", "notify_pending",
           "pending_switch_message", "space_changed"]
