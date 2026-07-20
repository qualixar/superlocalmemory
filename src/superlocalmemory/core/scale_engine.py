# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Staged lifecycle for optional CozoDB and LanceDB projections.

SQLite and sqlite-vec remain the canonical store.  The Scale Engine is a
derived projection: it is prepared outside the active paths, checked against
the canonical store, and only then promoted.  This prevents an installed
optional dependency from mutating an existing user's data root at daemon
startup.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.storage.logical_edges import count_logical_edges, iter_logical_edges
from superlocalmemory.storage.sqlite_vectors import (
    CanonicalVectorError,
    count_canonical_vectors,
    iter_canonical_vectors,
    load_sqlite_vec_extension,
)


class ScaleEngineError(RuntimeError):
    """A scale projection cannot safely advance to its next lifecycle state."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ScaleEngineManager:
    """Prepare, verify, promote, and roll back derived scale projections.

    ``backend_factory`` is intentionally injectable.  It keeps lifecycle tests
    independent from native optional packages while production uses the real
    CozoDB and LanceDB wrappers.
    """

    MANIFEST_NAME = "scale-engine.json"
    SCHEMA_VERSION = 1
    LIFECYCLE_LOCK = "scale-engine.lifecycle.lock"
    PROMOTION_JOURNAL = "scale-engine.promotion.json"

    def __init__(
        self,
        config: Any,
        *,
        backend_factory: Callable[[Path, Path], tuple[Any, Any]] | None = None,
        profile_id: str | None = None,
    ) -> None:
        self.config = config
        self.data_dir = Path(getattr(config, "data_dir", None) or config.base_dir)
        self.db_path = Path(getattr(config, "db_path", None) or self.data_dir / "memory.db")
        self.profile_id = profile_id or getattr(config, "active_profile", "default")
        self._backend_factory = backend_factory or self._real_backend_factory

    @property
    def staging_root(self) -> Path:
        return self.data_dir / "scale-staging"

    @property
    def backup_root(self) -> Path:
        return self.data_dir / "scale-backups"

    @property
    def active_paths(self) -> tuple[Path, Path]:
        return self.data_dir / "cozo", self.data_dir / "lance"

    @property
    def lifecycle_lock_path(self) -> Path:
        return self.data_dir / self.LIFECYCLE_LOCK

    @property
    def promotion_journal_path(self) -> Path:
        return self.data_dir / self.PROMOTION_JOURNAL

    def status(self) -> dict[str, Any]:
        manifests: list[dict[str, Any]] = []
        if self.staging_root.exists():
            for path in sorted(self.staging_root.glob(f"*/{self.MANIFEST_NAME}")):
                try:
                    manifests.append(json.loads(path.read_text()))
                except (OSError, json.JSONDecodeError):
                    manifests.append({"stage_id": path.parent.name, "state": "corrupt"})
        backups = (
            sorted(p.name for p in self.backup_root.glob("*") if p.is_dir())
            if self.backup_root.exists()
            else []
        )
        paths_present = {
            "cozo": self.active_paths[0].exists(),
            "lance": self.active_paths[1].exists(),
        }
        state = getattr(self.config, "scale_engine_state", "local_core")
        legacy_projection_candidate = (
            state == "local_core"
            and all(paths_present.values())
            and not backups
            and not self.promotion_journal_path.exists()
            and self._has_legacy_projection_layout()
        )
        runtime = self._runtime_backend_status()
        return {
            "state": state,
            # This command reads persisted state, not the live daemon. Never
            # turn a last-known backend row into a present-tense routing claim.
            "active": {"cozo": False, "lance": False},
            "last_daemon_observation": runtime,
            "paths_present": paths_present,
            "retrieval_routing": (
                "daemon_runtime_check_required" if state == "promoted" else "canonical_sqlite"
            ),
            "legacy_projection_candidate": legacy_projection_candidate,
            "legacy_candidate_requires_confirmation": legacy_projection_candidate,
            "migration_repair_required": (
                self.promotion_journal_path.exists()
                or any(
                    manifest.get("state") == "promoted" and state != "promoted"
                    for manifest in manifests
                )
                or (
                    state == "promoted"
                    and any(
                        manifest.get("state") in {"prepared", "verified"}
                        for manifest in manifests
                    )
                )
            ),
            "stages": manifests,
            "backups": backups,
        }

    def adopt_legacy_projection(self) -> dict[str, Any] | None:
        """Safely adopt a v3.5-era projection into the staged lifecycle.

        A legacy projection proves only that an older runtime created files.
        It cannot establish parity with today's canonical SQLite database.
        Rebuild a fresh stage, verify it while the canonical database is
        stable, and promote it atomically; the legacy directories become the
        explicit rollback copy.
        """
        if not self.status()["legacy_projection_candidate"]:
            return None
        lock_path = self._acquire_lifecycle_lock()
        prepared: dict[str, Any] | None = None
        try:
            # Re-check inside the lock. A concurrent command may have
            # completed promotion while this caller waited to acquire it.
            current_status = self.status()
            if not current_status["legacy_projection_candidate"]:
                return None
            retry_payloads = [
                manifest["stage_id"]
                for manifest in current_status["stages"]
                if manifest.get("state") in {"prepared", "verified"}
                and manifest.get("stage_id")
            ]
            prepared = self._prepare()
            self._verify(prepared["stage_id"])
            promoted = self._promote(prepared["stage_id"])
            retired: list[str] = []
            retirement_failures: dict[str, str] = {}
            for stage_id in retry_payloads:
                try:
                    self._retire_superseded_stage(stage_id)
                    retired.append(stage_id)
                except Exception as cleanup_exc:
                    retirement_failures[stage_id] = str(cleanup_exc)
            return {
                **promoted,
                "retired_stages": retired,
                "retirement_failures": retirement_failures,
            }
        except Exception as exc:
            # A failed adoption must leave the canonical path selected.  The
            # manifest is retained for inspection, but its replaceable Cozo
            # and Lance payloads are retired so repeated retries cannot grow
            # the data root without bound.
            retirement_error: Exception | None = None
            unresolved_promotion = self.promotion_journal_path.exists()
            if prepared is not None and not unresolved_promotion:
                try:
                    self._retire_rejected_stage(prepared["stage_id"], exc)
                except Exception as cleanup_exc:
                    retirement_error = cleanup_exc
            self.config.scale_engine_state = (
                "verified" if unresolved_promotion else "local_core"
            )
            self.config.graph_backend = "auto"
            self.config.vector_backend = "auto"
            self._save_config()
            if retirement_error is not None:
                raise ScaleEngineError(
                    f"{exc}; rejected stage retirement failed: {retirement_error}"
                ) from exc
            raise
        finally:
            self._release_lifecycle_lock(lock_path)

    def _retire_rejected_stage(self, stage_id: str, error: Exception) -> None:
        """Keep rejection evidence while removing replaceable projection data."""
        self._retire_stage_payload(
            stage_id,
            {
                "state": "rejected",
                "rejected_at": _utc_now(),
                "failure": f"{type(error).__name__}: {error}",
            },
        )

    def _retire_superseded_stage(self, stage_id: str) -> None:
        """Retain an old retry manifest after a newer projection is promoted."""
        self._retire_stage_payload(
            stage_id,
            {"state": "superseded", "superseded_at": _utc_now()},
        )

    def _retire_stage_payload(
        self, stage_id: str, manifest_updates: dict[str, Any]
    ) -> None:
        """Remove derived stage bytes while retaining its durable manifest."""
        stage_dir, manifest = self._load_stage(stage_id)
        for payload in (stage_dir / "cozo", stage_dir / "lance"):
            if payload.exists():
                shutil.rmtree(payload)
        self._fsync_directory(stage_dir)
        manifest.update(manifest_updates)
        self._write_manifest(stage_dir, manifest)

    def prepare(self) -> dict[str, Any]:
        """Build a new projection in a private staging directory."""
        lock_path = self._acquire_lifecycle_lock()
        try:
            self._recover_interrupted_promotion()
            return self._prepare()
        finally:
            self._release_lifecycle_lock(lock_path)

    def _prepare(self) -> dict[str, Any]:
        """Build a new projection while the caller owns the lifecycle lock."""
        self._require_default_profile()
        self._require_canonical_db()
        stage_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        stage_dir = self.staging_root / stage_id
        cozo_dir, lance_dir = stage_dir / "cozo", stage_dir / "lance"
        self._mkdir_durable(stage_dir, exist_ok=False)
        cozo = lance = None
        try:
            cozo, lance = self._backend_factory(cozo_dir, lance_dir)
            with self._readonly_connection() as conn:
                cozo.bulk_import_from_sqlite(conn, self.profile_id)
                lance.bulk_import_from_sqlite(conn, self.profile_id)
                canonical = self._canonical_counts(conn)
                source_fingerprint = self._projection_fingerprint(conn, canonical)
            observed = self._observed_counts(cozo, lance)
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "stage_id": stage_id,
                "state": "prepared",
                "created_at": _utc_now(),
                "profile_id": self.profile_id,
                "canonical": canonical,
                "observed": observed,
                "source_fingerprint": source_fingerprint,
            }
            self._write_manifest(stage_dir, manifest)
            self.config.scale_engine_state = "prepared"
            self._save_config()
            return manifest
        except CanonicalVectorError as exc:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ScaleEngineError(f"canonical vector projection failed: {exc}") from exc
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        finally:
            self._close(cozo)
            self._close(lance)

    def verify(self, stage_id: str) -> dict[str, Any]:
        """Prove a staged projection matches the current canonical SQLite data."""
        lock_path = self._acquire_lifecycle_lock()
        try:
            self._recover_interrupted_promotion()
            return self._verify(stage_id)
        except CanonicalVectorError as exc:
            raise ScaleEngineError(f"canonical vector projection failed: {exc}") from exc
        finally:
            self._release_lifecycle_lock(lock_path)

    def _verify(self, stage_id: str) -> dict[str, Any]:
        """Verify while the caller owns the lifecycle lock."""
        stage_dir, manifest = self._load_stage(stage_id)
        self._validate_manifest(manifest, state="prepared")
        self._require_default_profile()
        cozo = lance = None
        try:
            cozo, lance = self._backend_factory(stage_dir / "cozo", stage_dir / "lance")
            with self._readonly_connection() as conn:
                canonical = self._canonical_counts(conn)
                source_fingerprint = self._projection_fingerprint(conn, canonical)
            observed = self._observed_counts(cozo, lance)
            if manifest["source_fingerprint"] != source_fingerprint:
                raise ScaleEngineError(
                    "canonical SQLite changed after preparation; prepare a new stage"
                )
            if canonical != manifest["canonical"] or observed != canonical:
                raise ScaleEngineError(
                    f"projection parity failed: canonical={canonical}, observed={observed}"
                )
            manifest.update({"state": "verified", "verified_at": _utc_now(), "observed": observed})
            self._write_manifest(stage_dir, manifest)
            self.config.scale_engine_state = "verified"
            self._save_config()
            return manifest
        finally:
            self._close(cozo)
            self._close(lance)

    def promote(self, stage_id: str) -> dict[str, Any]:
        """Move a verified stage into active paths, preserving a rollback copy."""
        lock_path = self._acquire_lifecycle_lock()
        try:
            self._recover_interrupted_promotion()
            return self._promote(stage_id)
        except CanonicalVectorError as exc:
            raise ScaleEngineError(f"canonical vector projection failed: {exc}") from exc
        finally:
            self._release_lifecycle_lock(lock_path)

    def _promote(self, stage_id: str) -> dict[str, Any]:
        """Promote while the caller owns the lifecycle lock."""
        stage_dir, manifest = self._load_stage(stage_id)
        self._validate_manifest(manifest, state="verified")
        staged = (stage_dir / "cozo", stage_dir / "lance")
        if not all(path.exists() for path in staged):
            raise ScaleEngineError("verified stage is incomplete; prepare a new stage")
        backup_dir = self.backup_root / f"{stage_id}-{uuid.uuid4().hex[:6]}"
        active = self.active_paths
        gate = sqlite3.connect(self.db_path, timeout=30)
        try:
            # The stage was built from a point-in-time SQLite snapshot. Hold a
            # short writer fence for the final fingerprint check and directory
            # swap so no successful promotion can trail a canonical write.
            gate.execute("BEGIN IMMEDIATE")
            canonical = self._canonical_counts(gate)
            if manifest["source_fingerprint"] != self._projection_fingerprint(gate, canonical):
                raise ScaleEngineError(
                    "canonical SQLite changed after verification; prepare a new stage"
                )
            self._mkdir_durable(self.backup_root)
            journal = {
                "schema_version": self.SCHEMA_VERSION,
                "operation": "promotion",
                "state": "intent",
                "stage_id": stage_id,
                "backup_id": backup_dir.name,
                "moves": [],
            }
            self._write_promotion_journal(journal)
            self._mkdir_durable(backup_dir, exist_ok=False)
            for name, source, destination in zip(("cozo", "lance"), active, staged):
                if source.exists():
                    target = backup_dir / name
                    move = {"name": name, "kind": "active_to_backup", "state": "intent"}
                    journal["moves"].append(move)
                    self._write_promotion_journal(journal)
                    self._replace_durable(source, target)
                    move["state"] = "complete"
                    self._write_promotion_journal(journal)
                move = {"name": name, "kind": "stage_to_active", "state": "intent"}
                journal["moves"].append(move)
                self._write_promotion_journal(journal)
                self._replace_durable(destination, source)
                move["state"] = "complete"
                self._write_promotion_journal(journal)
            manifest.update(
                {
                    "state": "promoted",
                    "promoted_at": _utc_now(),
                    "backup_id": backup_dir.name,
                }
            )
            self._write_manifest(stage_dir, manifest)
            self.config.scale_engine_state = "promoted"
            self.config.graph_backend = "cozo"
            self.config.vector_backend = "lancedb"
            self._save_config()
            journal["state"] = "committed"
            self._write_promotion_journal(journal)
            self.promotion_journal_path.unlink(missing_ok=True)
            gate.rollback()
            return manifest
        except Exception as exc:
            try:
                gate.rollback()
            except sqlite3.Error:
                pass
            try:
                recovery = self._recover_interrupted_promotion()
            except Exception as recovery_error:
                raise ScaleEngineError(
                    f"promotion interrupted; automatic recovery needs repair: {recovery_error}"
                ) from exc
            if recovery == "finalized_committed_promotion":
                _, recovered_manifest = self._load_stage(stage_id)
                return recovered_manifest
            raise ScaleEngineError(f"promotion rolled back: {exc}") from exc
        finally:
            gate.close()

    def rollback(self, backup_id: str) -> dict[str, Any]:
        """Restore an explicitly named pre-promotion backup."""
        lock_path = self._acquire_lifecycle_lock()
        try:
            self._recover_interrupted_promotion()
            return self._rollback(backup_id)
        finally:
            self._release_lifecycle_lock(lock_path)

    def _rollback(self, backup_id: str) -> dict[str, Any]:
        """Roll back while the caller owns the lifecycle lock."""
        backup_dir = self.backup_root / backup_id
        if not backup_dir.is_dir():
            raise ScaleEngineError(f"backup does not exist: {backup_id}")
        backup_paths = (backup_dir / "cozo", backup_dir / "lance")
        active = self.active_paths
        displaced = self.backup_root / f"rollback-displaced-{uuid.uuid4().hex[:8]}"
        try:
            journal = {
                "schema_version": self.SCHEMA_VERSION,
                "operation": "rollback",
                "state": "intent",
                "backup_id": backup_id,
                "displaced_id": displaced.name,
                "moves": [],
            }
            self._write_promotion_journal(journal)
            for name, source, target in zip(("cozo", "lance"), active, backup_paths):
                if source.exists():
                    self._mkdir_durable(displaced)
                    move = {"name": name, "kind": "active_to_displaced", "state": "intent"}
                    journal["moves"].append(move)
                    self._write_promotion_journal(journal)
                    self._replace_durable(source, displaced / name)
                    move["state"] = "complete"
                    self._write_promotion_journal(journal)
                if target.exists():
                    move = {"name": name, "kind": "backup_to_active", "state": "intent"}
                    journal["moves"].append(move)
                    self._write_promotion_journal(journal)
                    self._replace_durable(target, source)
                    move["state"] = "complete"
                    self._write_promotion_journal(journal)
            self.config.scale_engine_state = "local_core"
            self.config.graph_backend = "auto"
            self.config.vector_backend = "auto"
            self._save_config()
            journal["state"] = "committed"
            self._write_promotion_journal(journal)
            self.promotion_journal_path.unlink(missing_ok=True)
            return {
                "state": "local_core",
                "restored_backup": backup_id,
                "displaced": displaced.name,
            }
        except Exception as exc:
            try:
                self._recover_interrupted_promotion()
            except ScaleEngineError as recovery_error:
                raise ScaleEngineError(
                    f"rollback interrupted; automatic recovery needs repair: {recovery_error}"
                ) from exc
            raise ScaleEngineError(f"rollback recovered: {exc}") from exc

    def _real_backend_factory(self, cozo_dir: Path, lance_dir: Path) -> tuple[Any, Any]:
        from superlocalmemory.graph.cozo_backend import CozoDBGraphBackend
        from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend
        # v3.7.6 (#72): promote the configured embedding width into the new
        # LanceDB store so a 1024d (or other) custom endpoint survives promotion.
        dimension = getattr(
            getattr(self.config, "embedding", None), "dimension", None
        )
        return (
            CozoDBGraphBackend(str(cozo_dir / "graph")),
            LanceDBVectorBackend(str(lance_dir), dimension=dimension),
        )

    def _readonly_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        load_sqlite_vec_extension(conn)
        return conn

    def _canonical_counts(self, conn: sqlite3.Connection) -> dict[str, int]:
        nodes = conn.execute(
            "SELECT COUNT(*) FROM canonical_entities WHERE profile_id=?",
            (self.profile_id,),
        ).fetchone()[0]
        edges = count_logical_edges(conn, self.profile_id)
        vectors = count_canonical_vectors(conn, self.profile_id)
        return {"entities": int(nodes), "edges": int(edges), "vectors": int(vectors)}

    def _observed_counts(self, cozo: Any, lance: Any) -> dict[str, int]:
        graph = cozo.health_check()
        vector = lance.health_check()
        if graph.get("status") != "active" or vector.get("status") != "active":
            raise ScaleEngineError(f"projection health failed: cozo={graph}, lancedb={vector}")
        return {
            "entities": int(graph["entities"]),
            "edges": int(graph["edges"]),
            "vectors": int(vector["vectors"]),
        }

    def _projection_fingerprint(
        self, conn: sqlite3.Connection, counts: dict[str, int]
    ) -> str:
        """Hash the exact projection source rows inside one SQLite snapshot."""
        digest = hashlib.sha256()
        digest.update(json.dumps(counts, sort_keys=True).encode())
        tables = (
            (
                "canonical_entities",
                "entity_id, canonical_name, entity_type, first_seen, last_seen, "
                "fact_count, profile_id",
                "entity_id",
            ),
            ("atomic_facts", "fact_id, canonical_entities_json, lifecycle, profile_id", "fact_id"),
        )
        for table, columns, ordering in tables:
            try:
                rows = conn.execute(
                    f"SELECT {columns} FROM {table} WHERE profile_id=? ORDER BY {ordering}",
                    (self.profile_id,),
                )
                for row in rows:
                    self._digest_row(digest, table, row)
            except sqlite3.OperationalError as exc:
                raise ScaleEngineError(f"canonical SQLite missing required {table} table") from exc
        try:
            for row in iter_logical_edges(conn, self.profile_id):
                self._digest_row(digest, "graph_edges", row)
        except sqlite3.OperationalError as exc:
            raise ScaleEngineError("canonical SQLite missing required graph_edges table") from exc
        for row in iter_canonical_vectors(conn, self.profile_id):
            self._digest_row(digest, "fact_embeddings", row)
        return digest.hexdigest()

    @staticmethod
    def _digest_row(digest: Any, table: str, row: Any) -> None:
        digest.update(table.encode())
        digest.update(json.dumps(list(row), default=str, separators=(",", ":")).encode())

    def _has_legacy_projection_layout(self) -> bool:
        cozo, lance = self.active_paths
        return (cozo / "graph").is_dir() and (lance / "embeddings.lance").exists()

    def _runtime_backend_status(self) -> dict[str, str]:
        status = {"cozo": "unknown", "lance": "unknown"}
        try:
            with self._readonly_connection() as conn:
                rows = conn.execute(
                    "SELECT backend_name, status FROM backend_status "
                    "WHERE backend_name IN ('cozo', 'lancedb')"
                )
                for name, value in rows:
                    if name == "cozo":
                        status["cozo"] = str(value)
                    elif name == "lancedb":
                        status["lance"] = str(value)
        except sqlite3.OperationalError:
            pass
        return status

    def _load_stage(self, stage_id: str) -> tuple[Path, dict[str, Any]]:
        stage_dir = self.staging_root / stage_id
        try:
            manifest = json.loads((stage_dir / self.MANIFEST_NAME).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ScaleEngineError(f"invalid scale stage: {stage_id}") from exc
        return stage_dir, manifest

    def _write_manifest(self, stage_dir: Path, manifest: dict[str, Any]) -> None:
        target = stage_dir / self.MANIFEST_NAME
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)

    def _validate_manifest(self, manifest: dict[str, Any], *, state: str) -> None:
        if manifest.get("schema_version") != self.SCHEMA_VERSION or manifest.get("state") != state:
            raise ScaleEngineError(f"stage must be {state}, got {manifest.get('state')!r}")

    def _require_default_profile(self) -> None:
        if self.profile_id != "default":
            raise ScaleEngineError(
                "Scale Engine promotion currently supports the default profile only"
            )

    def _require_canonical_db(self) -> None:
        if not self.db_path.exists():
            raise ScaleEngineError(f"canonical SQLite database not found: {self.db_path}")

    def _acquire_lifecycle_lock(self) -> Path:
        """Serialize every mutating lifecycle command across processes."""
        lock_path = self.lifecycle_lock_path
        descriptor = None
        for attempt in range(2):
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as exc:
                if attempt == 0 and self._clear_dead_legacy_adoption_lock(lock_path):
                    continue
                raise ScaleEngineError(
                    "Scale Engine lifecycle operation already in progress; retry after it completes"
                ) from exc
        if descriptor is None:
            raise ScaleEngineError("could not acquire Scale Engine lifecycle lock")
        try:
            with os.fdopen(descriptor, "w") as lock_file:
                json.dump({"pid": os.getpid(), "started_at": _utc_now()}, lock_file)
        except Exception:
            lock_path.unlink(missing_ok=True)
            raise
        return lock_path

    @staticmethod
    def _release_lifecycle_lock(lock_path: Path) -> None:
        lock_path.unlink(missing_ok=True)

    @staticmethod
    def _clear_dead_legacy_adoption_lock(lock_path: Path) -> bool:
        """Recover only a lock whose recorded process no longer exists."""
        try:
            owner = json.loads(lock_path.read_text())
            pid = owner.get("pid")
            if not isinstance(pid, int) or pid <= 0:
                return False
            os.kill(pid, 0)
        except ProcessLookupError:
            lock_path.unlink(missing_ok=True)
            return True
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return False

    def recover_interrupted_promotion(self) -> str | None:
        """Recover a durable promotion journal before opening projection paths."""
        lock_path = self._acquire_lifecycle_lock()
        try:
            return self._recover_interrupted_promotion()
        finally:
            self._release_lifecycle_lock(lock_path)

    def _recover_interrupted_promotion(self) -> str | None:
        """Finalize or reverse an interrupted directory swap under lifecycle lock."""
        if not self.promotion_journal_path.exists():
            return None
        try:
            journal = json.loads(self.promotion_journal_path.read_text())
            backup_id = str(journal["backup_id"])
            state = str(journal["state"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ScaleEngineError("invalid promotion journal; manual repair required") from exc
        operation = str(journal.get("operation", "promotion"))
        if state == "committed":
            if operation == "promotion":
                self.config.scale_engine_state = "promoted"
                self.config.graph_backend = "cozo"
                self.config.vector_backend = "lancedb"
            elif operation == "rollback":
                self.config.scale_engine_state = "local_core"
                self.config.graph_backend = "auto"
                self.config.vector_backend = "auto"
            else:
                raise ScaleEngineError(f"unknown journal operation: {operation!r}")
            self._save_config()
            self.promotion_journal_path.unlink(missing_ok=True)
            return f"finalized_committed_{operation}"
        if state != "intent":
            raise ScaleEngineError(f"unknown promotion journal state: {state!r}")
        backup_dir = self.backup_root / backup_id
        active = dict(zip(("cozo", "lance"), self.active_paths))
        stage_dir = self.staging_root / str(journal.get("stage_id", ""))
        displaced_dir = self.backup_root / str(journal.get("displaced_id", ""))
        moves = journal.get("moves")
        if not isinstance(moves, list):
            # Compatibility for journals written by the initial v3.7.3
            # candidate before per-rename intents existed.
            moves = [
                {"name": name, "kind": "active_to_backup", "state": "complete"}
                for name in journal.get("moved_active", [])
            ] + [
                {"name": name, "kind": "stage_to_active", "state": "complete"}
                for name in journal.get("moved_stage", [])
            ]
        for move in reversed(moves):
            self._reverse_journal_move(move, active, stage_dir, backup_dir, displaced_dir)
        if operation == "promotion":
            if backup_dir.exists() and not any(backup_dir.iterdir()):
                backup_dir.rmdir()
            self.config.scale_engine_state = "local_core"
            self.config.graph_backend = "auto"
            self.config.vector_backend = "auto"
        elif operation == "rollback":
            if displaced_dir.exists() and not any(displaced_dir.iterdir()):
                displaced_dir.rmdir()
            self.config.scale_engine_state = "promoted"
            self.config.graph_backend = "cozo"
            self.config.vector_backend = "lancedb"
        else:
            raise ScaleEngineError(f"unknown journal operation: {operation!r}")
        self._save_config()
        self.promotion_journal_path.unlink(missing_ok=True)
        return f"reversed_interrupted_{operation}"

    @staticmethod
    def _reverse_journal_move(
        move: Any,
        active: dict[str, Path],
        stage_dir: Path,
        backup_dir: Path,
        displaced_dir: Path,
    ) -> None:
        """Reverse one planned rename based on actual paths, not journal timing."""
        if not isinstance(move, dict):
            raise ScaleEngineError("invalid promotion journal move")
        name = move.get("name")
        kind = move.get("kind")
        active_path = active.get(name)
        if active_path is None:
            raise ScaleEngineError(f"invalid promotion journal backend: {name!r}")
        if kind == "stage_to_active":
            source, target = active_path, stage_dir / name
        elif kind == "active_to_backup":
            source, target = backup_dir / name, active_path
        elif kind == "active_to_displaced":
            source, target = displaced_dir / name, active_path
        elif kind == "backup_to_active":
            source, target = active_path, backup_dir / name
        else:
            raise ScaleEngineError(f"invalid promotion journal move kind: {kind!r}")
        if source.exists() and not target.exists():
            ScaleEngineManager._replace_durable(source, target)
        elif target.exists() and not source.exists():
            return
        else:
            raise ScaleEngineError(f"cannot safely reconcile {kind} for {name}")

    def _write_promotion_journal(self, journal: dict[str, Any]) -> None:
        self._write_json_durable(self.promotion_journal_path, journal)

    @staticmethod
    def _write_json_durable(target: Path, payload: dict[str, Any]) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        ScaleEngineManager._fsync_directory(target.parent)

    @staticmethod
    def _replace_durable(source: Path, target: Path) -> None:
        """Rename a projection path and persist both directory entries."""
        os.replace(source, target)
        ScaleEngineManager._fsync_directory(source.parent)
        if target.parent != source.parent:
            ScaleEngineManager._fsync_directory(target.parent)

    @staticmethod
    def _mkdir_durable(path: Path, *, exist_ok: bool = True) -> None:
        """Create a directory and persist every new parent entry before rename."""
        missing: list[Path] = []
        ancestor = path
        while not ancestor.exists():
            missing.append(ancestor)
            ancestor = ancestor.parent
        path.mkdir(parents=True, exist_ok=exist_ok)
        for created in reversed(missing):
            ScaleEngineManager._fsync_directory(created.parent)

    @staticmethod
    def _fsync_directory(directory_path: Path) -> None:
        """Best-effort directory-entry durability across local filesystems."""
        try:
            directory = os.open(directory_path, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # The file itself is already durable where directory fsync is not
            # supported by the local filesystem (notably some Windows setups).
            pass

    def _save_config(self) -> None:
        save = getattr(self.config, "save", None)
        if callable(save):
            save()

    @staticmethod
    def _close(resource: Any) -> None:
        if resource is not None:
            close = getattr(resource, "close", None)
            if callable(close):
                close()
