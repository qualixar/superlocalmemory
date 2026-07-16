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

    def __init__(
        self,
        config: Any,
        *,
        backend_factory: Callable[[Path, Path], tuple[Any, Any]] | None = None,
    ) -> None:
        self.config = config
        self.data_dir = Path(getattr(config, "data_dir", None) or config.base_dir)
        self.db_path = Path(getattr(config, "db_path", None) or self.data_dir / "memory.db")
        self.profile_id = getattr(config, "active_profile", "default")
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

    def status(self) -> dict[str, Any]:
        manifests: list[dict[str, Any]] = []
        if self.staging_root.exists():
            for path in sorted(self.staging_root.glob(f"*/{self.MANIFEST_NAME}")):
                try:
                    manifests.append(json.loads(path.read_text()))
                except (OSError, json.JSONDecodeError):
                    manifests.append({"stage_id": path.parent.name, "state": "corrupt"})
        return {
            "state": getattr(self.config, "scale_engine_state", "local_core"),
            "active": {"cozo": self.active_paths[0].exists(), "lance": self.active_paths[1].exists()},
            "stages": manifests,
            "backups": sorted(p.name for p in self.backup_root.glob("*") if p.is_dir()) if self.backup_root.exists() else [],
        }

    def prepare(self) -> dict[str, Any]:
        """Build a new projection in a private staging directory."""
        self._require_default_profile()
        self._require_canonical_db()
        stage_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        stage_dir = self.staging_root / stage_id
        cozo_dir, lance_dir = stage_dir / "cozo", stage_dir / "lance"
        stage_dir.mkdir(parents=True, exist_ok=False)
        cozo = lance = None
        try:
            cozo, lance = self._backend_factory(cozo_dir, lance_dir)
            with self._readonly_connection() as conn:
                cozo.bulk_import_from_sqlite(conn, self.profile_id)
                lance.bulk_import_from_sqlite(conn)
                canonical = self._canonical_counts(conn)
            observed = self._observed_counts(cozo, lance)
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "stage_id": stage_id,
                "state": "prepared",
                "created_at": _utc_now(),
                "profile_id": self.profile_id,
                "canonical": canonical,
                "observed": observed,
                "source_fingerprint": self._source_fingerprint(canonical),
            }
            self._write_manifest(stage_dir, manifest)
            self.config.scale_engine_state = "prepared"
            self._save_config()
            return manifest
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        finally:
            self._close(cozo)
            self._close(lance)

    def verify(self, stage_id: str) -> dict[str, Any]:
        """Prove a staged projection matches the current canonical SQLite data."""
        stage_dir, manifest = self._load_stage(stage_id)
        self._validate_manifest(manifest, state="prepared")
        self._require_default_profile()
        cozo = lance = None
        try:
            cozo, lance = self._backend_factory(stage_dir / "cozo", stage_dir / "lance")
            with self._readonly_connection() as conn:
                canonical = self._canonical_counts(conn)
            observed = self._observed_counts(cozo, lance)
            if manifest["source_fingerprint"] != self._source_fingerprint(canonical):
                raise ScaleEngineError("canonical SQLite changed after preparation; prepare a new stage")
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
        stage_dir, manifest = self._load_stage(stage_id)
        self._validate_manifest(manifest, state="verified")
        staged = (stage_dir / "cozo", stage_dir / "lance")
        if not all(path.exists() for path in staged):
            raise ScaleEngineError("verified stage is incomplete; prepare a new stage")
        backup_dir = self.backup_root / f"{stage_id}-{uuid.uuid4().hex[:6]}"
        active = self.active_paths
        self.backup_root.mkdir(parents=True, exist_ok=True)
        # Keep an explicit empty rollback point as well: a first promotion has
        # no former projection directories, but rollback must still be able to
        # return the installation to Local Core without deleting anything.
        backup_dir.mkdir(parents=True, exist_ok=False)
        moved_active: list[tuple[Path, Path]] = []
        moved_stage: list[tuple[Path, Path]] = []
        try:
            for name, source, destination in zip(("cozo", "lance"), active, staged):
                if source.exists():
                    target = backup_dir / name
                    os.replace(source, target)
                    moved_active.append((source, target))
                os.replace(destination, source)
                moved_stage.append((destination, source))
            manifest.update({"state": "promoted", "promoted_at": _utc_now(), "backup_id": backup_dir.name})
            self._write_manifest(stage_dir, manifest)
            self.config.scale_engine_state = "promoted"
            self.config.graph_backend = "cozo"
            self.config.vector_backend = "lancedb"
            self._save_config()
            return manifest
        except Exception as exc:
            for staged_path, active_path in reversed(moved_stage):
                if active_path.exists():
                    os.replace(active_path, staged_path)
            for active_path, backup_path in reversed(moved_active):
                if backup_path.exists():
                    os.replace(backup_path, active_path)
            raise ScaleEngineError(f"promotion rolled back: {exc}") from exc

    def rollback(self, backup_id: str) -> dict[str, Any]:
        """Restore an explicitly named pre-promotion backup."""
        backup_dir = self.backup_root / backup_id
        if not backup_dir.is_dir():
            raise ScaleEngineError(f"backup does not exist: {backup_id}")
        backup_paths = (backup_dir / "cozo", backup_dir / "lance")
        active = self.active_paths
        displaced = self.backup_root / f"rollback-displaced-{uuid.uuid4().hex[:8]}"
        try:
            for name, source, target in zip(("cozo", "lance"), active, backup_paths):
                if source.exists():
                    displaced.mkdir(parents=True, exist_ok=True)
                    os.replace(source, displaced / name)
                if target.exists():
                    os.replace(target, source)
            self.config.scale_engine_state = "local_core"
            self.config.graph_backend = "auto"
            self.config.vector_backend = "auto"
            self._save_config()
            return {"state": "local_core", "restored_backup": backup_id, "displaced": displaced.name}
        except Exception as exc:
            raise ScaleEngineError(f"rollback failed; inspect {backup_dir}: {exc}") from exc

    def _real_backend_factory(self, cozo_dir: Path, lance_dir: Path) -> tuple[Any, Any]:
        from superlocalmemory.graph.cozo_backend import CozoDBGraphBackend
        from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend
        return CozoDBGraphBackend(str(cozo_dir / "graph")), LanceDBVectorBackend(str(lance_dir))

    def _readonly_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        return conn

    def _canonical_counts(self, conn: sqlite3.Connection) -> dict[str, int]:
        nodes = conn.execute(
            "SELECT COUNT(*) FROM canonical_entities WHERE profile_id=?",
            (self.profile_id,),
        ).fetchone()[0]
        edges = conn.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE profile_id=?", (self.profile_id,)
        ).fetchone()[0]
        try:
            vectors = conn.execute("SELECT COUNT(*) FROM fact_embeddings_rowids").fetchone()[0]
        except sqlite3.OperationalError:
            vectors = 0
        return {"entities": int(nodes), "edges": int(edges), "vectors": int(vectors)}

    def _observed_counts(self, cozo: Any, lance: Any) -> dict[str, int]:
        graph = cozo.health_check()
        vector = lance.health_check()
        if graph.get("status") != "active" or vector.get("status") != "active":
            raise ScaleEngineError(f"projection health failed: cozo={graph}, lancedb={vector}")
        return {"entities": int(graph["entities"]), "edges": int(graph["edges"]), "vectors": int(vector["vectors"])}

    def _source_fingerprint(self, counts: dict[str, int]) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(counts, sort_keys=True).encode())
        for path in (self.db_path, self.db_path.with_name(self.db_path.name + "-wal")):
            if path.exists():
                digest.update(path.name.encode())
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
        return digest.hexdigest()

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
            raise ScaleEngineError("Scale Engine promotion currently supports the default profile only")

    def _require_canonical_db(self) -> None:
        if not self.db_path.exists():
            raise ScaleEngineError(f"canonical SQLite database not found: {self.db_path}")

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
