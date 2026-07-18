# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Canonical sqlite-vec export contract for derived vector projections."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import TypeAlias


class CanonicalVectorError(RuntimeError):
    """Canonical sqlite-vec data cannot be read without risking data loss."""


CanonicalVector: TypeAlias = tuple[int, str, str, str, bytes]


def load_sqlite_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec for this connection or fail the migration explicitly."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as exc:
        raise CanonicalVectorError(
            "sqlite-vec could not be loaded; refusing to treat canonical vectors as empty"
        ) from exc
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass


def count_canonical_vectors(conn: sqlite3.Connection, profile_id: str) -> int:
    """Count profile-scoped vectors that have a canonical fact identity."""
    if not _canonical_vector_table_exists(conn):
        return 0
    load_sqlite_vec_extension(conn)
    return _validate_canonical_vector_contract(conn, profile_id)


def iter_canonical_vectors(
    conn: sqlite3.Connection, profile_id: str
) -> Iterator[CanonicalVector]:
    """Yield supported vec0 rows joined to canonical fact identity and lifecycle."""
    if not _canonical_vector_table_exists(conn):
        return
    load_sqlite_vec_extension(conn)
    _validate_canonical_vector_contract(conn, profile_id)
    try:
        rows = conn.execute(
            "SELECT fe.rowid, em.fact_id, COALESCE(af.lifecycle, 'active'), "
            "af.profile_id, fe.embedding "
            "FROM fact_embeddings fe "
            "JOIN embedding_metadata em ON em.vec_rowid = fe.rowid "
            "JOIN atomic_facts af ON af.fact_id = em.fact_id "
            "AND af.profile_id = em.profile_id "
            "WHERE af.profile_id = ? AND fe.profile_id = af.profile_id "
            "ORDER BY fe.rowid",
            (profile_id,),
        )
        for rowid, fact_id, lifecycle, row_profile_id, blob in rows:
            if not isinstance(blob, bytes):
                raise CanonicalVectorError(
                    f"canonical vector {rowid} is not a float32 blob"
                )
            yield int(rowid), str(fact_id), str(lifecycle), str(row_profile_id), blob
    except sqlite3.Error as exc:
        raise CanonicalVectorError(
            "canonical sqlite-vec rows are unreadable; refusing a partial projection"
        ) from exc


def _validate_canonical_vector_contract(
    conn: sqlite3.Connection, profile_id: str
) -> int:
    """Prove every profile-owned metadata row maps to one vec0 partition row."""
    try:
        misowned = int(
            conn.execute(
                "SELECT COUNT(*) FROM embedding_metadata em "
                "LEFT JOIN atomic_facts af ON af.fact_id = em.fact_id "
                "WHERE (em.profile_id = ? OR af.profile_id = ?) "
                "AND (af.fact_id IS NULL OR af.profile_id <> em.profile_id)",
                (profile_id, profile_id),
            ).fetchone()[0]
        )
        expected = int(
            conn.execute(
                "SELECT COUNT(*) FROM embedding_metadata em "
                "JOIN atomic_facts af ON af.fact_id = em.fact_id "
                "AND af.profile_id = em.profile_id "
                "WHERE af.profile_id = ?",
                (profile_id,),
            ).fetchone()[0]
        )
        mapped = int(
            conn.execute(
                "SELECT COUNT(*) FROM embedding_metadata em "
                "JOIN atomic_facts af ON af.fact_id = em.fact_id "
                "AND af.profile_id = em.profile_id "
                "JOIN fact_embeddings fe ON fe.rowid = em.vec_rowid "
                "AND fe.profile_id = af.profile_id "
                "WHERE af.profile_id = ?",
                (profile_id,),
            ).fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise CanonicalVectorError(
            "canonical vector contract is unreadable; refusing a partial projection"
        ) from exc
    if misowned:
        raise CanonicalVectorError(
            f"canonical vector ownership mismatch for {misowned} metadata row(s)"
        )
    if mapped != expected:
        raise CanonicalVectorError(
            "canonical vector mapping is incomplete: "
            f"metadata={expected}, mapped_vec0={mapped}"
        )
    return expected


def _canonical_vector_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='fact_embeddings' "
        "AND type='table'"
    ).fetchone()
    if row is not None:
        return True
    metadata = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='embedding_metadata' "
        "AND type='table'"
    ).fetchone()
    if metadata is not None:
        try:
            metadata_count = int(
                conn.execute("SELECT COUNT(*) FROM embedding_metadata").fetchone()[0]
            )
        except sqlite3.Error as exc:
            raise CanonicalVectorError("embedding metadata is unreadable") from exc
        if metadata_count:
            raise CanonicalVectorError(
                "embedding metadata exists but fact_embeddings is missing"
            )
    shadow_payload = 0
    for table in ("fact_embeddings_rowids", "fact_embeddings_vector_chunks00"):
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name=? AND type='table'",
            (table,),
        ).fetchone()
        if exists is not None:
            try:
                shadow_payload += int(
                    conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
            except sqlite3.Error as exc:
                raise CanonicalVectorError(
                    "sqlite-vec shadow payload is unreadable"
                ) from exc
    if shadow_payload:
        raise CanonicalVectorError(
            "sqlite-vec shadow payload exists without the fact_embeddings virtual table"
        )
    return False
