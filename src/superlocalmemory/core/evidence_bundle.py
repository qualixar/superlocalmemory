# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Versioned, reviewable evidence export and rebuild contract.

The bundle keeps raw ingestion evidence and human-reviewable relational truth;
embeddings, Fisher parameters, lexical tokens, and optional backend indexes are
deliberately excluded because they are derived and rebuildable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BUNDLE_FORMAT = "superlocalmemory-evidence-bundle"
BUNDLE_SCHEMA_VERSION = 1

_FACT_DERIVED_COLUMNS = frozenset({
    "embedding", "fisher_mean", "fisher_variance", "langevin_position",
})

# Import order is dependency-safe. Every row is profile-scoped except the
# profile record; aliases are excluded until their cross-table identity
# contract is versioned.
_TABLES: tuple[tuple[str, str, str], ...] = (
    ("profile.jsonl", "profiles", "profile_id"),
    ("memories.jsonl", "memories", "memory_id"),
    ("facts.jsonl", "atomic_facts", "fact_id"),
    ("entities.jsonl", "canonical_entities", "entity_id"),
    ("entity_profiles.jsonl", "entity_profiles", "profile_entry_id"),
    ("memory_scenes.jsonl", "memory_scenes", "scene_id"),
    ("graph_edges.jsonl", "graph_edges", "edge_id"),
    ("temporal_events.jsonl", "temporal_events", "event_id"),
    ("provenance.jsonl", "provenance", "provenance_id"),
    ("ingestion_operations.jsonl", "ingestion_operations", "operation_id"),
    ("derivation_lineage.jsonl", "derivation_lineage", "lineage_id"),
)


@dataclass(frozen=True, slots=True)
class BundleReport:
    valid: bool
    bundle_id: str
    counts: dict[str, int]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _table_exists(db: Any, table: str) -> bool:
    rows = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    )
    if hasattr(rows, "fetchall"):
        rows = rows.fetchall()
    return bool(rows)


def _rows_for_profile(db: Any, table: str, profile_id: str, order: str) -> list[dict]:
    if not _table_exists(db, table):
        return []
    rows = db.execute(
        f'SELECT * FROM "{table}" WHERE profile_id=? ORDER BY "{order}"',
        (profile_id,),
    )
    if hasattr(rows, "fetchall"):
        rows = rows.fetchall()
    result = [dict(row) for row in rows]
    if table == "atomic_facts":
        result = [
            {key: value for key, value in row.items() if key not in _FACT_DERIVED_COLUMNS}
            for row in result
        ]
    return result


def _write_jsonl(path: Path, rows: Iterable[dict]) -> tuple[str, int]:
    payload = "".join(f"{_canonical(row)}\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return _sha256_bytes(payload), payload.count(b"\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _source_spans(
    operations: list[dict], facts: list[dict], lineage: list[dict],
) -> tuple[list[dict], dict[str, dict], list[str]]:
    facts_by_id = {str(row["fact_id"]): row for row in facts}
    spans: list[dict] = []
    by_fact: dict[str, dict] = {}
    unresolved: list[str] = []
    durable = {
        (str(row.get("object_id")), str(row.get("operation_id"))): row
        for row in lineage
        if row.get("object_type") == "fact"
    }
    for operation in operations:
        raw = str(operation.get("raw_content") or "")
        ids: list[str] = []
        for key in ("final_fact_ids_json", "queryable_fact_ids_json"):
            try:
                ids.extend(str(value) for value in json.loads(operation.get(key) or "[]"))
            except (TypeError, ValueError):
                continue
        for fact_id in dict.fromkeys(ids):
            fact = facts_by_id.get(fact_id)
            if fact is None:
                unresolved.append(
                    f"operation {operation['operation_id']} references "
                    f"missing fact {fact_id}"
                )
                continue
            content = str(fact.get("content") or "")
            recorded = durable.get((fact_id, str(operation["operation_id"])))
            if recorded is not None and recorded.get("source_status") == "unresolved":
                unresolved.append(
                    f"fact {fact_id} has no exact span in operation {operation['operation_id']}"
                )
                continue
            start = (
                int(recorded["source_start"])
                if recorded is not None
                and recorded.get("source_status") == "exact"
                and recorded.get("source_start") is not None
                else raw.find(content)
            )
            if start < 0:
                unresolved.append(
                    f"fact {fact_id} has no exact span in operation {operation['operation_id']}"
                )
                continue
            span = {
                "end": start + len(content),
                "fact_id": fact_id,
                "operation_id": str(operation["operation_id"]),
                "start": start,
                "text_sha256": _sha256_bytes(content.encode("utf-8")),
            }
            spans.append(span)
            by_fact[fact_id] = span
    spans.sort(key=lambda row: (row["fact_id"], row["operation_id"], row["start"]))
    return spans, by_fact, unresolved


def _lineage_coverage(
    rows_by_file: dict[str, list[dict]], span_map: dict[str, dict],
) -> dict[str, dict[str, int]]:
    lineage = rows_by_file.get("derivation_lineage.jsonl", [])
    by_type: dict[str, dict[str, list[dict]]] = {}
    for row in lineage:
        by_type.setdefault(str(row.get("object_type")), {}).setdefault(
            str(row.get("object_id")), []
        ).append(row)

    objects = {
        "fact": [str(row["fact_id"]) for row in rows_by_file.get("facts.jsonl", [])],
        "profile": [str(row["profile_id"]) for row in rows_by_file.get("profile.jsonl", [])],
        "entity_summary": [
            str(row["profile_entry_id"])
            for row in rows_by_file.get("entity_profiles.jsonl", [])
        ],
        "graph_edge": [
            str(row["edge_id"]) for row in rows_by_file.get("graph_edges.jsonl", [])
        ],
        "memory_scene": [
            str(row["scene_id"])
            for row in rows_by_file.get("memory_scenes.jsonl", [])
        ],
        "index_bm25": [
            str(row["fact_id"]) for row in rows_by_file.get("facts.jsonl", [])
        ],
    }
    coverage: dict[str, dict[str, int]] = {}
    for object_type, object_ids in objects.items():
        counts = {
            "total": len(object_ids),
            "durable": 0,
            "derived_from_facts": 0,
            "not_applicable": 0,
            "legacy_exact_inference": 0,
            "unresolved": 0,
        }
        for object_id in object_ids:
            records = by_type.get(object_type, {}).get(object_id, [])
            if records:
                statuses = {str(row.get("source_status")) for row in records}
                if statuses - {"unresolved"}:
                    counts["durable"] += 1
                    if "derived_from_facts" in statuses:
                        counts["derived_from_facts"] += 1
                    if "not_applicable" in statuses:
                        counts["not_applicable"] += 1
                else:
                    counts["unresolved"] += 1
            elif object_type == "fact" and object_id in span_map:
                counts["legacy_exact_inference"] += 1
            else:
                counts["unresolved"] += 1
        coverage[object_type] = counts
    return coverage


def export_evidence_bundle(
    db: Any, profile_id: str, destination: str | Path,
) -> dict:
    """Export deterministic JSONL truth plus a checksum manifest."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"destination must be empty: {root}")

    rows_by_file: dict[str, list[dict]] = {}
    for filename, table, order in _TABLES:
        rows_by_file[filename] = _rows_for_profile(db, table, profile_id, order)

    spans, span_map, unresolved = _source_spans(
        rows_by_file["ingestion_operations.jsonl"],
        rows_by_file["facts.jsonl"],
        rows_by_file["derivation_lineage.jsonl"],
    )
    rows_by_file["source_spans.jsonl"] = spans

    files: dict[str, dict[str, Any]] = {}
    for filename in sorted(rows_by_file):
        digest, count = _write_jsonl(root / filename, rows_by_file[filename])
        files[filename] = {"count": count, "sha256": digest}

    derivation_versions = sorted({
        str(row.get("derivation_version") or "")
        for row in rows_by_file["ingestion_operations.jsonl"]
        if row.get("derivation_version")
    })
    identity = {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "files": files,
        "derivation_versions": derivation_versions,
    }
    bundle_id = _sha256_bytes(_canonical(identity).encode("utf-8"))
    manifest = {
        **identity,
        "bundle_id": bundle_id,
        "source_spans": span_map,
        "lineage_coverage": _lineage_coverage(rows_by_file, span_map),
        "unresolved_source_links": unresolved,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_evidence_bundle(bundle: str | Path) -> BundleReport:
    """Verify checksums, counts, identities, and exact source spans."""
    root = Path(bundle)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return BundleReport(False, "", {}, (f"manifest: {exc}",), ())
    if manifest.get("format") != BUNDLE_FORMAT:
        errors.append("unsupported bundle format")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append("unsupported bundle schema version")

    counts: dict[str, int] = {}
    for filename, expected in sorted((manifest.get("files") or {}).items()):
        path = root / filename
        if not path.is_file():
            errors.append(f"missing file: {filename}")
            continue
        payload = path.read_bytes()
        actual_hash = _sha256_bytes(payload)
        if actual_hash != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {filename}")
        try:
            rows = _load_jsonl(path)
        except Exception as exc:
            errors.append(f"invalid JSONL {filename}: {exc}")
            continue
        counts[filename] = len(rows)
        if len(rows) != int(expected.get("count", -1)):
            errors.append(f"count mismatch: {filename}")

    identity = {
        "format": manifest.get("format"),
        "schema_version": manifest.get("schema_version"),
        "profile_id": manifest.get("profile_id"),
        "files": manifest.get("files"),
        "derivation_versions": manifest.get("derivation_versions") or [],
    }
    bundle_id = _sha256_bytes(_canonical(identity).encode("utf-8"))
    if bundle_id != manifest.get("bundle_id"):
        errors.append("bundle_id mismatch")

    try:
        facts = {str(row["fact_id"]): row for row in _load_jsonl(root / "facts.jsonl")}
        operations = {
            str(row["operation_id"]): row
            for row in _load_jsonl(root / "ingestion_operations.jsonl")
        }
        for span in _load_jsonl(root / "source_spans.jsonl"):
            fact = facts.get(str(span.get("fact_id")))
            operation = operations.get(str(span.get("operation_id")))
            if fact is None or operation is None:
                errors.append(f"orphan source span: {span.get('fact_id')}")
                continue
            content = str(fact.get("content") or "")
            raw = str(operation.get("raw_content") or "")
            start, end = int(span["start"]), int(span["end"])
            if raw[start:end] != content:
                errors.append(f"source span content mismatch: {span['fact_id']}")
            if _sha256_bytes(content.encode("utf-8")) != span.get("text_sha256"):
                errors.append(f"source span hash mismatch: {span['fact_id']}")
        warnings.extend(str(item) for item in manifest.get("unresolved_source_links") or [])
    except Exception as exc:
        errors.append(f"source reconciliation failed: {exc}")

    return BundleReport(
        valid=not errors,
        bundle_id=bundle_id,
        counts=counts,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _table_columns(db: Any, table: str) -> set[str]:
    rows = db.execute(f'PRAGMA table_info("{table}")')
    if hasattr(rows, "fetchall"):
        rows = rows.fetchall()
    return {str(row["name"] if isinstance(row, dict) else row[1]) for row in rows}


def _insert_rows(db: Any, table: str, rows: list[dict], profile_id: str) -> None:
    columns = _table_columns(db, table)
    for source in rows:
        row = dict(source)
        if "profile_id" in row:
            row["profile_id"] = profile_id
        row = {key: value for key, value in row.items() if key in columns}
        names = tuple(sorted(row))
        if not names:
            continue
        marks = ",".join("?" for _ in names)
        quoted = ",".join(f'"{name}"' for name in names)
        db.execute(
            f'INSERT OR REPLACE INTO "{table}" ({quoted}) VALUES ({marks})',
            tuple(row[name] for name in names),
        )


def import_evidence_bundle(
    db: Any,
    bundle: str | Path,
    *,
    target_profile_id: str | None = None,
    replace: bool = False,
    rollback_dir: str | Path | None = None,
) -> BundleReport:
    """Import relational truth; replacement always writes a rollback bundle."""
    report = verify_evidence_bundle(bundle)
    if not report.valid:
        raise ValueError("invalid evidence bundle: " + "; ".join(report.errors))
    manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
    profile_id = target_profile_id or str(manifest["profile_id"])
    existing = db.execute(
        "SELECT COUNT(*) AS count FROM atomic_facts WHERE profile_id=?", (profile_id,),
    )
    existing_count = int(existing[0]["count"]) if existing else 0
    if existing_count and not replace:
        raise ValueError("target profile is not empty; use replace with rollback_dir")
    if replace and rollback_dir is None:
        raise ValueError("rollback_dir is required for replace")
    if replace:
        export_evidence_bundle(db, profile_id, rollback_dir)

    delete_order = tuple(reversed(_TABLES[1:]))
    with db.transaction():
        if replace:
            for _filename, table, _order in delete_order:
                if _table_exists(db, table):
                    db.execute(f'DELETE FROM "{table}" WHERE profile_id=?', (profile_id,))
        for filename, table, _order in _TABLES:
            if not _table_exists(db, table):
                continue
            _insert_rows(db, table, _load_jsonl(Path(bundle) / filename), profile_id)
    return verify_evidence_bundle(bundle)


_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def rebuild_derived_state(
    db: Any, profile_id: str, *, embedder: Any | None = None,
) -> dict[str, int]:
    """Rebuild deterministic lexical state and optional content embeddings."""
    rows = db.execute(
        "SELECT fact_id, content FROM atomic_facts WHERE profile_id=? "
        "ORDER BY fact_id",
        (profile_id,),
    )
    facts = [dict(row) for row in rows]
    bm25_rows = 0
    embeddings = 0
    with db.transaction():
        db.execute("DELETE FROM bm25_tokens WHERE profile_id=?", (profile_id,))
        for fact in facts:
            tokens = [token.lower() for token in _TOKEN_RE.findall(str(fact["content"]))]
            db.execute(
                "INSERT INTO bm25_tokens (fact_id, profile_id, tokens) VALUES (?,?,?)",
                (fact["fact_id"], profile_id, json.dumps(tokens, separators=(",", ":"))),
            )
            bm25_rows += 1
            if embedder is not None:
                vector = embedder.embed(str(fact["content"]))
                db.execute(
                    "UPDATE atomic_facts SET embedding=? WHERE fact_id=? AND profile_id=?",
                    (json.dumps(vector, separators=(",", ":")), fact["fact_id"], profile_id),
                )
                embeddings += 1
    return {"bm25_rows": bm25_rows, "embeddings": embeddings}


def _normalized_bundle_rows(
    bundle: Path, filename: str, profile_id: str,
) -> list[dict]:
    rows = _load_jsonl(bundle / filename)
    for row in rows:
        if "profile_id" in row:
            row["profile_id"] = profile_id
    return rows


def verify_rebuild_equivalence(
    db: Any, bundle: str | Path, *, profile_id: str,
) -> dict[str, Any]:
    """Compare deterministic local surfaces; leave model-dependent retrieval unverified."""
    root = Path(bundle)
    bundle_report = verify_evidence_bundle(root)
    if not bundle_report.valid:
        raise ValueError("invalid evidence bundle: " + "; ".join(bundle_report.errors))

    mismatches: list[str] = []
    relational = (
        ("profile.jsonl", "profiles", "profile_id"),
        ("memories.jsonl", "memories", "memory_id"),
        ("facts.jsonl", "atomic_facts", "fact_id"),
        ("entities.jsonl", "canonical_entities", "entity_id"),
        ("entity_profiles.jsonl", "entity_profiles", "profile_entry_id"),
        ("memory_scenes.jsonl", "memory_scenes", "scene_id"),
        ("graph_edges.jsonl", "graph_edges", "edge_id"),
    )
    for filename, table, order in relational:
        expected = _normalized_bundle_rows(root, filename, profile_id)
        actual = _rows_for_profile(db, table, profile_id, order)
        if expected != actual:
            mismatches.append(table)

    facts = _normalized_bundle_rows(root, "facts.jsonl", profile_id)
    fact_ids = {str(row["fact_id"]) for row in facts}
    fts_rows = db.execute(
        "SELECT fact_id FROM atomic_facts_fts ORDER BY fact_id"
    ) if _table_exists(db, "atomic_facts_fts") else []
    fts_ids = {str(row["fact_id"]) for row in fts_rows}
    if not fact_ids <= fts_ids:
        mismatches.append("atomic_facts_fts")

    expected_tokens = {
        str(row["fact_id"]): [
            token.lower() for token in _TOKEN_RE.findall(str(row.get("content") or ""))
        ]
        for row in facts
    }
    actual_tokens = {
        str(row["fact_id"]): json.loads(row["tokens"] or "[]")
        for row in db.execute(
            "SELECT fact_id,tokens FROM bm25_tokens WHERE profile_id=? ORDER BY fact_id",
            (profile_id,),
        )
    }
    if expected_tokens != actual_tokens:
        mismatches.append("bm25_tokens")

    return {
        "status": "partial" if not mismatches else "failed",
        "equivalent_on_verified_surfaces": not mismatches,
        "verified_surfaces": [
            "relational_truth", "fts_membership", "bm25_tokens",
        ],
        "unverified_surfaces": [
            "semantic_vector_retrieval", "ann_index", "reranker_behavior",
        ],
        "mismatches": mismatches,
    }


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BundleReport",
    "export_evidence_bundle",
    "import_evidence_bundle",
    "rebuild_derived_state",
    "verify_evidence_bundle",
    "verify_rebuild_equivalence",
]
