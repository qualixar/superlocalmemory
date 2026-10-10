# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The full data export of one person includes their pictures, documents,
connected folders and background jobs (as records, never as file bytes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.media.store import MediaStore
from superlocalmemory.sources.store import SourceStore
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager

_SHA = "ab" * 32


def _memory_db(root: Path) -> DatabaseManager:
    mgr = DatabaseManager(root / "memory.db")
    mgr.initialize(real_schema)
    for pid in ("alice", "bob"):
        mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))
    mgr.execute("INSERT INTO memories (memory_id, profile_id, scope, content) "
                "VALUES ('m-pic', 'alice', 'shared', 'a picture')")
    return mgr


def _fill(root: Path) -> dict[str, str]:
    store = MediaStore(root / "media.db")
    ids = {}
    ids["pic"] = store.insert_item(
        profile_id="alice", kind="image", source_sha256=_SHA, mime="image/jpeg", bytes=2048,
        width=640, height=480, captured_at="2026-01-02T03:04:05Z", anchor_memory_id="m-pic",
        exif_json={"Make": "Acme"}, original_relpath="ab/cd/orig.bin", origin="tool",
        thumb_webp=b"thumbbytes")
    ids["bob_pic"] = store.insert_item(
        profile_id="bob", kind="image", source_sha256="cd" * 32, mime="image/png", bytes=9,
        origin="tool", original_relpath="bob/secret.bin")
    store.insert_document(document_id="d" * 32, profile_id="alice", sha256=_SHA, title="Lease",
                          mime="application/pdf", bytes=5000, source_relpath="")
    store.put_page("d" * 32, 1, media_id=None, memory_ids=[], fact_ids=[], text_origin="ocr",
                   char_count=10)
    store.insert_document(document_id="e" * 32, profile_id="bob", sha256="cd" * 32,
                          title="Bob private", mime="application/pdf", bytes=1,
                          source_relpath="")
    ids["job"] = store.enqueue_job("alice", "document", total=3, payload={"path": "/x/secret.pdf"})
    store.enqueue_job("bob", "document", total=1)
    sources = SourceStore(store)
    sources.create_source("alice", "folder", "/home/alice/notes", "notes", ["md"])
    sources.create_source("bob", "folder", "/home/bob/notes", "bob notes", ["md"])
    store.close()
    return ids


def test_export_carries_the_media_records_of_the_profile(tmp_path: Path) -> None:
    mgr = _memory_db(tmp_path)
    ids = _fill(tmp_path)
    media = GDPRCompliance(mgr, data_root=tmp_path).export_profile_data("alice")["media"]

    (pic,) = media["pictures"]
    assert pic["media_id"] == ids["pic"]
    assert (pic["mime"], pic["bytes"], pic["width"], pic["height"]) == ("image/jpeg", 2048, 640, 480)
    assert pic["captured_at"] == "2026-01-02T03:04:05Z"
    assert pic["exif"] == {"Make": "Acme"}
    assert pic["anchor_memory_id"] == "m-pic" and pic["anchor_scope"] == "shared"
    assert pic["has_thumbnail"] is True

    (doc,) = media["documents"]
    assert doc["title"] == "Lease" and doc["state"] == "processing"
    assert doc["pages"] == [{"page_no": 1, "text_origin": "ocr", "char_count": 10, "media_id": None}]

    (folder,) = media["folders"]
    assert folder["root_path"] == "/home/alice/notes" and folder["state"] == "active"

    (job,) = media["jobs"]
    assert (job["kind"], job["state"], job["total"]) == ("document", "queued", 3)


def test_export_has_no_file_bytes_paths_or_job_input(tmp_path: Path) -> None:
    mgr = _memory_db(tmp_path)
    _fill(tmp_path)
    text = json.dumps(GDPRCompliance(mgr, data_root=tmp_path)
                      .export_profile_data("alice")["media"], default=str)
    for leak in ("thumbbytes", "orig.bin", "secret.pdf", "payload"):
        assert leak not in text


def test_export_never_carries_another_profile(tmp_path: Path) -> None:
    mgr = _memory_db(tmp_path)
    _fill(tmp_path)
    text = json.dumps(GDPRCompliance(mgr, data_root=tmp_path)
                      .export_profile_data("alice")["media"], default=str)
    for other in ("Bob private", "/home/bob", "bob/secret.bin", "image/png"):
        assert other not in text


def test_export_without_media_db_has_no_media_section(tmp_path: Path) -> None:
    mgr = _memory_db(tmp_path)
    assert "media" not in GDPRCompliance(mgr, data_root=tmp_path).export_profile_data("alice")


def test_export_does_not_change_media_db(tmp_path: Path) -> None:
    mgr = _memory_db(tmp_path)
    _fill(tmp_path)
    before = (tmp_path / "media.db").read_bytes()
    GDPRCompliance(mgr, data_root=tmp_path).export_profile_data("alice")
    assert (tmp_path / "media.db").read_bytes() == before
