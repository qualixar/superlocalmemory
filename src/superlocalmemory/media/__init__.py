# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Images and documents: the store. Nothing here is created until the feature is turned on."""

from __future__ import annotations

import os
from pathlib import Path

from superlocalmemory.media.store import MediaStore, MediaStoreReadOnly

MEDIA_DB_NAME = "media.db"


def media_db_path(data_root: str | Path | None = None) -> Path:
    if data_root is None:
        from superlocalmemory.infra.data_root import canonical_data_root

        data_root = canonical_data_root()
    return Path(data_root) / MEDIA_DB_NAME


def media_db_exists(data_root: str | Path | None = None) -> bool:
    return media_db_path(data_root).is_file()


def open_media_store(*, create: bool = False, data_root: str | Path | None = None) -> MediaStore | None:
    """The store, or None when media.db does not exist and ``create`` is False.

    Reading never creates the file; only the turn-on step passes ``create=True``.
    """
    path = media_db_path(data_root)
    if not path.is_file():
        if not create:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(path, os.O_RDWR | os.O_CREAT, 0o600))
    return MediaStore(path)


__all__ = ["MEDIA_DB_NAME", "MediaStore", "MediaStoreReadOnly", "media_db_exists",
           "media_db_path", "open_media_store"]
