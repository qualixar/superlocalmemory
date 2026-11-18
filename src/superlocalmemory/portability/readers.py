# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Get the bytes of an export into memory safely: a file, several files, a folder or a zip.

An export arrives as one JSON file, a numbered set (``conversations-000.json``...), a zip of
those, or a folder of Markdown. This module turns each shape into a list of :class:`Doc`
(a name and bytes) under hard limits, so no importer ever opens a path or a zip itself.

* Every limit is a routing constant (``constants``); a file over a limit is refused with
  plain words, never cut short.
* A zip is read member by member with its declared sizes checked and the bytes actually read
  counted (a lying header cannot unpack more than the cap). Names with ``..`` or an absolute
  path are never used for anything: members are read into memory, never written to disk.
* A path is opened without following a link (``sources/safe_read``); a folder walk skips links
  and hidden directories and never enters SuperLocalMemory's own data folder.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from superlocalmemory.portability import constants as C

#: Files an importer can read. Anything else inside a zip or a folder is left alone.
READABLE = (".json", ".af", ".md", ".markdown", ".txt")
_SKIP_DIRS = frozenset({".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv"})


class IntakeRefused(ValueError):
    """The file or folder cannot be read; the message is plain words for a person."""


@dataclass(frozen=True)
class Doc:
    """One file of an export: its name (inside the export) and its bytes."""

    name: str
    data: bytes

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.name.replace("\\", "/")).suffix.lower()

    @property
    def base(self) -> str:
        return PurePosixPath(self.name.replace("\\", "/")).name.lower()

    def text(self) -> str:
        return self.data.decode("utf-8-sig", errors="replace")

    def json(self) -> Any:
        """The parsed JSON, read once; ``None`` when the file is not JSON."""
        if "_json" not in self.__dict__:
            try:
                value = json.loads(self.data.decode("utf-8-sig"))
            except (ValueError, UnicodeDecodeError, RecursionError):
                value = None
            object.__setattr__(self, "_json", value)
        return self.__dict__["_json"]


def _clean_name(name: str) -> str:
    """A label to show: the last parts of the name, no drive, no ``..``."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts[-3:]) or "file"


def from_bytes(name: str, data: bytes, *, limit: int | None = None) -> list[Doc]:
    """The docs of one upload: the file itself, or the readable members of a zip."""
    limit = C.MAX_UPLOAD_BYTES if limit is None else limit
    if len(data) > limit:
        raise IntakeRefused(f"That file is larger than {limit // (1024 * 1024)} MB. "
                            "Split it, or import it from a path on this computer.")
    if not data:
        raise IntakeRefused("That file is empty.")
    if data[:4] == b"PK\x03\x04":
        return _unzip(data)
    return [Doc(_clean_name(name or "upload"), data)]


def _unzip(data: bytes) -> list[Doc]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise IntakeRefused("That zip file could not be opened.") from None
    docs: list[Doc] = []
    total = 0
    with archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) > C.MAX_ARCHIVE_FILES:
            raise IntakeRefused("That zip holds too many files to import.")
        for member in members:
            name = _clean_name(member.filename)
            if not name.lower().endswith(READABLE) or member.flag_bits & 0x1:
                continue  # not a file we read, or encrypted
            if member.file_size > C.MAX_UNPACKED_BYTES - total:
                raise IntakeRefused("That zip unpacks to more than the size limit.")
            with archive.open(member) as handle:
                chunk = handle.read(C.MAX_UNPACKED_BYTES - total + 1)
            total += len(chunk)
            if total > C.MAX_UNPACKED_BYTES:
                raise IntakeRefused("That zip unpacks to more than the size limit.")
            docs.append(Doc(name, chunk))
    if not docs:
        raise IntakeRefused("That zip has no files this importer can read.")
    return docs


def from_path(path: str | Path) -> list[Doc]:
    """The docs of a file or folder on this computer (command line callers only)."""
    from superlocalmemory.infra.data_root import DATA_ROOT_REFUSAL, overlaps_data_root
    from superlocalmemory.sources.safe_read import open_regular

    target = Path(path).expanduser()
    if overlaps_data_root(target):
        raise IntakeRefused(DATA_ROOT_REFUSAL)
    try:
        if target.is_symlink():
            raise IntakeRefused("That path is a link; give the file itself.")
        if target.is_file():
            files = [target]
            root = target.parent
        elif target.is_dir():
            files, root = _walk(target), target
        else:
            raise IntakeRefused("That file or folder was not found.")
    except OSError:
        raise IntakeRefused("That file or folder could not be read.") from None
    docs: list[Doc] = []
    total = 0
    for file in files:
        try:
            with open_regular(file) as handle:
                data = handle.read(C.MAX_LOCAL_BYTES - total + 1)
        except OSError:
            raise IntakeRefused(f"{_clean_name(str(file))} could not be read.") from None
        total += len(data)
        if total > C.MAX_LOCAL_BYTES:
            raise IntakeRefused(f"That is more than {C.MAX_LOCAL_BYTES // (1024 * 1024)} MB.")
        if data[:4] == b"PK\x03\x04":
            docs.extend(_unzip(data))
        elif data:
            docs.append(Doc(_clean_name(os.path.relpath(file, root)), data))
    if not docs:
        raise IntakeRefused("There is nothing to read in that file or folder.")
    return docs


def _walk(folder: Path) -> list[Path]:
    found: list[Path] = []
    for current, dirs, names in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(names):
            path = Path(current, name)
            if name.lower().endswith(READABLE) and not path.is_symlink():
                found.append(path)
                if len(found) > C.MAX_FOLDER_FILES:
                    raise IntakeRefused("That folder holds too many files to import.")
    return found


__all__ = ["Doc", "IntakeRefused", "READABLE", "from_bytes", "from_path"]
