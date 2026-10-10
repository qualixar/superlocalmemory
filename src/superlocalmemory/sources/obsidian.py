# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Obsidian notes: front-matter properties, links and canvases. Pure parsing plus one save function.

The front matter is read by a minimal parser, never by a YAML library: a ``---`` block of at most
8 KB at the very top of the note, holding ``key: value`` lines, ``[a, b]`` lists and ``- item``
lists. Anything else (anchors, tags such as ``!!python/object``, block scalars, nesting) is kept
as plain text or ignored; nothing is ever evaluated, and every size is capped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from superlocalmemory.memory_core import (
    ContentOrigin, effective_pii_redaction, prepare_for_save, prepare_metadata)
from superlocalmemory.sources import ingest, links
from superlocalmemory.sources.host import SourceHost

MAX_FRONT_MATTER = 8 * 1024
_MAX_LINES, _MAX_KEYS, _MAX_ITEMS = 400, 200, 50
_MAX_TAGS, _MAX_TAG_CHARS, _MAX_PROPS, _MAX_VALUE = 20, 64, 30, 200
_OPEN = re.compile(r"---[ \t]*\r?\n")
_CLOSE = re.compile(r"^(?:---|\.\.\.)[ \t]*\r?$", re.M)
_KEY = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_ .\-]{0,63}):(?:[ \t]+(.*))?$")
_ITEM = re.compile(r"\s*-(?:[ \t]+(.*))?$")
_PIECES = re.compile(r"\"[^\"]*\"|'[^']*'|[^,]+")
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_TAG_KEYS, _ALIAS_KEYS, _DATE_KEYS = ("tags", "tag"), ("aliases", "alias"), ("date", "created")
_SPECIAL = frozenset(_TAG_KEYS + _ALIAS_KEYS + _DATE_KEYS)

Props = dict[str, "str | list[str]"]


def _scalar(value: str) -> str:
    value = value.strip()[:_MAX_VALUE * 3]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def _inline_list(value: str) -> list[str]:
    items = [_scalar(m.group(0)) for m in _PIECES.finditer(value[1:-1])]
    return [i for i in items if i][:_MAX_ITEMS]


def _parse_block(block: str) -> Props | None:
    """The keys of a front-matter block, or None when it is not a mapping at all."""
    props: Props = {}
    current: str | None = None
    for number, raw in enumerate(block.splitlines()[:_MAX_LINES]):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key = _KEY.match(line)
        item = _ITEM.match(line) if line[:1] in " \t-" else None
        if item is not None and current is not None and isinstance(props[current], list):
            if len(props[current]) < _MAX_ITEMS and item.group(1):
                props[current].append(_scalar(item.group(1)))
        elif key is not None and not line[0].isspace():
            name, value = key.group(1).strip(), (key.group(2) or "").strip()
            if name in props or len(props) >= _MAX_KEYS:
                current = None
            elif not value:
                props[name], current = [], name
            elif value.startswith("[") and value.endswith("]"):
                props[name], current = _inline_list(value), None
            else:
                props[name], current = _scalar(value), None
        elif number == 0 or not props:
            return None
    return props


def split_front_matter(text: str) -> tuple[Props | None, str]:
    """``(properties, body)``; ``(None, text)`` when the note has no usable front matter."""
    head = text[1:] if text.startswith("﻿") else text
    opened = _OPEN.match(head)
    if opened is None:
        return None, text
    window = head[opened.end():opened.end() + MAX_FRONT_MATTER + 8]
    closed = _CLOSE.search(window)
    if closed is None or closed.start() > MAX_FRONT_MATTER:
        return None, text
    props = _parse_block(window[:closed.start()])
    if props is None:
        return None, text
    start = opened.end() + closed.end()
    return props, head[start + 1 if head[start:start + 1] == "\n" else start:]


@dataclass
class NoteFields:
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    session_date: str = ""
    properties: dict[str, str] = field(default_factory=dict)


def _as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else [v for v in value if isinstance(v, str)]


def _tags(props: Props) -> list[str]:
    found: dict[str, None] = {}
    for key in _TAG_KEYS:
        for raw in _as_list(props.get(key, [])):
            for tag in re.split(r"[,\s]+", raw):
                tag = tag.lstrip("#").strip()
                if tag and len(tag) <= _MAX_TAG_CHARS:
                    found.setdefault(tag, None)
    return list(found)[:_MAX_TAGS]


def _session_date(props: Props) -> str:
    for key in _DATE_KEYS:
        value = props.get(key)
        found = _DATE.match(value) if isinstance(value, str) else None
        if found:
            try:
                return date.fromisoformat(found.group(1)).isoformat()
            except ValueError:
                continue
    return ""


def note_fields(props: Props) -> NoteFields:
    """Map parsed front matter to tags, aliases, a session date and the other properties."""
    fields = NoteFields(tags=_tags(props), session_date=_session_date(props))
    aliases: dict[str, None] = {}
    for key in _ALIAS_KEYS:
        for alias in _as_list(props.get(key, [])):
            if alias.strip():
                aliases.setdefault(alias.strip()[:_MAX_VALUE], None)
    fields.aliases = list(aliases)[:_MAX_ITEMS]
    for key, value in props.items():
        text = ", ".join(value) if isinstance(value, list) else value
        if key.lower() not in _SPECIAL and text and len(fields.properties) < _MAX_PROPS:
            fields.properties[key[:64]] = text[:_MAX_VALUE]
    return fields


def prepared_extras(fields: NoteFields, config: object | None) -> dict[str, Any]:
    """``_slm_properties`` as it is stored: derived-text secret stripping, then the normal redaction."""
    extras: dict[str, Any] = {}
    if fields.aliases:
        extras["aliases"] = fields.aliases
    if fields.properties:
        extras["properties"] = fields.properties
    if not extras:
        return {}

    def strip(value: Any) -> Any:
        if isinstance(value, str):
            return prepare_for_save(value, origin=ContentOrigin.DERIVED_TEXT, pii_redaction=False).text
        if isinstance(value, dict):
            return {strip(k): strip(v) for k, v in value.items()}
        return [strip(v) for v in value] if isinstance(value, list) else value

    return prepare_metadata(strip(extras), pii_redaction=effective_pii_redaction(config))[0]


def ingest_note(host: SourceHost, runtime: Any, source: dict, relpath: str, data: bytes, version: str,
                n: int, names: "links.NameIndex | None") -> ingest.Ingested:
    """Save a note or canvas: properties, tags and date from the front matter, links recorded."""
    if relpath.lower().endswith(".canvas"):
        try:
            canvas = links.parse_canvas(data)
        except links.CanvasError:
            return ingest.Ingested(skip_reason="canvas_unreadable", links=[])
        body, found, fields = canvas.text, canvas.links, NoteFields()
    else:
        props, body = split_front_matter(data.decode("utf-8", errors="replace"))
        fields = note_fields(props or {})
        found = links.extract_links(body, relpath, names)
    parts = ingest.split_text(relpath, body)
    if not parts:
        return ingest.Ingested(skip_reason="empty", links=found)
    extra = prepared_extras(fields, host.config())
    out = ingest.save_parts(host, runtime, source, relpath, parts, version, n, tags=",".join(fields.tags),
                            session_date=fields.session_date,
                            extra={"_slm_properties": extra} if extra else None)
    out.links = found
    return out


__all__ = ["NoteFields", "ingest_note", "note_fields", "prepared_extras", "split_front_matter"]
