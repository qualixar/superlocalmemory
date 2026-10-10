# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Links inside Obsidian notes and canvases. They are recorded only: nothing is fetched or followed,
and a target that leads outside the folder is dropped."""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote

MAX_LINKS = 2000
MAX_CANVAS_BYTES = 2 * 1024 * 1024
_WIKI = re.compile(r"(!?)\[\[([^\[\]\n|]{1,300})(?:\|([^\[\]\n]{0,200}))?\]\]")
_MD = re.compile(r"(?<!!)\[([^\]\n]{0,200})\]\(([^()\s]{1,300})\)")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class CanvasError(ValueError):
    """The canvas is not readable JSON of the expected shape."""


@dataclass(frozen=True)
class Link:
    target: str
    kind: str
    label: str | None = None


@dataclass
class Canvas:
    text: str = ""
    links: list[Link] = field(default_factory=list)


class NameIndex:
    """Files of one folder by name, so an embed finds the shortest matching path."""

    def __init__(self, relpaths: Iterable[str]) -> None:
        self._by_name: dict[str, list[str]] = {}
        for rel in set(relpaths):
            self._by_name.setdefault(rel.rsplit("/", 1)[-1].lower(), []).append(rel)

    def resolve(self, name: str) -> str | None:
        wanted = name.strip().replace("\\", "/").strip("/").lower()
        if not wanted:
            return None
        options = [wanted] if "." in wanted.rsplit("/", 1)[-1] else [wanted + ".md", wanted + ".canvas", wanted]
        found = [rel for w in options for rel in self._by_name.get(w.rsplit("/", 1)[-1], ())
                 if rel.lower() == w or rel.lower().endswith("/" + w)]
        return min(found, key=lambda r: (len(r), r)) if found else None


def _inside(path: str) -> str | None:
    """A folder-relative path, or None when it is empty, absolute or leaves the folder."""
    path = path.replace("\\", "/")
    if not path or path[0] in "/~" or _SCHEME.match(path) or _CONTROL.search(path):
        return None
    clean = posixpath.normpath(path)
    return None if clean in (".", "..") or clean.startswith("../") else clean


def _wiki_name(name: str) -> str | None:
    name = name.strip()
    if not name or ".." in name.split("/") or name[0] in "/\\" or "://" in name or _CONTROL.search(name):
        return None
    return name


def _strip_code(body: str) -> str:
    kept, fenced = [], False
    for line in body.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            kept.append(_INLINE_CODE.sub("", line))
    return "\n".join(kept)


def _wiki_link(m: re.Match, names: NameIndex | None) -> Link | None:
    name, _, fragment = m.group(2).partition("#")
    name = _wiki_name(name)
    label = (m.group(3) or "").strip() or None
    if name is None:
        return None
    if m.group(1):
        return Link((names.resolve(name) if names else None) or name, "embed", label)
    if fragment:
        return Link(f"{name}#{fragment.strip()}"[:400], "block" if fragment.startswith("^") else "heading", label)
    return Link(name, "wikilink", label)


def _md_link(m: re.Match, from_relpath: str) -> Link | None:
    target = unquote(m.group(2)).split("#", 1)[0].split("?", 1)[0]
    if not target or target[0] in "/\\~" or _SCHEME.match(target):
        return None
    joined = _inside(posixpath.join(posixpath.dirname(from_relpath), target.replace("\\", "/")))
    return Link(joined, "md_link", m.group(1).strip() or None) if joined else None


def extract_links(body: str, from_relpath: str, names: NameIndex | None) -> list[Link]:
    """The links of a note body, one per (kind, target), at most ``MAX_LINKS``."""
    text = _strip_code(body)
    found: dict[tuple[str, str], Link] = {}
    for m in _WIKI.finditer(text):
        link = _wiki_link(m, names)
        if link and (link.kind, link.target) not in found:
            found[(link.kind, link.target)] = link
        if len(found) >= MAX_LINKS:
            return list(found.values())
    for m in _MD.finditer(text):
        link = _md_link(m, from_relpath)
        if link and (link.kind, link.target) not in found:
            found[(link.kind, link.target)] = link
        if len(found) >= MAX_LINKS:
            break
    return list(found.values())


def _load_canvas(data: bytes) -> dict:
    if len(data) > MAX_CANVAS_BYTES:
        raise CanvasError("too large")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError, MemoryError) as exc:  # UnicodeDecodeError is a ValueError
        raise CanvasError("not json") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("nodes", []), list) \
            or not isinstance(doc.get("edges", []), list):
        raise CanvasError("unexpected shape")
    return doc


def _ref(refs: dict[str, str], node_id: object) -> str | None:
    return refs.get(node_id) if isinstance(node_id, str) else None


def parse_canvas(data: bytes) -> Canvas:
    """Text nodes joined as the canvas text; file nodes and edges as ``canvas_edge`` links."""
    doc = _load_canvas(data)
    texts: list[str] = []
    refs: dict[str, str] = {}
    out: list[Link] = []
    for node in [n for n in doc.get("nodes", [])[:MAX_LINKS] if isinstance(n, dict)]:
        node_id, kind = node.get("id"), node.get("type")
        ref = f"node:{node_id}"[:120]
        if kind == "text" and isinstance(node.get("text"), str) and node["text"].strip():
            texts.append(node["text"].strip())
        elif kind == "file" and isinstance(node.get("file"), str) and _inside(node["file"]):
            ref = _inside(node["file"]) or ref
            out.append(Link(ref, "canvas_edge", "file"))
        if isinstance(node_id, str):
            refs[node_id] = ref
    for edge in [e for e in doc.get("edges", [])[:MAX_LINKS] if isinstance(e, dict)]:
        src, dst = _ref(refs, edge.get("fromNode")), _ref(refs, edge.get("toNode"))
        label = edge.get("label")
        if src and dst:
            out.append(Link(f"{src}->{dst}"[:600], "canvas_edge",
                            label[:200] if isinstance(label, str) and label else None))
    return Canvas("\n\n".join(texts), out[:MAX_LINKS])


__all__ = ["Canvas", "CanvasError", "Link", "NameIndex", "extract_links", "parse_canvas"]
