# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Markdown of a projected page, and reading it back (D-17).

A projected page is a Markdown file whose every sentence is a block::

    <!-- slm:block fact=<ids> hash=<16 hex> slot=<slot_id>:<version>|- type=span|template -->
    The sentence.
    <!-- /slm:block -->

The marker carries the fact the block cites, the hash of its text as written and
the version of the slot that fact fills when the file was written (``-`` for a fact
in no slot): what a later edit is compared against (hash) and applied against
(slot version, compare-and-swap). The file's first line names the page. Text
between blocks is the reader's own and is ignored.

Memory text is written with ``<!--`` and ``-->`` escaped, and the escape is undone
on reading, so a memory that contains a marker cannot break a file, and an edit
that contains one cannot forge a block. Pure functions: no database, no files.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from superlocalmemory.compiler.page_text import block_hash

_PAGE = re.compile(r"<!-- slm:page ([^>\n]*?) -->")
_BLOCK = re.compile(r"<!-- slm:block ([^>\n]*?) -->\n(.*?)\n?<!-- /slm:block -->", re.S)
_ATTR = re.compile(r"([a-z_]+)=(\S*)")
_OPEN, _CLOSE = "<!--", "-->"
_ESC_OPEN, _ESC_CLOSE = "&lt;!--", "--&gt;"

HEADER_NOTE = (
    "<!-- This page is compiled from your memories. Change a sentence and run `slm wiki sync`: "
    "the change becomes a proposal you review, never a direct write. Deleting a line forgets "
    "nothing. -->")


def escape(text: str) -> str:
    return text.replace(_OPEN, _ESC_OPEN).replace(_CLOSE, _ESC_CLOSE)


def unescape(text: str) -> str:
    return text.replace(_ESC_OPEN, _OPEN).replace(_ESC_CLOSE, _CLOSE)


@dataclass(frozen=True)
class Block:
    fact_ids: tuple[str, ...]
    hash: str
    slot: str | None
    type: str
    text: str

    @property
    def slot_id(self) -> str | None:
        return self.slot.rsplit(":", 1)[0] if self.slot else None

    @property
    def slot_version(self) -> int | None:
        if not self.slot or ":" not in self.slot:
            return None
        tail = self.slot.rsplit(":", 1)[1]
        return int(tail) if tail.isdigit() else None


@dataclass(frozen=True)
class ParsedFile:
    page_id: str | None
    blocks: tuple[Block, ...]
    #: Block markers that could not be read (a hand-damaged marker): never treated as an edit.
    malformed: int = 0


def render_block(fact_ids: Sequence[str], text: str, slot: str | None, kind: str) -> str:
    marker = (f"<!-- slm:block fact={','.join(fact_ids)} hash={block_hash(text)} "
              f"slot={slot or '-'} type={kind} -->")
    return f"{marker}\n{escape(text)}\n<!-- /slm:block -->"


def render_page(page: Mapping[str, Any], slots: Mapping[str, str | None]) -> str:
    """The file text of one verified page view (``pages.show_page``); ``slots`` maps a
    cited fact id to ``<slot_id>:<version>`` for the facts that fill a slot."""
    lines = [f"<!-- slm:page id={page['page_id']} kind={page['kind']} version={page['version']} "
             f"origin=derived -->", f"# {escape(str(page['title']))}", "", HEADER_NOTE, ""]
    for item in page["sentences"]:
        facts = item["facts"]
        lines += [render_block(facts, item["text"], slots.get(facts[0]), item["type"]), ""]
    withheld = page["withheld"]
    if withheld["count"]:
        why = "; ".join(f"{v['count']} {v['why']}" for v in withheld["reasons"].values())
        lines += [f"_{withheld['count']} sentences withheld ({why})._", ""]
    lines.append("_Every sentence above is a word-for-word span of a memory, or a fixed template "
                 "over its values. No model wrote any of it._")
    return "\n".join(lines) + "\n"


def parse_file(text: str) -> ParsedFile:
    """Blocks and page id of a file as it is now (edited or not)."""
    page = _PAGE.search(text)
    page_id = dict(_ATTR.findall(page.group(1))).get("id") if page else None
    blocks, bad = [], 0
    for match in _BLOCK.finditer(text):
        attrs = dict(_ATTR.findall(match.group(1)))
        facts = tuple(f for f in attrs.get("fact", "").split(",") if f)
        if not facts or not re.fullmatch(r"[0-9a-f]{16}", attrs.get("hash", "")):
            bad += 1
            continue
        slot = attrs.get("slot")
        blocks.append(Block(facts, attrs["hash"], None if slot in (None, "-") else slot,
                            attrs.get("type", ""), unescape(match.group(2)).strip()))
    return ParsedFile(page_id, tuple(blocks), bad)


def edited(block: Block) -> bool:
    """The block's text no longer hashes to its marker: a person changed it (or emptied it)."""
    return bool(block.text) and block_hash(block.text) != block.hash


def emptied(block: Block) -> bool:
    """The text between the markers was deleted: a deleted line, never a forget."""
    return not block.text.strip()


__all__ = ["Block", "HEADER_NOTE", "ParsedFile", "edited", "emptied", "escape", "parse_file",
           "render_block", "render_page", "unescape"]
