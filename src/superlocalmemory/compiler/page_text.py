# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The words of a page: how a memory is cut into sentences, and how a page is identified (D-18).

Nothing here reads a database. A page sentence is one of two things:

* a **span**: one sentence of a fact's own text, exactly as stored (``sentences_of``);
* a **template**: a fixed frame filled with values from the slot tables and the
  verbatim text of cited facts (``compiler/page_templates``).

Pure functions, so the same input always gives the same page.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

KINDS = ("current_state", "story", "topic")
SPAN, TEMPLATE = "span", "template"

#: Routing only (how big a page is; never a threshold that changes memory).
MAX_SENTENCES = 60
MAX_SPANS_PER_FACT = 2
#: A sentence longer than this is not a sentence: it is withheld, not cut (a cut span would
#: still verify, but a page of fragments misleads).
MAX_SENTENCE_CHARS = 320
MIN_STORY_FACTS = 2
MIN_TOPIC_FACTS = 3

#: Why a candidate sentence is not on a page. The page shows the counts (D-18).
UNVERIFIABLE, OVER_LIMIT, TOO_LONG, SOURCE_CHANGED = (
    "unverifiable", "over_limit", "too_long", "source_changed")
REASON_WORDS: Mapping[str, str] = {
    UNVERIFIABLE: "not found word for word in the memory it cites",
    OVER_LIMIT: "over the size of a page",
    TOO_LONG: "too long to be one sentence",
    SOURCE_CHANGED: "its source file changed",
}

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences_of(content: str) -> list[str]:
    """The sentences of a fact's text, each a substring of it, in order (empty ones dropped)."""
    return [s for s in (part.strip() for part in _SPLIT.split(content or "")) if s]


def one_line(text: str) -> str:
    """Whitespace collapsed to single spaces: how a template quotes a fact."""
    return " ".join((text or "").split())


def humanize(key: str) -> str:
    """A slot key part in words: ``name:release_cadence`` -> ``release cadence``."""
    text = (key or "").split(":", 1)[-1] if (key or "").startswith(("name:", "entity:")) else (key or "")
    return text.replace("_", " ").replace("/", " / ").strip()


def day(stamp: Any) -> str:
    """The date part (``YYYY-MM-DD``) of a stored timestamp, or ``''``."""
    text = str(stamp or "")
    return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}", text) else ""


def page_id(profile_id: str, kind: str, subject_key: str) -> str:
    """The same page for the same profile, kind and subject, always."""
    return hashlib.sha256(f"{profile_id}\0{kind}\0{subject_key}".encode()).hexdigest()[:24]


def sentence(seq: int, kind: str, text: str, facts: Sequence[str], *,
             template: str | None = None, params: Mapping[str, Any] | None = None) -> dict:
    """One page sentence as stored: its words, its citations and, for a template, its inputs."""
    return {"seq": seq, "type": kind, "text": text, "facts": sorted(set(facts)),
            "template": template, "params": dict(params or {})}


def renumber(sentences: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [{**s, "seq": i} for i, s in enumerate(sentences)]


def content_hash(title: str, sentences: Sequence[Mapping[str, Any]]) -> str:
    """What makes two builds of a page the same page (words and citations, not times)."""
    body = [[s["type"], s["text"], list(s["facts"]), s.get("template"), s.get("params") or {}]
            for s in sentences]
    return hashlib.sha256(json.dumps([title, body], sort_keys=True).encode("utf-8")).hexdigest()


def block_hash(text: str) -> str:
    """The hash a projected block carries (``compiler/wiki``): of its text, whitespace folded."""
    return hashlib.sha256(one_line(text).encode("utf-8")).hexdigest()[:16]


__all__ = ["KINDS", "MAX_SENTENCES", "MAX_SENTENCE_CHARS", "MAX_SPANS_PER_FACT", "MIN_STORY_FACTS",
           "MIN_TOPIC_FACTS", "OVER_LIMIT", "REASON_WORDS", "SOURCE_CHANGED", "SPAN", "TEMPLATE",
           "TOO_LONG", "UNVERIFIABLE", "block_hash", "content_hash", "day", "humanize",
           "one_line", "page_id", "renumber", "sentence", "sentences_of"]
