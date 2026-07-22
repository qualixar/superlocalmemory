# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Fact-augmented key expansion (Phase 4, T3b).

Generates *alternate keys* for a fact — synonyms, aliases, and paraphrases —
that get indexed in ``fact_expansion_fts`` and UNION'd into BM25 retrieval, so a
query for "automobile" or "the Big Apple" can match a fact that only says "car"
or "NYC".

Two tiers, matching the rest of SLM:
  * Mode A (zero-LLM): pulls the fact's resolved entities' canonical names and
    aliases from SLM's own entity graph — no model, no cost, and it reuses the
    entity resolution already done at ingest.
  * Mode B/C: additionally asks the LLM for a few paraphrase keywords (own
    prompt, fail-open, bounded).

Keys already present in the fact's content are dropped — indexing them twice
adds nothing. Returns a single space-joined string ready for the FTS row.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)

# Bound the LLM enrichment so a pathological response can't bloat the index.
_MAX_LLM_KEYS = 8
_MAX_KEY_LEN = 60


class KeyExpander:
    """Produces alternate search keys for a fact (T3b)."""

    __slots__ = ("_db", "_llm")

    def __init__(self, db: DatabaseManager, llm: Any = None) -> None:
        self._db = db
        self._llm = llm

    def expand(self, fact: AtomicFact, profile_id: str, mode: str = "a") -> str:
        """Return space-joined alternate keys for ``fact`` (may be empty)."""
        keys: set[str] = set()
        keys |= self._alias_keys(fact, profile_id)
        if mode in ("b", "c") and self._llm_available():
            keys |= self._llm_keys(fact)

        content_low = (getattr(fact, "content", "") or "").lower()
        cleaned = {
            k.strip() for k in keys
            if k and k.strip() and len(k.strip()) <= _MAX_KEY_LEN
            and k.strip().lower() not in content_low
        }
        return " ".join(sorted(cleaned))

    # -- Mode A: entity aliases from SLM's own entity graph ------------------

    def _alias_keys(self, fact: AtomicFact, profile_id: str) -> set[str]:
        out: set[str] = set()
        for name in (getattr(fact, "canonical_entities", None) or []):
            if not name:
                continue
            try:
                ent = self._db.get_entity_by_name(name, profile_id)
            except Exception:
                ent = None
            if ent is None:
                continue
            canonical = getattr(ent, "canonical_name", None)
            if canonical:
                out.add(canonical)
            entity_id = getattr(ent, "entity_id", None)
            if not entity_id:
                continue
            try:
                for alias in self._db.get_aliases_for_entity(entity_id, profile_id):
                    a = getattr(alias, "alias", None)
                    if a:
                        out.add(a)
            except Exception:
                continue
        return out

    # -- Mode B/C: LLM paraphrases (own prompt, fail-open) -------------------

    def _llm_available(self) -> bool:
        if self._llm is None:
            return False
        check = getattr(self._llm, "is_available", None)
        try:
            return bool(check()) if callable(check) else bool(check)
        except Exception:
            return False

    def _llm_keys(self, fact: AtomicFact) -> set[str]:
        content = getattr(fact, "content", "") or ""
        if not content:
            return set()
        prompt = (
            "Give 3-6 short alternative search terms (synonyms, aliases, or "
            "paraphrases) that a person might use to look up the memory below. "
            "Output ONLY a comma-separated list, no numbering, no explanation.\n\n"
            f"Memory: {content}"
        )
        raw = self._invoke_llm(prompt)
        if not raw:
            return set()
        seen: list[str] = []
        for part in raw.replace("\n", ",").split(","):
            p = part.strip()
            if p and p not in seen:
                seen.append(p)
            if len(seen) >= _MAX_LLM_KEYS:
                break
        return set(seen)

    def _invoke_llm(self, prompt: str) -> str:
        """Call whatever generation method the injected LLM exposes; fail-open."""
        for meth in ("generate", "complete", "chat"):
            fn = getattr(self._llm, meth, None)
            if callable(fn):
                try:
                    out = fn(prompt)
                    return out if isinstance(out, str) else str(out or "")
                except Exception as exc:
                    logger.debug("KeyExpander LLM (%s) failed: %s", meth, exc)
                    return ""
        return ""
