# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.2

"""Topic signature — deterministic, Unicode-safe, ReDoS-resistant 16-char hex.

LLD reference: `.backup/active-brain/lld/LLD-01-context-cache-and-hot-path-hooks.md`
Section 4.2.

Hot-path contract:
  - stdlib-only imports (no third-party packages).
  - Same (NFC-normalized, lowercased) input → same output across Python
    versions, OSes, and locales. Enforced via CI matrix.
  - Patterns are O(n) — no catastrophic backtracking regardless of input.
  - Input is truncated to ``MAX_SIG_INPUT_CHARS`` BEFORE any regex to
    guarantee a hard upper bound on compute time.
  - Budget: <5 ms p95 at 2000 chars, <8 ms at 8000 (see tests).

No ``@lru_cache`` anywhere — per LLD-01 SEC-01-01 / PERF-01-04, caching on
raw prompts would leak secrets in memory and is useless across fresh hook
processes anyway.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# --------------------------------------------------------------------------
# Non-backtracking patterns. Each one is linear in input length.
# --------------------------------------------------------------------------

# CamelCase / PascalCase identifiers (>= 2 upper-case humps).
_CAMEL_PASCAL = re.compile(r"\b[A-Z][a-zA-Z0-9]+[A-Z][a-zA-Z0-9]*\b")
# URLs — capture up to whitespace / angle brackets. Linear in input.
_URL = re.compile(r"https?://[^\s<>]+")
# Paths — absolute POSIX paths. Linear; each segment is a simple char class.
_PATH = re.compile(r"/[^\s/<>]+(?:/[^\s/<>]+)*(?:\.[A-Za-z0-9]+)?")
# Quoted strings (capped at 200 chars to preserve linear time bound).
_QUOTED_DOUBLE = re.compile(r'"([^"]{1,200})"')
_QUOTED_SINGLE = re.compile(r"'([^']{1,200})'")
# Word tokens for content-word extraction. Length >= 3 below filters shorter.
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]{2,}")

# ~120 common English stopwords. Kept inline so the module is stdlib-only.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "did", "do", "does", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "let", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "use", "using", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours", "yourself", "yourselves",
})

MAX_SIG_INPUT_CHARS: int = 4000
_SIG_LEN: int = 16


def _canon(items: list[str]) -> str:
    """Canonicalize a list of tokens: sort, dedupe, join with a sentinel."""
    return "\0".join(sorted(set(items)))


def compute_topic_signature(
    text: str,
    *,
    entity_hits: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Compute a deterministic 16-char hex signature of ``text``.

    Returns ``"0" * 16`` for empty input. Always returns exactly 16
    lowercase hex characters.

    Algorithm:
      1. Truncate input to ``MAX_SIG_INPUT_CHARS`` (ReDoS safety).
      2. NFC-normalize so composed/decomposed Unicode hash identically.
      3. Extract structural tokens (identifiers, URLs, paths, quoted).
      4. Extract word tokens (lowercased); filter stopwords + len<3.
      5. Build bigrams over content words to resist stopword-only collisions.
      6. Sort-dedupe each group; join; SHA-256; take first 16 hex chars.

    ``entity_hits`` (LLD-13 Track C.1) — optional list of entity IDs
    produced by the inline trigram lookup. Backward-compatible default:
    when omitted or empty, the output is BYTE-IDENTICAL to the v3.4.22
    pre-Living-Brain signature. When non-empty, the sorted-deduped IDs
    are mixed into the canonical material as a seventh group so that
    cache probes differentiate semantically-distinct prompts that
    happen to share regex-level tokens.
    """
    if not text:
        return "0" * _SIG_LEN

    # 1. Hard truncation FIRST — bounds regex compute time.
    if len(text) > MAX_SIG_INPUT_CHARS:
        text = text[:MAX_SIG_INPUT_CHARS]

    # 2. NFC normalize. Different input encodings of the same glyph now
    # have identical codepoints before we extract or lowercase.
    text_nfc = unicodedata.normalize("NFC", text)
    lowered = text_nfc.lower()

    # 3. Structural tokens (case-preserving — camelCase carries meaning).
    identifiers = _CAMEL_PASCAL.findall(text_nfc)
    urls = _URL.findall(text_nfc)
    paths = _PATH.findall(text_nfc)
    quoted = _QUOTED_DOUBLE.findall(text_nfc) + _QUOTED_SINGLE.findall(text_nfc)

    # 4. Content words (lowered, stop-filtered, len >= 3).
    words = _WORD.findall(lowered)
    content_words = [w for w in words if w not in _STOPWORDS and len(w) >= 3]

    # 5. Bigrams from the ORIGINAL token stream order. Preserves "foo bar"
    # vs "bar foo" distinction and resists stopword-only differentiation.
    bigrams = [f"{a}_{b}" for a, b in zip(content_words, content_words[1:])]

    # 6. Materialize canonical form and hash.
    groups = [
        _canon(identifiers),
        _canon(urls),
        _canon(paths),
        _canon(quoted),
        _canon(content_words),
        _canon(bigrams),
    ]
    # LLD-13: append entity-hits group ONLY when non-empty. Empty/missing
    # preserves the byte-identical v3.4.22 pre-Living-Brain signature.
    if entity_hits:
        groups.append(_canon([str(e) for e in entity_hits]))
    material = "\0\0".join(groups)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_SIG_LEN]


__all__ = ("compute_topic_signature", "MAX_SIG_INPUT_CHARS")
