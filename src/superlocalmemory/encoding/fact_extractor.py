# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Fact extraction — converts raw conversation turns into structured AtomicFacts.

Three extraction strategies aligned to operating modes:
  Mode A  Zero LLM — regex entities, date inference, keyword type classification.
  Mode B  Local Ollama — LLM-guided extraction with JSON output, Mode A fallback.
  Mode C  Cloud LLM — narrative fact extraction (2-5 per chunk), richest quality.

This module is the primary driver of encoding quality. Competitor analysis
(EverMemOS 93%, Hindsight 89.6%, Mastra 94.9%) shows that structured
extraction at encoding time — not retrieval sophistication — accounts for
the majority of benchmark score differences.

Key patterns implemented:
  - Conversation chunking (5-10 turns, 2-turn overlap)
  - Three-date temporal model (observation, referenced, interval)
  - Typed fact classification (episodic / semantic / opinion / prospective)
  - Importance scoring (entity frequency + emotional markers + recency)
  - Narrative fact extraction in LLM modes (self-contained, context-rich)

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Protocol, runtime_checkable

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding import llm_kind_hint
from superlocalmemory.encoding.date_resolution import keep_supported_dates, try_parse_date
from superlocalmemory.encoding.entity_mentions import extract_mentions
from superlocalmemory.encoding.fact_extraction_result import (
    FactExtraction,
    for_review,
    partition_facts,
)
from superlocalmemory.encoding.interval_scan import interval_pattern
from superlocalmemory.encoding.prospective_markers import looks_prospective
from superlocalmemory.encoding.speaker_spans import bounded_turns, normalize_speaker, speaker_spans
from superlocalmemory.storage.models import AtomicFact, FactType, Mode, SignalType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — accept any LLM / embedder without importing concrete classes
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackboneProtocol(Protocol):
    """Minimal interface the fact extractor needs from an LLM."""

    def is_available(self) -> bool: ...
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str: ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Minimal interface for computing embeddings (Mode A type classification)."""

    def embed(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Constants — regex patterns, markers, templates
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})"  # ISO
    r"|\b(\d{1,2}/\d{1,2}/\d{2,4})"  # US
    r"|\b(\d{1,2}\s+(?:January|February|March|April|May|June|July"
    r"|August|September|October|November|December)\s+\d{4}\b)"
    r"|\b((?:January|February|March|April|May|June|July"
    r"|August|September|October|November|December)"
    r"\s+(?:\d{4}\b|\d{1,2}\b(?:,?\s+\d{4})?))"  # Month Day Year
    r"|\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+)\b",
    re.IGNORECASE,
)

_INTERVAL_RE = re.compile(
    r"\b(?:from|between)\s+(.+?)\s+(?:to|and|until|through)\s+(.+?)(?:[.,;]|$)",
    re.IGNORECASE,
)
#: Same result as ``_INTERVAL_RE.search``, in bounded time (interval_scan).
_INTERVAL = interval_pattern(_INTERVAL_RE, r"\b(?:from|between)", r"(?:to|and|until|through)")

_OPINION_MARKERS = re.compile(
    r"\b(?:I think|I believe|I feel|in my opinion|I prefer|I like|I love|"
    r"I hate|I want|I need|I wish|personally|my favorite|"
    r"probably|seems like|might be|could be|I guess|"
    r"thinks?|believes?|prefers?|preferred|likes?|liked|loves?|loved|hates?|hated|"
    r"overrated|underrated|best|worst|favorite|"
    r"should|shouldn't|ought to|better|rather)\b",
    re.IGNORECASE,
)

_EXPERIENCE_MARKERS = re.compile(
    r"\b(?:I went|I visited|I saw|I met|I did|I made|I had|I was|"
    r"we went|we visited|we had|I've been|I've done|I used to|"
    r"I remember|I once|last time I|when I was|my experience)\b",
    re.IGNORECASE,
)


_EMOTIONAL_KEYWORDS = frozenset(
    {
        "love",
        "hate",
        "amazing",
        "terrible",
        "wonderful",
        "awful",
        "excited",
        "angry",
        "happy",
        "sad",
        "scared",
        "thrilled",
        "devastated",
        "furious",
        "anxious",
        "grateful",
        "disappointed",
        "proud",
        "embarrassed",
        "jealous",
        "best",
        "worst",
        "incredible",
        "horrible",
        "fantastic",
        "miserable",
    }
)

_FILLER_PREFIXES = (
    "good to see",
    "nice to",
    "hello",
    "hi ",
    "hey ",
    "how are you",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "see you",
    "take care",
    "sure thing",
    "no problem",
    "okay",
)


# ---------------------------------------------------------------------------
# LLM Prompt Templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Extract 1-5 atomic statements supported by the source, or [] for no facts.\n"
    "Do not pad the answer. Copy the source wording whenever possible.\n"
    "Rewrite first person as 'the user' only for the user; use the explicit "
    "speaker label for another speaker. Never rewrite words inside quotations.\n"
    "Preserve speaker turns and quote attribution. Never infer names, employers, "
    "preferences, relationships or identities. Unresolved pronouns stay unresolved.\n"
    "Keep every number, unit, date, negation, qualifier and step order exactly.\n"
    "Leave relative date words in the text; deterministic code resolves date fields "
    "against the conversation date. Unknown date means no date inference.\n"
    "No calculations, implications, commentary, repeated or near-duplicate facts.\n"
    "Respond ONLY with a JSON array.\n"
    "Return objects with fields: text (supported statement), "
    "fact_type (episodic, semantic, opinion or prospective), entities (names copied "
    "from source), referenced_date (null unless explicit), importance (1-10), "
    "confidence (0-1). No example content is provided."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    from superlocalmemory.encoding.speaker_spans import split_sentences

    return split_sentences(text)


def _extract_date_string(text: str) -> str | None:
    """Extract the first recognizable date string from text."""
    match = _DATE_RE.search(text)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group.strip()
    return None


#: Date resolution lives in ``date_resolution`` with the guard that keeps
#: measurements ("2004.6 ms", "4.1.21", "$12.50") away from the date parser.
_try_parse_date = try_parse_date


def _extract_interval(text: str, ref_date: str | None = None) -> tuple[str | None, str | None]:
    """Extract temporal interval (start, end) from text."""
    found = _INTERVAL.groups(text)
    if found is None:
        return None, None
    start_raw, end_raw = found[0].strip(), found[1].strip()
    return _try_parse_date(start_raw, ref_date), _try_parse_date(end_raw, ref_date)


def _extract_entities(text: str) -> list[str]:
    """Candidate entity names in ``text``: the shared Mode A rules
    (encoding/entity_mentions), used by every write path and by questions."""
    return extract_mentions(text)


def _classify_sentence(sentence: str) -> FactType:
    """Classify a sentence into a FactType using keyword markers.

    Opinion is asked first, and the router asks in the same order. They used to
    differ, so "I think we should ship next week" was a plan on one path and an
    opinion on the other, for the same sentence. Opinion wins because an opinion
    about a plan is not a commitment, and the list of what is coming up should
    hold commitments.
    """
    if _OPINION_MARKERS.search(sentence):
        return FactType.OPINION
    if looks_prospective(sentence):
        return FactType.PROSPECTIVE
    if _EXPERIENCE_MARKERS.search(sentence):
        return FactType.EPISODIC
    return FactType.SEMANTIC


def _score_importance(
    text: str,
    entities: list[str],
    entity_frequency: dict[str, int],
    has_date: bool,
) -> float:
    """Score importance 0.0-1.0 based on entity frequency, emotion, temporality.

    Scoring formula:
      base  = 0.3
      +0.2  if contains emotional keywords
      +0.2  if temporally grounded (has a date reference)
      +0.3  scaled by entity prominence (max entity frequency / total)
    """
    score = 0.3

    # Emotional boost
    words = set(text.lower().split())
    if words & _EMOTIONAL_KEYWORDS:
        score += 0.2

    # Temporal boost
    if has_date:
        score += 0.2

    # Entity prominence boost (frequent entities are important)
    if entities and entity_frequency:
        total = sum(entity_frequency.values()) or 1
        max_freq = max((entity_frequency.get(e, 0) for e in entities), default=0)
        score += 0.3 * (max_freq / total)

    return min(1.0, round(score, 3))


def _signal_from_fact_type(ft: FactType) -> SignalType:
    """Map FactType to SignalType for V2 compatibility."""
    mapping = {
        FactType.EPISODIC: SignalType.FACTUAL,
        FactType.SEMANTIC: SignalType.FACTUAL,
        FactType.OPINION: SignalType.OPINION,
        FactType.PROSPECTIVE: SignalType.TEMPORAL,
    }
    return mapping.get(ft, SignalType.FACTUAL)


#: Model requests per save; the coverage pass keeps what they do not reach.
_MAX_EXTRACTION_CHUNKS = 5


def _is_filler(text: str) -> bool:
    """Return True if text is a greeting, filler, or social pleasantry."""
    low = text.strip().lower()
    return any(low.startswith(prefix) for prefix in _FILLER_PREFIXES)


def sentence_fact(
    normalized: str,
    session_id: str,
    session_date: str | None,
    entity_freq: dict[str, int],
) -> AtomicFact:
    """One source sentence as a typed, dated, scored fact (no model call)."""
    entities = _extract_entities(normalized)
    fact_type = _classify_sentence(normalized)
    raw_date = _extract_date_string(normalized)
    referenced_date = _try_parse_date(raw_date, session_date) if raw_date else None
    interval_start, interval_end = _extract_interval(normalized, session_date)
    importance = _score_importance(
        normalized,
        entities,
        entity_freq,
        referenced_date is not None or interval_start is not None,
    )
    return AtomicFact(
        fact_id=_new_id(),
        content=normalized,
        fact_type=fact_type,
        entities=entities,
        observation_date=session_date,
        referenced_date=referenced_date,
        interval_start=interval_start,
        interval_end=interval_end,
        confidence=0.7 if fact_type == FactType.SEMANTIC else 0.6,
        importance=importance,
        session_id=session_id,
        signal_type=_signal_from_fact_type(fact_type),
    )


# ---------------------------------------------------------------------------
# Chunk builder
# ---------------------------------------------------------------------------


def chunk_turns(
    turns: list[str],
    chunk_size: int = 10,
    overlap: int = 2,
) -> list[list[str]]:
    """Group conversation turns into overlapping chunks.

    Each chunk is up to ``chunk_size`` turns with ``overlap`` turns
    carried over from the previous chunk to preserve cross-boundary context.
    Trailing fragments smaller than ``overlap + 1`` are merged into the
    final chunk to avoid low-context extraction passes.
    """
    if not turns:
        return []
    if len(turns) <= chunk_size:
        return [list(turns)]

    chunks: list[list[str]] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(turns):
        end = min(start + chunk_size, len(turns))
        remaining_after = len(turns) - end
        # Merge tiny trailing fragment into current chunk
        if 0 < remaining_after < overlap + 1:
            end = len(turns)
        chunks.append(list(turns[start:end]))
        if end >= len(turns):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# FactExtractor
# ---------------------------------------------------------------------------


class FactExtractor:
    """Extract structured AtomicFacts from conversation turns.

    Strategies:
      Mode A — Rule-based: regex entities, keyword classification, heuristic importance.
      Mode B — Local LLM (Ollama): structured JSON extraction, Mode A fallback.
      Mode C — Cloud LLM: narrative fact extraction (2-5 per chunk), richest output.
    """

    def __init__(
        self,
        config: EncodingConfig,
        llm: LLMBackboneProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        mode: Mode = Mode.A,
        *,
        llm_extract_kinds: bool = True,
    ) -> None:
        self._config = config
        self._llm = llm
        self._embedder = embedder
        self._mode = mode
        #: memory_kinds.llm_extract_kinds — off sends exactly the 4.1.18 prompt.
        self._extract_kinds = llm_extract_kinds is True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_facts(
        self,
        turns: list[str],
        session_id: str,
        session_date: str | None = None,
        speaker_a: str = "",
        speaker_b: str = "",
    ) -> list[AtomicFact]:
        """Extract structured atomic facts from conversation turns.

        Chunks the conversation into overlapping windows, extracts facts from
        each chunk, and deduplicates the merged results.

        Args:
            turns: Raw conversation turn strings.
            session_id: Identifier for the conversation session.
            session_date: ISO-8601 date of the session (observation date).
            speaker_a: Name/identifier for the first speaker (e.g. user).
            speaker_b: Name/identifier for the second speaker (e.g. assistant).

        Returns:
            Deduplicated list of AtomicFact objects.
        """
        if not turns:
            return []

        chunks = [[turn] for turn in bounded_turns(turns, speaker_a=speaker_a, speaker_b=speaker_b)]
        all_facts: list[AtomicFact] = []
        review_candidates: list[AtomicFact] = []

        cap = self._config.max_facts_per_chunk
        # At most five model requests per save; sentences the extractor does
        # not reach are kept by the coverage pass (encoding.coverage_facts).
        for chunk in chunks[:_MAX_EXTRACTION_CHUNKS]:
            chunk_facts = self._extract_chunk(
                chunk,
                session_id,
                session_date,
                speaker_a,
                speaker_b,
            )
            all_facts.extend(chunk_facts)
            review_candidates.extend(getattr(chunk_facts, "review_candidates", ()))
            if len(self._deduplicate(all_facts)) >= cap:
                break

        return FactExtraction(self._deduplicate(all_facts)[:cap], review_candidates)

    # ------------------------------------------------------------------
    # Chunk-level dispatch
    # ------------------------------------------------------------------

    def _extract_chunk(
        self,
        turns: list[str],
        session_id: str,
        session_date: str | None,
        speaker_a: str,
        speaker_b: str,
    ) -> list[AtomicFact]:
        """Extract facts from a single chunk — dispatches by mode."""
        use_llm = (
            self._mode in (Mode.B, Mode.C) and self._llm is not None and self._llm.is_available()
        )
        review_candidates = ()
        if use_llm:
            facts = self._extract_llm(
                turns,
                session_id,
                session_date,
                speaker_a,
                speaker_b,
            )
            if facts:
                return facts
            review_candidates = getattr(facts, "review_candidates", ())
            # Fallback to local if LLM produced nothing
            logger.info("LLM extraction returned no facts, falling back to local.")

        return FactExtraction(self._extract_local(
            turns,
            session_id,
            session_date,
            speaker_a,
            speaker_b,
        ), review_candidates)

    # ------------------------------------------------------------------
    # Mode A: Rule-based extraction
    # ------------------------------------------------------------------

    def _extract_local(
        self,
        turns: list[str],
        session_id: str,
        session_date: str | None,
        speaker_a: str,
        speaker_b: str,
    ) -> list[AtomicFact]:
        """Rule-based extraction: regex entities, keyword classification, scoring."""
        sentences = []
        for turn in turns:
            for span in speaker_spans(turn, speaker_a or "the user"):
                sentences.extend(_split_sentences(normalize_speaker(span.text, span.speaker)))
        entity_freq: dict[str, int] = {}
        for sent in sentences:
            for ent in _extract_entities(sent):
                entity_freq[ent] = entity_freq.get(ent, 0) + 1

        facts: list[AtomicFact] = []
        seen_texts: set[str] = set()

        for sent in sentences:
            if _is_filler(sent):
                continue
            normalized = sent.strip()
            if normalized in seen_texts or len(normalized) < 20:
                continue
            seen_texts.add(normalized)

            fact = self._local_fact(normalized, session_id, session_date, entity_freq)
            if fact.importance >= self._config.min_fact_confidence:
                facts.append(fact)

        # Cap at max_facts_per_chunk, keeping highest importance
        facts.sort(key=lambda f: f.importance, reverse=True)
        return facts[: self._config.max_facts_per_chunk]

    def _local_fact(
        self,
        normalized: str,
        session_id: str,
        session_date: str | None,
        entity_freq: dict[str, int],
    ) -> AtomicFact:
        return sentence_fact(normalized, session_id, session_date, entity_freq)

    # ------------------------------------------------------------------
    # Mode B/C: LLM-based extraction
    # ------------------------------------------------------------------

    def _extract_llm(
        self,
        turns: list[str],
        session_id: str,
        session_date: str | None,
        speaker_a: str,
        speaker_b: str,
    ) -> list[AtomicFact]:
        """LLM-guided extraction: structured JSON prompt, parsed into AtomicFacts."""
        conversation_text = "\n".join(turns)
        speaker_info = ", ".join(name for name in (speaker_a, speaker_b) if name) or "unknown"
        prompt = (
            f"Extract supported atomic statements. Speakers: {speaker_info}\n"
            f"Conversation date: {session_date or 'unknown'}\n"
            f"--- SOURCE ---\n{conversation_text}\n--- END SOURCE ---\n"
            "Use only source information. Return 1-5 facts or [] in the JSON schema."
        )
        try:
            raw = self._llm.generate(  # type: ignore[union-attr]
                prompt=prompt,
                system=llm_kind_hint.system_prompt(_SYSTEM_PROMPT, self._extract_kinds),
                temperature=0.0,
                max_tokens=1024,
                think=False,
            )
            facts = self._parse_llm_response(raw, session_id, session_date)
            from superlocalmemory.encoding.fact_speakers import grounded_speakers

            grounded = grounded_speakers(facts, conversation_text)
            grounded_ids = {fact.fact_id for fact in grounded}
            ambiguous = [for_review(fact, ("attribution_changed",))
                         for fact in facts if fact.fact_id not in grounded_ids]
            checked = partition_facts(FactExtraction(grounded, ambiguous),
                                      conversation_text, session_date)
            return FactExtraction(keep_supported_dates(
                self._reflexion_refine(conversation_text, checked),
                conversation_text,
                reference_date=session_date,
            ), checked.review_candidates)
        except Exception as exc:
            logger.warning("LLM fact extraction failed: %s", exc)
            return []

    def _reflexion_refine(
        self,
        source_text: str,
        facts: list[AtomicFact],
    ) -> list[AtomicFact]:
        """Ground entities deterministically; a second model call adds no evidence."""
        from superlocalmemory.encoding.fact_grounding import grounded_entities

        return grounded_entities(facts, source_text)

    @staticmethod
    def _find_json_arrays(raw: str) -> list[list]:
        """All answer-candidate JSON arrays inside ``raw``, in trace order.

        Thinking-model traces interleave bracket noise (``[Alice,
        Google]``, ``[1]``, trial fragments) before the answer, so the
        legacy first-``[``-to-last-``]`` span is not valid JSON on real
        traces (#128). Every ``[`` candidate is decoded with
        ``raw_decode`` — which stops at the value's own end, ignoring
        trailing text. A plain single-array response yields exactly the
        array the old span produced.
        """
        decoder = json.JSONDecoder()
        found: list[list] = []
        for match in re.finditer(r"\[", raw):
            try:
                value, _ = decoder.raw_decode(raw, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(value, list) and value and any(isinstance(item, dict) for item in value):
                found.append(value)
        return found

    def _parse_llm_response(
        self,
        raw: str,
        session_id: str,
        session_date: str | None,
    ) -> list[AtomicFact]:
        """Parse JSON array from LLM response into AtomicFact list."""
        if not raw or not raw.strip():
            # #128: this used to fail silently, making Mode B look working
            # while pure Mode A ran underneath. Warn so the degradation
            # is visible in daemon.log.
            logger.warning(
                "LLM fact extraction got empty response text; nothing to parse.",
            )
            return []
        # Best schema fit wins, with later answers winning ties in thinking traces.
        try:
            candidates = self._find_json_arrays(raw)
            if not candidates:
                logger.warning("No JSON array found in LLM response.")
                return []
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("JSON parse error in LLM fact response: %s", exc)
            return []
        facts: list[AtomicFact] = []
        best_fit = -1
        for items in reversed(candidates):
            built: list[AtomicFact] = []
            for item in items[:10]:  # Hard cap at 10 per chunk
                if not isinstance(item, dict):
                    continue
                fact = self._item_to_fact(item, session_id, session_date)
                if fact is not None:
                    built.append(
                        llm_kind_hint.with_hint(fact, item) if self._extract_kinds else fact
                    )
            if not built:
                continue
            fit = sum(
                1
                for item in items[:10]
                if isinstance(item, dict)
                and item.get("text")
                and ("fact_type" in item or "type" in item)
            )
            if fit > best_fit:
                best_fit = fit
                facts = built
        return facts

    def _item_to_fact(
        self,
        item: dict[str, Any],
        session_id: str,
        session_date: str | None,
    ) -> AtomicFact | None:
        """Convert a single LLM JSON item to an AtomicFact.

        Returns None if the item is malformed or is filler.
        """
        text = str(item.get("text", "")).strip()
        from superlocalmemory.encoding.fact_fields import statement_shape

        if not text or len(text) < 8 or not statement_shape(text) or _is_filler(text):
            return None

        from superlocalmemory.encoding.fact_fields import llm_dates, llm_type, unit_score

        fact_type = llm_type(item)
        # Entities
        raw_entities = item.get("entities", [])
        if isinstance(raw_entities, list):
            entities = [str(e).strip() for e in raw_entities if str(e).strip()]
        elif isinstance(raw_entities, str):
            entities = [raw_entities.strip()] if raw_entities.strip() else []
        else:
            entities = _extract_entities(text)

        dates = llm_dates(item, session_date)
        importance = unit_score(item.get("importance", 5), 0.5, 10.0)
        confidence = unit_score(item.get("confidence", 0.8), 0.8)

        return AtomicFact(
            fact_id=_new_id(),
            content=text,
            fact_type=fact_type,
            entities=entities,
            observation_date=session_date,
            **dates,
            confidence=confidence,
            importance=importance,
            session_id=session_id,
            signal_type=_signal_from_fact_type(fact_type),
        )

    # ------------------------------------------------------------------
    # Speaker inference (Mode A heuristic)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_speaker(
        sentence: str,
        turns: list[str],
        speaker_a: str,
        speaker_b: str,
    ) -> str:
        """Infer which speaker said a sentence based on turn position.

        Checks which turn contains the sentence and uses even/odd indexing
        (even = speaker_a, odd = speaker_b by convention).
        """
        if not speaker_a and not speaker_b:
            return ""
        for i, turn in enumerate(turns):
            if sentence in turn:
                return speaker_a if i % 2 == 0 else speaker_b
        return speaker_a or speaker_b

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(facts: list[AtomicFact]) -> list[AtomicFact]:
        """Remove near-duplicate facts by content normalization.

        Uses lowercased, whitespace-collapsed content as dedup key.
        When duplicates exist, keeps the one with higher importance.
        """
        from superlocalmemory.encoding.fact_grounding import fact_key

        seen: dict[tuple, AtomicFact] = {}
        for fact in facts:
            key = fact_key(fact.content)
            existing = seen.get(key)
            if existing is None or fact.importance > existing.importance:
                seen[key] = fact
        from superlocalmemory.encoding.fact_grounding import collapse_partials

        return collapse_partials(list(seen.values()))
