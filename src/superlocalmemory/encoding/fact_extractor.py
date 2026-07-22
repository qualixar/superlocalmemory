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
  - Typed fact classification (episodic / semantic / opinion / temporal)
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
    ) -> str: ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Minimal interface for computing embeddings (Mode A type classification)."""

    def embed(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Constants — regex patterns, markers, templates
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})"                                    # ISO
    r"|\b(\d{1,2}/\d{1,2}/\d{2,4})"                             # US
    r"|\b((?:January|February|March|April|May|June|July"
    r"|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?)"                               # Month Day Year
    r"|\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+)\b",
    re.IGNORECASE,
)

_INTERVAL_RE = re.compile(
    r"\b(?:from|between)\s+(.+?)\s+(?:to|and|until|through)\s+(.+?)(?:[.,;]|$)",
    re.IGNORECASE,
)

_ENTITY_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3})\b"    # Capitalized word sequences
    r"|\b([A-Z]{2,})\b"                              # ALL-CAPS abbreviations (NYU, MIT)
)

_QUOTED_RE = re.compile(r'"([^"]+)"')  # Quoted strings as entities

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

_TEMPORAL_MARKERS = re.compile(
    r"\b(?:deadline|due date|expires?|scheduled|appointment|meeting|"
    r"on \w+day|at \d{1,2}:\d{2}|by \w+|until|before|after|"
    r"in \d+ (?:days?|weeks?|months?|years?)|"
    r"next week|next month|this weekend|tomorrow|yesterday)\b",
    re.IGNORECASE,
)

_EMOTIONAL_KEYWORDS = frozenset({
    "love", "hate", "amazing", "terrible", "wonderful", "awful", "excited",
    "angry", "happy", "sad", "scared", "thrilled", "devastated", "furious",
    "anxious", "grateful", "disappointed", "proud", "embarrassed", "jealous",
    "best", "worst", "incredible", "horrible", "fantastic", "miserable",
})

_FILLER_PREFIXES = (
    "good to see", "nice to", "hello", "hi ", "hey ", "how are you",
    "thanks", "thank you", "bye", "goodbye", "see you", "take care",
    "sure thing", "no problem", "okay",
)


# ---------------------------------------------------------------------------
# LLM Prompt Templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise fact extraction engine for a memory system.\n"
    "Given conversation turns, extract 2-5 atomic facts. Rules:\n"
    "1. Use EXPLICIT NAMES — never pronouns (he/she/they/it). Every fact "
    "must name the subject explicitly.\n"
    "2. Each fact must be a COMPLETE, STANDALONE statement understandable "
    "without the original conversation.\n"
    "3. Convert ALL relative time to ABSOLUTE dates when possible. "
    "'Yesterday' with session date 2024-01-15 becomes '2024-01-14'. "
    "'Next month' becomes the actual month and year.\n"
    "4. Resolve ALL coreferences. 'He went there' must become "
    "'[Person name] went to [Place name]'.\n"
    "5. Extract relationships between people when mentioned.\n"
    "6. Extract preferences, opinions, and experiences as SEPARATE facts.\n"
    "7. Skip greetings, filler, social pleasantries, and confirmations.\n"
    "8. For opinions, include a confidence between 0.0-1.0.\n\n"
    "Classify each fact:\n"
    "- episodic: personal event or experience (visited, attended, did)\n"
    "- semantic: objective fact about the world (jobs, locations, relations)\n"
    "- opinion: subjective belief or preference (likes, thinks, prefers)\n"
    "- temporal: time-bound fact with dates or deadlines\n\n"
    "Respond ONLY with a JSON array. Example:\n"
    '[{"text":"Alice works at Google as a software engineer",'
    '"fact_type":"semantic","entities":["Alice","Google"],'
    '"referenced_date":null,"importance":7,"confidence":0.95},'
    '{"text":"Alice prefers Python over Java",'
    '"fact_type":"opinion","entities":["Alice"],'
    '"referenced_date":null,"importance":5,"confidence":0.8}]'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def _extract_date_string(text: str) -> str | None:
    """Extract the first recognizable date string from text."""
    match = _DATE_RE.search(text)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group.strip()
    return None


def _try_parse_date(raw: str, reference_date: str | None = None) -> str | None:
    """Attempt to resolve a date string to ISO format.

    Uses dateutil.parser for structured dates and dateparser for
    relative expressions ("last Monday", "next week").
    Returns None on failure — never raises.
    """
    if not raw:
        return None

    # Fast path: already ISO
    iso_match = re.match(r"^\d{4}-\d{2}-\d{2}$", raw.strip())
    if iso_match:
        return raw.strip()

    # dateutil for structured dates (March 15, 2026 / 3/15/2026)
    try:
        from dateutil import parser as du_parser
        result = du_parser.parse(raw, fuzzy=True)
        return result.date().isoformat()
    except Exception:
        pass

    # V3.3.21: Rule-based relative date resolution (no dateparser dependency).
    # Handles the 90% case: yesterday, today, tomorrow, last X, next X.
    raw_lower = raw.strip().lower()
    if reference_date:
        try:
            from datetime import datetime, timedelta
            ref_dt = du_parser.parse(reference_date)
            _RELATIVE_MAP: dict[str, int] = {
                "yesterday": -1, "today": 0, "tomorrow": 1,
                "the day before": -2, "the other day": -2,
                "day before yesterday": -2,
            }
            if raw_lower in _RELATIVE_MAP:
                resolved = ref_dt + timedelta(days=_RELATIVE_MAP[raw_lower])
                return resolved.date().isoformat()
            # "last week" = -7, "last month" ≈ -30, "last year" = -365
            if raw_lower == "last week":
                return (ref_dt - timedelta(days=7)).date().isoformat()
            if raw_lower == "last month":
                month = ref_dt.month - 1 or 12
                year = ref_dt.year if ref_dt.month > 1 else ref_dt.year - 1
                return f"{year}-{month:02d}-{ref_dt.day:02d}"
            if raw_lower == "last year":
                return f"{ref_dt.year - 1}-{ref_dt.month:02d}-{ref_dt.day:02d}"
            if raw_lower == "next week":
                return (ref_dt + timedelta(days=7)).date().isoformat()
            if raw_lower == "next month":
                month = ref_dt.month + 1 if ref_dt.month < 12 else 1
                year = ref_dt.year if ref_dt.month < 12 else ref_dt.year + 1
                return f"{year}-{month:02d}-{ref_dt.day:02d}"
        except Exception:
            pass

    # dateparser for complex relative dates (optional dependency)
    try:
        import dateparser
        settings: dict[str, Any] = {"PREFER_DATES_FROM": "past"}
        if reference_date:
            ref = dateparser.parse(reference_date)
            if ref:
                settings["RELATIVE_BASE"] = ref
        result = dateparser.parse(raw, settings=settings)
        if result:
            return result.date().isoformat()
    except ImportError:
        pass
    except Exception:
        pass

    return None


def _extract_interval(text: str, ref_date: str | None = None) -> tuple[str | None, str | None]:
    """Extract temporal interval (start, end) from text."""
    match = _INTERVAL_RE.search(text)
    if not match:
        return None, None
    start_raw, end_raw = match.group(1).strip(), match.group(2).strip()
    return _try_parse_date(start_raw, ref_date), _try_parse_date(end_raw, ref_date)


def _extract_entities(text: str) -> list[str]:
    """Extract candidate entity names from text using regex heuristics."""
    entities: set[str] = set()

    # Capitalized word sequences (proper nouns)
    for match in _ENTITY_RE.finditer(text):
        candidate = (match.group(1) or match.group(2) or "").strip()
        # Filter common English words that start sentences
        # Check first word of multi-word candidates against stop list
        _first_word = candidate.split()[0].lower() if candidate else ""
        if _first_word not in {
            "the", "this", "that", "these", "those", "what", "when", "where",
            "which", "how", "who", "why", "also", "then", "just", "very",
            "really", "actually", "maybe", "well", "still", "even",
            "she", "he", "they", "them", "her", "him", "his", "its",
            "but", "and", "not", "yes", "yeah", "sure", "okay", "ok",
            "here", "there", "now", "today", "some", "all", "any",
            "been", "being", "have", "has", "had", "was", "were",
            "for", "with", "from", "about", "into", "over",
            # Sentence starters and conversational words
            "wow", "did", "so", "gonna", "got", "by", "thanks", "thank",
            "hey", "hi", "hello", "bye", "good", "great", "nice", "cool",
            "right", "like", "know", "think", "feel", "want", "need",
            "make", "take", "give", "tell", "said", "told", "get",
            "let", "can", "will", "would", "could", "should", "might",
            "much", "many", "more", "most", "lot", "way", "thing",
            "something", "anything", "everything", "nothing", "someone",
            "it", "my", "your", "our", "their", "me", "you", "we", "us",
            "do", "does", "if", "or", "no", "to", "at", "on", "in",
            "up", "out", "off", "too", "go", "come", "see", "look",
            "say", "ask", "try", "keep", "put", "run", "set", "move",
            "call", "end", "start", "find", "show", "hear", "play",
            "work", "read", "talk", "turn", "help", "miss", "hope",
            "love", "hate", "wish", "seem", "mean", "mind", "care",
        }:
            entities.add(candidate)

    # Quoted strings
    for match in _QUOTED_RE.finditer(text):
        quoted = match.group(1).strip()
        if len(quoted) >= 2:
            entities.add(quoted)

    return sorted(entities)


def _classify_sentence(sentence: str) -> FactType:
    """Classify a sentence into a FactType using keyword markers."""
    if _TEMPORAL_MARKERS.search(sentence):
        return FactType.TEMPORAL
    if _OPINION_MARKERS.search(sentence):
        return FactType.OPINION
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
        FactType.TEMPORAL: SignalType.TEMPORAL,
    }
    return mapping.get(ft, SignalType.FACTUAL)


def _is_filler(text: str) -> bool:
    """Return True if text is a greeting, filler, or social pleasantry."""
    low = text.strip().lower()
    return any(low.startswith(prefix) for prefix in _FILLER_PREFIXES)


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
    ) -> None:
        self._config = config
        self._llm = llm
        self._embedder = embedder
        self._mode = mode

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

        chunks = chunk_turns(turns, self._config.chunk_size, overlap=2)
        all_facts: list[AtomicFact] = []

        for chunk in chunks:
            chunk_facts = self._extract_chunk(
                chunk, session_id, session_date, speaker_a, speaker_b,
            )
            all_facts.extend(chunk_facts)

        return self._deduplicate(all_facts)

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
            self._mode in (Mode.B, Mode.C)
            and self._llm is not None
            and self._llm.is_available()
        )
        if use_llm:
            facts = self._extract_llm(
                turns, session_id, session_date, speaker_a, speaker_b,
            )
            if facts:
                return facts
            # Fallback to local if LLM produced nothing
            logger.info("LLM extraction returned no facts, falling back to local.")

        return self._extract_local(
            turns, session_id, session_date, speaker_a, speaker_b,
        )

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
        combined = "\n".join(turns)
        raw_sentences = _split_sentences(combined)
        if not raw_sentences:
            raw_sentences = [t.strip() for t in turns if len(t.strip()) >= 8]

        # V3.3.12: Sliding window of 2 sentences to preserve cross-sentence context.
        # "She enrolled at NYU. Starting January 2024." → becomes one combined fact.
        sentences = list(raw_sentences)  # Keep originals
        for i in range(len(raw_sentences) - 1):
            pair = raw_sentences[i].rstrip() + " " + raw_sentences[i + 1].lstrip()
            if len(pair) <= 300:  # Only combine if not too long
                sentences.append(pair)

        # Build entity frequency map for importance scoring
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

            # V3.3.21: Resolve [Speaker]: prefix AND first-person pronouns.
            # "[Caroline]: I went to the gym" → "Caroline went to the gym"
            # This makes facts self-contained and entity-rich for retrieval.
            # Previous: just prepended "Caroline: I went..." which left pronouns.
            import re as _re
            _resolved_speaker: str | None = None
            _spk_match = _re.match(r"^\[([A-Za-z ]+)\]:?\s*", normalized)
            if _spk_match:
                _resolved_speaker = _spk_match.group(1)
                rest = normalized[_spk_match.end():]
                # Replace first-person pronouns with speaker name
                rest = _re.sub(r"\bI'm\b", f"{_resolved_speaker} is", rest)
                rest = _re.sub(r"\bI've\b", f"{_resolved_speaker} has", rest)
                rest = _re.sub(r"\bI'll\b", f"{_resolved_speaker} will", rest)
                rest = _re.sub(r"\bI'd\b", f"{_resolved_speaker} would", rest)
                rest = _re.sub(r"\bI was\b", f"{_resolved_speaker} was", rest)
                rest = _re.sub(r"\bI am\b", f"{_resolved_speaker} is", rest)
                rest = _re.sub(r"\bI\b", _resolved_speaker, rest)
                rest = _re.sub(r"\bmy\b", f"{_resolved_speaker}'s", rest, flags=_re.IGNORECASE)
                rest = _re.sub(r"\bme\b", _resolved_speaker, rest)
                rest = _re.sub(r"\bmyself\b", _resolved_speaker, rest, flags=_re.IGNORECASE)
                normalized = rest

            # V3.3.21 R4: Post-resolution filler + length check.
            # After stripping [Speaker]: prefix, the remaining text may be
            # just "Hey!", "Yeah totally!", "Thanks Mel!" — pure noise.
            # These passed the pre-resolution filler check because the prefix
            # was still attached. Re-check after resolution.
            if _is_filler(normalized) or len(normalized.strip()) < 20:
                continue

            entities = _extract_entities(normalized)
            # Ensure resolved speaker is in entities list
            if _resolved_speaker and _resolved_speaker not in entities:
                entities = [_resolved_speaker] + entities
            fact_type = _classify_sentence(normalized)

            # Three-date model: extract and resolve relative dates
            raw_date = _extract_date_string(normalized)
            referenced_date = _try_parse_date(raw_date, session_date) if raw_date else None
            interval_start, interval_end = _extract_interval(normalized, session_date)

            # Resolve relative dates in content for better retrieval
            # "I went yesterday" + session_date=2023-05-08 → "I went on 2023-05-07"
            if raw_date and referenced_date and raw_date.lower() in (
                "yesterday", "today", "last week", "last month", "last year",
                "this morning", "this afternoon", "this evening",
                "the other day", "recently", "the day before",
            ):
                date_str = referenced_date[:10]  # YYYY-MM-DD
                normalized = normalized.replace(raw_date, f"on {date_str}")

            has_date = referenced_date is not None or interval_start is not None
            importance = _score_importance(normalized, entities, entity_freq, has_date)

            if importance < self._config.min_fact_confidence:
                continue

            # V3.3.12: Speaker inference removed — result was never stored in AtomicFact.
            # The speaker info is preserved in verbatim facts via [Speaker]: prefix.

            facts.append(AtomicFact(
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
            ))

        # Cap at max_facts_per_chunk, keeping highest importance
        facts.sort(key=lambda f: f.importance, reverse=True)
        return facts[: self._config.max_facts_per_chunk]

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
        speakers = []
        if speaker_a:
            speakers.append(f"Speaker A: {speaker_a}")
        if speaker_b:
            speakers.append(f"Speaker B: {speaker_b}")
        speaker_info = ", ".join(speakers) if speakers else "unknown"

        prompt = (
            f"Extract atomic facts from the following conversation.\n"
            f"Speakers: {speaker_info}\n"
            f"Conversation date: {session_date or 'unknown'}\n\n"
            f"--- CONVERSATION ---\n{conversation_text}\n--- END ---\n\n"
            f"Rules:\n"
            f"- Extract 2-5 comprehensive, self-contained facts.\n"
            f"- Use explicit names (never pronouns).\n"
            f"- Each fact must make sense WITHOUT the original conversation.\n"
            f"- For dates mentioned (\"yesterday\", \"next week\"), resolve to "
            f"ISO format relative to {session_date or 'today'}.\n"
            f"- Skip greetings, filler, and confirmations.\n"
            f"- importance: 1 (trivial) to 10 (critical)\n"
            f"- confidence: 0.0 (uncertain) to 1.0 (definite)\n\n"
            f"Respond with ONLY a JSON array."
        )

        try:
            raw = self._llm.generate(  # type: ignore[union-attr]
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )
            facts = self._parse_llm_response(raw, session_id, session_date)
            return self._reflexion_refine(conversation_text, facts)
        except Exception as exc:
            logger.warning("LLM fact extraction failed: %s", exc)
            return []

    def _reflexion_refine(
        self, source_text: str, facts: list[AtomicFact],
    ) -> list[AtomicFact]:
        """Wave Q1: bounded, fail-open entity self-review (Mode B/C only).

        Reached only from the LLM extraction path, so Mode A never runs this.
        Any failure returns the facts unchanged.
        """
        if not facts or not getattr(
            self._config, "enable_entity_reflexion", True,
        ):
            return facts
        try:
            from superlocalmemory.encoding.entity_reflexion import EntityReflexion

            reflexion = EntityReflexion(
                self._llm,
                max_facts=getattr(self._config, "reflexion_max_facts", 8),
            )
            return reflexion.refine(source_text, facts)
        except Exception as exc:
            logger.debug("Entity reflexion skipped: %s", exc)
            return facts

    def _parse_llm_response(
        self,
        raw: str,
        session_id: str,
        session_date: str | None,
    ) -> list[AtomicFact]:
        """Parse JSON array from LLM response into AtomicFact list."""
        if not raw or not raw.strip():
            return []

        # Extract JSON array from potentially wrapped response
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                logger.warning("No JSON array found in LLM response.")
                return []
            items = json.loads(match.group())
            if not isinstance(items, list):
                return []
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("JSON parse error in LLM fact response: %s", exc)
            return []

        facts: list[AtomicFact] = []
        for item in items[:10]:  # Hard cap at 10 per chunk
            if not isinstance(item, dict):
                continue
            fact = self._item_to_fact(item, session_id, session_date)
            if fact is not None:
                facts.append(fact)

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
        if not text or len(text) < 8 or _is_filler(text):
            return None

        # Fact type
        raw_type = str(item.get("fact_type", item.get("type", "semantic"))).lower()
        type_map = {
            "episodic": FactType.EPISODIC,
            "experience": FactType.EPISODIC,
            "semantic": FactType.SEMANTIC,
            "world": FactType.SEMANTIC,
            "opinion": FactType.OPINION,
            "temporal": FactType.TEMPORAL,
        }
        fact_type = type_map.get(raw_type, FactType.SEMANTIC)

        # Entities
        raw_entities = item.get("entities", [])
        if isinstance(raw_entities, list):
            entities = [str(e).strip() for e in raw_entities if str(e).strip()]
        elif isinstance(raw_entities, str):
            entities = [raw_entities.strip()] if raw_entities.strip() else []
        else:
            entities = _extract_entities(text)

        # Referenced date — from LLM or inferred
        ref_date_raw = item.get("referenced_date") or item.get("date")
        referenced_date: str | None = None
        if ref_date_raw and str(ref_date_raw).strip().lower() != "null":
            referenced_date = _try_parse_date(str(ref_date_raw), session_date)

        # Interval
        interval_start = item.get("interval_start")
        interval_end = item.get("interval_end")
        if interval_start:
            interval_start = _try_parse_date(str(interval_start), session_date)
        if interval_end:
            interval_end = _try_parse_date(str(interval_end), session_date)

        # Importance (LLM returns 1-10, we normalize to 0.0-1.0)
        raw_importance = item.get("importance", 5)
        try:
            importance = min(1.0, max(0.0, float(raw_importance) / 10.0))
        except (TypeError, ValueError):
            importance = 0.5

        # Confidence
        raw_conf = item.get("confidence", 0.8)
        try:
            confidence = min(1.0, max(0.0, float(raw_conf)))
        except (TypeError, ValueError):
            confidence = 0.8

        return AtomicFact(
            fact_id=_new_id(),
            content=text,
            fact_type=fact_type,
            entities=entities,
            observation_date=session_date,
            referenced_date=referenced_date,
            interval_start=interval_start,
            interval_end=interval_end,
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
        seen: dict[str, AtomicFact] = {}
        for fact in facts:
            key = re.sub(r"\s+", " ", fact.content.lower().strip())
            existing = seen.get(key)
            if existing is None or fact.importance > existing.importance:
                seen[key] = fact
        return list(seen.values())
