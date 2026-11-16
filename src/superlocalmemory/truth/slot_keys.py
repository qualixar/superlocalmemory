# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Tier 0 slot keys: which "value of what" a memory states, decided by rules.

A slot key is ``(subject, relation, qualifier)``: "the release cadence of SLM",
"the voice model Alex prefers for voiceovers". Only memories that state one
current value of something get a key; narratives, notes, questions, negations
and anything long stay slot-less, because a wrong key is worse than none (it
would let a new memory retire an unrelated one).

Rules recognise four shapes, in order:

* an explicit key the caller gives (``subject/relation[/qualifier]``), with the
  value it states (D-36);
* "the <attribute> of <subject> is <value>";
* "<subject>'s <attribute> is <value>";
* "<subject> prefers|chose|switched to|moved to|uses <value> for <purpose>".

Everything here is deterministic and offline. Memories the rules cannot key go
to the judge ladder (WS-2) as SAME_SLOT questions, never to a guess.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

#: Above this a memory is a narrative, not a value statement.
MAX_CHARS = 240
EXPLICIT_CONFIDENCE = 1.0
#: Descriptive only: what decides supersession is ``key_source`` (an explicit
#: key may supersede; a pattern key opens a dispute), never this number. The
#: 15 Nov shadow reading found pattern keys right about the slot but unable to
#: tell a changed value from a paraphrase (blueprint D-10, F-10..F-12).
PATTERN_CONFIDENCE = 0.6

_DETERMINERS = re.compile(r"^(?:the|a|an|my|our|its|their|his|her|your)\s+", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:not|never|no longer|isn't|aren't|wasn't|don't|doesn't|didn't|won't|cannot|can't)\b",
    re.IGNORECASE)
_HEDGE = re.compile(r"\b(?:maybe|might|perhaps|possibly|probably|i think|not sure)\b",
                    re.IGNORECASE)
#: Present tense only: "was"/"were" describe past events, not current values.
_COPULA = r"(?:is now|are now|is|are|has become|changed to|became|=)"
_NAME = r"[\w][\w .@/+#-]{0,59}?"
_ATTR = r"[a-z][\w -]{1,39}?"

_ATTRIBUTE_OF = re.compile(
    rf"^(?:the\s+)?(?P<attr>{_ATTR})\s+(?:of|for)\s+(?P<subj>{_NAME})\s+{_COPULA}\s+"
    r"(?P<val>.+)$", re.IGNORECASE)
_POSSESSIVE = re.compile(
    rf"^(?P<subj>{_NAME})['’]s\s+(?P<attr>{_ATTR})\s+{_COPULA}\s+(?P<val>.+)$",
    re.IGNORECASE)
_CHOICE = re.compile(
    rf"^(?P<subj>{_NAME})\s+(?P<verb>prefers|chose|switched to|moved to|uses|now uses)\s+"
    rf"(?P<val>.+?)\s+(?:for|as)\s+(?P<attr>{_ATTR})$", re.IGNORECASE)

#: Words that turn a would-be subject or attribute into a clause or a narrative.
_CLAUSE_WORDS = frozenset({
    "and", "or", "but", "which", "that", "who", "whom", "whose", "when", "where", "while",
    "because", "if", "then", "than", "after", "before", "since", "until", "asked", "said"})
_PRONOUNS = frozenset({
    "i", "we", "you", "he", "she", "it", "they", "this", "these", "those", "there", "here",
    "me", "us", "them", "someone", "something", "everyone", "everything"})
#: Event nouns: "the modification of X was ..." records something that happened,
#: not the current value of something, and must never supersede.
_EVENT_NOUNS = frozenset({
    "modification", "creation", "deletion", "fix", "update", "change", "commit", "edit",
    "rename", "move", "launch", "push", "merge", "upload", "download", "installation",
    "removal", "addition", "run", "attempt", "result", "outcome", "lot", "number", "part",
    "rest", "most", "some", "kind", "type", "example", "reason"})
#: "is managed by", "is running", "are stored in": the verb is part of the
#: relation, so "device is managed by X" and "device is a MacBook" are
#: different slots rather than one value replacing the other.
_ADVERBS = frozenset({"now", "fully", "currently", "still", "also", "always", "usually",
                      "mostly", "only", "already"})
_PARTICIPLES = frozenset({"built", "run", "made", "held", "kept", "set", "done", "written",
                          "known", "shown", "given", "owned", "led", "paid", "sold", "bound"})
_PREPOSITIONS = frozenset({"by", "in", "at", "on", "to", "for", "with", "from", "as", "into",
                           "under", "behind", "via"})
MAX_ATTRIBUTE_WORDS = 4
MAX_SUBJECT_WORDS = 6

#: Verbs that name one of several current values rather than the only one.
_PLURAL_VERBS = frozenset({"uses", "now uses"})
_VERB_RELATION = {"prefers": "prefers", "chose": "prefers", "switched to": "prefers",
                  "moved to": "prefers", "uses": "uses", "now uses": "uses"}


@dataclass(frozen=True)
class SlotKey:
    subject_key: str
    relation_key: str
    qualifier: str
    cardinality: str
    key_source: str
    confidence: float
    value_hash: str


def normalise_term(text: str) -> str:
    """Lower case, no determiner, words joined by '_', punctuation removed."""
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = _DETERMINERS.sub("", text)
    text = re.sub(r"[^\w\s.@/+#-]", "", text)
    return re.sub(r"\s+", "_", text).strip("_.")


def value_hash(value: str) -> str:
    """The same stated value gives the same hash, whatever its case and spacing."""
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = re.sub(r"\s+", " ", text).rstrip(" .;!")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _one_statement(content: str) -> str | None:
    text = " ".join(content.split())
    if not text or len(text) > MAX_CHARS or "?" in text or "\n" in content.strip():
        return None
    body = text.rstrip(".! ")
    if re.search(r"[.!;]\s+\S", body):  # more than one sentence
        return None
    if _NEGATION.search(body) or _HEDGE.search(body):
        return None
    return body


def _plausible(subject: str, attribute: str) -> bool:
    """A short noun phrase each: no clause words, no leading pronoun, no event noun."""
    subj, attr = subject.lower().split(), attribute.lower().split()
    if not subj or not attr or len(subj) > MAX_SUBJECT_WORDS or len(attr) > MAX_ATTRIBUTE_WORDS:
        return False
    if _CLAUSE_WORDS & set(subj + attr) or subj[0] in _PRONOUNS:
        return False
    head = [w for w in attr if w not in ("the", "a", "an", "my", "our", "its", "their")]
    return bool(head) and head[0] not in _PRONOUNS and head[-1] not in _EVENT_NOUNS


def _verb_relation(value: str) -> tuple[str, str]:
    """Split a leading participle (and its preposition) off the value."""
    words = value.split()
    i = 0
    while i < len(words) - 1 and words[i].lower() in _ADVERBS:
        i += 1
    if i >= len(words):
        return "", value
    verb = words[i].lower().strip(",")
    if not (verb in _PARTICIPLES or (len(verb) > 4 and verb.endswith(("ed", "ing")))):
        return "", value
    parts = [verb]
    if i + 1 < len(words) and words[i + 1].lower() in _PREPOSITIONS:
        parts.append(words[i + 1].lower())
        i += 1
    return "_".join(parts), " ".join(words[i + 1:])


def _subject(text: str, resolve: Callable[[str], str | None] | None) -> str | None:
    name = text.strip()
    if resolve is not None:
        entity_id = resolve(name)
        if entity_id:
            return f"entity:{entity_id}"
    term = normalise_term(name)
    return f"name:{term}" if term else None


#: An explicit key with the value the caller stated: a different value on its
#: slot is a ``replaces`` question for the rule tier ``explicit-v1`` (D-36).
EXPLICIT = "explicit"
#: An explicit key with no stated value: the slot is the caller's, but what the
#: memory says about it is only its wording, so a different wording disputes.
EXPLICIT_UNVALUED = "explicit_unvalued"
MAX_SPEC_CHARS = 200
#: Marks the hash of a value the caller stated, so a revision records whether it
#: holds a stated value or only a wording: the two never compare equal, and only
#: a stated value against a stated value can be a ``replaces`` question.
STATED_PREFIX = "stated:"
MAX_VALUE_CHARS = MAX_CHARS


class SlotSpecError(ValueError):
    """A ``slot`` or ``slot_value`` a caller gave that cannot be a key."""


def parse_spec(spec: str | None, value: str | None = None) -> tuple[str, str | None] | None:
    """The caller's ``slot`` and ``slot_value``, checked; None when no slot is given.

    Raises ``SlotSpecError`` for a malformed one, so a door can refuse the save
    before anything is stored, as it does for ``replaces``.
    """
    spec = (spec or "").strip()
    value = value.strip() if isinstance(value, str) else None
    if not spec:
        if value:
            raise SlotSpecError("slot_value needs a slot (subject/relation[/qualifier])")
        return None
    if len(spec) > MAX_SPEC_CHARS:
        raise SlotSpecError(f"slot is longer than {MAX_SPEC_CHARS} characters")
    parts = [normalise_term(p) for p in spec.split("/")]
    if len(parts) not in (2, 3) or not all(parts):
        raise SlotSpecError("slot must be subject/relation or subject/relation/qualifier")
    if value is not None and len(value) > MAX_VALUE_CHARS:
        raise SlotSpecError(f"slot_value is longer than {MAX_VALUE_CHARS} characters")
    return spec, (value or None)


def explicit_key(spec: str, content: str, *, value: str | None = None) -> SlotKey | None:
    """``subject/relation[/qualifier]`` given by the caller, taken as stated.

    The value is the one the caller stated, never the memory's wording (R-34):
    "deploys run on Fridays" and "we deploy every Friday" with ``value="Friday"``
    are the same value. Without a stated value the key is ``explicit_unvalued``
    and its value is the wording, which can only ever open a dispute.
    """
    try:
        parsed = parse_spec(spec, value)
    except SlotSpecError:
        return None
    if parsed is None:
        return None
    parts = [normalise_term(p) for p in parsed[0].split("/")]
    stated = parsed[1]
    if not (stated or content.strip()):
        return None
    return SlotKey(subject_key=f"name:{parts[0]}", relation_key=parts[1],
                   qualifier=parts[2] if len(parts) == 3 else "", cardinality="single",
                   key_source=EXPLICIT if stated else EXPLICIT_UNVALUED,
                   confidence=EXPLICIT_CONFIDENCE,
                   value_hash=(STATED_PREFIX + value_hash(stated)) if stated
                   else value_hash(content))


def is_stated(stored_hash: str | None) -> bool:
    """Whether a revision's value hash is of a value the caller stated."""
    return bool(stored_hash) and str(stored_hash).startswith(STATED_PREFIX)


def extract(content: str, *, resolve_subject: Callable[[str], str | None] | None = None,
            explicit: str | None = None, explicit_value: str | None = None) -> SlotKey | None:
    """The slot key ``content`` states, or None when the rules cannot tell."""
    if explicit:
        return explicit_key(explicit, content, value=explicit_value)
    body = _one_statement(content)
    if body is None:
        return None
    for pattern in (_ATTRIBUTE_OF, _POSSESSIVE, _CHOICE):
        match = pattern.match(body)
        if match is None:
            continue
        if not _plausible(match["subj"], match["attr"]):
            return None
        subject = _subject(match["subj"], resolve_subject)
        relation = normalise_term(match["attr"])
        val = match["val"].strip()
        verb = (match.groupdict().get("verb") or "").lower()
        if not verb:
            participle, val = _verb_relation(val)
            if participle:
                relation = f"{relation}__{participle}"
        if not subject or not relation or not val.strip():
            return None
        qualifier = ""
        if verb:
            qualifier, relation = relation, _VERB_RELATION[verb]
        return SlotKey(subject_key=subject, relation_key=relation, qualifier=qualifier,
                       cardinality="plural" if verb in _PLURAL_VERBS else "single",
                       key_source="pattern", confidence=PATTERN_CONFIDENCE,
                       value_hash=value_hash(val))
    return None


__all__ = ["EXPLICIT", "EXPLICIT_CONFIDENCE", "EXPLICIT_UNVALUED", "MAX_CHARS", "MAX_SPEC_CHARS",
           "MAX_VALUE_CHARS", "PATTERN_CONFIDENCE", "STATED_PREFIX", "SlotKey", "SlotSpecError", "explicit_key",
           "extract", "is_stated", "normalise_term", "parse_spec", "value_hash"]
