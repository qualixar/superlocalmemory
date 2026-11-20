# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Whether a string can be the name of an entity at all.

Measured on a clone of the author's store after the first entity repair
(4.1.25): relinking from the names the old extractor had saved recreated
"Users" (529 facts, from /Users/... paths), "LIVE", "VERIFIED", "PASS",
"Final", "IST", 16-hex fact ids (211 facts each), "2026-09-14 00:55",
"/api/v3/consolidation/trigger", "Xenobot Section Still Has", "Check
Acme" and "spill over". None of those names a person, product, place or
thing. Every name, from the rule-based extractor, a model or an old store,
passes through ``valid_name`` before it can become or reach an entity.

Rejected:
* ids and hashes (hex runs of 8+ with a digit, UUIDs), dates and times,
  numbers, file and API paths, URLs;
* file names ("gate.py", "notes.md") and anything with path or markup
  characters ("/api/*", "[Agantic_official]");
* an ALL-CAPS word that is an everyday English word (PASS, LIVE, VERIFIED,
  DONE, FAIL), alone or in a phrase ("SCREEN RECORDING"), or a time zone /
  unit (IST, UTC, AM, MB);
* a lower-case everyday word ("plan") or a phrase of them ("spill over");
* a phrase that starts or ends with a sentence word, or has one inside other
  than a connector ("Xenobot Section Still Has", "Check Acme");
* a single Capitalised everyday word ("Final", "Users", "Tested") unless the
  text shows it capitalised where a sentence does not start ("the Atlas
  report"): capitalised at the start of a sentence, a line or a label it is
  not evidence of a name.
"""

from __future__ import annotations

import re

from superlocalmemory.encoding.common_words import is_common_word

_HEXISH = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8,}\b", re.IGNORECASE)
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_DATE_TIME = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\b\d{1,2}:\d{2}(?::\d{2})?\b|"
                        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_NO_LETTERS = re.compile(r"^[^A-Za-z]*$")
_PATHISH = re.compile(r"^(?:/|~|\.{1,2}/)|://|\w/\w[^ ]*/|^www\.|[/\\*\[\]{}<>|=]")
_ACRONYM_PAIR = re.compile(r"^[A-Z0-9]{1,6}/[A-Z0-9]{1,6}$")
_FILE = re.compile(r"\.(?:md|py|js|ts|tsx|jsx|json|toml|ya?ml|txt|sh|log|db|csv|html?|css|"
                   r"mp3|mp4|wav|png|jpe?g|gif|svg|pdf|zip|lock|cfg|ini|env|sql|ipynb)$", re.I)
_TITLE = re.compile(r"^[A-Z][a-z]+$")
_CAPS = re.compile(r"^[A-Z]{2,}$")

#: ALL-CAPS tokens that are units, time zones or log words, not names.
_CAPS_NOISE = frozenset("""
ist utc gmt pst pdt est edt cst cdt mst mdt cet cest bst jst aest am pm kb mb gb tb ms
ok todo fixme wip tbd na eta faq fyi asap tldr imo btw
""".split())

#: Generic words of tech notes that read as names when capitalised.
_GENERIC = frozenset("""
url urls uri href id ids todo
""".split())

#: Lower-case words a name may contain between its other words.
_CONNECTORS = frozenset({"of", "and", "the", "for", "de", "la", "le", "van", "von",
                         "der", "den", "du", "da", "del", "di", "y", "&", "on", "in", "at"})

#: Words a multi-word name may start with although a sentence word on their own.
_LEADS = frozenset({"the", "mr", "mrs", "ms", "dr", "prof", "sir", "st"})

#: A lower-case phrase ending in one of these is a system's name ("billing service").
_SYSTEM_NOUNS = frozenset("""
service server database db cluster queue api gateway worker cache bucket repo
repository library plugin extension app bot daemon scheduler endpoint pipeline
""".split())


def _sentence_word(word: str) -> bool:
    from superlocalmemory.encoding.entity_mentions import STOP_WORDS  # (import cycle)

    return word.lower().rstrip(".") in STOP_WORDS


def _shape_ok(name: str) -> bool:
    """Ids, hashes, dates, times, numbers and paths are not names."""
    if _NO_LETTERS.match(name) or _HEXISH.search(name) or _UUID.search(name):
        return False
    if _ACRONYM_PAIR.match(name):  # "TCP/IP", "A/B"
        return True
    return not (_DATE_TIME.search(name) or _PATHISH.search(name) or _FILE.search(name))


def _single_ok(word: str, capital_evidence: bool) -> bool:
    if word.lower() in _GENERIC:
        return False
    parts = word.split("-")
    if len(parts) > 1 and all(_CAPS.match(p) and is_common_word(p.lower()) for p in parts):
        return False  # "AUTO-CREATED", "READ-ONLY": a shouted label
    if _CAPS.match(word):
        low = word.lower()
        return low not in _CAPS_NOISE and not (len(word) >= 3 and is_common_word(low))
    if _sentence_word(word):
        return False
    if word.islower():
        return not is_common_word(word)
    if _TITLE.match(word) and is_common_word(word):
        return capital_evidence
    return True


def _phrase_ok(words: list[str]) -> bool:
    if words[0].lower().rstrip(".") in _LEADS:  # "The Kestrel", "Dr. Iyer"
        words = words[1:]
        if len(words) == 1:
            return _single_ok(words[0], True)
    if _sentence_word(words[0]) or _sentence_word(words[-1]):
        return False
    if any(_sentence_word(w) and w.lower() not in _CONNECTORS for w in words[1:-1]):
        return False
    # Shouted text, not a name: "SCREEN RECORDING", "KEYS AVAILABLE".
    if any(len(w) >= 3 and _CAPS.match(w) and is_common_word(w.lower()) for w in words):
        return False
    if all(w.islower() for w in words):
        return words[-1] in _SYSTEM_NOUNS or not all(is_common_word(w) for w in words)
    return True


def valid_name(name: str, *, capital_evidence: bool = False) -> bool:
    """Whether ``name`` may be an entity's name.

    ``capital_evidence``: the text shows this single Capitalised everyday
    word capitalised somewhere a sentence does not start (see module doc).
    """
    text = (name or "").strip()
    if not text or len(text) > 80 or not _shape_ok(text):
        return False
    words = text.split()
    if len(words) > 6:
        return False
    if len(words) == 1:
        return _single_ok(words[0], capital_evidence)
    return _phrase_ok(words)


def opens_sentence(text: str, pos: int) -> bool:
    """Whether a capital at ``pos`` says nothing about a name: true unless the
    nearest character before it on the same line is a letter, digit or comma
    ("check the Atlas report", "Paris, Atlas and Rome"). After a full stop, a
    colon, a dash, a bullet, a table bar, an emoji or a line start, a capital
    is just how the text starts."""
    i = pos - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i < 0 or not (text[i].isalnum() or text[i] == ",")


def capitalised_mid_sentence(word: str, text: str) -> bool:
    """Whether ``text`` has ``word`` capitalised where no sentence starts."""
    for m in re.finditer(rf"(?<![\w/.~-]){re.escape(word)}(?![\w/-])", text or ""):
        if not opens_sentence(text, m.start()):
            return True
    # An explicit naming role is evidence even at a sentence boundary.
    role = (r"\s+(?:is|was)\s+(?:(?:our|my|the|a|an)\s+)?"
            r"(?:project|tool|application|service|colleague|teammate|client|company)\b")
    return bool(re.search(rf"(?<![\w/.~-]){re.escape(word)}{role}", text or "", re.I))


def valid_names(names: list[str], text: str | None) -> list[str]:
    """The names (from a model or an old store) that may name an entity,
    judged against the text they came from."""
    out: list[str] = []
    for name in names or ():
        word = (name or "").strip()
        evidence = bool(text) and bool(_TITLE.match(word)) and capitalised_mid_sentence(
            word, text or "")
        if valid_name(word, capital_evidence=evidence):
            out.append(name)
    return out


__all__ = ["capitalised_mid_sentence", "opens_sentence", "valid_name", "valid_names"]
