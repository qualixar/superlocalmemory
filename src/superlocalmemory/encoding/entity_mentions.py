# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which words in a text name something: Mode A's rule-based entity mentions.

One extractor for every place that needs one: the rule-based fact extractor,
the first searchable fact of a save, the verbatim fact, the fast store path,
and the question side of the entity channel. Before 4.1.25 each had its own
copy of ``\\b[A-Z][a-z]+...\\b`` and ``\\b[A-Z]{2,}\\b``, which never matched
inside "SuperLocalMemory" or "AgentAssert", split "Wi-Fi" into "Wi" and "Fi",
and kept "The", "Our", "My", "Dr" and "Decision" as names.

A mention is any of:

* a run of up to four name words joined by single spaces: Capitalised words,
  CamelCase and mixed case ("SuperLocalMemory", "GitHub", "iPhone", "OpenAI"),
  ALL-CAPS ("NASA", "EC2"), hyphenated names ("Wi-Fi", "Jean-Luc"); a version
  may follow a name word ("SLM 4.1.25", "Gemini 3.1 Pro"), in which case the
  name before the version is a mention too ("SLM");
* a lower-case technical word: letters then digits ("llama3.2", "s3"), an inner
  "_", ":" or "." ("snake_case", "superpowers:brainstorming"), or a word no
  English word starts like ("pgbouncer", "nginx", "ffmpeg");
* a lower-case word with a single capital letter after it ("vitamin D",
  "plan B"), or a lower-case modifier with a system noun ("billing service",
  "auth server");
* a quoted phrase.

Words capitalised only because they open a sentence, pronouns, honorifics,
months, weekdays and note labels ("Decision:", "Status:") are dropped wherever
they sit in a run, so "The Kestrel" keeps "Kestrel" and "Dr. Iyer" keeps
"Iyer". Single regex passes over the text: this runs on every write.
"""

from __future__ import annotations

import re

from superlocalmemory.encoding.lower_tech_scan import lower_tech_spans

#: Words that are capitalised only by position, address someone, or label a
#: note. Never a name on their own, removed from inside runs as well.
STOP_WORDS: frozenset[str] = frozenset("""
the this that these those what when where which how who whom whose why also then
just very really actually maybe well still even she he they them her him his its
but and not yes yeah sure okay ok here there now today tomorrow yesterday some all
any been being have has had was were for with from about into over wow did so gonna
got by thanks thank hey hi hello bye good great nice cool right like know think feel
want need make take give tell said told get let can will would could should might
much many more most lot way thing something anything everything nothing someone it
my your our their me you we us i do does if or no to at on in up out off too go come
see look say ask try keep put run set move call end start find show hear play work
read talk turn help miss hope love hate wish seem mean mind care use deploy every
never quickly please always consider check install restart enable ignore verify
finally basically specifically clearly obviously immediately apply avoid update
upgrade note remember ensure skip a an is are be am as of after before since until
while because though although if unless however therefore meanwhile perhaps once
monday tuesday wednesday thursday friday saturday sunday january february march
april may june july august september october november december morning evening
night week month year mr mrs ms dr prof sir madam st decision rule status summary
todo action context goal result reason question answer problem issue fix outcome
lesson takeaway warning tip idea step owner deadline priority task update reminder
configure configured reconfigure refactor refactored rerun reran restarted summarised
""".split())

#: Lower-case nouns that make the word before them a system name.
_SYSTEM_NOUNS = frozenset("""
service server database db cluster queue api gateway worker cache bucket repo
repository library plugin extension app bot daemon scheduler endpoint pipeline
""".split())

#: Modifiers that make "X service" generic rather than a name.
_GENERIC_MODIFIERS = frozenset("""
main primary secondary local remote old new same other whole entire single shared
default current previous next first last separate own real test prod production
staging dev web our my your their the a an this that every each any some no
""".split())

#: Two-letter starts an English word can have. A lower-case word starting any
#: other way, with a vowel somewhere, is a coined technical name ("pgbouncer").
_ONSETS = frozenset("""
bl br ch cl cr dr dw fl fr gh gl gn gr kn kr ph pl pn pr ps qu rh sc sh sk sl sm
sn sp sq st sw th tr ts tw wh wr mn kh sch scr shr sph spl spr str thr chr
""".split())
_VOWELS = frozenset("aeiouy")

_NAME_WORD = (r"(?:[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)*|[a-z]+[A-Z][A-Za-z0-9]*)"
              r"(?:-[A-Za-z0-9]+)*")
_VERSION = r"v?\d+(?:\.\d+)+|\d+"
_RUN = re.compile(rf"(?<![\w./~\\-])({_NAME_WORD}(?:[ ](?:{_NAME_WORD}|{_VERSION}))*)"
                  r"(?![\w/\\-]|\.\w)")
_LETTER_AFTER = re.compile(r"\b([a-z]{3,}) ([A-Z])(?![\w-])")
_SYSTEM_PHRASE = re.compile(r"(?<![\w-])([a-z]{3,}) (" + "|".join(sorted(_SYSTEM_NOUNS)) + r")\b")
_QUOTED = re.compile(r"\"([^\"]{2,60})\"")
_ROLE = re.compile(r"\b([A-Z][a-z]+)\s+(?:is|was)\s+(?:(?:our|my|the|a|an)\s+)?"
                   r"(?:project|tool|application|service|colleague|teammate|client|company)\b")
_IS_VERSION = re.compile(rf"^(?:{_VERSION})$")
_LABEL = re.compile(r"(?:^|\n)\s*([A-Z][a-z]+):")


def _is_stop(word: str) -> bool:
    return word.lower().rstrip(".") in STOP_WORDS


class _Found(dict):
    """name -> whether some occurrence was capitalised mid-sentence."""

    def add(self, name: str, evidence: bool = True) -> None:
        self[name] = self.get(name, False) or evidence


def _split_run(words: list[str], out: _Found, initial: bool) -> None:
    """Add the stop-free spans of a run, and each span's name before a version.

    Only a span that opens the run can sit where a sentence starts."""
    span: list[str] = []
    span_initial = initial
    for i, word in enumerate([*words, ""]):
        if word and not _is_stop(word) and not (not span and _IS_VERSION.match(word)):
            if not span:
                span_initial = initial and i == 0
            span.append(word)
            continue
        if span:
            _add_span(span, out, not span_initial)
        span = []


def _add_span(span: list[str], out: _Found, evidence: bool) -> None:
    if len(span) > 5:
        span = span[:5]
    names = [w for w in span if not _IS_VERSION.match(w)]
    if not names:
        return
    text = " ".join(span)
    if len(text) >= 2 and not (len(span) == 1 and len(text) == 1):
        out.add(text, evidence)
    first_version = next((i for i, w in enumerate(span) if _IS_VERSION.match(w)), None)
    if first_version:  # a name with a version is a product's ("Python 3.12")
        out.add(" ".join(span[:first_version]), True)
        out.add(text, True)
    caps = [_is_caps(w) for w in names]
    if len(names) > 1 and any(caps) and not all(caps):
        _add_kind_groups(names, caps, out, evidence)


def _is_caps(word: str) -> bool:
    return len(word) >= 2 and word.isupper()


def _add_kind_groups(names: list[str], caps: list[bool], out: _Found,
                     evidence: bool) -> None:
    """'Google HQ' also names 'Google' and 'HQ': a run mixing ALL-CAPS and
    other words is often two names side by side. Only the first group can
    open a sentence."""
    group = [names[0]]
    first = True
    for word, kind, prev in zip(names[1:], caps[1:], caps[:-1]):
        if kind != prev:
            out.add(" ".join(group), evidence or not first)
            group, first = [], False
        group.append(word)
    out.add(" ".join(group), evidence or not first)


def _coined(word: str) -> bool:
    """A lower-case word no English word starts like ("pgbouncer")."""
    if len(word) < 5 or word[:2] in _ONSETS or word[0] in _VOWELS or word[1] in _VOWELS:
        return False
    return word[:3] not in _ONSETS and any(c in _VOWELS for c in word[2:])


def _technical(word: str) -> bool:
    """'llama3.2', 'snake_case', 'pkg:name' -- not 'e.g', 'v2' or 'a1'."""
    if re.match(r"^(?:v?\d|[a-z]\d*$)", word):
        return False
    return any(len(part) >= 3 and part.isalpha() for part in re.split(r"[_:.\d]+", word))


def _lower_mentions(text: str, out: _Found) -> None:
    # Bounded time, identical words (#310): see encoding/lower_tech_scan.py.
    for start, end in lower_tech_spans(text):
        word = text[start:end]
        if word.isalpha():
            if _coined(word):
                out.add(word)
        elif _technical(word):
            out.add(word)
    for m in _LETTER_AFTER.finditer(text):
        if m.group(2) not in ("I", "A") and m.group(1) not in STOP_WORDS:
            out.add(f"{m.group(1)} {m.group(2)}")
    for m in _SYSTEM_PHRASE.finditer(text):
        modifier, noun = m.group(1), m.group(2)
        if modifier not in STOP_WORDS and modifier not in _GENERIC_MODIFIERS \
                and modifier not in _SYSTEM_NOUNS:
            out.add(f"{modifier} {noun}")


def extract_mentions(text: str, *, question: bool = False) -> list[str]:
    """Entity mentions in ``text``, sorted and without duplicates.

    Every candidate passes ``entity_validity.valid_name``; a single
    Capitalised everyday word counts only where no sentence starts. A
    ``question`` ("Scope of the audit?") gets the benefit of the doubt there:
    it only looks up entities that exist, and it creates none."""
    if not text:
        return []
    from superlocalmemory.encoding.entity_validity import opens_sentence, valid_name

    out = _Found()
    labels = {m.group(1) for m in _LABEL.finditer(text)}
    for m in _RUN.finditer(text):
        words = [w for w in m.group(1).split(" ") if w not in labels]
        _split_run(words, out, opens_sentence(text, m.start()))
    # A name introduced by an explicit role ("Zephyr is our project") counts
    # even at the start of a sentence, where the capitalisation rule abstains.
    for m in _ROLE.finditer(text):
        out.add(m.group(1), True)
    _lower_mentions(text, out)
    for m in _QUOTED.finditer(text):
        quoted = m.group(1).strip()
        if len(quoted) >= 2:
            out.add(quoted)
    return sorted(n for n, evidence in out.items()
                  if valid_name(n, capital_evidence=evidence or question))


__all__ = ["STOP_WORDS", "extract_mentions"]
