# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Base types for the #113 bounded summary layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Sequence

#: Names the facts a caller may not see, among the ids it is given.
HiddenOf = Callable[[Sequence[str]], "set[str]"]


def without_hidden(rows: list[dict], hidden_of: HiddenOf | None) -> list[dict]:
    """``rows`` (each with a ``fact_id``) minus the ones ``hidden_of`` names. The same list without it."""
    if hidden_of is None or not rows:
        return rows
    hidden = hidden_of([r["fact_id"] for r in rows])
    return [r for r in rows if r["fact_id"] not in hidden]


@dataclass
class SummaryResult:
    """A bounded, profile-scoped, traceable summary of user memories.

    Maintainer's binding constraint (issue #113 reply):
        "views must be customizable, profile-scoped, privacy-aware, and
         traceable back to the underlying memories rather than becoming
         opaque generic summaries"

    This dataclass enforces three of those four constraints structurally:

    Traceability
        ``source_fact_ids`` carries the atomic_facts.fact_id for every fact
        that contributed to this summary.  A user can always drill back to
        the raw memories.

    Profile scope
        ``profile_id`` is mandatory; callers must never mix profiles.

    Honesty / non-opaqueness
        ``coverage`` must be set to an accurate value.  See the constants
        below.  A summary over 3.9% of facts that presents itself as "your
        session" is precisely the opaque generic summary the maintainer said
        to avoid.

    Generated-by
        ``generated_by`` records whether the content is extractive
        (deterministic, always available, Mode A default) or came from an
        LLM (Mode B Ollama / Mode C cloud).

    Attributes:
        kind:            "session" | "daily" | "project"
        profile_id:      Owning profile — never expose across profiles.
        content:         Human-readable summary text.
        source_fact_ids: IDs of the atomic_facts that contributed.
                         Empty only when the underlying data does not exist.
        coverage:        One of the COVERAGE_* constants below.
        generated_by:    One of the GENERATED_BY_* constants below.
        metadata:        Extra context: date, project_path, session_id, etc.
    """

    kind: str
    profile_id: str
    content: str
    source_fact_ids: list[str]
    coverage: str
    generated_by: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── coverage constants ──────────────────────────────────────────────────────
#
# Use these strings; the tests check for their presence
# and the values must be human-interpretable without this file.

COVERAGE_FULL = "full"
"""All relevant data was available and contributed to the summary."""

COVERAGE_PARTIAL = "partial"
"""Some data was available.  Session summaries are always at most partial
because only ~3.9% of facts carry a session_id on a real store."""

COVERAGE_INSUFFICIENT = "insufficient"
"""Too few facts to produce a meaningful summary (below MIN_FACTS threshold)."""

COVERAGE_NO_SESSION = "no_session"
"""Session ID not found, or the session has no associated facts."""

COVERAGE_UNAVAILABLE = "unavailable"
"""Required data does not exist or a query error prevented access."""


# ── generated_by constants ──────────────────────────────────────────────────
#
# extractive is the deterministic fallback, ALWAYS available.
# Mode A users never get anything else.  Mode B/C users fall back when
# Ollama or the cloud is down — silence is not an option.

GENERATED_BY_EXTRACTIVE = "extractive"
"""Deterministic extractive summary — no LLM.  Always available."""

GENERATED_BY_LLM_B = "llm_b"
"""Ollama local LLM (Mode B).  Falls back to extractive if unavailable."""

GENERATED_BY_LLM_C = "llm_c"
"""Cloud LLM (Mode C).  Falls back via llm_b to extractive."""


# ── local calendar day ──────────────────────────────────────────────────────

#: Furthest a real time zone sits from UTC, in minutes (UTC-12 .. UTC+14).
MAX_TZ_OFFSET_MINUTES = 840


def local_day_modifier(tz_offset_minutes: int) -> str:
    """SQLite date modifier that turns a stored UTC instant into the caller's day.

    ``created_at`` is stored in UTC. A day summary asked for "5 October" by a
    person in India means 5 October in India, which starts at 18:30 UTC on the
    4th. ``DATE(created_at, '+330 minutes')`` is that day.

    Refuses anything that is not a whole number of minutes within the range a
    real time zone can have, so no caller text ever reaches the SQL.
    """
    if (not isinstance(tz_offset_minutes, int) or isinstance(tz_offset_minutes, bool)
            or abs(tz_offset_minutes) > MAX_TZ_OFFSET_MINUTES):
        raise ValueError(
            "tz_offset_minutes must be a whole number of minutes between "
            f"-{MAX_TZ_OFFSET_MINUTES} and {MAX_TZ_OFFSET_MINUTES} (the offset east of UTC)")
    return f"{tz_offset_minutes:+d} minutes"


def local_offset_minutes(day: str | None = None) -> int:
    """This computer's offset east of UTC on ``day`` (ISO date; default today).

    Taken at local noon of that day so a daylight-saving change on the day itself
    cannot pick the wrong side. Used by the CLI and the MCP tool, whose "today"
    is this computer's today; the dashboard sends the browser's own offset.
    """
    from datetime import datetime

    base = date.fromisoformat(day) if day else date.today()
    noon = datetime(base.year, base.month, base.day, 12).astimezone()
    offset = noon.utcoffset()
    return int(offset.total_seconds() // 60) if offset is not None else 0


# ── highlight formatting ────────────────────────────────────────────────────

#: Display width for one bullet in a summary body.
#:
#: Chosen for a bullet, not for a paragraph. The generators originally truncated
#: at 300 characters and nothing else, which looks fine on a synthetic corpus of
#: one-line facts and falls apart on a real store: agent-written facts routinely
#: contain blank lines and markdown headings, so a 300-character slice rendered
#: as six or more display lines and the bullet list stopped being a list.
HIGHLIGHT_CHARS = 180


def format_highlight(content: str, limit: int = HIGHLIGHT_CHARS) -> str:
    """Collapse *content* to a single readable line for a summary bullet.

    Three things, in order:

    1. **Flatten whitespace.** Newlines, blank lines and runs of spaces all
       become one space. This is the fix that matters: character truncation
       alone cannot keep a multi-paragraph fact on one line, and every
       generator here writes into a bullet list.
    2. **Prefer a whole first sentence** when there is one and it fits. A
       complete sentence reads better than a slice of one, and the first
       sentence of a report is usually its summary.
    3. **Otherwise cut at a word boundary** and mark the cut with an ellipsis,
       so it is visible that text was dropped rather than that a fact ended
       mid-word.

    Markdown heading markers are stripped because a flattened ``**Summary**``
    mid-sentence reads as noise.
    """
    import re

    text = re.sub(r"\s+", " ", (content or "")).strip()
    # Leading/inline markdown emphasis and heading marks, once flattened, add
    # nothing but clutter to a one-line bullet.
    text = re.sub(r"(?:^|\s)#{1,6}\s+", " ", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""
    if len(text) <= limit:
        return text

    # A complete first sentence, if it fits comfortably.
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    if match:
        sentence = match.group(1).strip()
        if len(sentence) <= limit:
            return sentence

    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:          # don't cut a long unbroken token to a stub
        cut = cut[:space]
    return cut.rstrip(" ,;:—-") + "…"


# ── LLM output cleanup ──────────────────────────────────────────────────────

#: Sentences a chat-tuned model emits *around* the answer rather than as part of
#: it. Anchored to the start of a paragraph so they cannot match mid-content.
#:
#: Measured, not guessed: a Mode B summary on the author's own store opened with
#: "I apologize for the previous confusion. It seems that I misunderstood the
#: context of the texts provided.\n\nTo provide a concise summary paragraph, here
#: is a merge of all the key information:" — 180 characters of the model talking
#: to itself, shown to the user as their daily reflection.
_LLM_PREAMBLE = (
    r"^(?:"
    r"i\s+apologi[sz]e\b.*"
    r"|i'?m\s+sorry\b.*"
    r"|it\s+seems\s+that\s+i\b.*"
    r"|sure[,!.]?\s*(?:thing)?\b.*"
    r"|certainly[,!.]?\b.*"
    r"|of\s+course[,!.]?\b.*"
    r"|here(?:'s|\s+is|\s+are)\b[^.!?]*:"
    r"|to\s+(?:provide|summari[sz]e|answer)\b[^.!?]*:"
    r"|based\s+on\s+the\s+(?:facts|texts|information|data)\s+provided[,:]?"
    r"|as\s+(?:an|a)\s+(?:ai|language\s+model)\b.*"
    r")\s*$"
)

#: Closing pleasantries. Same anchoring rule.
_LLM_POSTAMBLE = (
    r"^(?:"
    r"(?:i\s+hope|hope)\s+(?:this|that)\s+helps\b.*"
    r"|let\s+me\s+know\b.*"
    r"|feel\s+free\s+to\b.*"
    r"|would\s+you\s+like\s+me\s+to\b.*"
    r")$"
)


def clean_llm_summary(text: str) -> str:
    """Strip chat-assistant scaffolding from a model-written summary.

    WHY THIS EXISTS
    ---------------
    Mode B/C summaries are shown to the user as *their* memory, with no chat
    framing around them. A chat-tuned model does not know that: it opens with an
    apology or "Here is a concise summary:" and closes with "Let me know if you
    want more detail". Both are addressed to a conversation that the reader
    cannot see, and both make the product look broken.

    The system prompt now asks for bare prose, which handles most of it. This is
    the second line of defence, because instruction-following on a 3B local
    model is not something to bet the displayed output on.

    Conservative by construction: patterns are anchored to whole paragraphs or
    whole leading sentences, so a summary that legitimately contains the word
    "sure" mid-paragraph is untouched. If stripping would empty the text, the
    original is returned — a scaffolded summary beats a blank one.
    """
    import re

    original = (text or "").strip()
    if not original:
        return ""

    # Fenced code blocks wrapping the whole answer: keep the contents.
    fenced = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", original, re.DOTALL)
    if fenced:
        original = fenced.group(1).strip()

    paras = [p.strip() for p in re.split(r"\n\s*\n", original) if p.strip()]

    while paras and re.match(_LLM_PREAMBLE, paras[0], re.IGNORECASE | re.DOTALL):
        paras.pop(0)
    while paras and re.match(_LLM_POSTAMBLE, paras[-1], re.IGNORECASE | re.DOTALL):
        paras.pop()

    if not paras:
        return original

    # A preamble that shares a paragraph with real content: drop just the leading
    # sentence, and only when what follows is substantial enough to stand alone.
    lead = re.match(r"(.+?[.:!?])\s+(\S.*)$", paras[0], re.DOTALL)
    if lead and re.match(_LLM_PREAMBLE, lead.group(1), re.IGNORECASE | re.DOTALL):
        if len(lead.group(2).strip()) > 40:
            paras[0] = lead.group(2).strip()

    cleaned = "\n\n".join(paras).strip()
    return cleaned or original


#: System prompt for every summary generator, Mode B and Mode C alike.
#:
#: Mode C always sent one; Mode B sent none at all, which is why local-model
#: output arrived wrapped in chat scaffolding while cloud output did not.
SUMMARY_SYSTEM_PROMPT = (
    "You summarise a person's own saved notes for them. "
    "Reply with the summary text only — no preamble, no apologies, no sign-off, "
    "no markdown headings, and never refer to yourself or to these instructions. "
    "Write plain prose in the third person about the work described."
)


def get_mode_str(config: object | None) -> str:
    """Extract the operating mode string ('a', 'b', or 'c') from a config."""
    if config is None:
        return "a"
    m = getattr(config, "mode", None)
    if m is None:
        return "a"
    return getattr(m, "value", str(m)).lower()
