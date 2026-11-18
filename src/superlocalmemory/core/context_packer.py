# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Pack memories into an agent's context as complete, labelled evidence (WS-4, W3-7a).

Two failures this fixes:

* **A cut can change what a memory says.** ``clamp_content`` sliced a memory
  at a character count, so "Never deploy on Friday unless the release lead
  approves" could arrive as "Never deploy on Friday", and a cut can fall right
  after a "not". Text is now packed in whole sentences; when even the first
  sentence does not fit it is cut at a word and says so, so a shortened rule
  never passes as complete.
* **Evidence came without its standing.** Each unit now carries what an agent
  needs to weigh it: how old it is, whether its value is disputed (with the
  other side, D-39), and whether the file it came from changed (D-38), from the
  same read-only truth notes recall uses (``truth.recall_notes``).

Everything stays inside the untrusted-evidence delimiters ``render_context``
already draws.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

#: Ends a memory shortened between sentences.
CUT_MARK = " [… shortened to fit]"
#: Ends a memory whose first sentence alone did not fit.
CUT_MID_MARK = " [… cut mid-sentence; the rest is missing, recall it for the full text]"
#: The mark when the budget is too small for a sentence about the cut.
SHORT_MARK = " […]"
#: Characters of the other side of a dispute shown beside a memory.
OTHER_SIDE_CHARS = 240

#: A sentence ends at . ! or ? followed by whitespace or the end, or at a line
#: break; a dot inside a token ("v1.2", "e.g") does not end it.
_UNIT = re.compile(r"(?s).*?(?:[.!?]+(?=\s|\Z)|\n|\Z)")


@dataclass(frozen=True)
class Packed:
    text: str
    cut: bool


def units(text: str) -> list[str]:
    """Sentences (and lines) in order, each with its own ending."""
    return [u for u in (m.group(0) for m in _UNIT.finditer(text)) if u.strip()]


def pack_text(text: str, max_tokens: int) -> Packed:
    """``text`` within ``max_tokens`` in whole sentences, or visibly cut."""
    from superlocalmemory.core.injection import estimate_tokens

    if not text or estimate_tokens(text) <= max_tokens:
        return Packed(text, False)
    limit = max(0, max_tokens * 4)
    mark = CUT_MARK if len(CUT_MARK) <= limit // 2 else SHORT_MARK
    kept = ""
    for unit in units(text):
        if len(kept) + len(unit) > limit - len(mark):
            break
        kept += unit
    if kept.strip():
        return Packed(kept.rstrip() + mark, True)
    mark = CUT_MID_MARK if len(CUT_MID_MARK) <= limit // 2 else SHORT_MARK
    head = text[: max(0, limit - len(mark))]
    if " " in head:
        head = head[: head.rfind(" ")]
    return Packed((head.rstrip() + mark)[:limit], True)


#: The sycophancy rule (WS-4; PersistBench, research frontier §11: a stored
#: view makes models agree instead of answer): a remembered opinion or
#: preference is shown as what someone said, never as evidence it is right.
OPINION_KIND = "opinion"
VIEW_LABEL = "a view or preference someone stated, not a checked fact: weigh it, do not defer to it"


def standing(memory) -> str:
    """The unit's label: age, dispute, source state (empty when there is none)."""
    from superlocalmemory.core.injection import sanitize_untrusted_content
    from superlocalmemory.retrieval.temporal_frame import relative_age

    parts = []
    if getattr(memory, "created_at", ""):
        age = relative_age(memory.created_at)
        if age:
            parts.append(f"said {age}")
    if getattr(memory, "disputed", False):
        other = getattr(memory, "other_side", "")
        parts.append("DISPUTED" + (": another memory says \""
                                   + sanitize_untrusted_content(other[:OTHER_SIDE_CHARS])
                                   + "\"; ask which is true now" if other else ""))
    state = getattr(memory, "source_state", "")
    if state in ("changed", "missing"):
        parts.append(f"its source file is {state}: re-read it before relying on this")
    if getattr(memory, "memory_kind", "") == OPINION_KIND:
        parts.append(VIEW_LABEL)
    return f" [{'; '.join(parts)}]" if parts else ""


def annotate(memories: list, db_path: Path | str | None, profile_id: str, *,
             slm_home: Path | None = None) -> list:
    """The memories with dispute and source state from the truth layer
    (read-only, fail-open: any error leaves them as they were)."""
    if not memories or db_path is None:
        return memories
    try:
        from superlocalmemory.truth import recall_notes

        notes = recall_notes.annotate(db_path, profile_id,
                                      [{"fact_id": m.fact_id} for m in memories],
                                      slm_home=slm_home)
    except Exception:  # noqa: BLE001 - presentation only
        return memories
    kinds = _kinds(db_path, [m.fact_id for m in memories])
    out = []
    for memory, note in zip(memories, notes, strict=True):
        changes = {}
        if note.get("disputed"):
            changes["disputed"] = True
            changes["other_side"] = _other_side(memory.fact_id, note.get("disputes") or [])
        state = (note.get("source") or {}).get("state")
        if state:
            changes["source_state"] = str(state)
        kind = kinds.get(memory.fact_id)
        if kind and not getattr(memory, "memory_kind", ""):
            changes["memory_kind"] = kind
        out.append(replace(memory, **changes) if changes else memory)
    return out


def _kinds(db_path: Path | str, fact_ids: list[str]) -> dict[str, str]:
    """Each fact's memory kind, read-only; nothing on any error."""
    ids = [f for f in fact_ids if f]
    if not ids:
        return {}
    try:
        from superlocalmemory.truth.invariants import open_read_only

        conn = open_read_only(str(db_path))
        try:
            marks = ", ".join("?" * len(ids))
            return {r[0]: r[1] for r in conn.execute(
                f"SELECT fact_id, memory_kind FROM atomic_facts WHERE fact_id IN ({marks})"
                " AND memory_kind IS NOT NULL", ids)}
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - presentation only
        return {}


def _other_side(fact_id: str, disputes: list[dict]) -> str:
    for dispute in disputes:
        for side in dispute.get("sides") or []:
            if side.get("fact_id") != fact_id and side.get("visible") and side.get("content"):
                return str(side["content"])
    return ""


__all__ = ["CUT_MARK", "CUT_MID_MARK", "OPINION_KIND", "OTHER_SIDE_CHARS", "SHORT_MARK",
           "VIEW_LABEL", "Packed", "annotate", "pack_text", "standing", "units"]
