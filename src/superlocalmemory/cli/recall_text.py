# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Plain-text lines for ``slm recall`` that the JSON already carries.

An incomplete recall (a channel abandoned at the hang guard, an embedding model
still loading on a just-started daemon, or the daemon's over-budget keyword
fallback) used to print exactly what a complete one prints. With no results
that was "No confident match." — a confident statement about the store, made
by a search that did not look everywhere. These lines say what was skipped.
"""

from __future__ import annotations

__all__ = ["empty_result_line", "incomplete_line", "project_line", "remember_receipt_text",
           "status_line", "tag_line"]


def incomplete_line(result: dict) -> str:
    """One line naming the channels this answer ran without, or ""."""
    skipped = [str(c) for c in (result.get("incomplete_channels") or [])]
    if not skipped:
        return ""
    status = result.get("channel_status") or {}
    warming = any(status.get(c) == "warming" for c in skipped)
    if any(status.get(c) == "needs_service" for c in skipped):
        from superlocalmemory.core.daemon_text_embedder import NEEDS_SERVICE

        return f"Incomplete search: {NEEDS_SERVICE}"
    reason = (
        "the embedding model is still loading"
        if warming else "part of the search did not finish"
    )
    return (
        f"Incomplete search: {reason}; searched without "
        f"{', '.join(sorted(skipped))}. Ask again in a moment for a full answer."
    )


def project_line(result: dict) -> str:
    """Says when ``--project`` matched nothing found and the results shown are
    therefore NOT narrowed to it (#150), or ""."""
    scope = result.get("project_scope") or {}
    filt = scope.get("filter") if isinstance(scope, dict) else None
    if isinstance(filt, dict) and not filt.get("applied", True):
        return str(filt.get("note") or "These results are not narrowed to the project.")
    return ""


def tag_line(result: dict) -> str:
    """Says why ``--tag`` left nothing: no memory carries the tag at all, or
    some do but none matched the question (4.1.22), or ""."""
    scope = result.get("tag_scope") or {}
    if isinstance(scope, dict) and not scope.get("matched") and scope.get("note"):
        return str(scope["note"])
    return ""


def result_line(position: int, item: dict) -> str:
    """One recalled memory; one that does not answer the question says which
    kind it is: [related] (a close memory of a real question) or [weak]."""
    score = item.get("score") or 0  # a present-but-null score prints as 0
    tier = item.get("tier") or ""
    label = (" [related]" if tier == "related"
             else " [weak]" if item.get("weak") or tier == "weak" else "")
    return f"  {position}. [{score:.2f}]{label} {item.get('content', '')}"


def context_lines(item: dict) -> list[str]:
    """Depth 2: the memory's surroundings, one indented line each."""
    ctx = item.get("context") or {}
    out = []
    if ctx.get("source"):
        out.append(f"       source: {' '.join(str(ctx['source']).split())[:200]}")
    for label, key in (("same session", "same_session"), ("same entities", "same_entities")):
        for row in ctx.get(key) or ():
            out.append(f"       {label}: {row.get('content', '')}")
    if ctx.get("superseded_by"):
        out.append(f"       replaced by: {ctx['superseded_by'].get('content', '')}")
    return out


def truth_lines(item: dict) -> list[str]:
    """An open dispute (both sides, how to settle) and a changed source file (D-39, D-38)."""
    out = []
    for d in item.get("disputes") or ():
        out.append(f"       disputed: {d.get('question', 'Which is true now?')}")
        for side in d.get("sides") or ():
            text = side.get("content") if side.get("visible") else "(not visible to you)"
            when = str(side.get("said_at") or "")[:10]
            now = " (current)" if side.get("current") else ""
            out.append(f"         - {when}{now}: {' '.join(str(text or '').split())[:160]}"
                       f"  [fact {side.get('fact_id')}]")
        out.append(f"         settle: slm disputes settle {d.get('dispute_id')} "
                   "--keep FACT_ID|both|unrelated")
    source = item.get("source") or {}
    if source.get("state") in ("changed", "missing"):
        word = "changed since it was saved" if source["state"] == "changed" else "is gone"
        out.append(f"       source file {word}: {source.get('file')} (read it again before trusting this)")
    return out


def status_line(result: dict) -> str:
    """The warming-up note, when the recall ran before SLM was ready and no
    other line printed says so already."""
    note = str(result.get("status_note") or "") if result.get("warming_up") else ""
    return "" if note == str(result.get("abstention_note") or "") else note


def empty_result_line(result: dict) -> str:
    """What to print when no memory came back. While warming up that is the
    warming note, never "That isn't in SuperLocalMemory."."""
    if result.get("warming_up") and result.get("status_note"):
        return str(result["status_note"])
    status = result.get("source_fidelity_status")
    if status in {"degraded", "unavailable"}:
        return f"Source fidelity {status}: no verified results available."
    if incomplete_line(result):
        return "Nothing found yet."
    if result.get("abstention_note"):
        return str(result["abstention_note"])
    if result.get("no_confident_match"):
        return "No confident match."
    return "No matching memories found."


def fidelity_line(result: dict) -> str:
    """Expose withheld derivatives and give the operator a recovery action."""
    note = str(result.get("source_fidelity_note") or "")
    if result.get("source_fidelity_status") in {"degraded", "unavailable"}:
        return (note + " Run slm doctor to diagnose the source store; repair any reported "
                "storage/schema errors, then retry recall.").strip()
    return note


def remember_receipt_text(result: dict) -> str:
    """The ``slm remember`` receipt, saying which searches can reach it yet.

    "Queryable" is true from the moment of admission for search by the
    memory's own words (the full-text index is written in the same
    transaction). Meaning-based search needs the memory's vector, which the
    background indexer adds a moment later — so a paraphrased question can
    miss it until then, and the receipt says so instead of implying more.
    """
    state = str(result.get("materialization_state") or "queryable")
    if state == "accepted":
        return (
            "Saved ✓ (durable). The memory writer is busy, so it is being "
            "indexed now and will be searchable within seconds "
            f"(admission={result.get('admission_id', 'unknown')})."
        )
    line = (f"{state.capitalize()} ✓ {result.get('count', 0)} facts "
            f"(operation={result.get('operation_id', 'unknown')}).")
    if state in ("queryable", "enriching"):
        line += ("\nFindable now by its words; meaning-based search catches "
                 "up when background indexing finishes (usually seconds).")
    conflict = result.get("kind_conflict")
    if isinstance(conflict, dict):
        line += (f"\nThese exact words were already saved as '{conflict.get('kept')}', "
                 f"which was kept; '{conflict.get('requested')}' was not recorded.")
    return line


def replaced_text(replaced: dict) -> str:
    """What ``slm remember --replaces`` replaced, and the exact way to undo it.

    The undo hint used to say "call review_correction with each case_id ...
    and that case's version" without printing either, so a person at a shell
    had nothing to type. Each case now gets the literal command.
    """
    count = len(replaced.get("fact_ids") or [])
    lines = [f"Replaced ✓ {count} fact(s) of {replaced.get('replaces')}."]
    cases = [c for c in (replaced.get("cases") or []) if isinstance(c, dict)
             and c.get("case_id") and isinstance(c.get("version"), int)]
    if cases:
        lines.append("To undo, run:")
        lines.extend(f"  slm review-correction {c['case_id']} rollback {c['version']}"
                     for c in cases)
    elif replaced.get("undo"):
        lines.append(str(replaced["undo"]))
    return "\n".join(lines)
