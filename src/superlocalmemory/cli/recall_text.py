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
           "tag_line"]


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


def empty_result_line(result: dict) -> str:
    """What to print when no memory came back."""
    if incomplete_line(result):
        return "Nothing found yet."
    if result.get("no_confident_match"):
        return "No confident match."
    return "No matching memories found."


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
