# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""One answer to "was this person erased?", shared by every surface.

The erasure returns a flat dict of counts and markers. Three surfaces read it
(the HTTP route, the command line, and the dashboard that calls the route), and
each used to decide for itself what counted as a failure: the command line
printed success=true with ``erasure_complete: false``, and the dashboard looked
only for an ``error`` key and told the person "Erase failed: unknown" under a
receipt that said COMPLETE.

``erasure_succeeded`` is the single verdict. ``describe`` turns the same dict
into the statement a person can act on: what was removed, what remains and why,
and what to do about it.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Counts that are not failures even though they end in a failure-looking word.
_NOT_FAILURES = frozenset({"erasure_complete", "erasure_provable"})

#: Markers that do not end in ``_failed``/``_failures`` but mean something is left.
_RESIDUE_MARKERS = (
    "residue_rows", "fts_residue_rows", "owner_erasure_incomplete",
    "backup_obligations_pending", "erasure_aborted", "backup_snapshots_unreadable",
    "optimize_capture_present",
)

#: What each marker means to the person, and the step that retries it.
_REMAINING: dict[str, tuple[str, str]] = {
    "residue_rows": ("Some rows of this profile are still in the memory database.",
                     "Run the erasure again."),
    "table_delete_failures": ("Some tables could not be emptied.", "Run the erasure again."),
    "fts_residue_rows": ("The search index still holds words of this profile.",
                         "Run the erasure again, then `slm doctor`."),
    "fts_purge_failed": ("The search index could not be rewritten without the erased words.",
                         "Run `slm db repair`, then the erasure again."),
    "vector_store_failures": ("Some stored vectors of this profile could not be removed.",
                              "Run the erasure again; if it repeats, run `slm doctor`."),
    "graph_projection_failures": ("The graph store still holds some of this profile.",
                                  "Run the erasure again."),
    "write_receipts_failed": ("The write receipts of this profile could not be removed.",
                              "Run the erasure again."),
    "optimize_cache_failed": ("The MCP cache or compression originals could not be cleared.",
                              "Run the erasure again."),
    "profile_registry_failed": ("The workspace is still listed in profiles.json.",
                                "Run the erasure again."),
    "pending_memories_failed": ("Queued saves of this profile could not be removed.",
                                "Run the erasure again."),
    "optimize_capture_present": (
        "optimize_capture.jsonl holds recorded proxy traffic. It names no profile, so it "
        "cannot be erased selectively.", "Delete optimize_capture.jsonl if you do not need it."),
    "event_previews_failed": ("Previews of the profile's memories remain in the event log.",
                              "Run the erasure again."),
    "restore_journal_failed": (
        "The erasure could not be journalled outside the backups, so restoring a backup could "
        "bring the erased memories back.", "Run the erasure again."),
    "recall_queue_failed": ("Queued recall requests could not be cleared.",
                            "Run the erasure again."),
    "admission_journal_failed": ("Saved requests of this profile could not be removed.",
                                 "Run the erasure again."),
    "sidecar_scrub_failed": ("Deleted text may still sit in free space inside a side database.",
                             "Stop the SLM service, then run the erasure again."),
    "backup_obligations_pending": (
        "Backups still hold this profile. They are listed as erasure obligations and are "
        "removed from the data on any restore.",
        "Delete those backups, or restore none of them."),
    "backup_snapshots_unreadable": (
        "Some backup files could not be read, so it is unknown whether they hold this profile. "
        "They are listed as erasure obligations.", "Delete or repair those backups."),
    "backup_scan_failed": ("The backups could not be checked.", "Run the erasure again."),
    "context_cache_failed": ("The context cache could not be cleared.", "Run the erasure again."),
    "media_failed": ("Some pictures or documents could not be removed.",
                     "Run the erasure again."),
    "learning_db_failed": ("Learning data could not be removed.", "Run the erasure again."),
    "code_graph_failed": ("The code graph could not be cleaned.", "Run the erasure again."),
    "audit_request_failed": ("The erasure could not be recorded, so nothing was erased.",
                             "Check disk space and run the erasure again."),
    "erasure_aborted": ("Nothing was erased.", "Run the erasure again."),
}


#: Markers that mean the erasure cannot be SHOWN afterwards, not that data is left.
PROVABILITY_ONLY = frozenset({"receipt_persist_failed", "audit_completion_failed"})


def apply_proofs(counts: dict, proofs: Any) -> None:
    """Turn the unproven owners of a targeted erasure into named markers.

    A snapshot still holding the facts is an outstanding obligation, a journal
    that could not be written is a failure, and anything else is an owner that
    could not prove its projection gone. (These were all counted as
    ``vector_store_failures``, which named none of them.)
    """
    other = 0
    for proof in proofs:
        if proof.erased:
            continue
        if proof.owner == "backup_copies":
            counts["backup_obligations_pending"] = len(proof.residue)
        elif proof.owner == "restore_journal":
            counts["restore_journal_failed"] = 1
        else:
            other += 1
    if other:
        counts["vector_store_failures"] = other


def subject_ref(kind: str, name: str) -> str:
    """A stable reference to an erased subject that does not contain its name.

    The receipt, the audit chain and the log must be able to say WHICH erasure
    this was without keeping the words that were erased: an entity's name is the
    personal data. The same name always gives the same reference, so a person
    who knows the name can find the receipt; nobody reading the receipt learns it.
    """
    import hashlib

    digest = hashlib.sha256(f"{kind}:{name.strip().casefold()}".encode()).hexdigest()
    return f"{kind}:{digest[:32]}"


def remaining_markers(result: Mapping[str, Any], *, data_only: bool = False) -> list[str]:
    """Every key of ``result`` that says something is left, in a stable order.

    ``data_only`` leaves out the markers that only affect whether the erasure
    can be proven (see :data:`PROVABILITY_ONLY`): they decide
    ``erasure_provable``, not ``erasure_complete``.
    """
    found = []
    for key, value in result.items():
        if key in _NOT_FAILURES or not value or isinstance(value, (dict, list)):
            continue
        if data_only and key in PROVABILITY_ONLY:
            continue
        if key.endswith(("_failed", "_failures")) or key in _RESIDUE_MARKERS \
                or key == "table_delete_failures":
            found.append(key)
    return sorted(found)


def erasure_succeeded(result: Mapping[str, Any]) -> bool:
    """True only when the data is gone AND the erasure can be shown to have happened.

    A result with no verdict keys (the entity path) is judged on its markers
    alone; their absence is not a failure.
    """
    if remaining_markers(result):
        return False
    for verdict in ("erasure_complete", "erasure_provable"):
        if verdict in result and not result.get(verdict):
            return False
    return True


def _step_text(key: str) -> tuple[str, str]:
    if key in _REMAINING:
        return _REMAINING[key]
    return (f"Step '{key.removesuffix('_failed').removesuffix('_failures')}' did not finish.",
            "Run the erasure again.")


def describe(result: Mapping[str, Any]) -> dict[str, Any]:
    """The statement of an erasure: ``complete``, what was ``removed``, what ``remains``."""
    markers = remaining_markers(result)
    removed = {k: v for k, v in result.items()
               if isinstance(v, int) and not isinstance(v, bool) and v > 0
               and k not in markers and k not in _NOT_FAILURES
               and not k.endswith(("_pending", "_recorded", "_scanned", "_registered"))}
    remains = []
    for key in markers:
        what, retry = _step_text(key)
        remains.append({"step": key, "what": what, "retry": retry, "count": result.get(key)})
    if not remains and "erasure_provable" in result and not result.get("erasure_provable"):
        remains.append({
            "step": "receipt",
            "what": "The data is gone, but the tamper-evident receipt could not be written, "
                    "so the erasure cannot be shown afterwards.",
            "retry": "Check disk space; the receipt is written again on the next erasure run.",
            "count": 1})
    return {
        "complete": bool(result.get("erasure_complete", not remains)) and not remains,
        "provable": bool(result.get("erasure_provable", not remains)),
        "removed": removed,
        "remains": remains,
    }


def failure_message(result: Mapping[str, Any]) -> str:
    """One plain sentence for a surface that has room for only one."""
    statement = describe(result)
    if not statement["remains"]:
        return ""
    steps = "; ".join(item["what"] for item in statement["remains"])
    return f"The erasure is not complete. {steps}"
