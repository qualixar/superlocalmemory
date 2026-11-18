# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Why each recalled memory was returned, in one block an agent can read (WS-4, W3-7b).

A recall result already carries the raw parts (``channel_scores``, the answer
check's ``tier``, ``age_label``, and from the truth layer ``disputed`` and
``source``), spread across a dozen fields. Hosts and agents asked one thing:
should I rely on this? ``why`` answers it in four keys, from those same fields
(nothing new is computed, so it cannot disagree with them):

* ``found_by``: the retrieval channels that found it, strongest first (up to 3);
* ``answer_check``: ``supported`` / ``weak`` / ``related`` when the answer check
  ran, ``not_checked`` when it did not;
* ``standing``: ``current``, ``disputed`` (two memories disagree: ask which is
  true now) or ``source changed`` / ``source missing`` (re-read the file);
* ``said``: how long ago it was saved ("" when undated);
* ``origin`` (only while ``trust.taint`` is on or in shadow, WS-5): ``class``, ``taint``
  and ``label`` of where it came from, copied from the result's ``origin``, which the
  truth notes fill from one batched read (never a read per result here).
"""

from __future__ import annotations

#: Channels named in ``found_by``.
TOP_CHANNELS = 3


def why(item: dict, found_by: tuple[str, ...] = ()) -> dict:
    """``found_by`` names the channels when the result carries no scores (the
    keyword fallback); a scored result always speaks for itself."""
    scores = item.get("channel_scores") or {}
    found = sorted(((k, v) for k, v in scores.items()
                    if isinstance(v, (int, float)) and v > 0), key=lambda kv: (-kv[1], kv[0]))
    state = (item.get("source") or {}).get("state") if isinstance(item.get("source"), dict) \
        else None
    if item.get("disputed"):
        standing = "disputed"
    elif state in ("changed", "missing"):
        standing = f"source {state}"
    else:
        standing = "current"
    check = str(item.get("tier") or "not_checked")
    if item.get("weak") and check == "supported":
        check = "weak"  # below the answer check's threshold though labelled supported
    block = {"found_by": [k for k, _v in found[:TOP_CHANNELS]] or list(found_by),
             "answer_check": check,
             "standing": standing,
             "said": str(item.get("age_label") or "")}
    origin = item.get("origin")
    if isinstance(origin, dict) and origin.get("class"):  # filled by truth/recall_notes
        block["origin"] = {k: str(origin[k]) for k in ("class", "taint", "label") if k in origin}
    return block


def _killed() -> bool:
    try:
        from superlocalmemory.core.feature_flags import kill_switch_on
        from superlocalmemory.infra.data_root import canonical_data_root

        return kill_switch_on(canonical_data_root())
    except Exception:  # noqa: BLE001 - a label is never worth a failed recall
        return False


def with_why(results: list[dict], *, found_by: tuple[str, ...] = ()) -> list[dict]:
    """Each result with its ``why`` block; never raises (presentation only).

    With the kill switch on (D-23) results come back as they were before 4.2.30: no ``why``,
    so no standing and no origin either."""
    if _killed():
        return results
    out = []
    for item in results:
        try:
            out.append({**item, "why": why(item, found_by)} if isinstance(item, dict) else item)
        except Exception:  # noqa: BLE001 - never fail a recall over a label
            out.append(item)
    return out


__all__ = ["TOP_CHANNELS", "why", "with_why"]
