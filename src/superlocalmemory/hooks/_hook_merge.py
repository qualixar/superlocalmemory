# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Replace-and-add merge of SLM-owned hook entries (4.2.30).

Hooks already live in people's harnesses, so SLM may only replace its own
entries and add missing ones. This helper is pure: it never mutates its
arguments, keeps every non-SLM entry (and every non-SLM command inside a
mixed group) where it was, and puts the new SLM entries where the first old
SLM entry stood, so a second run over the result changes nothing.
"""

from __future__ import annotations

from typing import Any, Callable


def replace_owned(
    existing: list[Any],
    new_entries: list[dict[str, Any]],
    is_owned_command: Callable[[Any], bool],
    *,
    is_owned_hook: Callable[[Any], bool] | None = None,
) -> list[Any]:
    """Return ``existing`` with SLM-owned hooks swapped for ``new_entries``.

    ``is_owned_hook`` (the whole hook object) decides ownership when given; otherwise the
    command alone does.
    """
    result: list[Any] = []
    insert_at: int | None = None
    for entry in existing:
        hooks = entry.get("hooks") if isinstance(entry, dict) else None
        if not isinstance(hooks, list):
            result.append(entry)
            continue
        kept = [h for h in hooks if not _owned(h, is_owned_command, is_owned_hook)]
        if len(kept) == len(hooks):
            result.append(entry)
            continue
        if kept:  # a mixed group: the user's own hook stays, SLM's goes
            result.append({**entry, "hooks": kept})
            if insert_at is None:
                insert_at = len(result)
        elif insert_at is None:
            insert_at = len(result)
    fresh = [dict(e) for e in new_entries]
    if insert_at is None:
        return [*result, *fresh]
    return [*result[:insert_at], *fresh, *result[insert_at:]]


def _owned(hook: Any, by_command: Callable[[Any], bool],
           by_hook: Callable[[Any], bool] | None) -> bool:
    if by_hook is not None:
        return bool(by_hook(hook))
    return isinstance(hook, dict) and by_command(hook.get("command", ""))


def has_owned(entries: Any, is_owned_command: Callable[[Any], bool]) -> bool:
    """True when any group in ``entries`` holds an SLM-owned command."""
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(entry, dict) and isinstance(entry.get("hooks"), list)
        and any(isinstance(h, dict) and is_owned_command(h.get("command", ""))
                for h in entry["hooks"])
        for entry in entries
    )
