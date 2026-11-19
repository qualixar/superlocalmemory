# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A credential is stored as a typed placeholder, never in clear (WS-5, ``trust.secret_screen``).

Before this, memory text was kept exactly as written: ``redact_secrets`` ran on
text leaving the machine (prompts, mesh, dashboards) and on text SLM produced
itself (``ContentOrigin.DERIVED_TEXT``), but not on what a person or an agent
asked to remember. A pasted API key was stored, indexed for search and journaled.

With the flag on, the one function every save door shares
(``memory_core.save_path.prepare_for_save``, reached by ``remember`` on every
door, ``observe``, the hooks that save, imports, documents and media) calls
:func:`screen` first. A match becomes ``[REDACTED:TYPE:LAST4]``, the house
placeholder (``core.credential_shapes.local_marker``). The original is not kept
anywhere: not in a table, the full-text index, the journal, the outbox or a log
line, because the text those writers see is already the screened text, and only
counts per type are ever logged.

What is screened is what a key looks like: the vendor shapes and key formats of
``security_primitives`` at normal aggression, and a long high-entropy run. A
sentence such as "the wifi password is hunter2" is not a credential shape and
is kept, as it always was: redacting at storage cannot be undone
(``core/credential_shapes``). The four characters a placeholder keeps are the
last four of a long key (the house format), never of a short value.

States: ``on`` is the default (D-50) and screens; ``off`` changes nothing; ``shadow`` counts what would be
screened and logs the counts, changing nothing. The kill switch
reads as off. A screen that cannot run while it is ``on`` refuses the save
(the error propagates) rather than storing the text in clear.

The side fields of a save (``slot``, ``slot_value`` and every cue) are screened too, by
:func:`screen_field`, at the daemon's save route and at the engine's slot door: they are
stored outside the memory text, so ``prepare_for_save`` never sees them.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core.feature_flags import FeatureFlags, FlagState

logger = logging.getLogger(__name__)

FLAG = "trust.secret_screen"
_PLACEHOLDER = re.compile(r"\[REDACTED:([A-Z0-9_]+):[^\]\n]{1,16}\]")
#: Strings shorter than this cannot hold a vendor key or a 32-character run.
_MIN_LEN = 20
#: Metadata keys the server itself sets (hashes, ids): never screened.
_SERVER_KEY_PREFIX = "_slm_"


def state() -> FlagState:
    """The flag as the daemon, a CLI or a hook sees it (environment, file, kill switch)."""
    try:
        from superlocalmemory.infra.data_root import canonical_data_root

        return FeatureFlags(canonical_data_root()).state(FLAG)
    except Exception:  # noqa: BLE001 - an unreadable flag file never stops a save
        logger.debug("secret screen flag unreadable; read as off", exc_info=True)
        return FlagState.OFF


#: The house rule's own example (``[REDACTED:TYPE:LAST4]``): text that quotes the rule
#: holds no credential, so it is never counted as one.
_EXAMPLE = re.compile(r"\[REDACTED:TYPE:LAST4\]", re.IGNORECASE)


def types_in(text: str) -> Counter:
    """How many placeholders of each type ``text`` holds (never any value)."""
    return Counter(_PLACEHOLDER.findall(_EXAMPLE.sub("", text or "")))


@dataclass(frozen=True)
class Screened:
    text: str
    #: Credentials newly replaced in this text, by type.
    counts: Mapping[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def screen(text: str) -> Screened:
    """``text`` with credentials replaced by typed placeholders; pure."""
    if not isinstance(text, str) or len(text) < _MIN_LEN:
        return Screened(text, {})
    from superlocalmemory.core.security_primitives import redact_secrets

    out = redact_secrets(text)
    if out == text:
        return Screened(text, {})
    found = types_in(out)
    found.subtract(types_in(text))  # placeholders already there are not new
    return Screened(out, {kind: n for kind, n in found.items() if n > 0})


def screen_tree(value: Any) -> tuple[Any, Counter]:
    """Every string of a dict, list or tuple screened (keys too); never mutates.

    Keys the server sets (``_slm_*``) and non-string leaves are left as they are.
    """
    if isinstance(value, str):
        got = screen(value)
        return got.text, Counter(got.counts)
    if isinstance(value, dict):
        out: dict = {}
        total: Counter = Counter()
        for key, item in value.items():
            if isinstance(key, str) and key.startswith(_SERVER_KEY_PREFIX):
                out[key] = item
                continue
            new_key, k_counts = screen_tree(key)
            new_item, v_counts = screen_tree(item)
            out[new_key] = new_item
            total += k_counts + v_counts
        return out, total
    if isinstance(value, (list, tuple)):
        pairs = [screen_tree(item) for item in value]
        items = [p[0] for p in pairs]
        total = Counter()
        for _item, counts in pairs:
            total += counts
        return (tuple(items) if isinstance(value, tuple) else items), total
    return value, Counter()


def screen_field(text: str | None) -> str | None:
    """One caller-supplied side field (a slot, a slot value, a cue) as the flag says: replaced
    when the screen is on, counted when in shadow, as written when off. ``None`` stays ``None``.

    These fields travel outside the memory text (reserved ``_slm_*`` metadata, the cue table),
    so the text screen never sees them; every door that stores one calls this first
    (Fable audit A-11). A screen that cannot run while on raises: the save is refused.
    """
    mode = state()
    if mode is FlagState.OFF or not isinstance(text, str):
        return text
    got = screen(text)
    note(got.counts, acting=mode is FlagState.ON)
    return got.text if mode is FlagState.ON else text


def screen_fields(texts: list[str] | None) -> list[str] | None:
    """:func:`screen_field` over a list (cue specs); ``None`` stays ``None``."""
    return None if texts is None else [screen_field(t) for t in texts]


def note(counts: Mapping[str, int], *, acting: bool) -> None:
    """One log line with counts per type: never the text, never a value."""
    if not counts:
        return
    kinds = ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
    logger.info("secret screen: %s %d credential(s) before save (%s)",
                "replaced" if acting else "would replace", sum(counts.values()), kinds)


__all__ = ["FLAG", "Screened", "note", "screen", "screen_field", "screen_fields", "screen_tree",
           "state", "types_in"]
