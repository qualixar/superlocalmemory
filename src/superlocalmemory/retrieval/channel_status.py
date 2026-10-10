# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""What happened to each retrieval channel on one recall.

WHY THIS EXISTS
---------------
A channel that crashes on every query and a channel that correctly found
nothing produced the same observable result: absence. Both simply had no entry
in the fused candidate map, and the only trace was a log line on a machine
nobody is reading.

That matters because the channels are not interchangeable. Lexical search
finding nothing for a conceptual question is the system working. Lexical search
raising on every question is an outage that looks, from the outside, like a
store with nothing in it — the user sees weaker answers and has no way to tell
which of the two they are getting.

There is a third case the absence hid, and it is the worst of them: a channel
that never ran. When the query embedding is unavailable, three of the five
channels are never even dispatched. Nothing in the answer said so.

WHY MORE THAN "ok / empty / error"
----------------------------------
Because the remedies differ, and a status whose remedy is ambiguous is a status
nobody acts on. ``error`` is a bug to fix. ``timeout`` is a capacity or data-size
problem. ``no_embedding`` means the embedding provider is down and several
channels are silently offline together. ``disabled`` and ``not_configured`` are
someone's deliberate choice and must not read as faults — which is the point:
without naming them, an operator reading a list of missing channels cannot tell
their own configuration apart from a failure.

STRINGS, NOT AN ENUM
--------------------
This crosses the MCP and HTTP surfaces, where it is JSON either way. A str-valued
constant serialises without a custom encoder and compares equal to what a client
sends back.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "ALL_STATUSES",
    "CHANNEL_NAMES",
    "ChannelStatus",
    "DISABLED",
    "EMPTY",
    "ERROR",
    "NEEDS_SERVICE",
    "NOT_CONFIGURED",
    "NO_CANDIDATES",
    "NO_EMBEDDING",
    "OK",
    "TIMEOUT",
    "WARMING",
    "is_fault",
]

ChannelStatus = Literal[
    "ok",
    "empty",
    "error",
    "timeout",
    "disabled",
    "not_configured",
    "no_embedding",
    "no_candidates",
    "warming",
    "needs_service",
]

#: Ran and contributed candidates.
OK: ChannelStatus = "ok"
#: Ran, contributed nothing. A legitimate answer, not a fault.
EMPTY: ChannelStatus = "empty"
#: Raised. This answer is missing whatever this channel alone could see.
ERROR: ChannelStatus = "error"
#: Abandoned at the hang guard. The answer is incomplete, not merely late.
TIMEOUT: ChannelStatus = "timeout"
#: Excluded by configuration or by a per-recall ablation flag.
DISABLED: ChannelStatus = "disabled"
#: No such channel on this engine — not built, or its dependency is absent.
NOT_CONFIGURED: ChannelStatus = "not_configured"
#: Needed the query embedding, which was unavailable. Several channels fail
#: together this way, and none of them individually did anything wrong.
NO_EMBEDDING: ChannelStatus = "no_embedding"
#: Had nothing to work on. Distinct from ``empty``, which claims a search
#: happened: a channel that re-scores other channels' candidates never searched
#: at all when there were none, and if the reason there were none is that the
#: others failed, calling this "found nothing" hides the actual fault.
NO_CANDIDATES: ChannelStatus = "no_candidates"
#: Needed the query embedding, but the embedding model was still loading when
#: this recall stopped waiting (a just-started daemon) -- or, for
#: ``entity_graph``, the daemon's start-up build of the graph held it
#: (retrieval/entity_graph_warmup). Transient: the same question shortly
#: afterwards gets this channel back. Reported as incomplete, never as
#: "found nothing".
WARMING: ChannelStatus = "warming"
#: Needed the query embedding, but the SLM service that makes text vectors for this
#: process is not running (not loading: not running). It stays so until the service
#: is started, so the answer says to start it.
NEEDS_SERVICE: ChannelStatus = "needs_service"

ALL_STATUSES: frozenset[str] = frozenset({
    OK, EMPTY, ERROR, TIMEOUT, DISABLED, NOT_CONFIGURED, NO_EMBEDDING,
    NO_CANDIDATES, WARMING, NEEDS_SERVICE,
})

#: Every channel a recall can report on, so a caller can tell "this channel had
#: no status recorded" (a gap in the reporting) from "this channel reported that
#: it did nothing" (an answer). A missing key is a bug; ``empty`` is not.
CHANNEL_NAMES: tuple[str, ...] = (
    "semantic",
    "bm25",
    "temporal",
    "hopfield",
    "spreading_activation",
    "entity_graph",
    "profile",
)

#: Statuses that mean the answer is worse than it should have been. ``empty``,
#: ``disabled`` and ``not_configured`` are deliberately absent: the first is a
#: valid finding and the other two are somebody's decision.
_FAULTS: frozenset[str] = frozenset({ERROR, TIMEOUT, NO_EMBEDDING, WARMING, NEEDS_SERVICE})


def is_fault(status: str | None) -> bool:
    """Whether this status means the answer was degraded."""
    return (status or "") in _FAULTS
