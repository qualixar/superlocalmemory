# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""SuperLocalMemory V3.2 -- Channel Registry.

Self-registration system for retrieval channels. Prevents merge
conflicts when Phases 2/3/4 add new channels in parallel.

Design: Registry Pattern + Protocol (structural subtyping).
Each channel registers itself with a name and implements the
RetrievalChannel protocol. The registry dispatches search()
calls to all registered channels and collects results.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalChannel(Protocol):
    """Protocol for retrieval channels.

    Any object with a search() method matching this signature
    is a valid channel. No inheritance required.
    """

    def search(
        self,
        query: Any,
        profile_id: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Search for relevant facts. Returns [(fact_id, score)]."""
        ...


# Type alias for post-retrieval filter functions.
# Filters operate on the FULL channel results dict, NOT per-channel.
# Signature: (all_channel_results, profile_id, context) -> filtered_results
FilterFn = Callable[
    [dict[str, list[tuple[str, float]]], str, Any],
    dict[str, list[tuple[str, float]]]
]


class ChannelRegistry:
    """Registry for retrieval channels with self-registration.

    Usage:
        registry = ChannelRegistry()
        registry.register_channel("semantic", semantic_ch, needs_embedding=True)
        registry.register_channel("bm25", bm25_ch)
        registry.register_filter(temporal_filter)
        results = registry.run_all(query, "default", embedder=emb)

    BACKWARD COMPATIBILITY:
    - register_channel() defaults needs_embedding=False (existing channels unaffected)
    - run_all() is a new method (no existing callers to break)
    - Filters use dict-based signature for Phase 4 temporal filter compatibility
    """

    def __init__(self) -> None:
        self._channels: dict[str, RetrievalChannel] = {}
        self._filters: list[FilterFn] = []
        self._embedding_channels: set[str] = set()

    def register_channel(
        self, name: str, channel: RetrievalChannel, needs_embedding: bool = False,
    ) -> None:
        """Register a retrieval channel by name.

        Args:
            name: Channel identifier (e.g., "semantic", "spreading_activation").
            channel: Object implementing RetrievalChannel protocol.
            needs_embedding: If True, raw query string is embedded into a vector
                before passing to channel.search(). Required for channels that
                expect vector input (semantic, spreading_activation).
        """
        self._channels[name] = channel
        if needs_embedding:
            self._embedding_channels.add(name)
        logger.debug("Registered channel: %s (needs_embedding=%s)", name, needs_embedding)

    def register_filter(self, fn: FilterFn) -> None:
        """Register a post-retrieval filter function.

        Filters run after all channels, before fusion. Used for
        temporal validity filtering (Phase 4) and other concerns.

        Filter signature: (channel_results_dict, profile_id, context) -> filtered_dict
        """
        self._filters.append(fn)
        logger.debug("Registered filter: %s", getattr(fn, '__name__', str(fn)))

    def run_all(
        self,
        query: str,
        profile_id: str,
        *,
        embedder: Any | None = None,
        disabled: set[str] | None = None,
        top_k: int = 50,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run all registered channels and return results.

        Channels in `disabled` are skipped. Channels in _embedding_channels
        receive embedder.embed(query) instead of raw query text.
        Errors in channels are logged, not raised (Rule 19).

        Returns dict of channel_name to [(fact_id, score)].
        """
        disabled = disabled or set()
        out: dict[str, list[tuple[str, float]]] = {}

        for name, channel in self._channels.items():
            if name in disabled:
                continue
            try:
                if name in self._embedding_channels and embedder is not None:
                    q_emb = embedder.embed(query)
                    results = channel.search(q_emb, profile_id, top_k)
                else:
                    results = channel.search(query, profile_id, top_k)
                if results:
                    out[name] = results
            except Exception as exc:
                logger.warning("Channel %s failed: %s", name, exc)

        # Filters operate on the FULL results dict (not per-channel).
        for fn in self._filters:
            try:
                out = fn(out, profile_id, None)
            except Exception as exc:
                logger.debug("Filter %s failed: %s",
                            getattr(fn, '__name__', str(fn)), exc)

        return out

    @property
    def channel_names(self) -> list[str]:
        """List of registered channel names."""
        return list(self._channels.keys())

    @property
    def channel_count(self) -> int:
        """Number of registered channels."""
        return len(self._channels)
