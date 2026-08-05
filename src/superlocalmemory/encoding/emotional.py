# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Emotional Tagging (VADER).

Extracts emotional valence and arousal from text.
Emotionally charged memories are stored more strongly and retrieved more easily
(amygdala tagging principle from neuroscience).

Ported from V1 — VADER-based, zero-LLM, works in all modes.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe, fork-safe VADER singleton
#
# Design notes:
#   • _vader_analyzer = None means "not yet initialized for this process" OR
#     "initialized but vaderSentiment is unavailable" (both cases return None
#     from _get_vader, which causes tag_emotion to fall back to the keyword
#     heuristic — the correct, observable behavior in both cases).
#
#   • _vader_pid tracks which OS process last ran initialization. On fork()
#     (e.g., under pytest-xdist, multiprocessing, or uwsgi) the child inherits
#     the parent's _vader_analyzer pointer but the native SentimentIntensityAnalyzer
#     object may reference memory that is no longer safe in the child's address
#     space. Re-initializing per-PID guarantees each process gets a fresh, safe
#     object constructed in its own address space.
#
#   • _vader_lock (threading.Lock) prevents the double-initialization race in a
#     multi-threaded caller (e.g., the ingestion thread-pool in store_pipeline.py
#     where tag_emotion is called from multiple MaterializerWorker threads
#     concurrently). Without the lock, Thread A and Thread B can both see
#     _vader_analyzer is None, both enter the import block, and both try to load
#     vaderSentiment's lexicon simultaneously — causing memory corruption when
#     the native C-extension in the same process's numpy/BLAS layer is already
#     running in a third thread (the LanceDB tokio background thread). The lock
#     serializes VADER initialization so only one thread loads the lexicon.
#
#   • After initialization, reads bypass the lock entirely (fast-path at top of
#     _get_vader). The GIL guarantees atomic reads of _vader_pid and
#     _vader_analyzer in CPython; writing _vader_pid AFTER _vader_analyzer
#     (always, inside the lock) ensures a reader that sees a matching _vader_pid
#     is guaranteed to also see the correctly initialized _vader_analyzer.
# ---------------------------------------------------------------------------

_vader_analyzer = None          # None = uninitialized OR unavailable
_vader_pid: int | None = None   # PID in which _vader_analyzer was initialized
_vader_lock = threading.Lock()  # Serializes initialization only; not scoring


def _get_vader():
    """Lazy-load VADER — thread-safe via lock, fork-safe via PID tracking.

    Returns a SentimentIntensityAnalyzer instance, or None if vaderSentiment
    is not installed.  Callers must treat None as "fallback to keyword heuristic."
    """
    global _vader_analyzer, _vader_pid
    current_pid = os.getpid()

    # Fast-path: already initialized in this process — no lock needed.
    # We read _vader_pid first; if it matches current_pid, _vader_analyzer was
    # written before _vader_pid was set (see ordering in slow-path below),
    # so the value we read is fully initialized.
    if _vader_pid == current_pid:
        return _vader_analyzer

    # Slow-path: either first call in this process, or a forked child.
    with _vader_lock:
        # Double-checked locking: another thread may have raced us here.
        if _vader_pid == current_pid:
            return _vader_analyzer

        # Initialize (or re-initialize) for this process.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning, module="vaderSentiment",
                )
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed — emotional tagging disabled")
            _vader_analyzer = None

        # Write _vader_pid LAST so that the fast-path can safely use it as a
        # "initialization complete" signal.
        _vader_pid = current_pid

    return _vader_analyzer


@dataclass(frozen=True)
class EmotionalTag:
    """Emotional metadata for a memory or fact."""

    valence: float   # -1.0 (negative) to +1.0 (positive)
    arousal: float   # 0.0 (calm) to 1.0 (intense)


def tag_emotion(text: str) -> EmotionalTag:
    """Extract emotional valence and arousal from text.

    Valence: VADER compound score [-1, +1].
    Arousal: absolute compound + max(pos, neg) — higher = more emotional intensity.
    Falls back to keyword heuristic when VADER is unavailable.
    """
    analyzer = _get_vader()
    if analyzer is None:
        return _keyword_fallback(text)

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]     # -1 to +1
    pos = scores["pos"]               # 0 to 1
    neg = scores["neg"]               # 0 to 1

    valence = compound
    # Arousal = emotional intensity regardless of direction
    arousal = min(1.0, abs(compound) * 0.6 + max(pos, neg) * 0.4)

    return EmotionalTag(valence=round(valence, 4), arousal=round(arousal, 4))


_POSITIVE_WORDS: frozenset[str] = frozenset({
    "love", "amazing", "wonderful", "great", "happy", "fantastic",
    "excellent", "beautiful", "awesome", "brilliant", "incredible",
    "joy", "thrilled", "grateful", "delighted", "superb",
})

_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "hate", "terrible", "horrible", "awful", "bad", "worst",
    "angry", "frustrated", "disappointed", "sad", "miserable",
    "disgusting", "dreadful", "pathetic", "furious", "outraged",
})


def _keyword_fallback(text: str) -> EmotionalTag:
    """Lightweight sentiment heuristic when VADER is unavailable.

    Counts positive/negative keywords and derives approximate valence/arousal.
    """
    if not text.strip():
        return EmotionalTag(valence=0.0, arousal=0.0)

    words = set(text.lower().split())
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return EmotionalTag(valence=0.0, arousal=0.0)

    # Valence: positive - negative, normalised to [-1, 1]
    raw_valence = (pos_count - neg_count) / total
    valence = max(-1.0, min(1.0, raw_valence))

    # Arousal: how many emotional words relative to total word count
    word_count = max(len(words), 1)
    arousal = min(1.0, total / word_count * 2.0)

    return EmotionalTag(valence=round(valence, 4), arousal=round(arousal, 4))


def is_emotionally_significant(tag: EmotionalTag, threshold: float = 0.3) -> bool:
    """Check if the emotional signal is strong enough to boost importance."""
    return tag.arousal >= threshold


def emotional_importance_boost(tag: EmotionalTag) -> float:
    """Compute importance boost from emotional signal.

    Returns 0.0-0.3 boost. High arousal memories get stored more strongly
    (amygdala-inspired encoding enhancement).
    """
    if tag.arousal <= 0.2:
        return 0.0
    return min(0.3, tag.arousal * 0.3)
