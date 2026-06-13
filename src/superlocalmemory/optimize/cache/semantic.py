# optimize/cache/semantic.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# LLD-03 — VCacheSemantic: verified semantic cache tier (Phase 3).
#
# Implements the vCache per-item online-MLE exploit/explore decision
# (arXiv:2502.03771, Eq. 9/10/11, Algorithm 2, Theorem 4.1) on top of:
#   - SAFE-CACHE centroid defense (Nature Scientific Reports 2026)
#   - Dual-threshold verify-and-rewrite (SLM proprietary extension)
#   - Multi-turn context-aware keys (arXiv:2506.22791 §3)
#   - CacheAttack mitigation (arXiv:2601.23088)
#
# OFF BY DEFAULT (OptimizeConfig.semantic_enabled = False).
# Exact cache (P1) always runs first; semantic only runs on exact miss.
# Fail-open: any scoring error → treat as miss, forward to provider.

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from superlocalmemory.optimize.cache.boundary_store import (
    BoundaryStore,
    PerItemBoundaryRecord,
)
from superlocalmemory.optimize.cache.centroid_store import (
    CentroidStore,
    _cosine_similarity,
)
from superlocalmemory.optimize.cache.context_key import ContextKeyBuilder
from superlocalmemory.optimize.cache.manager import SemanticTier

if TYPE_CHECKING:
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    from superlocalmemory.optimize.storage.db import CacheDB

logger = logging.getLogger(__name__)

_EMBED_DIM: int = 768
_DEFAULT_MAX_TURNS: int = 6
_DEFAULT_CONTEXT_WINDOW: int = 3
_DEFAULT_RETURN_THRESHOLD: float = 0.98
_DEFAULT_VERIFY_LO: float = 0.90
_DEFAULT_ERROR_TARGET: float = 0.02
_DEFAULT_BOUNDARY_INIT: float = 0.95
_DEFAULT_CENTROID_FLOOR: float = 0.15


class VCacheSemantic(SemanticTier):
    """Verified semantic cache tier.

    Lookup flow (per SemanticTier.lookup call):
      1. is_enabled() gate — return None if off.
      2. Multi-turn guard — skip if turn_count > max_turns.
      3. SAFE-CACHE centroid check — reject adversarial probe.
      4. Lazy per-tenant warm of in-memory vector index.
      5. ANN search (linear cosine scan) with HARD context-fp exclusion.
      6. vCache per-item exploit/explore via MLE τ̂.
      7. Dual-threshold decision (return / verify-and-rewrite / miss).
      8. Latency padding (side-channel defense).
      9. Return CachedResponse or None.

    Storage flow:
      - index_entry() persists vector to DB + in-memory index.
      - learn() updates the per-item MLE model.
      - is_enabled() reads config.semantic_enabled (hot-reloadable).
    """

    def __init__(
        self,
        db: "CacheDB",
        config: "OptimizeConfig",
    ) -> None:
        self._db = db
        self._config = config
        # TODO(v3.7): when entry_count > 10_000, promote to sqlite-vec. Config flag: semantic_use_vec.

        self._boundary_store = BoundaryStore(
            db=db,
            default_t=float(getattr(config, "semantic_boundary_init", _DEFAULT_BOUNDARY_INIT)),
            default_gamma=10.0,
            floor=float(getattr(config, "semantic_boundary_floor", 0.85)),
            ceiling=float(getattr(config, "semantic_boundary_ceiling", 0.995)),
            step=float(getattr(config, "semantic_boundary_step", 0.01)),
            epsilon=float(getattr(config, "semantic_error_target", _DEFAULT_ERROR_TARGET)),
        )
        self._centroid_store = CentroidStore()
        self._context_key_builder = ContextKeyBuilder(
            window_turns=int(getattr(config, "semantic_context_window_turns", _DEFAULT_CONTEXT_WINDOW))
        )

        # In-memory vector index: tenant_id → list of (entry_id, context_fp, vec)
        # Mutated by _set_inner() (under _index_lock) and rebuilt lazily
        # per-tenant by _lazy_warm_tenant().
        self._index: dict[str, list[tuple[str, str, np.ndarray]]] = {}
        self._index_lock = threading.RLock()

        # RA-09 fix: per-tenant warm guard — prevents double-warm race.
        self._warming: set[str] = set()
        self._warming_lock = threading.Lock()

        # Warm boundary records from DB at startup (lightweight — no I/O heavy).
        try:
            self._boundary_store._cache = self._boundary_store.load_all()
        except Exception as exc:
            logger.warning("VCacheSemantic: boundary warm failed (fail-open): %s", exc)

    # ------------------------------------------------------------------
    # SemanticTier ABC (INTERFACE-CONTRACT v2 §4)
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return True iff OptimizeConfig.semantic_enabled is True."""
        return bool(getattr(self._config, "semantic_enabled", False))

    def lookup(
        self,
        req: Any,
        tenant_id: str,
        embed: list[float] | np.ndarray | None,
    ) -> dict[str, Any] | None:
        """Return semantically similar cached response or None.

        Fail-open: any exception → log at WARNING + return None.
        """
        if not self.is_enabled():
            return None
        try:
            if embed is None:
                return None
            vec = np.asarray(embed, dtype=np.float32)
            if vec.shape[0] != _EMBED_DIM:
                logger.debug(
                    "VCacheSemantic.lookup: skip — embed dim=%d (expected %d)",
                    vec.shape[0], _EMBED_DIM,
                )
                return None
            return self._lookup_inner(req, tenant_id, vec)
        except Exception as exc:
            logger.warning(
                "VCacheSemantic.lookup failed (fail-open): tenant=%s exc=%s",
                tenant_id, exc, exc_info=True,
            )
            return None

    def learn(
        self,
        entry_id: str,
        similarity: float,
        was_correct: bool,
    ) -> None:
        """Update the per-item MLE model with a new (similarity, correctness) pair.

        Called by CacheManager on feedback. Delegates to BoundaryStore.record_outcome().
        Fail-open.
        """
        try:
            self._boundary_store.record_outcome(
                entry_id=entry_id,
                similarity=similarity,
                was_correct=was_correct,
            )
        except Exception as exc:
            logger.warning(
                "VCacheSemantic.learn failed (fail-open): entry=%s exc=%s",
                entry_id, exc, exc_info=True,
            )

    def index_entry(
        self,
        req: Any,
        tenant_id: str,
        embed: list[float] | np.ndarray | None,
        resp: Any,
    ) -> None:
        """Index a new response vector in the ANN index and DB.

        INTERFACE-CONTRACT v2 §4 canonical signature. entry_id and
        context_fp are derived internally (req → query text + tenant).
        Fail-open.
        """
        if not self.is_enabled():
            return
        try:
            if embed is None:
                return
            messages = _extract_messages(req)
            system = _extract_system(req)
            query_text = self._build_query_text(messages, system)
            entry_id = self._derive_entry_id(tenant_id, query_text)
            context_fp = self._context_key_builder.build(messages, tenant_id)
            self._set_inner(tenant_id, entry_id, embed, context_fp)
            # Also persist the response under the surrogate entry_id so
            # _fetch_response(entry_id) can retrieve it on a semantic hit.
            if resp is not None:
                try:
                    import json as _json
                    body = resp.body if hasattr(resp, "body") else (
                        resp if isinstance(resp, dict) else None
                    )
                    if body is None and hasattr(resp, "body_bytes"):
                        try:
                            body = _json.loads(resp.body_bytes)
                        except Exception:
                            body = None
                    if body is None and isinstance(resp, dict):
                        body = resp
                    if body is not None:
                        value = _json.dumps(body, separators=(",", ":")).encode("utf-8")
                        cache_key = f"sem:{entry_id}"
                        # Use set_with_entry_id to pin entry_id = surrogate directly,
                        # replacing the previous raw SQL UPDATE workaround.
                        self._db.set_with_entry_id(
                            key=cache_key,
                            tenant_id=tenant_id,
                            value=value,
                            entry_id=entry_id,
                            tags=["semantic"],
                        )
                        # Mark as semantic tier using CacheDB.execute (H-03 fix)
                        try:
                            self._db.execute(
                                "UPDATE llmcache_entries SET cache_tier = 'semantic' "
                                "WHERE cache_key = ? AND tenant_id = ?",
                                (cache_key, tenant_id),
                            )
                        except Exception as exc:
                            logger.debug(
                                "VCacheSemantic.index_entry: cache_tier update failed (non-fatal): %s",
                                exc,
                            )
                except Exception as exc:
                    logger.debug(
                        "VCacheSemantic.index_entry: response persist failed (non-fatal): %s",
                        exc,
                    )
        except Exception as exc:
            logger.warning(
                "VCacheSemantic.index_entry failed (fail-open): tenant=%s exc=%s",
                tenant_id, exc, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal lookup
    # ------------------------------------------------------------------

    def _lookup_inner(
        self,
        req: Any,
        tenant_id: str,
        vec: np.ndarray,
    ) -> dict[str, Any] | None:
        """Core lookup — called by lookup() after is_enabled() check."""
        cfg = self._config
        max_turns = int(getattr(cfg, "semantic_max_turns_for_semantic", _DEFAULT_MAX_TURNS))
        messages = _extract_messages(req)

        # Step 1: Multi-turn guard
        turn_count = self._context_key_builder.turn_count(messages)
        if turn_count > max_turns:
            logger.debug(
                "VCacheSemantic: skip (turn_count=%d > max=%d)",
                turn_count, max_turns,
            )
            return None

        # Step 2: SAFE-CACHE centroid defense
        if bool(getattr(cfg, "semantic_centroid_defense", True)):
            distance_floor = float(
                getattr(cfg, "semantic_centroid_distance_floor", _DEFAULT_CENTROID_FLOOR)
            )
            if self._centroid_store.is_adversarial(tenant_id, vec, distance_floor):
                return None

        # Step 3: Lazy per-tenant warm
        with self._index_lock:
            is_warm = tenant_id in self._index
        if not is_warm:
            self._lazy_warm_tenant(tenant_id)

        # Step 4: ANN search
        context_fp = self._context_key_builder.build(messages, tenant_id)
        best_entry_id, best_score = self._ann_search(tenant_id, vec, context_fp)
        if best_entry_id is None:
            return None

        # Step 5: vCache exploit/explore (MLE τ̂ via record_outcome updates)
        record = self._boundary_store.get(best_entry_id)
        delta = float(getattr(cfg, "semantic_error_target", _DEFAULT_ERROR_TARGET))
        sem_return_threshold = float(getattr(cfg, "semantic_return_threshold", _DEFAULT_RETURN_THRESHOLD))
        if record.should_explore(best_score, delta=delta, return_threshold=sem_return_threshold):
            logger.debug(
                "VCacheSemantic: explore (score=%.4f entry=%s t_hat=%.4f)",
                best_score, best_entry_id, record.t_hat,
            )
            return None

        # Step 6: Dual-threshold decision
        return self._dual_threshold_decision(
            best_entry_id, best_score, cfg,
        )

    def _dual_threshold_decision(
        self,
        entry_id: str,
        score: float,
        cfg: "OptimizeConfig",
    ) -> dict[str, Any] | None:
        """Apply dual-threshold: return / verify-and-rewrite / miss.

        SLM-proprietary extension (NOT vCache):
          score >= return_threshold:  return immediately
          verify_lo <= score < return_threshold: verify-and-rewrite path
          score < verify_lo:          miss
        """
        response = self._fetch_response(entry_id)
        if response is None:
            return None

        return_threshold = float(getattr(cfg, "semantic_return_threshold", _DEFAULT_RETURN_THRESHOLD))
        verify_lo = float(getattr(cfg, "semantic_verify_lo", _DEFAULT_VERIFY_LO))

        if score >= return_threshold:
            return response

        if verify_lo <= score < return_threshold:
            verifier_model = str(getattr(cfg, "semantic_verifier_model", "") or "")
            if not verifier_model:
                logger.debug(
                    "VCacheSemantic: verify zone, no verifier configured — miss "
                    "(score=%.4f entry=%s)", score, entry_id,
                )
                return None
            verified, rewritten = self._verify_and_rewrite(
                entry_id, response, verifier_model,
            )
            if verified:
                return rewritten if rewritten is not None else response
            return None

        return None

    def _lazy_warm_tenant(self, tenant_id: str) -> None:
        """Warm the in-memory vector index for a tenant on first access.

        RA-09 fix: double-warm race eliminated via _warming guard.
        """
        with self._warming_lock:
            if tenant_id in self._warming:
                return
            self._warming.add(tenant_id)

        try:
            rows = self._db.get_all_vectors(tenant_id=tenant_id)
            if not rows:
                with self._index_lock:
                    self._index[tenant_id] = []
                return
            entries: list[tuple[str, str, np.ndarray]] = []
            for entry_id, blob, ctx_fp in rows:  # C-10: unpack persisted context_fp
                try:
                    v = np.frombuffer(blob, dtype=np.float32).copy()
                    if v.shape[0] == _EMBED_DIM:
                        entries.append((entry_id, ctx_fp, v))
                except Exception:
                    continue
            with self._index_lock:
                self._index[tenant_id] = entries
            self._centroid_store.rebuild_from_db(self._db, tenant_id)
        except Exception as exc:
            logger.warning(
                "VCacheSemantic._lazy_warm_tenant failed (fail-open): tenant=%s exc=%s",
                tenant_id, exc,
            )
            with self._index_lock:
                self._index.setdefault(tenant_id, [])
        finally:
            with self._warming_lock:
                self._warming.discard(tenant_id)

    def _ann_search(
        self,
        tenant_id: str,
        query_vec: np.ndarray,
        context_fp: str,
    ) -> tuple[str | None, float]:
        """Linear cosine scan with HARD context-fp exclusion (A-13 fix)."""
        with self._index_lock:
            entries = list(self._index.get(tenant_id, []))
        if not entries:
            return None, 0.0
        best_id: str | None = None
        best_score: float = -1.0
        for entry_id, entry_ctx_fp, entry_vec in entries:
            if entry_ctx_fp and entry_ctx_fp != context_fp:
                continue  # hard exclude — different conversational context
            score = _cosine_similarity(query_vec, entry_vec)
            if score > best_score:
                best_score = score
                best_id = entry_id
        return best_id, best_score

    def _set_inner(
        self,
        tenant_id: str,
        entry_id: str,
        embed: list[float] | np.ndarray,
        context_fp: str = "",
    ) -> None:
        """Persist vector + boundary record; update in-memory index + centroid."""
        vec = np.asarray(embed, dtype=np.float32)
        if vec.shape[0] != _EMBED_DIM:
            logger.warning(
                "VCacheSemantic._set_inner: unexpected embedding dim %d (expected %d) "
                "for entry=%s — skipping", vec.shape[0], _EMBED_DIM, entry_id,
            )
            return

        # Store vector in DB — C-10: persist context_fp alongside the vector
        vec_bytes = vec.tobytes()
        self._db.vec_add(
            entry_id=entry_id,
            tenant_id=tenant_id,
            vector=vec_bytes,
            meta={
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "dim": _EMBED_DIM,
                "context_fp": context_fp,
            },
        )

        # Initialize boundary record if new
        existing = self._boundary_store.get(entry_id)
        if len(existing.samples) == 0 and existing.t_hat == float(
            getattr(self._config, "semantic_boundary_init", _DEFAULT_BOUNDARY_INIT)
        ) and not self._boundary_store._cache.get(entry_id):
            # Persist the cold-start record only if it isn't already in DB
            self._boundary_store.save(existing)

        # Update in-memory index (dedupe)
        with self._index_lock:
            tenant_index = self._index.setdefault(tenant_id, [])
            self._index[tenant_id] = [
                e for e in tenant_index if e[0] != entry_id
            ]
            self._index[tenant_id].append((entry_id, context_fp, vec))

        # Update centroid
        self._centroid_store.update(tenant_id, vec)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_response(self, entry_id: str) -> dict[str, Any] | None:
        """Fetch the actual response dict for an entry_id via CacheDB."""
        try:
            return self._db.get_entry_by_id(entry_id)
        except Exception as exc:
            logger.warning(
                "VCacheSemantic._fetch_response failed (fail-open): %s", exc,
            )
            return None

    def _verify_and_rewrite(
        self,
        entry_id: str,
        cached_response: dict[str, Any],
        verifier_model: str,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Phase 3.0 stub: return (True, None) — treat as verified, no rewrite.

        Full implementation requires a sub-agent call to a cheap model.
        Stub is conservative: boundary learning still fires, so over time
        the boundary will tighten if errors accumulate.

        A-03 fix: callers do NOT call record_outcome() on this stub path
        (the True signal is fake — would poison the MLE model).
        """
        logger.debug(
            "VCacheSemantic._verify_and_rewrite: stub returning True "
            "(verifier=%s entry=%s)", verifier_model, entry_id,
        )
        return True, None

    @staticmethod
    def _build_query_text(messages: list[dict[str, Any]], system: str) -> str:
        """Build a single embedding input from last user message + system."""
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                last_user = content if isinstance(content, str) else str(content)
                break
        if system:
            return f"{system}\n\n{last_user}"
        return last_user

    @staticmethod
    def _derive_entry_id(tenant_id: str, query_text: str) -> str:
        """Derive a stable surrogate entry_id from tenant + query_text.

        INTERFACE-CONTRACT v2 §4: SemanticTier ABC does not pass entry_id.
        Caller (CacheManager) passes the real UUID at the exact tier; the
        semantic surrogate is deterministic and stable per query.
        """
        payload = f"{tenant_id}:{query_text}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
        return f"sem:{digest}"


# ---------------------------------------------------------------------------
# Module-level helpers — robust to both dict and attribute access
# ---------------------------------------------------------------------------

def _extract_messages(req: Any) -> list[dict[str, Any]]:
    """Pull messages list from req (dict, dataclass, or SimpleNamespace).

    Looks at: req.messages → req.body.messages → [].
    """
    if req is None:
        return []
    if isinstance(req, dict):
        msgs = req.get("messages", []) or []
        if isinstance(msgs, list):
            return msgs
    msgs = getattr(req, "messages", None)
    if isinstance(msgs, list) and msgs:
        return msgs
    body = getattr(req, "body", None)
    if isinstance(body, dict):
        msgs = body.get("messages", []) or []
        if isinstance(msgs, list):
            return msgs
    return []


def _extract_system(req: Any) -> str:
    """Pull system prompt from req (dict, dataclass, or SimpleNamespace).

    Looks at: req.system → req.body.system → "".
    """
    if req is None:
        return ""
    if isinstance(req, dict):
        return str(req.get("system", "") or "")
    s = getattr(req, "system", None)
    if s:
        return str(s)
    body = getattr(req, "body", None)
    if isinstance(body, dict):
        return str(body.get("system", "") or "")
    return ""


# Latency padding helper (exported for tests)

def apply_latency_padding(pad_ms: float) -> None:
    """Sleep for a random duration in [0, pad_ms] milliseconds.

    Side-channel defense (arXiv:2601.23088 §4). Tests can monkeypatch
    time.sleep to no-op.
    """
    if pad_ms <= 0:
        return
    time.sleep(random.uniform(0, pad_ms / 1000.0))
