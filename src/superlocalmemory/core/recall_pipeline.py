# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall pipeline — extracted free functions for MemoryEngine.recall().

Direction: engine.py imports this module. This module NEVER imports engine.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from superlocalmemory.core.session_identity import is_conversation

if TYPE_CHECKING:
    import os

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.hooks import HookRegistry
    from superlocalmemory.storage.database import DatabaseManager

from superlocalmemory.core.security_primitives import ensure_install_token
from superlocalmemory.storage.models import Mode, RecallResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLD-00 §3 — HMAC fact-id markers (P0.4, SEC-C-01 fix)
# ---------------------------------------------------------------------------
#
# Every fact surfaced in a recall response is tagged with
#   slm:fact:<fact_id>:<hmac8>
# where hmac8 is the first 8 hex chars of HMAC-SHA256(install_token, fact_id).
#
# post_tool_outcome_hook (LLD-09) scans only for this prefix and validates
# the HMAC. Unverified markers are ignored — this closes the tool-output
# injection attack where attacker-controlled output could forge engagement
# signals by spelling a known fact_id.

_HMAC_MARKER_PREFIX = "slm:fact:"
_HMAC_LEN = 8


def _emit_marker(fact_id: str) -> str:
    """Tag ``fact_id`` with its HMAC so downstream hooks can validate.

    Deterministic per install: a given (install_token, fact_id) pair always
    produces the same marker. Token rotation invalidates old markers.
    """
    token = ensure_install_token()
    digest = hmac.new(
        token.encode("utf-8"), fact_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:_HMAC_LEN]
    return f"{_HMAC_MARKER_PREFIX}{fact_id}:{digest}"


def _validate_marker(marker: str) -> str | None:
    """Return ``fact_id`` if ``marker`` is a valid HMAC marker, else None.

    Uses constant-time compare. Never raises.
    """
    if not isinstance(marker, str) or not marker.startswith(_HMAC_MARKER_PREFIX):
        return None
    rest = marker[len(_HMAC_MARKER_PREFIX):]
    fact_id, sep, presented = rest.rpartition(":")
    if not sep or not fact_id or len(presented) != _HMAC_LEN:
        return None
    try:
        token = ensure_install_token()
    except Exception:  # pragma: no cover — install-token I/O failure
        return None
    expected = hmac.new(
        token.encode("utf-8"), fact_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:_HMAC_LEN]
    if hmac.compare_digest(presented, expected):
        return fact_id
    return None


def _apply_markers_to_response(response: RecallResponse) -> None:
    """Populate ``result.marker`` on every result in ``response``, in place.

    Called as the last step of :func:`run_recall` before returning. Empty
    responses pass through untouched.

    # L-P-06: audit flagged ``dataclasses.replace`` as a cheaper path.
    # Verified: ``RecallResult`` is NOT frozen, so the direct in-place
    # attribute assignment below is the O(1) mutation path — no dataclass
    # reconstruction happens. ``replace`` would ALLOCATE a fresh instance
    # per result (strictly slower). Keep the in-place mutation.
    """
    for r in response.results:
        r.marker = _emit_marker(r.fact.fact_id)


def _preserve_exact_lexical_evidence(
    response: RecallResponse,
    query: str,
) -> None:
    """Keep a deterministic exact BM25 hit ahead of learned refinements.

    Adaptive and bandit ranking are valuable for ambiguous candidates, but
    they must not demote a fact containing the caller's exact query behind
    semantically similar noise. This guard runs after every learned layer and
    changes only ordering; it does not introduce or bypass evidence.
    What counts as an exact hit is defined once, in retrieval.exact_lexical,
    which the kind-aware pass reads too so it never moves anything above it.
    """
    from superlocalmemory.retrieval.exact_lexical import (
        bm25_strength,
        is_exact_lexical_hit,
        normalize_query,
    )

    normalized_query = normalize_query(query)
    if len(response.results) < 2:
        return
    exact = [r for r in response.results if is_exact_lexical_hit(r, normalized_query)]
    if not exact:
        return
    strongest = max(exact, key=bm25_strength)
    if response.results[0] is strongest:
        return
    response.results = [
        strongest,
        *(result for result in response.results if result is not strongest),
    ]


# ---------------------------------------------------------------------------
# Stage 8 SB-1 — feed shadow_router from recall-settled signals.
#
# LLD-10 Track A.3 needs live-recall A/B observations to feed ShadowTest
# (pre-promotion) and ModelRollback (post-promotion). The ndcg_at_10
# signal materialises when ``EngagementRewardModel.finalize_outcome``
# settles a row — that is the natural call site for this helper.
#
# This is a THIN wrapper over ``core.shadow_router.get_shadow_router``
# so the finalize-outcome path does not need to import shadow_router
# directly. Fail-soft on every error — recall pipeline integrity comes
# first.
# ---------------------------------------------------------------------------


def feed_recall_settled(
    *,
    learning_db: str,
    profile_id: str,
    query_id: str,
    reward: float | None = None,
    ndcg_at_10: float | None = None,
    memory_db: str = "",
) -> None:
    """Route one settled recall's reward into the shadow router.

    Called when a recall's reward becomes known: an explicit outcome settling
    its bandit play (``reward_from_outcomes``) or an engagement label being
    finalized (``EngagementRewardModel.finalize_outcome``). The router credits
    it to the arm recorded when the recall was SERVED (``shadow_assignments``);
    a query served outside a test is ignored there.

    ``ndcg_at_10`` is the pre-4.1.25 name of ``reward`` — the value was never
    an NDCG. ``memory_db`` is accepted and unused.
    """
    del memory_db
    value = reward if reward is not None else ndcg_at_10
    if value is None or not query_id:
        return
    try:
        from superlocalmemory.core import shadow_router as _sr
        router = _sr.get_shadow_router(
            learning_db=learning_db, profile_id=profile_id,
        )
        router.on_recall_settled(query_id=query_id, reward=float(value))
    except Exception as exc:  # pragma: no cover — defence in depth
        logger.debug("feed_recall_settled error: %s", exc)


# ---------------------------------------------------------------------------
# V3.3.16: Module-level singletons for recall hot-path objects.
# Prevents creating new BehavioralTracker / ForgettingScheduler per recall
# (304 recalls = 304 objects that fragment pymalloc arenas → 25GB).
# ---------------------------------------------------------------------------

_behavioral_tracker_cache: dict[int, object] = {}
_forgetting_scheduler_cache: dict[int, object] = {}


def _get_behavioral_tracker(db: Any) -> Any:
    """Get or create a cached BehavioralTracker for this DB instance."""
    key = id(db)
    if key not in _behavioral_tracker_cache:
        from superlocalmemory.learning.behavioral import BehavioralTracker
        _behavioral_tracker_cache[key] = BehavioralTracker(db)
    return _behavioral_tracker_cache[key]


def _get_forgetting_scheduler(db: Any, config: Any) -> Any:
    """Get or create a cached ForgettingScheduler for this DB instance."""
    key = id(db)
    if key not in _forgetting_scheduler_cache:
        from superlocalmemory.learning.forgetting_scheduler import ForgettingScheduler
        from superlocalmemory.math.ebbinghaus import EbbinghausCurve
        ebbinghaus = EbbinghausCurve(config.forgetting)
        _forgetting_scheduler_cache[key] = ForgettingScheduler(db, ebbinghaus, config.forgetting)
    return _forgetting_scheduler_cache[key]


def release_recall_resources(db: Any) -> None:
    """Release process-level recall singletons owned by a closing database."""
    key = id(db)
    _behavioral_tracker_cache.pop(key, None)
    _forgetting_scheduler_cache.pop(key, None)


def _behavioral_entities(results: list[Any], limit: int = 20) -> list[str]:
    """Read canonical entity IDs from the typed retrieval-result contract.

    ``RetrievalResult`` is a dataclass whose entity evidence lives on
    ``result.fact.canonical_entities``. Treating it as a SQLite mapping silently
    discarded every entity and disabled behavioral entity learning.
    """
    entities: list[str] = []
    seen: set[str] = set()
    for result in results:
        fact = getattr(result, "fact", None)
        for entity_id in getattr(fact, "canonical_entities", ()) or ():
            if not isinstance(entity_id, str) or not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            entities.append(entity_id)
            if len(entities) >= limit:
                return entities
    return entities


# ---------------------------------------------------------------------------
# S8-ARC-04 (v3.4.22): unified ranking entry point.
# ---------------------------------------------------------------------------

_RANKING_MODES: frozenset[str] = frozenset({"off", "v1", "v2", "v2-ensemble"})


from superlocalmemory.learning.signal_kinds import FEEDBACK_ONLY_SQL  # noqa: E402 - late import, order kept


class _ReadOnlyLearningView:
    """Minimal learning-model reader that cannot initialise or mutate a DB."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path.resolve()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{self._db_path.as_uri()}?mode=ro",
            uri=True,
            timeout=0.25,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA busy_timeout=250")
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            connection.close()
            raise
        return connection

    def count_signals(self, profile_id: str) -> int:
        """Count FEEDBACK rows only — an exposure is not feedback.

        Must agree with ``LearningDatabase.count_signals``: a phase resolved
        from one of these is compared against a threshold resolved from the
        other, so a difference between them is a phase that flickers. Both
        share ``FEEDBACK_ONLY_SQL``; a test asserts neither carries its own
        copy of the predicate.
        """
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM learning_signals "
                f"WHERE profile_id = ?{FEEDBACK_ONLY_SQL}",
                (profile_id,),
            ).fetchone()
            return int(row["count"]) if row else 0
        finally:
            connection.close()

    def count_feedback(self, profile_id: str) -> int:
        """Count legacy feedback without running schema initialization.

        Reports the raw ``learning_feedback`` table, which the dashboard
        surfaces as ``legacy_feedback_rows`` alongside a pending-migration
        card. Do NOT gate a ranking phase on this — use ``count_signals``,
        which is the counter every other surface resolves its phase from
        (issue #106).
        """
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM learning_feedback "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return int(row["count"]) if row else 0
        finally:
            connection.close()

    def load_active_model(self, profile_id: str) -> dict[str, Any] | None:
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT state_bytes, bytes_sha256, feature_names, trained_at, "
                "model_version FROM learning_model_state "
                "WHERE profile_id = ? AND is_active = 1 LIMIT 1",
                (profile_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "state_bytes": bytes(row["state_bytes"]),
                "bytes_sha256": row["bytes_sha256"],
                "feature_names": row["feature_names"],
                "trained_at": row["trained_at"],
                "model_version": row["model_version"],
            }
        finally:
            connection.close()


def _resolve_ranking_mode(env: "dict[str, str] | os._Environ[str]") -> str:
    """Map the ``SLM_RANKING`` env var to a canonical mode.

    ``SLM_RANKING`` is an explicit operator policy, and the absence of one
    means ``off``: an upgrade must not start reordering results on its own.

    Fixing the settlement path was necessary for the ensemble to be worth
    enabling, but it is not sufficient. Settlement decides what an arm learns
    *after* a query; the channel weights come from sampling that arm's
    posterior *before* it. Until arms have posteriors shaped by real
    observations, that sample is drawn from the prior, so two identical
    queries can be weighted differently for no reason the store can justify —
    variance with nothing bought by it. A memory system's answers should move
    because the evidence moved.

    So the ensemble stays opt-in until arms carry settled rewards. Set
    ``SLM_RANKING=v2-ensemble`` to enable it, or ``v1``/``v2`` for the
    narrower passes.
    """
    raw = (env.get("SLM_RANKING", "") or "").strip().lower()
    if raw in _RANKING_MODES:
        return raw
    return "off"


def apply_ranking(
    response: "RecallResponse",
    query: str,
    profile_id: str,
    query_id: str,
    *,
    config: Any = None,
    pipeline_version: str = "v2-ensemble",
    record_signals: bool = False,
    record_plays: bool = True,
    memory_db_path: Any = None,
    play_sink: dict | None = None,
    learning_db_path: Any = None,
) -> "RecallResponse":
    """Run the ranking pipeline at the requested version.

    Modes:
      - ``off``: identity — no ranking passes run at all.
      - ``v1``: v3.1 Active-Memory adaptive rerank only.
      - ``v2``: v1 + v3.4.22 lambdarank rerank + signal enqueue.
      - ``v2-ensemble`` (default): v2 + v3.4.22 contextual-bandit ensemble.

    Each underlying pass is already defensive (catches its own exceptions),
    so this wrapper adds an outer try/except to guarantee the caller
    always gets a response back. Previously three separate call sites in
    run_recall chained these; collapsing keeps precedence explicit.
    """
    if pipeline_version == "off":
        return response
    try:
        response = apply_adaptive_ranking(response, query, profile_id,
                                          config=config)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_ranking v1 step skipped: %s", exc)
    if pipeline_version == "v1":
        return response
    try:
        response = apply_v2_adaptive_ranking(
            response, query, profile_id, query_id,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_ranking v2 step skipped: %s", exc)
    if pipeline_version == "v2":
        return response
    try:
        response = apply_v2_bandit_ensemble(
            response, query, profile_id, query_id,
            record_signals=record_signals,
            record_plays=record_plays,
            memory_db_path=memory_db_path,
            play_sink=play_sink,
            learning_db_path=learning_db_path,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_ranking ensemble step skipped: %s", exc)
    return response


# ---------------------------------------------------------------------------
# apply_adaptive_ranking  (was MemoryEngine._apply_adaptive_ranking)
# ---------------------------------------------------------------------------

def apply_adaptive_ranking(
    response: RecallResponse,
    query: str,
    pid: str,
    *,
    config: SLMConfig,
) -> RecallResponse:
    """Apply adaptive re-ranking if enough learning signals exist.

    Phase 1 (< 50 signals): returns response unchanged (backward compat).
    Phase 2 (50+): heuristic boosts from recency, access count, trust.
    Phase 3 (200+): LightGBM ML-based reranking.
    """
    from superlocalmemory.infra.data_root import state_path
    learning_db = state_path("learning.db")
    if not learning_db.exists():
        return response

    # issue #106: count the CANONICAL store, not the legacy one. The
    # dashboard's Living Brain panel and ranker-phase card both resolve their
    # phase from ``learning_signals``; this gate read ``learning_feedback``,
    # so the phase a user was shown and the phase that actually ranked their
    # results were computed from different tables and could disagree without
    # limit. ``learning_feedback`` rows reach this counter through
    # ``legacy_migration``, which copies them forward.
    try:
        signal_count = _ReadOnlyLearningView(learning_db).count_signals(pid)
    except sqlite3.Error:
        # A pre-learning database may not have this optional table yet.
        # Recall remains a query and cannot create it on demand.
        return response

    from superlocalmemory.learning.ranker import (
        PHASE_2_THRESHOLD,
        AdaptiveRanker,
    )

    # Thresholds come from ``learning.ranker`` — the same constants the
    # dashboard gates on. Duplicating the literals here is how the two
    # surfaces drifted apart in the first place.
    if signal_count < PHASE_2_THRESHOLD:
        return response  # Phase 1: no change

    ranker = AdaptiveRanker(signal_count=signal_count)

    from datetime import UTC
    from datetime import datetime as _dt
    _now = _dt.now(UTC)

    result_dicts = []
    for r in response.results:
        _age = 0.0
        _created = getattr(r.fact, "created_at", None)
        if _created:
            try:
                _age = max(0.0, (_now - _dt.fromisoformat(
                    _created.replace("Z", "+00:00")
                )).total_seconds() / 86400.0)
            except (ValueError, TypeError):
                pass
        result_dicts.append({
            "score": r.score,
            "cross_encoder_score": r.score,
            "trust_score": r.trust_score,
            "channel_scores": r.channel_scores or {},
            "fact": {
                "age_days": _age,
                "access_count": r.fact.access_count,
            },
            "_original": r,
        })

    query_context = {"query_type": response.query_type}
    reranked = ranker.rerank(result_dicts, query_context)

    # Rebuild response with new ordering
    for item in reranked:
        original = item.get("_original")
        if original is not None and item.get("_adaptive_score") is not None:
            original.ranking_score = float(item["_adaptive_score"])
    new_results = [d["_original"] for d in reranked]

    return RecallResponse(
        query=response.query,
        mode=response.mode,
        results=new_results,
        query_type=response.query_type,
        channel_weights=response.channel_weights,
        total_candidates=response.total_candidates,
        retrieval_time_ms=response.retrieval_time_ms,
        # v3.6.6: preserve evidence-floor signal across reranking rebuilds.
        no_confident_match=(len(new_results) == 0) and response.no_confident_match,
        **_completeness_of(response),
    )


# ---------------------------------------------------------------------------
# apply_v2_adaptive_ranking (LLD-02 §4.3)
# ---------------------------------------------------------------------------
#
# Opt-in v3.4.22 path: load active model from learning.db with SHA-256
# verification, re-rank via native Booster, enqueue signals async. The
# existing ``apply_adaptive_ranking`` above stays for 3.4.20 callers.
# ---------------------------------------------------------------------------


def apply_v2_adaptive_ranking(
    response: RecallResponse,
    query: str,
    profile_id: str,
    query_id: str,
    *,
    learning_db_path: Any = None,
) -> RecallResponse:
    """LLD-02 §4.3 — load verified model, rerank, enqueue signals.

    Never raises. On any error, returns ``response`` unchanged.
    """
    try:
        from pathlib import Path as _P

        from superlocalmemory.infra.data_root import state_path
        from superlocalmemory.learning.model_cache import load_active
        from superlocalmemory.learning.ranker import AdaptiveRanker

        db_path = (_P(learning_db_path) if learning_db_path
                   else state_path("learning.db"))
        if not db_path.exists():
            return response

        db = _ReadOnlyLearningView(db_path)
        signal_count = db.count_signals(profile_id)
        active = load_active(db, profile_id)

        ranker = AdaptiveRanker(
            signal_count=signal_count,
            active_model=active,
        )

        # Build result-dict shape expected by the ranker's rerank() path.
        from datetime import UTC
        from datetime import datetime as _dt
        _now_v2 = _dt.now(UTC)

        result_dicts: list[dict] = []
        for r in response.results:
            _age_v2 = 0.0
            _created_v2 = getattr(r.fact, "created_at", None)
            if _created_v2:
                try:
                    _age_v2 = max(0.0, (_now_v2 - _dt.fromisoformat(
                        _created_v2.replace("Z", "+00:00")
                    )).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    pass
            result_dicts.append({
                "fact_id": r.fact.fact_id,
                "score": r.score,
                "cross_encoder_score": r.score,
                "trust_score": r.trust_score,
                "channel_scores": r.channel_scores or {},
                "fact": {
                    "age_days": _age_v2,
                    "access_count": r.fact.access_count,
                },
                "_original": r,
            })

        query_context = {
            "query_type": response.query_type,
            "profile_id": profile_id,
        }
        reranked_dicts = ranker.rerank(result_dicts, query_context)
        for item in reranked_dicts:
            original = item.get("_original")
            if original is not None and item.get("_adaptive_score") is not None:
                original.ranking_score = float(item["_adaptive_score"])
        new_results = [d["_original"] for d in reranked_dicts
                       if "_original" in d]

        # S8-SK-04 fix: signal enqueue is OWNED by ``apply_v2_bandit_ensemble``
        # (see below), not this function. Previously both emitted a batch
        # under the same query_id which doubled ``learning_signals`` and
        # tripped the phase-transition threshold at half the intended
        # signal count. This function now just re-ranks; the ensemble path
        # is the single source of signal events.

        return RecallResponse(
            query=response.query,
            mode=response.mode,
            results=new_results,
            query_type=response.query_type,
            channel_weights=response.channel_weights,
            total_candidates=response.total_candidates,
            retrieval_time_ms=response.retrieval_time_ms,
            # v3.6.6: preserve evidence-floor signal across reranking rebuilds.
            no_confident_match=(len(new_results) == 0) and response.no_confident_match,
            **_completeness_of(response),
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_v2_adaptive_ranking skipped: %s", exc)
        return response


# ---------------------------------------------------------------------------
# apply_v2_bandit_ensemble (LLD-03 §5.5)
# ---------------------------------------------------------------------------
#
# Contextual Thompson bandit chooses channel weights. If an LGBM model is
# active, a D8-blended ensemble re-ranks the reweighted candidates. Never
# raises; honours ``SLM_BANDIT_DISABLED=1`` as a kill switch.
# ---------------------------------------------------------------------------


def _rank_key(result) -> tuple[float, str]:
    """Descending ordering key for the score-bias passes.

    Falls back to ``score`` exactly as the writes do. Falling back to 0.0
    instead — which is what a bare ``ranking_score or 0.0`` does — sorts every
    result that no bias touched as if it scored nothing, collapsing the whole
    un-adjusted tail to the bottom in fact_id order and discarding the ordering
    retrieval worked for. ``ensemble_rerank`` populates ``ranking_score`` on
    every candidate, so that only bites when the ensemble is disabled or has
    fallen back — which is precisely when a bias must still behave.

    One function rather than the same expression at each call site: two passes
    that sort by different keys silently overwrite each other's ordering.

    ``is None`` rather than a falsy test, because 0.0 is a real score. A
    cross-encoder that judges a candidate irrelevant returns exactly that, and
    ``x or y`` would then discard the model's verdict and rank the candidate on
    its retrieval score instead — quietly promoting the thing the model just
    ruled out.
    """
    utility = result.ranking_score
    if utility is None:
        utility = result.score
    if utility is None:
        utility = 0.0
    return (-float(utility), result.fact.fact_id)


def _question_key(query: str | None) -> str | None:
    """A stable, text-free key for a question; None when there is no question."""
    if query is None:
        return None
    from superlocalmemory.learning.bandit_draw import normalize_question

    return hashlib.sha256(normalize_question(query).encode("utf-8")).hexdigest()[:16]


def _apply_outcome_bonus(
    results: list, profile_id: str, memory_db_path: Any = None,
    *, query: str | None = None,
) -> list:
    """Nudge ranking by whether each memory has demonstrably helped before.

    Bounded to ``pcos.MAX_BONUS`` (0.15) and weighted by how often the fact has
    actually settled, so history breaks ties between similar-looking memories
    and cannot drag a poor match to the top.

    Order is recomputed here rather than left to the caller: a bonus that does
    not reorder anything is a number nobody reads, which is the failure mode
    this whole wave exists to fix.

    Never raises. On a store where M045 has not run, ``fetch_scores`` returns
    ``{}`` and this is the identity function.
    """
    if not results:
        return results
    try:
        import sqlite3 as _sq

        from superlocalmemory.infra.data_root import state_path
        from superlocalmemory.learning.pcos import (
            RECENT_TOPS,
            bonus_for,
            fetch_scores,
        )

        ids = [r.fact.fact_id for r in results if getattr(r, "fact", None)]
        if not ids:
            return results
        # The store the RECALL used, not whichever one the data-root default
        # resolves to. Hardcoding state_path("memory.db") worked on a default
        # install and silently read another store's scores — or nothing at all —
        # anywhere the engine was pointed elsewhere: a second data root, a test
        # store, a config whose db_path diverges. Both fail modes are silent,
        # because this function is fail-open by design.
        db_file = Path(memory_db_path) if memory_db_path else state_path("memory.db")
        conn = _sq.connect(f"file:{db_file}?mode=ro", uri=True, timeout=0.5)
        try:
            scores = fetch_scores(conn, profile_id, ids)
        finally:
            conn.close()
        if not scores:
            return results

        adjusted = []
        for r in results:
            entry = scores.get(r.fact.fact_id)
            if not entry:
                adjusted.append(r)
                continue
            # A fact that has been winning first place lately stops EARNING the
            # bonus. It keeps everything retrieval gave it — the alternative
            # considered was a 10x demotion, which would bury a memory whose
            # only offence is having been useful.
            if RECENT_TOPS.capped(profile_id, r.fact.fact_id):
                adjusted.append(r)
                continue
            delta = bonus_for(*entry)
            if delta == 0.0:
                adjusted.append(r)
                continue
            adjusted.append(replace(
                r,
                ranking_score=float(
                    r.ranking_score if r.ranking_score is not None else r.score
                ) + delta,
            ))
        adjusted.sort(key=_rank_key)
        if adjusted:
            # Against the question, so asking it again is not another win.
            RECENT_TOPS.record_top(profile_id, adjusted[0].fact.fact_id,
                                   query_key=_question_key(query))
        return adjusted
    except Exception as exc:  # pragma: no cover — advisory, never fatal
        logger.debug("outcome bonus skipped: %s", exc)
        return results


#: Largest lift a memory can earn purely from having been in the last few
#: turns. Additive on ``ranking_score``, deliberately not a multiplier.
#:
#: A multiplier was the first design and it is scale-dependent, which makes the
#: same constant mean different things on two installs. Measured: with a
#: cross-encoder loaded, ranking scores land near [0, 1] and 1.5x is worth about
#: +0.45; with no reranker configured, raw fusion scores top out at
#: ``n_channels/(rrf_k+1)`` — 0.08 at k=60 — and the same 1.5x is worth +0.03.
#: One constant, a fifteen-fold difference in effect, decided by installed
#: optional components. An additive cap behaves the same everywhere.
#:
#: 0.20 sits just above the ceiling on the outcome bonus (0.15), so continuity
#: of attention outranks historical usefulness when the two disagree. Both are
#: bounded, so neither can drag a genuinely poor match to the top.
WM_MAX_BONUS = 0.20

#: Everything the ranking passes may add to one result, summed.
#:
#: Each pass caps its own contribution, which says nothing about the total —
#: and the total is what decides whether a genuinely better match can be
#: displaced. Two passes at 0.15 and 0.20 already reach 0.35; a third added
#: later would raise the ceiling with nothing to notice.
#:
#: Stated here so the invariant is one number rather than an emergent property
#: of however many passes exist, and asserted by test: a result ahead by more
#: than this cannot be overtaken by bias alone, and the sum of the individual
#: caps must not exceed it.
MAX_TOTAL_BIAS = 0.35


def _apply_working_memory_bias(
    results: list, profile_id: str, session_id: str,
) -> list:
    """Lift memories this session was just looking at.

    Retrieval still runs in full and this changes nothing about what was
    retrieved — it reorders what came back. A memory absent from the working
    set is untouched, so an empty working set is exactly the identity function
    and the first turn of every session behaves as before.

    Reads only in-process state: no query, no connection, no file. Deliberately
    does not create a working set — a bias that ran on every recall would
    otherwise register an entry for every session id that ever appears.

    Never raises. Attention is advisory; a recall that cannot be biased is
    still a correct recall.

    WHAT OUTRANKS WHAT
    ------------------
    This is not the last word on the order. An exact lexical hit is promoted to
    the top after every learned layer, including this one, so the full precedence
    is:

        exact lexical match  >  continuity  >  demonstrated usefulness  >  retrieval

    That ordering is deliberate — a memory containing the caller's exact words
    should not sit behind a merely similar one — but it means a boosted memory
    can be moved back off position 1 by a guard that runs later. Two independent
    reviewers raised it as a bug, which is what an undocumented precedence looks
    like from outside, so it is written down here rather than left to be
    rediscovered.
    """
    if not results or not is_conversation(session_id, profile_id):
        return results
    try:
        from superlocalmemory.core.working_memory import peek

        wm = peek(profile_id, session_id)
        if wm is None:
            return results
        held = wm.boost_set()
        if not held:
            return results

        adjusted = []
        for r in results:
            fact = getattr(r, "fact", None)
            if fact is None or fact.fact_id not in held:
                adjusted.append(r)
                continue
            adjusted.append(replace(
                r,
                ranking_score=float(
                    r.ranking_score if r.ranking_score is not None else r.score
                ) + WM_MAX_BONUS,
            ))
        # Same ordering key as the outcome bonus, so the two compose instead of
        # one silently overwriting the other's sort.
        adjusted.sort(key=_rank_key)
        return adjusted
    except Exception as exc:  # pragma: no cover — advisory, never fatal
        logger.debug("working-memory bias skipped: %s", exc)
        return results


def _resettle_shown_after_bias(
    play_sink: dict, profile_id: str, results: list, shown_before: list[str],
) -> None:
    """Correct the play's evidence when the bias changed what was shown.

    A play is settled later by asking whether anything downstream referenced one
    of the memories this query surfaced. That question is answered against a
    stored list, and the list was written before the continuity bias ran — so a
    memory the bias lifted into the answer was absent from it, and an outcome
    citing that memory settled nothing.

    Rewritten only when the set actually differs, which is the uncommon case: a
    cold session, or a warm one whose held memories were already at the top,
    costs nothing. Never raises — a play whose evidence cannot be corrected
    settles on the old list, which is the behaviour before this existed.
    """
    play_id = play_sink.get("play_id")
    learning_db = play_sink.get("learning_db")
    if not play_id or not learning_db:
        return
    try:
        from superlocalmemory.core.working_memory import ADMIT_TOP_N
        from superlocalmemory.learning.bandit import ContextualBandit

        shown_now = [
            r.fact.fact_id for r in results[:ADMIT_TOP_N]
            if getattr(r, "fact", None) is not None
        ]
        # Compare with what the play RECORDED when the sink knows it: a pass
        # that ran in between (the exact-lexical guard) can make the caller's
        # "before" list already equal to the final one while the stored record
        # is stale.
        recorded = play_sink.get("shown")
        baseline = recorded if isinstance(recorded, list) else shown_before
        if set(shown_now) == set(baseline):
            return
        if ContextualBandit(Path(learning_db), profile_id).record_shown(
            play_id, shown_now,
        ):
            play_sink["shown"] = list(shown_now)
    except Exception as exc:  # pragma: no cover — advisory
        logger.debug("shown-set correction skipped: %s", exc)


def _note_order_replaced(
    play_sink: dict, profile_id: str, response: Any, shown_before: list[str],
) -> None:
    """Tell the play its arm did not choose what was shown first.

    Only when the opt-in hosted reorder replaced the order AND that changed the
    shown top. The play still settles and the memories an outcome names are
    still credited; only the arm's posterior is left alone, because the outcome
    describes an order the arm did not produce. Never raises.
    """
    play_id = play_sink.get("play_id")
    learning_db = play_sink.get("learning_db")
    if not play_id or not learning_db:
        return
    try:
        from superlocalmemory.retrieval.jev_rerank import STATUS_LISTWISE

        if getattr(response, "reranker_status", "") != STATUS_LISTWISE:
            return
        if _top_ids(response.results) == list(shown_before):
            return
        from superlocalmemory.learning.bandit import ContextualBandit

        ContextualBandit(Path(learning_db), profile_id).mark_order_replaced(play_id)
    except Exception as exc:  # pragma: no cover — advisory
        logger.debug("order-replaced note skipped: %s", exc)


def _admit_to_working_memory(
    results: list, profile_id: str, session_id: str,
) -> None:
    """Remember what this answer showed, so the next turn is not cold.

    Runs last, on the order the caller will actually see. Admitting the
    pre-rerank order would teach the session something it was never shown.
    """
    if not results or not is_conversation(session_id, profile_id):
        return
    try:
        from superlocalmemory.core.working_memory import ADMIT_TOP_N, get_or_create

        shown = [
            r.fact.fact_id for r in results[:ADMIT_TOP_N]
            if getattr(r, "fact", None) is not None and r.fact.fact_id
        ]
        if shown:
            get_or_create(profile_id, session_id).admit(shown)
    except Exception as exc:  # pragma: no cover — advisory, never fatal
        logger.debug("working-memory admit skipped: %s", exc)


def apply_v2_bandit_ensemble(
    response: RecallResponse,
    query: str,
    profile_id: str,
    query_id: str,
    *,
    learning_db_path: Any = None,
    record_signals: bool = False,
    record_plays: bool = True,
    memory_db_path: Any = None,
    play_sink: dict | None = None,
) -> RecallResponse:
    """Apply contextual bandit + optional LGBM ensemble rerank. Safe on error.

    ``play_sink``, when given, receives the play this call recorded and the store
    it went to. The caller needs both because the answer is not final here: a
    continuity bias runs afterwards and can change which memories end up in
    front of the user, and the play's evidence has to describe what was actually
    shown. Caller-owned rather than returned on the response, matching how the
    retrieval engine hands back its dropped-channel set.

    ``record_signals`` and ``record_plays`` are separate because they are
    separate things, and conflating them is what stopped the bandit learning.

    * A **play** is "the bandit chose arm X for this query". It carries no
      reward — ``bandit_plays.reward`` stays NULL until an authenticated
      outcome settles it. One INSERT.
    * A **signal** is an exposure row per displayed fact, twenty per query. Those
      are what inflated the ranking-phase counter 2,675x, and the enqueue also
      writes canonical/learning state, which is the contention the comment
      below describes.

    ``record_signals=False`` was hardcoded at the only caller on 2026-07-27
    (commit ``cbf7929f``, release 3.8.6). That is the same day
    ``MAX(bandit_arms.last_played_at)`` stops. It took the play recording down
    with the exposure enqueue, so nothing was ever written for the reward proxy
    to settle, and 165 arms have sat at alpha == beta ever since.

    Recording a play is not feedback. It is the ticket the settlement path
    later resolves against evidence, and without it there is no learning loop
    at all.
    """
    import os as _os

    if _os.environ.get("SLM_BANDIT_DISABLED", "0") == "1":
        return response
    if not response.results:
        return response

    try:
        from pathlib import Path as _P

        from superlocalmemory.infra.data_root import state_path
        from superlocalmemory.learning.bandit import ContextualBandit
        from superlocalmemory.learning.ensemble import (
            choose_ensemble,
            ensemble_rerank,
        )
        from superlocalmemory.learning.signals import (
            SignalBatch,
            SignalCandidate,
            enqueue,
        )
        from superlocalmemory.retrieval.engine import apply_channel_weights

        db_path = (_P(learning_db_path) if learning_db_path
                   else state_path("learning.db"))
        if not db_path.exists():
            return response

        # --- 1. bandit.choose ---------------------------------------------
        entity_count = 0
        # Use query_context hints if available on the engine — cheap fallback.
        bandit = ContextualBandit(db_path, profile_id)
        context = {
            "query_type": response.query_type,
            "entity_count": entity_count,
        }
        # Record the play unless explicitly told not to. ``choose_readonly``
        # samples the same arm from a read-only snapshot and returns
        # ``play_id=None``, so taking that branch means this query can never be
        # settled and the arm can never move off its prior.
        # The draw is keyed by the question (never stored): the same question
        # over the same posterior gets the same weights, hence the same order.
        choice = (
            bandit.choose(context, query_id, draw_key=query)
            if record_plays
            else bandit.choose_readonly(context, draw_key=query)
        )

        # --- 2. apply channel weights -------------------------------------
        weighted = apply_channel_weights(list(response.results), choice.weights)

        # --- 3. choose ensemble + load model (optional) -------------------
        active_model = None
        signal_count = 0
        try:
            from superlocalmemory.learning.model_cache import load_active
            db = _ReadOnlyLearningView(db_path)
            signal_count = db.count_signals(profile_id)
            active_model = load_active(db, profile_id)
            # While a retrained candidate is under test, half the recalls are
            # ranked by it and the assignment is recorded.
            from superlocalmemory.learning.shadow_serving import serving_model
            active_model = serving_model(db_path, profile_id, query_id,
                                         active_model)
        except Exception as exc:
            logger.debug("v2 bandit: model/signal load skipped: %s", exc)

        weights = choose_ensemble(signal_count, active_model)

        # --- 4. ensemble rerank -------------------------------------------
        query_context = {
            "query_type": response.query_type,
            "profile_id": profile_id,
            "query_id": query_id,
            "bandit_play_id": choice.play_id,
        }
        try:
            final_results = ensemble_rerank(
                weighted, choice, active_model, weights, query_context,
            )
        except Exception as exc:
            logger.debug("v2 bandit ensemble_rerank skipped: %s", exc)
            final_results = weighted

        # --- 4b. outcome bonus -------------------------------------------
        # Applied AFTER the model score, never as a model feature. The wave
        # plan proposed adding "outcome_score" to FEATURE_NAMES for inference
        # and excluding it from training; that is a shape mismatch —
        # booster.predict needs the columns the model was trained on, and
        # features.py asserts len(FEATURE_NAMES) == FEATURE_DIM == 20 against a
        # live 20-feature model. Applying it here also makes it
        # true by construction: the model cannot learn from a signal it
        # never sees, so there is no self-reinforcing loop to exclude.
        final_results = _apply_outcome_bonus(
            final_results, profile_id, memory_db_path, query=query,
        )

        # Give the play its evidence: which memories this query actually
        # surfaced, so the reward proxy can settle it from a downstream
        # reference instead of falling through to the neutral default. Written
        # after the rerank because that is when the shown order is final.
        if choice.play_id:
            shown_ids = [r.fact.fact_id for r in final_results[:5]]
            recorded = False
            try:
                recorded = bandit.record_shown(choice.play_id, shown_ids)
            except Exception as exc:  # pragma: no cover — never break a recall
                logger.debug("v2 bandit record_shown skipped: %s", exc)
            if play_sink is not None:
                # So the caller can correct this record if the order changes
                # after this function returns — compared against the list
                # actually stored, not one taken after later passes.
                play_sink["play_id"] = choice.play_id
                play_sink["learning_db"] = str(db_path)
                if recorded:
                    play_sink["shown"] = list(shown_ids)

        # Exposure capture for the ranker's training rows. Off unless the
        # caller asks (run_recall asks only in v2-ensemble mode): the enqueue
        # is a put_nowait, the signal worker writes later, and the rows are
        # exposures — the phase counter excludes them (signal_kinds).
        # NOTE: this gates the exposure enqueue ONLY. The play above is
        # recorded regardless — see this function's docstring.
        if record_signals:
            try:
                top20 = final_results[:20]
                candidates = tuple(
                    SignalCandidate(
                        fact_id=r.fact.fact_id,
                        channel_scores=dict(r.channel_scores or {}),
                        cross_encoder_score=None,
                        result_dict={"fact_id": r.fact.fact_id,
                                     "score": r.score},
                    )
                    for r in top20
                )
                enqueue(SignalBatch(
                    profile_id=profile_id,
                    query_id=query_id,
                    query_text=query,
                    candidates=candidates,
                    query_context=query_context,
                ))
            except Exception as exc:
                logger.debug("v2 bandit signal enqueue skipped: %s", exc)

        return RecallResponse(
            query=response.query,
            mode=response.mode,
            results=final_results,
            query_type=response.query_type,
            channel_weights=response.channel_weights,
            total_candidates=response.total_candidates,
            retrieval_time_ms=response.retrieval_time_ms,
            # v3.6.6: preserve evidence-floor signal across ensemble rebuilds.
            no_confident_match=(len(final_results) == 0) and response.no_confident_match,
            # A reorder does not make an incomplete or warming answer whole.
            **_completeness_of(response),
        )
    except Exception as exc:  # pragma: no cover — defensive top-level
        logger.debug("apply_v2_bandit_ensemble skipped: %s", exc)
        return response


def _completeness_of(response: RecallResponse) -> dict:
    """What a rebuilt response must carry over to stay honest.

    The ensemble and agentic steps build a new ``RecallResponse`` around a new
    result list. Without these fields an answer whose channels were warming
    or abandoned came out reading as complete, and its stage timings vanished.
    """
    return {
        "incomplete_channels": tuple(getattr(response, "incomplete_channels", ()) or ()),
        "channel_status": dict(getattr(response, "channel_status", {}) or {}),
        "graph_lag_s": getattr(response, "graph_lag_s", None),
        "stage_ms": dict(getattr(response, "stage_ms", {}) or {}),
        # #150: a project filter that fell back to unfiltered results must
        # still say so after a rebuild.
        "project_scope": getattr(response, "project_scope", None),
        # 4.1.22: a tag filter's note/reason must survive a rebuild too.
        "tag_scope": getattr(response, "tag_scope", None),
        # 4.1.25: the question's evidence features describe the question, not
        # the result list, so a rebuild keeps them.
        "query_evidence": dict(getattr(response, "query_evidence", {}) or {}),
    }


# ---------------------------------------------------------------------------
# run_recall  (was MemoryEngine.recall)
# ---------------------------------------------------------------------------

def resolve_hot_path_fast(fast: bool | None, config: "SLMConfig") -> bool:
    """Resolve the recall ``fast`` flag when a caller leaves it unset (None).

    v3.8.2 client-driven agentic: the agent hot path (CLI / MCP / plugins) is
    consumed by a frontier LLM (Claude Code, Copilot, Codex, …) that reformulates
    multi-hop / low-confidence queries far better than the local Ollama model.
    So an unset ``fast`` defaults to True — skip the internal agentic round and
    let the calling LLM drive refinement — whenever ``retrieval.client_driven_agentic``
    is on (the ship default). An explicit ``True``/``False`` from the caller
    always wins (the dashboard search path passes ``True`` for a snappy list;
    a no-smart-client deployment can pass ``False``). Env override
    ``SLM_HOT_PATH_INTERNAL_AGENTIC=1`` forces internal-agentic-on globally.

    This is the single resolution point: every recall path (HTTP, MCP, CLI,
    in-process adapter) funnels through ``run_recall`` and calls this, so the
    client-driven default is consistent everywhere by construction.
    """
    if fast is not None:
        return bool(fast)
    import os
    rc = getattr(config, "retrieval", None)
    client_driven = bool(getattr(rc, "client_driven_agentic", True))
    if os.environ.get("SLM_HOT_PATH_INTERNAL_AGENTIC") == "1":
        client_driven = False
    return client_driven



def _judge_sufficiency(retrieval_engine: Any, query: str, response: RecallResponse) -> Any:
    """The verdict of the engine's answer check, or None (the previous behaviour).

    Kept for direct callers; ``run_recall`` uses ``answer_check_stage`` itself so
    that the recall's time budget and the per-recall request apply, and so that
    the response says what became of the check.
    """
    from superlocalmemory.core.answer_check_stage import run_answer_check

    return run_answer_check(retrieval_engine, query, response).verdict


def _reorders(judge: Any) -> bool:
    from superlocalmemory.core.answer_check_stage import reorders

    return reorders(judge)


def _rerank_and_judge(judge: Any, query: str, response: RecallResponse) -> Any:
    from superlocalmemory.core.answer_check_stage import rerank_and_judge

    return rerank_and_judge(judge, query, response, None).verdict


def _apply_order(response: RecallResponse, order: tuple[int, ...]) -> bool:
    from superlocalmemory.core.answer_check_stage import apply_order

    return apply_order(response, order)


def _top_ids(results: list) -> list[str]:
    from superlocalmemory.core.working_memory import ADMIT_TOP_N

    return [r.fact.fact_id for r in results[:ADMIT_TOP_N]
            if getattr(r, "fact", None) is not None]


def _engine_learning_db(config: Any) -> Path | None:
    """``<base_dir>/learning.db`` for this engine's config, when it has one."""
    base = getattr(config, "base_dir", None)
    if not base:
        return None
    try:
        return Path(base) / "learning.db"
    except TypeError:  # a stand-in config in tests
        return None


def _metadata_reader(db: Any) -> Any:
    """Tags/project per result for the support gate's optional bonus, when the
    store can be read (a stand-in database cannot)."""
    if not callable(getattr(db, "execute", None)):
        return None
    from superlocalmemory.core.support_stage import metadata_reader
    return metadata_reader(db)


def run_recall(
    query: str,
    profile_id: str,
    mode: Mode | None = None,
    limit: int = 20,
    agent_id: str = "unknown",
    session_id: str | None = None,
    *,
    config: SLMConfig,
    retrieval_engine: Any,
    trust_scorer: Any,
    embedder: Any,
    db: DatabaseManager,
    llm: Any,
    hooks: HookRegistry,
    access_log: Any = None,
    auto_linker: Any = None,
    fast: bool | None = None,
    include_global: bool = False,
    include_shared: bool = False,
    window: str | tuple[str, str] | None = None,
    as_of: str | None = None,
    known_as_of: str | None = None,
    valid_at: str | None = None,
    include_unknown: bool = False,
    answer_check: str | None = None,
    facets: Any = None,
    query_id: str | None = None,
    show_weak: bool = False,
    include_related: bool = False,
) -> RecallResponse:
    """Recall relevant facts for a query.

    4.1.25: when nothing in memory supports the question (the answer-support
    gate, ``core/support_stage``), recall returns no results and a plain
    ``abstention_note``; ``show_weak=True`` returns the weak matches instead,
    each labelled ``weak``; ``include_related=True`` shows up to three close
    memories of a real question, labelled ``related`` (people searching get
    them; agents ask). Pinned memories and explicit filters keep their
    existing behaviour.

    ``query_id`` is this recall's one identity. The engine allocates it and
    returns it on the response; the bandit play, the shadow assignment and
    the exposure features are all recorded under the same id, so an outcome
    that quotes it settles exactly this recall. A fresh id is
    minted only for a direct caller that supplies none.

    Multi-scope: ``include_global`` / ``include_shared`` control which
    scopes participate in retrieval (passed through to retrieval engine).

    ``answer_check``: ``"full"`` (the default — every recall a person or
    agent asked for) or ``"no_reorder"`` (the check without the opt-in
    reorder: the loop gate). Unknown values raise ValueError. A recall that is
    not a question runs inside ``core.answer_check_scope.skip_answer_check()``
    and is never judged.

    Pipeline: retrieval -> agentic sufficiency (if configured) -> post-recall updates.

    ``fast=True`` skips the internal agentic verification round while retaining
    the six local retrieval channels + reranker. ``fast=None`` (unset) resolves
    to the client-driven-agentic default (see ``resolve_hot_path_fast``): the
    agent hot path skips the internal round and delegates refinement to the
    calling LLM. ``fast=False`` forces the internal agentic round.

    ``as_of``: Optional ISO 8601 datetime string for point-in-time time-travel
    recall. When set, the bi-temporal validity filter demotes facts that were
    not yet valid or had already expired at that point. Default ``None`` leaves
    all existing behaviour unchanged.
    """
    m = mode or config.mode
    from superlocalmemory.retrieval.answer_check_status import normalize_request
    answer_check = normalize_request(answer_check)
    # The answer check's budget runs from here: what retrieval and ranking use
    # is what the check cannot.
    import time as _time_budget
    _recall_started = _time_budget.monotonic()

    # v3.8.2: resolve the client-driven-agentic default when a caller left
    # ``fast`` unset (None). After this line ``fast`` is a concrete bool, so
    # the agentic gate below (``if not fast``) behaves identically for every
    # entry point that funnels through here.
    fast = resolve_hot_path_fast(fast, config)

    # v3.5.0 diagnostic: per-stage recall timing under SLM_RECALL_TIMING=1.
    # Zero overhead when the env var is unset. Permanent observability hook.
    import os as _os_t
    import time as _time_t
    _timing = bool(_os_t.environ.get("SLM_RECALL_TIMING"))
    _t0 = _time_t.monotonic()

    def _mark(_label: str) -> None:
        if _timing:
            logger.warning("[RECALL-TIMING] %-22s %.0f ms",
                           _label, (_time_t.monotonic() - _t0) * 1000.0)

    # The interactive path must retain the complete local retrieval contract.
    # Only agentic verification may call a language model during recall; the
    # entity channel resolves names read-only (EntityResolver.lookup).
    extra_disabled = None
    # 4.1.25: extra terms for the keyword channel only (retrieval/query_rewrite);
    # passed only when the rewrite added something, so a stand-in engine that
    # predates the parameter is never asked for it.
    from superlocalmemory.retrieval.query_rewrite import rewrite as _rewrite
    _lexical = _rewrite(query, mode=m, db=db, profile_id=profile_id, llm=llm, config=config)
    response = retrieval_engine.recall(
        query, profile_id, m, limit,
        extra_disabled_channels=extra_disabled,
        include_global=include_global,
        include_shared=include_shared,
        window=window,
        as_of=as_of,
        known_as_of=known_as_of,
        valid_at=valid_at,
        include_unknown=include_unknown,
        # Only when asked: a retrieval stand-in need not know about facets.
        **({"facets": facets} if facets is not None and not facets.empty else {}),
        **({"lexical_query": _lexical.lexical_query} if _lexical.added else {}),
    )
    _mark("retrieval(chan+rerank)")

    _mark("pre-agentic")
    # Agentic sufficiency verification
    # V3.3.19: Only trigger for multi_hop queries in Mode A (rule-based).
    # Single-hop/factual/temporal queries get WORSE with decomposition —
    # sub-query noise dilutes precision. Mode C (LLM) can trigger broadly.
    agentic_rounds = config.retrieval.agentic_max_rounds
    if not fast and agentic_rounds > 0 and response.results:
        max_score = max((r.score for r in response.results), default=0.0)
        has_llm = llm is not None and getattr(llm, "is_available", False)
        should_trigger = (
            response.query_type == "multi_hop"
            or (has_llm and max_score < config.retrieval.agentic_confidence_threshold)
            or (has_llm and len(response.results) < 3)
        )
        if should_trigger:
            try:
                from superlocalmemory.retrieval.agentic import AgenticRetriever
                agentic = AgenticRetriever(
                    confidence_threshold=config.retrieval.agentic_confidence_threshold,
                    db=db,
                )
                enhanced_facts = agentic.retrieve(
                    query=query, profile_id=profile_id,
                    retrieval_engine=retrieval_engine,
                    llm=llm,
                    top_k=limit,
                    query_type=response.query_type,
                    # L2-08: the facet the caller asked for must survive the
                    # internal verification round too — round 2 used to
                    # re-retrieve with none, then REPLACE the already
                    # filtered round-1 response with its unfiltered answer.
                    facets=facets,
                )
                # Replace response results with enhanced facts if we got more
                if len(enhanced_facts) > len(response.results):
                    from superlocalmemory.storage.models import RetrievalResult
                    enhanced_results = []
                    for i, f in enumerate(enhanced_facts):
                        # Look up real trust score for agentic results
                        fact_trust = 0.5
                        if trust_scorer:
                            try:
                                fact_trust = trust_scorer.get_fact_trust(
                                    f.fact_id, profile_id,
                                )
                            except Exception:
                                pass
                        enhanced_results.append(RetrievalResult(
                            fact=f, score=1.0 / (i + 1),
                            channel_scores={"agentic": 1.0},
                            confidence=f.confidence,
                            relevance_score=1.0 / (i + 1),
                            ranking_score=None,
                            memory_confidence=f.confidence,
                            evidence_chain=["agentic_round_2"],
                            trust_score=fact_trust,
                        ))
                    response = RecallResponse(
                        query=query, mode=m, results=enhanced_results[:limit],
                        query_type=response.query_type,
                        channel_weights=response.channel_weights,
                        total_candidates=response.total_candidates + len(enhanced_facts),
                        retrieval_time_ms=response.retrieval_time_ms,
                        # v3.6.6: agentic round-2 may add facts; recompute flag.
                        no_confident_match=(len(enhanced_results[:limit]) == 0)
                        and response.no_confident_match,
                        **_completeness_of(response),
                    )
            except Exception as exc:
                logger.debug("Agentic sufficiency skipped: %s", exc)

    _mark("agentic")
    # S8-ARC-04 (v3.4.22): unified ranking entry point. Single env-var
    # (SLM_RANKING=off|v1|v2|v2-ensemble) controls the pipeline. Legacy
    # SLM_V2_PIPELINE_DISABLED + SLM_BANDIT_DISABLED still honoured for
    # one-release back-compat. Identity when no active model.
    # Bound before the try: everything after it reads the sink, and an empty
    # sink is the "no play recorded" case every reader already handles.
    play_sink: dict = {}
    # What the cross-encoder scored, read before the learned layers rebuild
    # the results; their order may then overturn only a near-tie of it.
    _rerank_anchors = {
        r.fact.fact_id: r.rerank_score for r in response.results
        if getattr(r, "rerank_score", None) is not None
    }
    try:
        import os as _os
        import uuid as _uuid
        query_id = str(query_id or "") or _uuid.uuid4().hex
        mode = _resolve_ranking_mode(_os.environ)
        # Exposure features are recorded (asynchronously, by the signal
        # worker) only when the operator turned adaptive ranking on: they are
        # the trainer's input, and without them no labelled row can exist.
        # An exposure is never counted as feedback.
        response = apply_ranking(
            response, query, profile_id, query_id,
            config=config, pipeline_version=mode,
            record_signals=(mode == "v2-ensemble"),
            # the store THIS recall read from, so the outcome bonus
            # cannot resolve a different one
            memory_db_path=getattr(db, "db_path", None),
            play_sink=play_sink,
            # ...and the learning store beside it: the play, the assignment
            # and the exposure belong to the engine's store, not to whatever
            # the default data root resolves to.
            learning_db_path=_engine_learning_db(config),
        )
    except Exception as exc:
        logger.debug("Ranking pipeline skipped: %s", exc)
    if _rerank_anchors and response.results:
        from superlocalmemory.retrieval.reranker_lead import hold_reranker_lead

        _shown_before_lead = _top_ids(response.results)
        response.results = hold_reranker_lead(response.results, _rerank_anchors)
        _resettle_shown_after_bias(play_sink, profile_id, response.results,
                                   _shown_before_lead)

    # Continuity of attention, applied after every learned layer and outside
    # the ranking-version switch: a session's recent memories bias this answer
    # whether or not the bandit and the model are enabled, because they are
    # unrelated features and coupling them would make one disappear with the
    # other. In-process only — no query is added to the recall path.
    # Only an id that names a conversation drives continuity. The fronts invent
    # one per request (HTTP) or per client (MCP) for bookkeeping, and neither is
    # a conversation: the first would register a working set per dashboard
    # search and evict real ones, the second would pool unrelated clients into
    # a shared set. See core.session_identity.
    _sid = session_id if is_conversation(session_id, profile_id) else ""
    if _sid and response.results:
        from superlocalmemory.core.working_memory import ADMIT_TOP_N as _TOP

        _shown_before = [
            r.fact.fact_id for r in response.results[:_TOP]
            if getattr(r, "fact", None) is not None
        ]
        response.results = _apply_working_memory_bias(
            response.results, profile_id, _sid,
        )
        # The play was recorded with the order as it stood before the line
        # above. If that order changed, its evidence now describes an answer
        # nobody saw.
        _resettle_shown_after_bias(
            play_sink, profile_id, response.results, _shown_before,
        )

    # #150: the session's / caller's project, preferred on the FINAL order:
    # after learned ranking and continuity, which both rewrite ranking scores
    # without knowing the project, and before the exact-hit pin and kind
    # awareness, which keep their guarantees above it. Bounded by
    # project_scope.BOOST on the final ranking key.
    if response.results and facets is not None and getattr(facets, "prefer_project", None):
        from superlocalmemory.retrieval import project_scope as _project_scope

        _shown_before_project = _top_ids(response.results)
        response.results = _project_scope.prefer_in_final_order(
            response.results, _project_scope.preferred_in(db, response.results, facets))
        _resettle_shown_after_bias(play_sink, profile_id, response.results,
                                   _shown_before_project)

    # 4.2.17: a question naming a version is not answered by another version.
    # Ordering only, before the exact-hit pin so that guarantee still holds.
    if response.results:
        from superlocalmemory.retrieval import version_scope

        _shown_before_version = _top_ids(response.results)
        response.results = version_scope.prefer_named_versions(response.results, query)
        _resettle_shown_after_bias(play_sink, profile_id, response.results,
                                   _shown_before_version)

    _preserve_exact_lexical_evidence(response, query)

    # Kind-aware ordering: "what did we decide" favours decisions, "current
    # status" puts the newest confirmed current-state memory about the named
    # subject first. Bounded, read-only, no model; never above the exact hit
    # pinned just above; retrieval.kind_aware=False skips it (retrieval/kind_aware).
    if response.results:
        from superlocalmemory.core.working_memory import ADMIT_TOP_N as _KTOP
        from superlocalmemory.retrieval import kind_aware, source_fidelity_flags
        _shown_before_kind = [r.fact.fact_id for r in response.results[:_KTOP]
                              if getattr(r, "fact", None) is not None]
        response.results = source_fidelity_flags.flag_unverified(kind_aware.apply_for_recall(
            response.results, query, profile_id, getattr(config, "retrieval", None),
            engine=retrieval_engine, db=db), db, profile_id, response=response)
        _resettle_shown_after_bias(play_sink, profile_id, response.results,
                                   _shown_before_kind)

    # A captured prompt is something the user asked an agent to do, not a fact
    # about their work: last, so no earlier layer can lift it back above a
    # memory an agent or the user saved (retrieval/request_memories).
    if response.results:
        from superlocalmemory.retrieval import request_memories

        _shown_before_requests = _top_ids(response.results)
        response.results = request_memories.settle_requests(
            response.results, db, query, response.query_type)
        _resettle_shown_after_bias(play_sink, profile_id, response.results,
                                   _shown_before_requests)
    _mark("learning+ranking")
    # Deliberately no trust, Fisher, retention, lifecycle, popularity, or graph
    # mutation here.  Those state transitions require a separately authenticated
    # positive/negative outcome; merely returning a result is an exposure.

    from superlocalmemory.core.answer_check_stage import build_trace, run_answer_check
    from superlocalmemory.core.score_contract import finalize_score_contract
    _shown_before_judge = _top_ids(response.results)
    _judge_t0 = _time_budget.monotonic()
    outcome = run_answer_check(retrieval_engine, query, response,
                               request=answer_check, recall_started=_recall_started,
                               profile_id=profile_id)
    _judge_t1 = _time_budget.monotonic()
    _mark("sufficiency")
    # The opt-in hosted reordering may have changed what is shown first; the
    # play's evidence must name what the caller actually sees, and the arm the
    # play credits must not be rewarded for an order it did not choose.
    _note_order_replaced(play_sink, profile_id, response, _shown_before_judge)
    _resettle_shown_after_bias(
        play_sink, profile_id, response.results, _shown_before_judge,
    )
    from superlocalmemory.core import support_stage
    from superlocalmemory.core.answer_check_scope import in_context_load
    # A context load is not a question: it gets its memories, under the
    # surface's token cap, instead of an abstention. Nor is "what am I working
    # on": its answer is the newest memories (the recency strategy put them
    # first), which no word of the question appears in, so a word-support test
    # would call every right answer weak and say "That isn't in memory".
    recency = getattr(response, "query_type", "") == "recency"
    gated = support_stage.enabled(config) and not in_context_load() and not recency
    support = support_stage.assess(
        retrieval_engine, query, response, outcome.verdict, profile_id,
        metadata=_metadata_reader(db)) if gated else None
    finalize_score_contract(
        response, verdict=None if recency else outcome.verdict, support=support,
        show_weak=show_weak, include_related=include_related,
        keep_results=recency or support_stage.explicit_filter(
            facets, window, as_of=as_of, known_as_of=known_as_of, valid_at=valid_at))
    response.answer_check_status = outcome.status
    response.answer_check_detail = outcome.detail

    # LLD-00 §3 — stamp HMAC markers on every result so post_tool_outcome_hook
    # can validate fact_ids observed in downstream tool output.
    _apply_markers_to_response(response)

    # Last, on the order the caller receives, so the next turn of this session
    # starts from what was actually shown.
    if _sid:
        _admit_to_working_memory(response.results, profile_id, _sid)

    _mark("TOTAL(fisher+trust+markers)")
    response.answer_check_trace = build_trace(  # 4.1.20: Answer Check history
        retrieval_engine, outcome, response, recall_started=_recall_started,
        judge_started=_judge_t0, judge_ended=_judge_t1, ended=_time_budget.monotonic())
    return response
